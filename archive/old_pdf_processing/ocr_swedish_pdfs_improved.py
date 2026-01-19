#!/usr/bin/env python3
"""
OCR Processing Script for Swedish RSA PDF Documents.

This module processes PDF files that failed regular text extraction (e.g., via
readtext in R). It uses Tesseract OCR with Swedish language support to extract
text from scanned or image-based PDFs.

The script is designed for the Swedish Risk Analysis Text-as-Data Project,
processing municipal RSA (Risk and Vulnerability Analysis) reports.

IMPORTANT:
    - Swedish language support in Tesseract is REQUIRED
    - All outputs are in Parquet format for Python workflows
    - Default: 4 parallel workers for faster processing
    - Recommended: Run from Terminal for best performance

Output Format:
    The primary output (ocr_readtext_format.parquet) matches the readtext R 
    package structure:
    - file: PDF filename (e.g., "RSA Ale 2015 Maskad.pdf")
    - text: Extracted text content

    This enables direct concatenation with converted readtext output before
    Stanza NLP preprocessing.

Example Usage (Terminal - Recommended):
    # Process with default 4 workers
    python ocr_swedish_pdfs_improved.py -i ./pdfs -o ./output -f failed_files.txt

    # Process with 8 workers (faster on multi-core systems)
    python ocr_swedish_pdfs_improved.py -i ./pdfs -o ./output -f failed_files.txt -w 8

Example Usage (Python/Jupyter):
    from pathlib import Path
    from ocr_swedish_pdfs_improved import run_pipeline, ProcessingConfig
    
    config = ProcessingConfig(
        language="swe",
        dpi=300,
        workers=4,
    )
    
    results, summary = run_pipeline(
        input_dir=Path("./pdfs"),
        output_dir=Path("./output"),
        config=config,
    )

Requirements:
    Python packages: pytesseract, pdf2image, pillow, pandas, tqdm, pyarrow
    System: Tesseract OCR with Swedish language pack (tesseract-ocr-swe), Poppler

Author: Swedish Risk Analysis Project
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, Sequence

# =============================================================================
# Constants and Configuration
# =============================================================================

# Default processing parameters
DEFAULT_DPI = 300
DEFAULT_LANGUAGE = "swe"  # Swedish only (all documents are in Swedish)
DEFAULT_MIN_TEXT_LENGTH = 500
DEFAULT_WORKERS = 4  # Parallel processing for faster OCR

# Jupyter Lab detection
IS_JUPYTER = False
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        IS_JUPYTER = True
except (ImportError, NameError):
    pass

# Tesseract configuration
# PSM 1: Automatic page segmentation with OSD (Orientation and Script Detection)
# OEM 3: Default, based on what is available (LSTM + Legacy)
TESSERACT_CONFIG = "--psm 1 --oem 3"

# File patterns
PDF_EXTENSIONS = {".pdf", ".PDF"}

# RSA filename parsing pattern
# Matches: RSA [Municipality] [Year] [Maskad].pdf
RSA_FILENAME_PATTERN = re.compile(
    r"^RSA\s+(?P<municipality>.+?)\s+(?P<year>(?:19|20)\d{2})(?:\s+(?P<maskad>[Mm]askad))?\s*\.pdf$",
    re.IGNORECASE,
)

# Fallback year pattern for non-standard filenames
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


# =============================================================================
# Enums and Data Classes
# =============================================================================


class ProcessingStatus(Enum):
    """Status codes for PDF processing results."""

    SUCCESS = auto()
    FAILED_SHORT = auto()
    FAILED_OCR = auto()
    FAILED_CONVERSION = auto()
    FAILED_IO = auto()
    SKIPPED = auto()

    def is_success(self) -> bool:
        """Check if status indicates successful extraction."""
        return self == ProcessingStatus.SUCCESS


@dataclass
class DocumentMetadata:
    """Metadata extracted from RSA document filename.

    Attributes:
        filename: Original PDF filename
        municipality: Swedish municipality name
        year: Publication year (4-digit string or 'unknown')
        is_masked: Whether document is marked as 'Maskad' (redacted)
    """

    filename: str
    municipality: str = "unknown"
    year: str = "unknown"
    is_masked: bool = False

    @classmethod
    def from_filename(cls, filename: str) -> DocumentMetadata:
        """Parse metadata from RSA document filename.

        Expected format: RSA [municipality] [year] [Maskad].pdf

        Args:
            filename: PDF filename to parse

        Returns:
            DocumentMetadata instance with extracted fields

        Examples:
            >>> DocumentMetadata.from_filename("RSA Ale 2015 Maskad.pdf")
            DocumentMetadata(filename='RSA Ale 2015 Maskad.pdf',
                           municipality='Ale', year='2015', is_masked=True)
            >>> DocumentMetadata.from_filename("RSA Gnosjö 2019.pdf")
            DocumentMetadata(filename='RSA Gnosjö 2019.pdf',
                           municipality='Gnosjö', year='2019', is_masked=False)
        """
        # Try structured pattern first
        match = RSA_FILENAME_PATTERN.match(filename)
        if match:
            return cls(
                filename=filename,
                municipality=match.group("municipality").strip(),
                year=match.group("year"),
                is_masked=match.group("maskad") is not None,
            )

        # Fallback: extract what we can
        name_without_ext = Path(filename).stem
        is_masked = "maskad" in name_without_ext.lower()

        # Remove 'RSA' prefix and 'Maskad' suffix
        clean_name = re.sub(r"^RSA\s*", "", name_without_ext, flags=re.IGNORECASE)
        clean_name = re.sub(r"\s*[Mm]askad\s*$", "", clean_name)

        # Extract year
        year_match = YEAR_PATTERN.search(clean_name)
        year = year_match.group(0) if year_match else "unknown"

        # Remaining text is municipality
        municipality = YEAR_PATTERN.sub("", clean_name).strip()
        municipality = municipality if municipality else "unknown"

        return cls(
            filename=filename,
            municipality=municipality,
            year=year,
            is_masked=is_masked,
        )


@dataclass
class OCRResult:
    """Result of OCR processing for a single document.

    Attributes:
        file: Original PDF filename (for readtext compatibility)
        text: Extracted text content
        status: Processing status code
        page_count: Number of pages processed
        word_count: Approximate word count
        text_length: Character count of extracted text
        error_message: Error description if processing failed
        metadata: Parsed document metadata
        processing_time_seconds: Time taken to process
    """

    file: str
    text: str = ""
    status: ProcessingStatus = ProcessingStatus.SKIPPED
    page_count: int = 0
    word_count: int = 0
    text_length: int = 0
    error_message: str | None = None
    metadata: DocumentMetadata | None = None
    processing_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Compute derived fields after initialization."""
        if self.text:
            self.text_length = len(self.text)
            self.word_count = len(self.text.split())

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        result = {
            "file": self.file,
            "text": self.text,
            "status": self.status.name.lower(),
            "page_count": self.page_count,
            "word_count": self.word_count,
            "text_length": self.text_length,
            "error": self.error_message,
            "processing_time_seconds": self.processing_time_seconds,
        }

        # Add metadata fields
        if self.metadata:
            result["municipality"] = self.metadata.municipality
            result["year"] = self.metadata.year
            result["maskad"] = self.metadata.is_masked

        return result


@dataclass
class ProcessingConfig:
    """Configuration for OCR processing pipeline.

    Attributes:
        language: Tesseract language code(s)
        dpi: Resolution for PDF to image conversion
        min_text_length: Minimum characters for successful extraction
        save_individual_txt: Whether to save per-document text files
        workers: Number of parallel processing workers
        tesseract_config: Additional Tesseract configuration
    """

    language: str = DEFAULT_LANGUAGE
    dpi: int = DEFAULT_DPI
    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH
    save_individual_txt: bool = True
    workers: int = DEFAULT_WORKERS
    tesseract_config: str = TESSERACT_CONFIG

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.dpi < 72 or self.dpi > 600:
            raise ValueError(f"DPI must be between 72 and 600, got {self.dpi}")
        if self.min_text_length < 0:
            raise ValueError(f"min_text_length must be non-negative")
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1")


# =============================================================================
# Dependency Management
# =============================================================================


class DependencyError(Exception):
    """Raised when required dependencies are missing."""

    pass


def check_python_dependencies() -> dict[str, bool]:
    """Check availability of required Python packages.

    Returns:
        Dictionary mapping package names to availability status
    """
    packages = {
        "pytesseract": False,
        "pdf2image": False,
        "PIL": False,
        "pandas": False,
        "tqdm": False,
        "pyarrow": False,
    }

    for package in packages:
        try:
            __import__(package)
            packages[package] = True
        except ImportError:
            pass

    return packages


def verify_dependencies() -> None:
    """Verify all required dependencies are available.

    Raises:
        DependencyError: If any required package is missing
    """
    status = check_python_dependencies()
    missing = [pkg for pkg, available in status.items() if not available]

    if missing:
        install_cmd = f"pip install {' '.join(missing)}"
        raise DependencyError(
            f"Missing required packages: {', '.join(missing)}\n"
            f"Install with: {install_cmd}"
        )


def check_tesseract_installation() -> list[str]:
    """Check Tesseract OCR installation and verify Swedish language support.

    Returns:
        List of available languages

    Raises:
        DependencyError: If Tesseract is not installed or Swedish is unavailable
    """
    import pytesseract

    try:
        version = pytesseract.get_tesseract_version()
        languages = pytesseract.get_languages()

        logging.info(f"Tesseract version: {version}")
        logging.info(f"Available languages: {', '.join(languages)}")

        if "swe" not in languages:
            raise DependencyError(
                "Swedish language pack (swe) not found in Tesseract.\n"
                "All documents are in Swedish - cannot proceed without Swedish support.\n"
                "Install with:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr-swe\n"
                "  macOS: brew install tesseract-lang"
            )

        logging.info("Swedish language support: OK")
        return languages

    except pytesseract.TesseractNotFoundError as e:
        raise DependencyError(
            "Tesseract OCR not found. Install with:\n"
            "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-swe\n"
            "  macOS: brew install tesseract tesseract-lang\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        ) from e


# =============================================================================
# Image Preprocessing
# =============================================================================


class ImagePreprocessor(ABC):
    """Abstract base class for image preprocessing strategies."""

    @abstractmethod
    def process(self, image: "Image.Image") -> "Image.Image":
        """Apply preprocessing to an image.

        Args:
            image: PIL Image to process

        Returns:
            Processed PIL Image
        """
        pass


class GrayscalePreprocessor(ImagePreprocessor):
    """Convert image to grayscale."""

    def process(self, image: "Image.Image") -> "Image.Image":
        """Convert image to grayscale mode.

        Args:
            image: PIL Image (any mode)

        Returns:
            Grayscale PIL Image
        """
        if image.mode != "L":
            return image.convert("L")
        return image


class ThresholdPreprocessor(ImagePreprocessor):
    """Apply binary threshold to image.

    Useful for improving OCR on low-contrast scanned documents.
    """

    def __init__(self, threshold: int = 150):
        """Initialize with threshold value.

        Args:
            threshold: Pixel values above this become white (255),
                      below become black (0)
        """
        self.threshold = threshold

    def process(self, image: "Image.Image") -> "Image.Image":
        """Apply binary thresholding.

        Args:
            image: PIL Image (should be grayscale)

        Returns:
            Thresholded PIL Image
        """
        return image.point(lambda x: 255 if x > self.threshold else 0)


class CompositePreprocessor(ImagePreprocessor):
    """Apply multiple preprocessing steps in sequence."""

    def __init__(self, preprocessors: Sequence[ImagePreprocessor]):
        """Initialize with sequence of preprocessors.

        Args:
            preprocessors: Ordered sequence of preprocessors to apply
        """
        self._preprocessors = list(preprocessors)

    def process(self, image: "Image.Image") -> "Image.Image":
        """Apply all preprocessors in sequence.

        Args:
            image: PIL Image to process

        Returns:
            Processed PIL Image
        """
        for preprocessor in self._preprocessors:
            image = preprocessor.process(image)
        return image


def get_default_preprocessor() -> ImagePreprocessor:
    """Get the default image preprocessing pipeline.

    Returns:
        Configured ImagePreprocessor instance
    """
    return GrayscalePreprocessor()


# =============================================================================
# OCR Engine
# =============================================================================


class OCREngine:
    """Wrapper for Tesseract OCR operations.

    Handles PDF to image conversion and text extraction with
    configurable preprocessing and error handling.
    """

    def __init__(
        self,
        config: ProcessingConfig,
        preprocessor: ImagePreprocessor | None = None,
    ):
        """Initialize OCR engine.

        Args:
            config: Processing configuration
            preprocessor: Image preprocessor (uses default if None)
        """
        self._config = config
        self._preprocessor = preprocessor or get_default_preprocessor()

        # Lazy imports to allow dependency checking first
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image

        self._pytesseract = pytesseract
        self._convert_from_path = convert_from_path
        self._Image = Image

    def extract_text(self, pdf_path: Path) -> tuple[str, int]:
        """Extract text from a PDF using OCR.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (extracted_text, page_count)

        Raises:
            RuntimeError: If PDF conversion fails
            IOError: If file cannot be read
        """
        if not pdf_path.exists():
            raise IOError(f"PDF file not found: {pdf_path}")

        # Convert PDF pages to images
        try:
            images = self._convert_from_path(
                str(pdf_path),
                dpi=self._config.dpi,
                thread_count=1,  # Avoid nested parallelism
            )
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}") from e

        # Process each page
        page_texts = []
        for page_num, image in enumerate(images, start=1):
            page_text = self._ocr_single_page(image, page_num)
            page_texts.append(page_text)

        full_text = "\n\n".join(page_texts)
        return full_text, len(images)

    def _ocr_single_page(
        self,
        image: "Image.Image",
        page_num: int,
    ) -> str:
        """Perform OCR on a single page image.

        Args:
            image: PIL Image of the page
            page_num: Page number for logging

        Returns:
            Extracted text with page header
        """
        # Apply preprocessing
        processed_image = self._preprocessor.process(image)

        try:
            text = self._pytesseract.image_to_string(
                processed_image,
                lang=self._config.language,
                config=self._config.tesseract_config,
            )
            return f"--- Page {page_num} ---\n{text}"

        except Exception as e:
            logging.warning(f"OCR failed on page {page_num}: {e}")
            return f"--- Page {page_num} ---\n[OCR ERROR: {e}]"


# =============================================================================
# File Discovery
# =============================================================================


def discover_pdf_files(
    input_dir: Path,
    file_list: Path | None = None,
) -> tuple[list[Path], list[str]]:
    """Discover PDF files to process.

    Args:
        input_dir: Directory containing PDF files
        file_list: Optional text file with specific filenames

    Returns:
        Tuple of (found_files, missing_filenames)

    Raises:
        ValueError: If input_dir doesn't exist
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    if file_list and file_list.exists():
        return _discover_from_list(input_dir, file_list)
    else:
        return _discover_from_directory(input_dir)


def _discover_from_list(
    input_dir: Path,
    file_list: Path,
) -> tuple[list[Path], list[str]]:
    """Discover PDFs from a file list.

    Args:
        input_dir: Base directory for PDF files
        file_list: Text file with filenames (one per line)

    Returns:
        Tuple of (found_files, missing_filenames)
    """
    with open(file_list, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f if line.strip()]

    found = []
    missing = []

    for filename in filenames:
        pdf_path = input_dir / filename
        if pdf_path.exists():
            found.append(pdf_path)
        else:
            missing.append(filename)

    return found, missing


def _discover_from_directory(input_dir: Path) -> tuple[list[Path], list[str]]:
    """Discover all PDFs in a directory.

    Args:
        input_dir: Directory to scan

    Returns:
        Tuple of (found_files, empty_list)
    """
    pdf_files = [
        f for f in input_dir.iterdir() if f.suffix in PDF_EXTENSIONS and f.is_file()
    ]
    return sorted(pdf_files), []


# =============================================================================
# Document Processor
# =============================================================================


def process_single_document(
    pdf_path: Path,
    config: ProcessingConfig,
    output_dir: Path | None = None,
) -> OCRResult:
    """Process a single PDF document.

    This function is designed to be called in parallel workers.

    Args:
        pdf_path: Path to the PDF file
        config: Processing configuration
        output_dir: Optional directory for individual text files

    Returns:
        OCRResult with processing outcome
    """
    import time

    start_time = time.perf_counter()

    # Parse metadata from filename
    metadata = DocumentMetadata.from_filename(pdf_path.name)

    result = OCRResult(
        file=pdf_path.name,
        metadata=metadata,
    )

    try:
        # Initialize OCR engine
        engine = OCREngine(config)

        # Extract text
        text, page_count = engine.extract_text(pdf_path)

        result.text = text
        result.page_count = page_count
        result.status = ProcessingStatus.SUCCESS

        # Check minimum length threshold
        if result.text_length < config.min_text_length:
            result.status = ProcessingStatus.FAILED_SHORT

        # Save individual text file if requested
        if (
            output_dir
            and config.save_individual_txt
            and result.status == ProcessingStatus.SUCCESS
        ):
            _save_text_file(output_dir, pdf_path.stem, text)

    except IOError as e:
        result.status = ProcessingStatus.FAILED_IO
        result.error_message = str(e)
        logging.error(f"IO error processing {pdf_path.name}: {e}")

    except RuntimeError as e:
        result.status = ProcessingStatus.FAILED_CONVERSION
        result.error_message = str(e)
        logging.error(f"Conversion error processing {pdf_path.name}: {e}")

    except Exception as e:
        result.status = ProcessingStatus.FAILED_OCR
        result.error_message = str(e)
        logging.exception(f"Unexpected error processing {pdf_path.name}")

    finally:
        result.processing_time_seconds = time.perf_counter() - start_time

    return result


def _save_text_file(output_dir: Path, stem: str, text: str) -> None:
    """Save extracted text to individual file.

    Args:
        output_dir: Output directory
        stem: Filename without extension
        text: Text content to save
    """
    txt_path = output_dir / f"{stem}.txt"
    try:
        txt_path.write_text(text, encoding="utf-8")
    except IOError as e:
        logging.warning(f"Failed to save text file {txt_path}: {e}")


# =============================================================================
# Batch Processor
# =============================================================================


class BatchProcessor:
    """Orchestrates batch processing of multiple PDF documents.

    Supports both sequential and parallel processing modes.
    
    Note:
        Parallel processing uses multiprocessing which may have limitations
        in Jupyter notebooks. Sequential processing is recommended for Jupyter.
    """

    def __init__(
        self,
        config: ProcessingConfig,
        output_dir: Path,
    ):
        """Initialize batch processor.

        Args:
            config: Processing configuration
            output_dir: Directory for output files
        """
        self._config = config
        self._output_dir = output_dir
        self._results: list[OCRResult] = []

    def process_files(self, pdf_files: Sequence[Path]) -> list[OCRResult]:
        """Process multiple PDF files.

        Args:
            pdf_files: Sequence of PDF paths to process

        Returns:
            List of OCRResult objects
        """
        # Use tqdm.auto for compatibility with various environments
        from tqdm.auto import tqdm

        self._results = []

        if self._config.workers == 1:
            # Sequential processing
            for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
                result = process_single_document(
                    pdf_path,
                    self._config,
                    self._output_dir,
                )
                self._results.append(result)
                self._log_result(result)
        else:
            # Parallel processing (recommended for large batches)
            self._results = self._process_parallel(pdf_files)

        return self._results

    def _process_parallel(self, pdf_files: Sequence[Path]) -> list[OCRResult]:
        """Process files in parallel using ProcessPoolExecutor.

        Args:
            pdf_files: Sequence of PDF paths

        Returns:
            List of OCRResult objects
            
        Note:
            May not work correctly in Jupyter notebooks due to 
            multiprocessing limitations. Use sequential processing instead.
        """
        from tqdm.auto import tqdm

        results = []

        with ProcessPoolExecutor(max_workers=self._config.workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(
                    process_single_document,
                    pdf_path,
                    self._config,
                    self._output_dir,
                ): pdf_path
                for pdf_path in pdf_files
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_path),
                total=len(pdf_files),
                desc="Processing PDFs",
            ):
                try:
                    result = future.result()
                    results.append(result)
                    self._log_result(result)
                except Exception as e:
                    pdf_path = future_to_path[future]
                    logging.error(f"Worker failed for {pdf_path.name}: {e}")
                    results.append(
                        OCRResult(
                            file=pdf_path.name,
                            status=ProcessingStatus.FAILED_OCR,
                            error_message=str(e),
                            metadata=DocumentMetadata.from_filename(pdf_path.name),
                        )
                    )

        return results

    def _log_result(self, result: OCRResult) -> None:
        """Log processing result.

        Args:
            result: OCRResult to log
        """
        if result.status.is_success():
            logging.info(
                f"Success: {result.file} "
                f"({result.page_count} pages, {result.word_count} words)"
            )
        else:
            logging.warning(
                f"Failed: {result.file} "
                f"(status={result.status.name}, error={result.error_message})"
            )


# =============================================================================
# Output Generation
# =============================================================================


@dataclass
class ProcessingSummary:
    """Summary statistics for batch processing."""

    total_files: int = 0
    successful: int = 0
    failed_short: int = 0
    failed_other: int = 0
    total_pages: int = 0
    total_words: int = 0
    total_time_seconds: float = 0.0

    @classmethod
    def from_results(cls, results: Sequence[OCRResult]) -> ProcessingSummary:
        """Compute summary from processing results.

        Args:
            results: Sequence of OCRResult objects

        Returns:
            ProcessingSummary instance
        """
        summary = cls(total_files=len(results))

        for result in results:
            if result.status == ProcessingStatus.SUCCESS:
                summary.successful += 1
                summary.total_pages += result.page_count
                summary.total_words += result.word_count
            elif result.status == ProcessingStatus.FAILED_SHORT:
                summary.failed_short += 1
            else:
                summary.failed_other += 1

            summary.total_time_seconds += result.processing_time_seconds

        return summary

    def log(self) -> None:
        """Log summary statistics."""
        logging.info("=" * 60)
        logging.info("Processing Summary")
        logging.info("=" * 60)
        logging.info(f"Total files processed: {self.total_files}")
        logging.info(f"Successful extractions: {self.successful}")
        logging.info(f"Failed (too short): {self.failed_short}")
        logging.info(f"Failed (other errors): {self.failed_other}")
        logging.info(f"Total pages processed: {self.total_pages}")
        logging.info(f"Total words extracted: {self.total_words:,}")
        logging.info(f"Total processing time: {self.total_time_seconds:.1f}s")


class OutputWriter:
    """Handles writing processing results to various formats.
    
    All outputs are in Parquet format for efficient Python workflows.
    """

    def __init__(self, output_dir: Path, min_text_length: int):
        """Initialize output writer.

        Args:
            output_dir: Directory for output files
            min_text_length: Threshold for success classification
        """
        self._output_dir = output_dir
        self._min_text_length = min_text_length

    def write_all(self, results: Sequence[OCRResult]) -> dict[str, Path]:
        """Write all output files.

        Args:
            results: Processing results to write

        Returns:
            Dictionary mapping output type to file path
        """
        import pandas as pd

        # Convert results to DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])

        output_paths = {}

        # 1. readtext-compatible format (primary output) - Parquet only
        output_paths.update(self._write_readtext_format(df))

        # 2. Full results with metadata
        output_paths.update(self._write_full_results(df))

        # 3. Summary CSV (without text)
        output_paths.update(self._write_summary(df))

        # 4. Failed files list
        output_paths.update(self._write_failed_list(df))

        return output_paths

    def _write_readtext_format(self, df: "pd.DataFrame") -> dict[str, Path]:
        """Write readtext-compatible output file (Parquet format).

        Args:
            df: Full results DataFrame

        Returns:
            Dictionary of output paths
        """
        # Filter to successful only, keep only file and text columns
        df_readtext = df[df["status"] == "success"][["file", "text"]].copy()
        df_readtext = df_readtext.reset_index(drop=True)

        paths = {}

        # Parquet format (primary output for Python workflows)
        parquet_path = self._output_dir / "ocr_readtext_format.parquet"
        df_readtext.to_parquet(parquet_path, index=False)
        paths["readtext_parquet"] = parquet_path
        logging.info(f"Saved Parquet (readtext format): {parquet_path}")
        logging.info(f"  → {len(df_readtext)} documents ready for Stanza pipeline")

        return paths

    def _write_full_results(self, df: "pd.DataFrame") -> dict[str, Path]:
        """Write full results with all metadata.

        Args:
            df: Full results DataFrame

        Returns:
            Dictionary of output paths
        """
        parquet_path = self._output_dir / "ocr_full_results.parquet"
        df.to_parquet(parquet_path, index=False)
        logging.info(f"Saved full results: {parquet_path}")
        return {"full_results": parquet_path}

    def _write_summary(self, df: "pd.DataFrame") -> dict[str, Path]:
        """Write summary CSV without text content.

        Args:
            df: Full results DataFrame

        Returns:
            Dictionary of output paths
        """
        csv_path = self._output_dir / "ocr_results_summary.csv"
        df_summary = df.drop(columns=["text"], errors="ignore")
        df_summary.to_csv(csv_path, index=False, encoding="utf-8")
        logging.info(f"Saved summary CSV: {csv_path}")
        return {"summary_csv": csv_path}

    def _write_failed_list(self, df: "pd.DataFrame") -> dict[str, Path]:
        """Write list of files that still failed.

        Args:
            df: Full results DataFrame

        Returns:
            Dictionary of output paths
        """
        failed_files = df[df["status"] != "success"]["file"].tolist()

        if not failed_files:
            return {}

        failed_path = self._output_dir / "still_failed_files.txt"
        failed_path.write_text("\n".join(failed_files) + "\n", encoding="utf-8")
        logging.info(f"Saved failed files list: {failed_path}")
        return {"failed_list": failed_path}


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(
    output_dir: Path | None = None,
    level: int = logging.INFO,
    jupyter_mode: bool = False,
) -> logging.Logger:
    """Configure logging to file and console.

    Args:
        output_dir: Directory for log file (None for no file logging)
        level: Logging level
        jupyter_mode: If True, configure for Jupyter notebook display

    Returns:
        Configured logger instance
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    handlers = []
    
    # Console/Jupyter handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    handlers.append(console_handler)
    
    # File handler (if output_dir provided)
    if output_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"ocr_processing_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    return logging.getLogger(__name__)


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    file_list: Path | None = None,
    config: ProcessingConfig | None = None,
) -> tuple[list[OCRResult], ProcessingSummary]:
    """Run the complete OCR processing pipeline.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for output files
        file_list: Optional text file with specific filenames
        config: Processing configuration (uses defaults if None)

    Returns:
        Tuple of (results_list, processing_summary)

    Raises:
        DependencyError: If required dependencies are missing (including Swedish)
        ValueError: If input validation fails
    """
    # Use default config if not provided
    if config is None:
        config = ProcessingConfig()

    # Validate configuration
    config.validate()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging (with Jupyter detection)
    logger = setup_logging(output_dir, jupyter_mode=IS_JUPYTER)
    logger.info("Starting OCR processing pipeline")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config}")
    
    if IS_JUPYTER:
        logger.info("Jupyter environment detected")

    # Verify dependencies (will raise if Swedish not available)
    verify_dependencies()
    languages = check_tesseract_installation()  # Raises if Swedish missing

    # Discover PDF files
    pdf_files, missing_files = discover_pdf_files(input_dir, file_list)

    if missing_files:
        logger.warning(f"Missing {len(missing_files)} files from list:")
        for filename in missing_files[:10]:
            logger.warning(f"  - {filename}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")

    if not pdf_files:
        logger.error("No PDF files found to process")
        return [], ProcessingSummary()

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process files
    processor = BatchProcessor(config, output_dir)
    results = processor.process_files(pdf_files)

    # Generate summary
    summary = ProcessingSummary.from_results(results)
    summary.log()

    # Write output files
    writer = OutputWriter(output_dir, config.min_text_length)
    output_paths = writer.write_all(results)

    logger.info("Pipeline complete")

    return results, summary


# =============================================================================
# CLI Interface
# =============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="ocr_swedish_pdfs",
        description="OCR processing for Swedish RSA PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all PDFs in a directory
    python ocr_swedish_pdfs.py -i ./pdfs -o ./output

    # Process specific files from a list
    python ocr_swedish_pdfs.py -i ./pdfs -o ./output -f failed_files.txt

    # Use only Swedish language (faster, recommended)
    python ocr_swedish_pdfs.py -i ./pdfs -o ./output --lang swe

    # Check Tesseract installation only
    python ocr_swedish_pdfs.py --check-only

Output Files:
    ocr_readtext_format.parquet - Primary output matching readtext format (file, text)
    ocr_full_results.parquet    - Full results with metadata
    ocr_results_summary.csv     - Summary without text content
    still_failed_files.txt      - Files that still failed OCR

Note:
    Swedish language support in Tesseract is REQUIRED. The script will fail
    if tesseract-ocr-swe is not installed.
    
    For Jupyter Lab usage, import and use run_pipeline() directly instead.
        """,
    )

    parser.add_argument(
        "--input_dir",
        "-i",
        type=Path,
        help="Directory containing PDF files",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        help="Directory to save output files",
    )

    parser.add_argument(
        "--file_list",
        "-f",
        type=Path,
        default=None,
        help="Text file with specific PDF filenames to process (one per line)",
    )

    parser.add_argument(
        "--lang",
        "-l",
        type=str,
        default=DEFAULT_LANGUAGE,
        help=f"Tesseract language code(s) (default: {DEFAULT_LANGUAGE})",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"DPI for PDF to image conversion (default: {DEFAULT_DPI})",
    )

    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_TEXT_LENGTH,
        help=f"Minimum text length for success (default: {DEFAULT_MIN_TEXT_LENGTH})",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )

    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Do not save individual text files for each PDF",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check Tesseract installation, do not process files",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    return parser


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle check-only mode
    if args.check_only:
        try:
            verify_dependencies()
            languages = check_tesseract_installation()
            print("\nDependency check passed!")
            print(f"Swedish language support: Yes")
            print(f"Available languages: {', '.join(languages)}")
            return 0
        except DependencyError as e:
            print(f"\nDependency check failed:\n{e}")
            return 1

    # Validate required arguments
    if not args.input_dir:
        parser.error("--input_dir is required unless using --check-only")
    if not args.output_dir:
        parser.error("--output_dir is required unless using --check-only")

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    # Build configuration
    config = ProcessingConfig(
        language=args.lang,
        dpi=args.dpi,
        min_text_length=args.min_length,
        save_individual_txt=not args.no_individual,
        workers=args.workers,
    )

    try:
        # Run pipeline
        results, summary = run_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_list=args.file_list,
            config=config,
        )

        # Print final summary
        print(f"\n{'=' * 60}")
        print("OCR Processing Complete!")
        print(f"{'=' * 60}")
        print(f"Output directory: {args.output_dir}")
        print(f"Files processed: {summary.total_files}")
        print(f"Successful: {summary.successful}")
        print(f"Failed (short): {summary.failed_short}")
        print(f"Failed (other): {summary.failed_other}")

        return 0 if summary.successful > 0 else 1

    except DependencyError as e:
        print(f"\nDependency error:\n{e}")
        return 1

    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        return 1

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 130

    except Exception as e:
        logging.exception("Unexpected error during processing")
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
