#!/usr/bin/env python3
"""
Enhanced PDF Reader for Swedish RSA Documents

This script handles multiple PDF formats including:
1. Standard PDFs with embedded text (using pypdf, pdfplumber, pdfminer)
2. ZIP archives masquerading as PDFs (containing JPEG scans)
3. Image-based PDFs requiring OCR

The script provides:
- Automatic format detection
- Multiple extraction methods with fallback
- OCR support for scanned documents
- Output format compatible with readtext R package
- Detailed logging and error handling

Usage:
    python pdf_reader_enhanced.py --input-dir /path/to/pdfs --output-dir /path/to/output
    python pdf_reader_enhanced.py --input-dir /path/to/pdfs --output-dir /path/to/output --ocr
    python pdf_reader_enhanced.py --help

Output:
    - pdf_texts.parquet: Successfully extracted documents (file, text columns)
      Ready to use with preprocessing.py:
      $ python preprocessing.py --input pdf_texts.parquet --output sentences_lemmatized.parquet
    - failed_files.txt: Files that need manual review
    - processing_log.json: Detailed processing information

Author: Swedish Risk Analysis Text-as-Data Project
Date: 2026-01-14
"""

import argparse
import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

# PDF libraries
import pypdf
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract

# Image and OCR (optional)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Data handling
import pandas as pd


# =============================================================================
# Configuration and Data Classes
# =============================================================================

@dataclass
class ExtractionResult:
    """Result of text extraction from a single file."""
    filename: str
    success: bool
    text: str = ""
    method: str = ""  # pypdf, pdfplumber, pdfminer, zip_ocr, etc.
    file_type: str = ""  # pdf, zip, unknown
    page_count: int = 0
    char_count: int = 0
    word_count: int = 0
    error_message: str = ""
    processing_time: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.text:
            self.char_count = len(self.text)
            self.word_count = len(self.text.split())


@dataclass
class ProcessingConfig:
    """Configuration for PDF processing."""
    min_text_length: int = 500  # Minimum chars for success
    use_ocr: bool = False  # Whether to use OCR for image-based content
    ocr_language: str = "swe"  # Tesseract language
    ocr_dpi: int = 300  # DPI for image conversion
    verbose: bool = False


# =============================================================================
# File Type Detection
# =============================================================================

def detect_file_type(file_path: Path) -> str:
    """
    Detect the actual file type regardless of extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        File type: 'pdf', 'zip', or 'unknown'
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(10)
            
        # PDF signature
        if header.startswith(b'%PDF'):
            return 'pdf'
        
        # ZIP signature (PK\x03\x04)
        if header.startswith(b'PK\x03\x04'):
            return 'zip'
            
        return 'unknown'
    except Exception:
        return 'unknown'


# =============================================================================
# Standard PDF Extraction Methods
# =============================================================================

def extract_with_pypdf(file_path: Path) -> Tuple[bool, str, str]:
    """
    Extract text using pypdf library.
    
    Returns:
        (success, text, error_message)
    """
    try:
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            
            if reader.is_encrypted:
                return False, "", "PDF is encrypted"
            
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = "\n\n".join(text_parts)
            return True, full_text, ""
            
    except Exception as e:
        return False, "", str(e)


def extract_with_pdfplumber(file_path: Path) -> Tuple[bool, str, str]:
    """
    Extract text using pdfplumber library.
    
    Returns:
        (success, text, error_message)
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = "\n\n".join(text_parts)
            return True, full_text, ""
            
    except Exception as e:
        return False, "", str(e)


def extract_with_pdfminer(file_path: Path) -> Tuple[bool, str, str]:
    """
    Extract text using pdfminer library.
    
    Returns:
        (success, text, error_message)
    """
    try:
        text = pdfminer_extract(str(file_path))
        return True, text, ""
    except Exception as e:
        return False, "", str(e)


# =============================================================================
# PDF to Image OCR (for scanned PDFs)
# =============================================================================

def extract_from_scanned_pdf(file_path: Path, config: ProcessingConfig) -> Tuple[bool, str, str, int]:
    """
    Extract text from scanned PDF using OCR.
    
    For PDFs that have no text layer - converts pages to images and performs OCR.
    
    Args:
        file_path: Path to PDF file
        config: Processing configuration
        
    Returns:
        (success, text, error_message, page_count)
    """
    if not OCR_AVAILABLE:
        return False, "", "OCR libraries not available (PIL/pytesseract required)", 0
    
    if not config.use_ocr:
        return False, "", "Scanned PDF detected but OCR is disabled", 0
    
    try:
        # Try PyMuPDF first (best option - easy to install, good quality)
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            page_count = len(doc)
            
            if page_count == 0:
                return False, "", "PDF has no pages", 0
            
            page_texts = []
            
            for page_num in range(page_count):
                try:
                    page = doc[page_num]
                    
                    # Render page to image
                    pix = page.get_pixmap(dpi=config.ocr_dpi)
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(
                        img,
                        lang=config.ocr_language,
                        config=f'--psm 1 --dpi {config.ocr_dpi}'
                    )
                    
                    if text.strip():
                        page_texts.append(text)
                        
                except Exception as e:
                    logging.warning(f"Failed to OCR page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            
            if not page_texts:
                return False, "", "OCR produced no text", page_count
            
            full_text = "\n\n".join(page_texts)
            return True, full_text, "", page_count
            
        except ImportError:
            logging.debug("PyMuPDF not available, trying pdf2image")
            
            # Try pdf2image as alternative
            try:
                from pdf2image import convert_from_path
                
                images = convert_from_path(
                    file_path,
                    dpi=config.ocr_dpi,
                    fmt='jpeg'
                )
                
                page_texts = []
                for i, img in enumerate(images, 1):
                    try:
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Perform OCR
                        text = pytesseract.image_to_string(
                            img,
                            lang=config.ocr_language,
                            config=f'--psm 1 --dpi {config.ocr_dpi}'
                        )
                        
                        if text.strip():
                            page_texts.append(text)
                    except Exception as e:
                        logging.warning(f"Failed to OCR page {i}: {e}")
                        continue
                
                if not page_texts:
                    return False, "", "OCR produced no text", len(images)
                
                full_text = "\n\n".join(page_texts)
                return True, full_text, "", len(images)
                
            except ImportError:
                return False, "", "Neither PyMuPDF nor pdf2image available. Install with: pip install pymupdf", 0
            except Exception as e:
                return False, "", f"pdf2image failed: {str(e)}", 0
                
    except Exception as e:
        return False, "", str(e), 0


# =============================================================================
# ZIP-based PDF Extraction (with OCR)
# =============================================================================

def extract_from_zip_pdf(file_path: Path, config: ProcessingConfig) -> Tuple[bool, str, str, int]:
    """
    Extract text from ZIP archive masquerading as PDF.
    These files contain JPEG images that need OCR.
    
    Args:
        file_path: Path to ZIP file
        config: Processing configuration
        
    Returns:
        (success, text, error_message, page_count)
    """
    if not OCR_AVAILABLE and config.use_ocr:
        return False, "", "OCR libraries not available (PIL/pytesseract required)", 0
    
    if not config.use_ocr:
        return False, "", "ZIP-based PDF detected but OCR is disabled", 0
    
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # List all files
            file_list = zip_ref.namelist()
            
            # Filter for JPEG images and sort by number
            jpeg_files = sorted(
                [f for f in file_list if f.lower().endswith(('.jpeg', '.jpg'))],
                key=lambda x: int(''.join(filter(str.isdigit, x)) or '0')
            )
            
            if not jpeg_files:
                return False, "", "No JPEG files found in ZIP archive", 0
            
            # Extract and OCR each image
            page_texts = []
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                for jpeg_file in jpeg_files:
                    try:
                        # Extract image
                        image_data = zip_ref.read(jpeg_file)
                        image_path = temp_path / jpeg_file
                        image_path.write_bytes(image_data)
                        
                        # Open and OCR
                        with Image.open(image_path) as img:
                            # Convert to RGB if needed
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Perform OCR
                            text = pytesseract.image_to_string(
                                img,
                                lang=config.ocr_language,
                                config=f'--psm 1 --dpi {config.ocr_dpi}'
                            )
                            
                            if text.strip():
                                page_texts.append(text)
                                
                    except Exception as e:
                        logging.warning(f"Failed to OCR {jpeg_file}: {e}")
                        continue
            
            if not page_texts:
                return False, "", "OCR produced no text", len(jpeg_files)
            
            full_text = "\n\n".join(page_texts)
            return True, full_text, "", len(jpeg_files)
            
    except Exception as e:
        return False, "", str(e), 0


# =============================================================================
# Main Extraction Orchestrator
# =============================================================================

def extract_text_from_file(file_path: Path, config: ProcessingConfig) -> ExtractionResult:
    """
    Extract text from a file using the best available method.
    
    This function:
    1. Detects the actual file type
    2. Tries multiple extraction methods in order
    3. Returns the first successful extraction
    
    Args:
        file_path: Path to PDF file
        config: Processing configuration
        
    Returns:
        ExtractionResult with extraction details
    """
    start_time = time.time()
    filename = file_path.name
    
    # Detect file type
    file_type = detect_file_type(file_path)
    
    # Handle ZIP files (scanned PDFs)
    if file_type == 'zip':
        success, text, error, page_count = extract_from_zip_pdf(file_path, config)
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            filename=filename,
            success=success and len(text) >= config.min_text_length,
            text=text if success else "",
            method="zip_ocr" if success else "zip_failed",
            file_type=file_type,
            page_count=page_count,
            error_message=error,
            processing_time=processing_time
        )
    
    # Handle standard PDFs - try multiple methods
    if file_type == 'pdf':
        methods = [
            ('pypdf', extract_with_pypdf),
            ('pdfplumber', extract_with_pdfplumber),
            ('pdfminer', extract_with_pdfminer),
        ]
        
        # Try standard text extraction methods
        for method_name, method_func in methods:
            success, text, error = method_func(file_path)
            
            # Check if extraction was successful and sufficient
            if success and len(text) >= config.min_text_length:
                processing_time = time.time() - start_time
                return ExtractionResult(
                    filename=filename,
                    success=True,
                    text=text,
                    method=method_name,
                    file_type=file_type,
                    error_message="",
                    processing_time=processing_time
                )
        
        # All standard methods failed or produced insufficient text
        # Try OCR as fallback if enabled (for scanned PDFs)
        if config.use_ocr and OCR_AVAILABLE:
            logging.info(f"  Standard extraction insufficient, trying OCR fallback...")
            success, text, error, page_count = extract_from_scanned_pdf(file_path, config)
            
            if success and len(text) >= config.min_text_length:
                processing_time = time.time() - start_time
                return ExtractionResult(
                    filename=filename,
                    success=True,
                    text=text,
                    method="scanned_pdf_ocr",
                    file_type=file_type,
                    page_count=page_count,
                    error_message="",
                    processing_time=processing_time
                )
            else:
                # OCR also failed
                processing_time = time.time() - start_time
                return ExtractionResult(
                    filename=filename,
                    success=False,
                    text="",
                    method="all_failed_including_ocr",
                    file_type=file_type,
                    page_count=page_count,
                    error_message=error or f"All methods including OCR failed or produced <{config.min_text_length} chars",
                    processing_time=processing_time
                )
        
        # OCR not available or not enabled
        processing_time = time.time() - start_time
        return ExtractionResult(
            filename=filename,
            success=False,
            text="",
            method="all_failed",
            file_type=file_type,
            error_message=f"All extraction methods failed or produced <{config.min_text_length} chars" + 
                         (" (OCR disabled - use --ocr flag)" if not config.use_ocr else " (OCR not available)"),
            processing_time=processing_time
        )
    
    # Unknown file type
    processing_time = time.time() - start_time
    return ExtractionResult(
        filename=filename,
        success=False,
        text="",
        method="none",
        file_type=file_type,
        error_message=f"Unknown file type: {file_type}",
        processing_time=processing_time
    )


# =============================================================================
# Batch Processing
# =============================================================================

def process_directory(
    input_dir: Path,
    output_dir: Path,
    config: ProcessingConfig
) -> Tuple[List[ExtractionResult], Dict]:
    """
    Process all PDF files in a directory.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for output files
        config: Processing configuration
        
    Returns:
        (results, summary_stats)
    """
    # Setup logging
    # Don't create processing.log - it gets too large with pdfminer DEBUG output
    # Only log to console, keep it concise
    logging.basicConfig(
        level=logging.INFO,  # Always INFO for root logger
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Silence verbose third-party libraries
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('pypdf').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    # Set our logger to DEBUG only if verbose requested
    if config.verbose:
        logger.setLevel(logging.DEBUG)
    logger.info(f"Starting PDF processing")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"OCR enabled: {config.use_ocr}")
    logger.info(f"Min text length: {config.min_text_length}")
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        logger.warning("No PDF files found!")
        return [], {}
    
    # Process each file
    results = []
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        result = extract_text_from_file(pdf_file, config)
        results.append(result)
        
        if result.success:
            logger.info(f"  ✓ Success using {result.method} ({result.char_count} chars)")
        else:
            logger.warning(f"  ✗ Failed: {result.error_message}")
    
    # Calculate summary statistics
    summary = {
        'total_files': len(results),
        'successful': sum(1 for r in results if r.success),
        'failed': sum(1 for r in results if not r.success),
        'success_rate': sum(1 for r in results if r.success) / len(results) * 100 if results else 0,
        'methods_used': {},
        'file_types': {},
        'total_chars_extracted': sum(r.char_count for r in results if r.success),
        'total_processing_time': sum(r.processing_time for r in results),
    }
    
    # Count by method and file type
    for result in results:
        summary['methods_used'][result.method] = summary['methods_used'].get(result.method, 0) + 1
        summary['file_types'][result.file_type] = summary['file_types'].get(result.file_type, 0) + 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {summary['total_files']}")
    logger.info(f"Successful: {summary['successful']} ({summary['success_rate']:.1f}%)")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Total chars extracted: {summary['total_chars_extracted']:,}")
    logger.info(f"Total time: {summary['total_processing_time']:.1f}s")
    logger.info(f"\nMethods used: {summary['methods_used']}")
    logger.info(f"File types: {summary['file_types']}")
    
    return results, summary


def save_results(
    results: List[ExtractionResult],
    summary: Dict,
    output_dir: Path
):
    """
    Save processing results to various output formats.
    
    Args:
        results: List of extraction results
        summary: Summary statistics
        output_dir: Output directory
    """
    # 1. Save successful extractions in preprocessing-compatible format
    # Column names: 'file' and 'text' (exactly as preprocessing.py expects)
    successful = [r for r in results if r.success]
    if successful:
        df_readtext = pd.DataFrame({
            'file': [r.filename for r in successful],
            'text': [r.text for r in successful]
        })
        # Try parquet first, fall back to CSV
        try:
            readtext_path = output_dir / "pdf_texts.parquet"
            df_readtext.to_parquet(readtext_path, index=False)
            print(f"✓ Saved {len(successful)} texts for preprocessing: {readtext_path}")
            print(f"  Format: Parquet with columns ['file', 'text']")
            print(f"  Ready for: python preprocessing.py --input {readtext_path}")
        except:
            readtext_path = output_dir / "pdf_texts.csv"
            df_readtext.to_csv(readtext_path, index=False)
            print(f"✓ Saved {len(successful)} texts for preprocessing: {readtext_path}")
            print(f"  Format: CSV with columns ['file', 'text']")
            print(f"  Note: Use updated preprocessing.py that accepts CSV")
    
    # 2. Save full results with metadata
    df_full = pd.DataFrame([asdict(r) for r in results])
    # Try parquet first, fall back to CSV
    try:
        full_path = output_dir / "pdf_extraction_full.parquet"
        df_full.to_parquet(full_path, index=False)
        print(f"✓ Saved full results: {full_path}")
    except:
        full_path = output_dir / "pdf_extraction_full.csv"
        df_full.to_csv(full_path, index=False)
        print(f"✓ Saved full results: {full_path}")
    
    # 3. Save summary statistics
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")
    
    # 4. Save list of failed files
    failed = [r for r in results if not r.success]
    if failed:
        failed_path = output_dir / "failed_files.txt"
        with open(failed_path, 'w') as f:
            for result in failed:
                f.write(f"{result.filename}\n")
        print(f"✓ Saved {len(failed)} failed files list: {failed_path}")
        
        # Also save detailed failure info
        failed_details_path = output_dir / "failed_files_details.csv"
        df_failed = pd.DataFrame([{
            'filename': r.filename,
            'file_type': r.file_type,
            'method_attempted': r.method,
            'error_message': r.error_message
        } for r in failed])
        df_failed.to_csv(failed_details_path, index=False)
        print(f"✓ Saved failure details: {failed_details_path}")


# =============================================================================
# CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced PDF text extraction for Swedish RSA documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic extraction (no OCR)
    python pdf_reader_enhanced.py --input-dir ./pdfs --output-dir ./output
    
    # With OCR for scanned documents
    python pdf_reader_enhanced.py --input-dir ./pdfs --output-dir ./output --ocr
    
    # Custom minimum text length
    python pdf_reader_enhanced.py --input-dir ./pdfs --output-dir ./output --min-length 1000 --ocr

Output Files:
    pdf_texts.parquet              - Main output, ready for preprocessing.py
                                     Format: columns ['file', 'text']
                                     Usage: python preprocessing.py --input pdf_texts.parquet
    pdf_extraction_full.csv        - Full results with all metadata
    processing_summary.json        - Summary statistics
    failed_files.txt              - List of files that failed
    failed_files_details.csv      - Detailed failure information
    processing.log                - Detailed processing log

File Type Detection:
    The script automatically detects:
    - Standard PDFs with embedded text
    - ZIP archives with .pdf extension (common for scanned documents)
    - Other file types

Extraction Methods:
    For standard PDFs, tries in order: pypdf → pdfplumber → pdfminer
    For ZIP-based PDFs: Extracts JPEGs and performs OCR (if --ocr enabled)
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing PDF files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=500,
        help='Minimum text length (chars) for success (default: 500)'
    )
    
    parser.add_argument(
        '--ocr',
        action='store_true',
        help='Enable OCR for image-based PDFs (requires pytesseract and tesseract-ocr-swe)'
    )
    
    parser.add_argument(
        '--ocr-lang',
        type=str,
        default='swe',
        help='Tesseract language code (default: swe)'
    )
    
    parser.add_argument(
        '--ocr-dpi',
        type=int,
        default=300,
        help='DPI for OCR processing (default: 300)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate directories
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check OCR availability
    if args.ocr and not OCR_AVAILABLE:
        print("Error: OCR enabled but required libraries not available")
        print("Install with: pip install Pillow pytesseract")
        print("System: apt-get install tesseract-ocr tesseract-ocr-swe")
        return 1
    
    # Create configuration
    config = ProcessingConfig(
        min_text_length=args.min_length,
        use_ocr=args.ocr,
        ocr_language=args.ocr_lang,
        ocr_dpi=args.ocr_dpi,
        verbose=args.verbose
    )
    
    # Process files
    print(f"\n{'='*60}")
    print("Enhanced PDF Text Extraction")
    print(f"{'='*60}\n")
    
    results, summary = process_directory(args.input_dir, args.output_dir, config)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}\n")
    save_results(results, summary, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}\n")
    
    return 0 if summary['successful'] > 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
