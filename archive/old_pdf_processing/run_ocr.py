#!/usr/bin/env python3
"""
Simple OCR Runner Script for Swedish RSA PDFs.

This script is pre-configured with your specific paths.
Run from Terminal for best performance with parallel processing.

Usage (from Terminal):
    cd "/Users/theodorselimovic/Library/CloudStorage/OneDrive-Personal/Sciences Po/Master Thesis/Text analysis code/Text-as-data-master-thesis-repo"
    python run_ocr.py

Or with custom number of workers:
    python run_ocr.py --workers 4
"""

import argparse
import sys
from pathlib import Path

# =============================================================================
# CONFIGURE YOUR PATHS HERE
# =============================================================================

# Where your PDFs are located
PDF_DIRECTORY = Path(
    "/Users/theodorselimovic/Sciences Po/Material/Risk analyses/Kommunala RSA"
)

# Where this script is located (output will be saved here)
SCRIPT_DIRECTORY = Path(
    "/Users/theodorselimovic/Library/CloudStorage/OneDrive-Personal/"
    "Sciences Po/Master Thesis/Text analysis code/"
    "Text-as-data-master-thesis-repo"
)

# List of failed files to process (should be in SCRIPT_DIRECTORY)
FAILED_FILES_LIST = SCRIPT_DIRECTORY / "failed_files.txt"

# Output will be saved to this subdirectory
OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "ocr_output"

# =============================================================================
# MAIN SCRIPT
# =============================================================================


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run OCR on failed Swedish PDFs")
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,  # Default to 4 workers for Terminal usage
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF conversion (default: 300)"
    )
    args = parser.parse_args()

    # Import the OCR module
    # Make sure ocr_swedish_pdfs_improved.py is in the same directory
    try:
        from ocr_swedish_pdfs_improved import run_pipeline, ProcessingConfig
    except ImportError:
        print("ERROR: Cannot find ocr_swedish_pdfs_improved.py")
        print(f"Make sure it's in: {SCRIPT_DIRECTORY}")
        sys.exit(1)

    # Validate paths
    if not PDF_DIRECTORY.exists():
        print(f"ERROR: PDF directory not found: {PDF_DIRECTORY}")
        sys.exit(1)

    if not FAILED_FILES_LIST.exists():
        print(f"ERROR: Failed files list not found: {FAILED_FILES_LIST}")
        print("Make sure failed_files.txt exists in your script directory.")
        sys.exit(1)

    # Show configuration
    print("=" * 60)
    print("OCR Processing Configuration")
    print("=" * 60)
    print(f"PDF Directory:    {PDF_DIRECTORY}")
    print(f"Output Directory: {OUTPUT_DIRECTORY}")
    print(f"File List:        {FAILED_FILES_LIST}")
    print(f"Workers:          {args.workers}")
    print(f"DPI:              {args.dpi}")
    print("=" * 60)
    print()

    # Configure processing
    config = ProcessingConfig(
        language="swe",
        dpi=args.dpi,
        min_text_length=500,
        workers=args.workers,
        save_individual_txt=False,  # Don't save individual .txt files
    )

    # Run the pipeline
    results, summary = run_pipeline(
        input_dir=PDF_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        file_list=FAILED_FILES_LIST,
        config=config,
    )

    # Final summary
    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total files:      {summary.total_files}")
    print(f"Successful:       {summary.successful}")
    print(f"Failed (short):   {summary.failed_short}")
    print(f"Failed (other):   {summary.failed_other}")
    print(f"Total words:      {summary.total_words:,}")
    print(f"Processing time:  {summary.total_time_seconds:.1f} seconds")
    print()
    print(f"Output saved to: {OUTPUT_DIRECTORY}")
    print(f"Main output file: {OUTPUT_DIRECTORY / 'ocr_readtext_format.parquet'}")

    return 0 if summary.successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
