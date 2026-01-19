#!/usr/bin/env Rscript
#
# PDF Text Extraction for Swedish RSA Documents
#
# This script extracts text from PDF files using the readtext package.
# It handles both standard text extraction and flags files that may need OCR.
#
# Usage:
#   Rscript pdf_reading.R --pdf-folder /path/to/pdfs
#   Rscript pdf_reading.R --pdf-folder /path/to/pdfs --min-length 500 --output-dir ./output
#   Rscript pdf_reading.R --help
#
# Output:
#   - readtext_success.rds: Successfully extracted documents
#   - failed_files.txt: List of files that need OCR processing
#
# Author: Swedish Risk Analysis Text-as-Data Project
# Date: 2025-01-04
# Version: 2.0

# =============================================================================
# DEPENDENCIES
# =============================================================================

# Check and load required packages
required_packages <- c("readtext", "tidyverse", "argparse")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    message(sprintf("Installing missing package: %s", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default values
DEFAULT_MIN_LENGTH <- 500
DEFAULT_OUTPUT_DIR <- "."

# =============================================================================
# COMMAND-LINE ARGUMENT PARSER
# =============================================================================

# Create argument parser
parser <- ArgumentParser(
  description = "Extract text from Swedish RSA PDF documents",
  epilog = "
Examples:
  # Basic usage
  Rscript pdf_reading.R --pdf-folder /path/to/pdfs
  
  # Custom minimum length threshold
  Rscript pdf_reading.R --pdf-folder /path/to/pdfs --min-length 1000
  
  # Custom output directory
  Rscript pdf_reading.R --pdf-folder /path/to/pdfs --output-dir ./results
  
  # All options
  Rscript pdf_reading.R \\
    --pdf-folder /path/to/pdfs \\
    --output-dir ./output \\
    --min-length 500 \\
    --verbose

Output Files:
  - readtext_success.rds: Successfully extracted documents (RDS format)
  - failed_files.txt: List of files that need OCR (text format)
  - failed_files_list.rds: Failed files list (RDS format)

Note:
  The minimum length threshold filters out empty or nearly-empty extractions
  that likely need OCR processing. Default is 500 characters, which typically
  represents ~80-100 words.
"
)

# Add arguments
parser$add_argument(
  "--pdf-folder",
  type = "character",
  required = TRUE,
  help = "Path to directory containing PDF files"
)

parser$add_argument(
  "--output-dir",
  type = "character",
  default = DEFAULT_OUTPUT_DIR,
  help = sprintf("Directory to save output files (default: %s)", DEFAULT_OUTPUT_DIR)
)

parser$add_argument(
  "--min-length",
  type = "integer",
  default = DEFAULT_MIN_LENGTH,
  help = sprintf("Minimum text length in characters for successful extraction (default: %d)", DEFAULT_MIN_LENGTH)
)

parser$add_argument(
  "--verbose",
  action = "store_true",
  default = FALSE,
  help = "Enable verbose output"
)

# Parse arguments
args <- parser$parse_args()

# =============================================================================
# VALIDATION
# =============================================================================

validate_inputs <- function(pdf_folder, output_dir, min_length) {
  # Check PDF folder exists
  if (!dir.exists(pdf_folder)) {
    stop(sprintf("PDF folder does not exist: %s", pdf_folder))
  }
  
  # Create output directory if needed
  if (!dir.exists(output_dir)) {
    message(sprintf("Creating output directory: %s", output_dir))
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Check minimum length is positive
  if (min_length <= 0) {
    stop(sprintf("Minimum length must be positive, got: %d", min_length))
  }
  
  # Check for PDF files
  pdf_files <- list.files(pdf_folder, pattern = "\\.pdf$", ignore.case = TRUE, full.names = TRUE)
  if (length(pdf_files) == 0) {
    stop(sprintf("No PDF files found in: %s", pdf_folder))
  }
  
  return(pdf_files)
}

# =============================================================================
# SAFE PDF READING FUNCTION
# =============================================================================

#' Safely read a PDF and capture warnings/messages
#'
#' This function attempts to read a PDF using readtext, capturing any
#' warnings, messages, or errors that occur. It classifies results as
#' either "success" or "failed" based on whether text was extracted
#' and whether any issues occurred.
#'
#' @param file_path Character. Full path to PDF file
#' @param verbose Logical. If TRUE, print detailed progress
#'
#' @return Tibble with columns:
#'   - file: Filename (basename)
#'   - status: "success" or "failed"
#'   - text: Extracted text content (NA if failed)
#'   - error: Error/warning message (NA if success)
#'
safe_read_pdf <- function(file_path, verbose = FALSE) {
  if (verbose) {
    message(sprintf("Processing: %s", basename(file_path)))
  }
  
  # Variables to capture warnings and messages
  warning_messages <- character()
  message_text <- character()
  result <- NULL
  
  # Capture output, warnings, and messages
  result <- tryCatch(
    withCallingHandlers(
      {
        # Suppress console output and capture it
        output <- capture.output({
          text_result <- readtext(file_path)
        }, type = "message")
        
        # Store any captured output
        if (length(output) > 0) {
          message_text <<- c(message_text, output)
        }
        
        text_result
      },
      warning = function(w) {
        warning_messages <<- c(warning_messages, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) {
        message_text <<- c(message_text, conditionMessage(m))
        invokeRestart("muffleMessage")
      }
    ),
    error = function(e) {
      return(list(error = as.character(e$message)))
    }
  )
  
  # Check if there was an error
  if (!is.null(result$error)) {
    return(tibble(
      file = basename(file_path),
      status = "failed",
      text = NA_character_,
      error = result$error
    ))
  }
  
  # Get text content
  text_content <- result$text
  is_empty <- is.na(text_content) || str_trim(text_content) == ""
  
  # Combine all captured issues
  all_issues <- c(warning_messages, message_text)
  has_issues <- length(all_issues) > 0
  
  # Determine status
  if (has_issues || is_empty) {
    status <- "failed"
    # Prioritize captured issues over empty text
    error_msg <- if (has_issues) {
      paste(all_issues, collapse = "; ")
    } else {
      "Empty text extracted"
    }
  } else {
    status <- "success"
    error_msg <- NA_character_
  }
  
  return(tibble(
    file = basename(file_path),
    status = status,
    text = text_content,
    error = error_msg
  ))
}

# =============================================================================
# CLASSIFICATION FUNCTION
# =============================================================================

#' Classify extraction results as successful or failed
#'
#' @param results_df Tibble. Results from safe_read_pdf
#' @param min_length Integer. Minimum character count for success
#'
#' @return List with two tibbles:
#'   - successful: Documents with sufficient text
#'   - failed: Documents needing OCR
#'
classify_results <- function(results_df, min_length) {
  # Calculate text length
  results_with_length <- results_df %>%
    mutate(text_length = str_length(text))
  
  # Classify as successful if:
  # 1. Status is "success" OR
  # 2. Status is "failed" but error is not "Empty text extracted" (some text was extracted)
  # AND text length exceeds threshold
  successful <- results_with_length %>%
    filter(
      (status == "success" | (status == "failed" & error != "Empty text extracted")) &
      text_length > min_length
    ) %>%
    select(file, text)
  
  # Failed files need OCR
  failed <- results_with_length %>%
    filter(text_length <= min_length) %>%
    select(file)
  
  return(list(
    successful = successful,
    failed = failed
  ))
}

# =============================================================================
# MAIN PROCESSING
# =============================================================================

main <- function() {
  # Print configuration
  cat("\n")
  cat(strrep("=", 80), "\n")
  cat("PDF TEXT EXTRACTION\n")
  cat(strrep("=", 80), "\n")
  cat(sprintf("PDF folder:      %s\n", args$pdf_folder))
  cat(sprintf("Output directory: %s\n", args$output_dir))
  cat(sprintf("Min text length:  %d characters\n", args$min_length))
  cat(sprintf("Verbose:          %s\n", args$verbose))
  cat(strrep("=", 80), "\n\n")
  
  # Validate inputs and get PDF files
  pdf_files <- validate_inputs(args$pdf_folder, args$output_dir, args$min_length)
  
  cat(sprintf("Found %d PDF files\n\n", length(pdf_files)))
  
  # Process all PDFs
  cat("Processing PDFs...\n")
  start_time <- Sys.time()
  
  results_df <- map_dfr(pdf_files, ~safe_read_pdf(.x, verbose = args$verbose))
  
  elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  
  cat(sprintf("\nProcessing complete in %.1f seconds\n", elapsed_time))
  cat(sprintf("Average: %.2f PDFs/second\n\n", length(pdf_files) / elapsed_time))
  
  # Classify results
  cat("Classifying results...\n")
  classified <- classify_results(results_df, args$min_length)
  
  # Print summary
  cat("\n")
  cat(strrep("=", 80), "\n")
  cat("SUMMARY\n")
  cat(strrep("=", 80), "\n")
  cat(sprintf("Total PDFs:       %d\n", nrow(results_df)))
  cat(sprintf("Successful:       %d (%.1f%%)\n", 
              nrow(classified$successful),
              100 * nrow(classified$successful) / nrow(results_df)))
  cat(sprintf("Failed (need OCR): %d (%.1f%%)\n", 
              nrow(classified$failed),
              100 * nrow(classified$failed) / nrow(results_df)))
  cat(strrep("=", 80), "\n\n")
  
  # Show some examples of failed files
  if (nrow(classified$failed) > 0) {
    cat("Sample of failed files (first 10):\n")
    sample_failed <- head(classified$failed$file, 10)
    for (filename in sample_failed) {
      cat(sprintf("  - %s\n", filename))
    }
    if (nrow(classified$failed) > 10) {
      cat(sprintf("  ... and %d more\n", nrow(classified$failed) - 10))
    }
    cat("\n")
  }
  
  # Save results
  cat("Saving results...\n")
  
  # Successful extractions (RDS)
  success_path <- file.path(args$output_dir, "readtext_success.rds")
  saveRDS(classified$successful, file = success_path)
  cat(sprintf("  ✓ Saved successful extractions: %s\n", success_path))
  
  # Failed files (text file for OCR script)
  failed_txt_path <- file.path(args$output_dir, "failed_files.txt")
  write_lines(classified$failed$file, failed_txt_path)
  cat(sprintf("  ✓ Saved failed files list (txt): %s\n", failed_txt_path))
  
  # Failed files (RDS for R workflows)
  failed_rds_path <- file.path(args$output_dir, "failed_files_list.rds")
  saveRDS(classified$failed, file = failed_rds_path)
  cat(sprintf("  ✓ Saved failed files list (rds): %s\n", failed_rds_path))
  
  cat("\n")
  cat(strrep("=", 80), "\n")
  cat("NEXT STEPS\n")
  cat(strrep("=", 80), "\n")
  cat("\n1. Successfully extracted documents are ready for preprocessing:\n")
  cat(sprintf("     %s\n", success_path))
  cat("\n2. Failed files need OCR processing:\n")
  cat(sprintf("     python scripts/01_ocr/ocr_swedish_pdfs_improved.py \\\n"))
  cat(sprintf("         --input %s \\\n", args$pdf_folder))
  cat(sprintf("         --output ocr_output \\\n"))
  cat(sprintf("         --file-list %s\n", failed_txt_path))
  cat("\n3. After OCR, merge the results with readtext_success.rds\n")
  cat("   and proceed to preprocessing (readingtexts.py)\n\n")
  
  # Return invisibly for testing
  invisible(list(
    successful = classified$successful,
    failed = classified$failed,
    summary = list(
      total = nrow(results_df),
      successful = nrow(classified$successful),
      failed = nrow(classified$failed)
    )
  ))
}

# =============================================================================
# EXECUTE
# =============================================================================

# Run main function and handle errors
tryCatch(
  {
    result <- main()
    quit(status = 0)  # Success
  },
  error = function(e) {
    cat("\n")
    cat(strrep("=", 80), "\n")
    cat("ERROR\n")
    cat(strrep("=", 80), "\n")
    cat(sprintf("An error occurred: %s\n", e$message))
    cat("\nPlease check:\n")
    cat("  - PDF folder path is correct\n")
    cat("  - You have read permissions for the PDF folder\n")
    cat("  - Output directory is writable\n")
    cat("  - All required packages are installed\n")
    cat("\n")
    quit(status = 1)  # Error
  }
)
