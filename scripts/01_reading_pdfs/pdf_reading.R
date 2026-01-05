#!/usr/bin/env Rscript
# =============================================================================
# PDF Text Extraction Script for Swedish RSA Documents
# =============================================================================
# 
# This script reads PDF files from a directory using the readtext package,
# categorizes them as successful or failed based on text extraction quality,
# and saves the results to RDS files.
#
# Author: Theodor Selimovic
# Project: Swedish Risk Analysis Text-as-Data
# Date: 2024-12-31
#
# Outputs:
#   - readtext_success.rds: Successfully extracted texts (>500 characters)
#   - failed_files_list.rds: List of files that failed extraction
#
# Usage:
#   Rscript pdf_reading.R
#   Or source("pdf_reading.R") from R console
#
# =============================================================================

# Load required packages
library(tidyverse)
library(readtext)
library(pdftools)
library(tesseract)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set your PDF folder path
pdf_folder <- '/Users/theodorselimovic/Sciences Po/Material/Risk analyses/Kommunala RSA'

# Minimum text length threshold for successful extraction
MIN_TEXT_LENGTH <- 500

# =============================================================================
# FUNCTIONS
# =============================================================================

# Function to safely read a PDF and capture warnings/messages
safe_read_pdf <- function(file_path) {
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
          # Core text extraction function
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
    return(list(
      file = basename(file_path),
      status = "failed",
      text = NA,
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
    error_msg <- NA
  }
  
  return(tibble(
    file = basename(file_path),
    status = status,
    text = text_content,
    error = error_msg
  ))
}

# =============================================================================
# MAIN PROCESSING
# =============================================================================

cat("=============================================================================\n")
cat("PDF TEXT EXTRACTION\n")
cat("=============================================================================\n")
cat("PDF Folder:", pdf_folder, "\n")
cat("Minimum text length:", MIN_TEXT_LENGTH, "characters\n\n")

# Get all PDF files
pdf_files <- list.files(pdf_folder, pattern = "\\.pdf$", full.names = TRUE)
cat("Found", length(pdf_files), "PDF files\n\n")

# Process all PDFs
cat("Processing PDFs...\n")
results_df <- map_dfr(pdf_files, safe_read_pdf)

# Separate successful and failed reads
# Note: Anything with >500 characters is considered successful, 
# since some files may have warnings but still extract text properly
true_successful <- results_df %>% 
  filter(
    status == "success" | (status == "failed" & error != "Empty text extracted")
  ) %>% 
  mutate(text_length = str_length(text)) %>% 
  filter(text_length > MIN_TEXT_LENGTH) %>% 
  select(file, text)

failed <- results_df %>% 
  mutate(text_length = str_length(text)) %>%
  filter(text_length < MIN_TEXT_LENGTH) %>% 
  select(file)

# =============================================================================
# SAVE RESULTS
# =============================================================================

# Save successful results
saveRDS(object = true_successful, file = "readtext_success.rds")
cat("✓ Saved successful extractions to: readtext_success.rds\n")

# Save list of failed files
saveRDS(object = failed, file = "failed_files_list.rds")
cat("✓ Saved failed file list to: failed_files_list.rds\n\n")

# Also save failed files as plain text for OCR pipeline
failed_txt_path <- "failed_files.txt"
write_lines(failed$file, failed_txt_path)
cat("✓ Saved failed files list to:", failed_txt_path, "\n")
cat("  (Use this file with ocr_swedish_pdfs_improved.py)\n\n")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

cat("=============================================================================\n")
cat("SUMMARY\n")
cat("=============================================================================\n")
cat("Total PDFs processed:", nrow(results_df), "\n")
cat("Successful extractions:", nrow(true_successful), 
    sprintf("(%.1f%%)", nrow(true_successful) / nrow(results_df) * 100), "\n")
cat("Failed extractions:", nrow(failed), 
    sprintf("(%.1f%%)", nrow(failed) / nrow(results_df) * 100), "\n\n")

if (nrow(failed) > 0) {
  cat("Next step: Run OCR on failed files\n")
  cat("  python scripts/01_ocr/ocr_swedish_pdfs_improved.py \\\n")
  cat("    --input", pdf_folder, "\\\n")
  cat("    --output ocr_output \\\n")
  cat("    --file-list", failed_txt_path, "\n")
}

cat("=============================================================================\n")