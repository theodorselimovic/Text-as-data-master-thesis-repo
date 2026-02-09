#!/usr/bin/env python3
"""
Document Preview Generator for Quality Inspection

This script creates preview files showing the first N characters of each document
from a parquet file for visual quality inspection.

Usage:
    python document_preview_generator.py --input data.parquet --output preview --chars 1000

Output formats:
    - CSV file with document metadata and preview text
    - HTML file with formatted table for easy viewing
    - Optional: Text file with one document per section

Author: Theodor Selimovic
Date: 2025-01-19
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd
import sys


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class DocumentPreviewGenerator:
    """
    Generate preview files for quality inspection of processed documents.
    """
    
    def __init__(self, input_path: Path, output_prefix: Path, 
                 char_limit: int = 1000, text_column: str = 'text'):
        """
        Initialize the preview generator.
        
        Parameters:
        -----------
        input_path : Path
            Path to input parquet file
        output_prefix : Path
            Prefix for output files (without extension)
        char_limit : int
            Number of characters to show per document
        text_column : str
            Name of the column containing text data
        """
        self.input_path = input_path
        self.output_prefix = output_prefix
        self.char_limit = char_limit
        self.text_column = text_column
        self.df = None
        self.preview_df = None
        
    def load_data(self) -> None:
        """Load parquet file into DataFrame."""
        logging.info("=" * 80)
        logging.info("LOADING DATA")
        logging.info("=" * 80)
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        logging.info(f"Reading: {self.input_path}")
        self.df = pd.read_parquet(self.input_path)
        
        logging.info(f"✓ Loaded {len(self.df):,} rows")
        logging.info(f"  Columns: {list(self.df.columns)}")
        logging.info(f"  Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Verify text column exists
        if self.text_column not in self.df.columns:
            available_cols = [col for col in self.df.columns if 'text' in col.lower()]
            if available_cols:
                logging.warning(f"Column '{self.text_column}' not found.")
                logging.warning(f"Available text columns: {available_cols}")
                raise ValueError(f"Text column '{self.text_column}' not found in data")
            else:
                raise ValueError(f"No text columns found. Available: {list(self.df.columns)}")
    
    def create_previews(self) -> None:
        """
        Create preview DataFrame with first N characters per document.
        """
        logging.info("\n" + "=" * 80)
        logging.info("CREATING DOCUMENT PREVIEWS")
        logging.info("=" * 80)
        
        # Determine grouping column
        if 'doc_id' in self.df.columns:
            group_col = 'doc_id'
        elif 'document_id' in self.df.columns:
            group_col = 'document_id'
        elif 'file' in self.df.columns:
            group_col = 'file'
        else:
            raise ValueError("No document ID column found (tried: doc_id, document_id, file)")
        
        logging.info(f"Grouping by: {group_col}")
        
        # Metadata columns to include
        metadata_cols = [col for col in self.df.columns 
                        if col in ['municipality', 'year', 'maskad', 'category']]
        
        logging.info(f"Including metadata: {metadata_cols}")
        
        # Group by document and concatenate text
        logging.info("Concatenating sentences by document...")
        grouped = self.df.groupby(group_col, as_index=False).agg({
            self.text_column: lambda x: ' '.join(str(s) for s in x if pd.notna(s)),
            **{col: 'first' for col in metadata_cols}
        })
        
        logging.info(f"✓ Created previews for {len(grouped):,} documents")
        
        # Create preview text (first N characters)
        logging.info(f"Extracting first {self.char_limit} characters...")
        grouped['preview_text'] = grouped[self.text_column].apply(
            lambda x: x[:self.char_limit] if len(x) > self.char_limit else x
        )
        
        # Add character counts
        grouped['total_chars'] = grouped[self.text_column].str.len()
        grouped['preview_chars'] = grouped['preview_text'].str.len()
        grouped['is_truncated'] = grouped['total_chars'] > self.char_limit
        
        # Drop full text column (we only need preview)
        grouped = grouped.drop(columns=[self.text_column])
        
        # Reorder columns for clarity
        col_order = [group_col] + metadata_cols + [
            'total_chars', 'preview_chars', 'is_truncated', 'preview_text'
        ]
        self.preview_df = grouped[col_order]
        
        logging.info("✓ Preview DataFrame created")
        self._print_preview_stats()
    
    def _print_preview_stats(self) -> None:
        """Print statistics about the previews."""
        logging.info("\nPreview Statistics:")
        logging.info(f"  Total documents: {len(self.preview_df):,}")
        logging.info(f"  Truncated: {self.preview_df['is_truncated'].sum():,}")
        logging.info(f"  Complete: {(~self.preview_df['is_truncated']).sum():,}")
        
        if 'municipality' in self.preview_df.columns:
            n_municipalities = self.preview_df['municipality'].nunique()
            logging.info(f"  Municipalities: {n_municipalities}")
        
        if 'year' in self.preview_df.columns:
            years = sorted(self.preview_df['year'].unique())
            logging.info(f"  Years: {min(years)} - {max(years)}")
    
    def save_csv(self) -> Path:
        """
        Save preview to CSV file.
        
        Returns:
        --------
        Path : Path to saved CSV file
        """
        output_path = self.output_prefix.with_suffix('.csv')
        logging.info("\n" + "=" * 80)
        logging.info("SAVING CSV")
        logging.info("=" * 80)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.preview_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logging.info(f"✓ Saved to: {output_path}")
        logging.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_path
    
    def save_html(self) -> Path:
        """
        Save preview to HTML file with formatting.
        
        Returns:
        --------
        Path : Path to saved HTML file
        """
        output_path = self.output_prefix.with_suffix('.html')
        logging.info("\n" + "=" * 80)
        logging.info("SAVING HTML")
        logging.info("=" * 80)
        
        # Create HTML with custom styling
        html = self._create_html()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logging.info(f"✓ Saved to: {output_path}")
        logging.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        logging.info(f"  Open in browser: file://{output_path.absolute()}")
        
        return output_path
    
    def _create_html(self) -> str:
        """Create formatted HTML output."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Preview - Quality Inspection</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .stats {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .stats p {
            margin: 5px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        thead {
            background-color: #4CAF50;
            color: white;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            font-weight: 600;
            position: sticky;
            top: 0;
            background-color: #4CAF50;
        }
        tbody tr:hover {
            background-color: #f5f5f5;
        }
        .preview-text {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            background-color: #f9f9f9;
            padding: 10px;
            border-left: 3px solid #4CAF50;
            max-width: 800px;
        }
        .truncated {
            color: #ff9800;
            font-weight: bold;
        }
        .complete {
            color: #4CAF50;
            font-weight: bold;
        }
        .metadata {
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 Document Preview - Quality Inspection</h1>
        <div class="stats">
            <p><strong>Total Documents:</strong> {total_docs}</p>
            <p><strong>Character Limit:</strong> {char_limit} characters per document</p>
            <p><strong>Truncated:</strong> {truncated_count} documents</p>
            <p><strong>Complete:</strong> {complete_count} documents</p>
            {extra_stats}
        </div>
        
        <table>
            <thead>
                <tr>
                    {headers}
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
</body>
</html>
        """
        
        # Generate statistics
        total_docs = len(self.preview_df)
        truncated_count = self.preview_df['is_truncated'].sum()
        complete_count = total_docs - truncated_count
        
        extra_stats = ""
        if 'municipality' in self.preview_df.columns:
            n_muni = self.preview_df['municipality'].nunique()
            extra_stats += f"<p><strong>Municipalities:</strong> {n_muni}</p>"
        if 'year' in self.preview_df.columns:
            years = sorted(self.preview_df['year'].unique())
            extra_stats += f"<p><strong>Years:</strong> {min(years)} - {max(years)}</p>"
        
        # Generate table headers
        headers = "".join([f"<th>{col}</th>" for col in self.preview_df.columns if col != 'preview_text'])
        headers += "<th>Preview Text</th>"
        
        # Generate table rows
        rows = []
        for _, row in self.preview_df.iterrows():
            row_html = "<tr>"
            
            # Add metadata columns
            for col in self.preview_df.columns:
                if col == 'preview_text':
                    continue
                elif col == 'is_truncated':
                    status = "Truncated" if row[col] else "Complete"
                    status_class = "truncated" if row[col] else "complete"
                    row_html += f'<td class="{status_class}">{status}</td>'
                else:
                    row_html += f"<td>{row[col]}</td>"
            
            # Add preview text in its own cell
            row_html += f'<td><div class="preview-text">{row["preview_text"]}</div></td>'
            row_html += "</tr>"
            rows.append(row_html)
        
        rows_html = "\n".join(rows)
        
        # Fill in template
        html = html.format(
            total_docs=total_docs,
            char_limit=self.char_limit,
            truncated_count=truncated_count,
            complete_count=complete_count,
            extra_stats=extra_stats,
            headers=headers,
            rows=rows_html
        )
        
        return html
    
    def save_text(self) -> Path:
        """
        Save preview to plain text file.
        
        Returns:
        --------
        Path : Path to saved text file
        """
        output_path = self.output_prefix.with_suffix('.txt')
        logging.info("\n" + "=" * 80)
        logging.info("SAVING TEXT FILE")
        logging.info("=" * 80)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DOCUMENT PREVIEW - QUALITY INSPECTION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Documents: {len(self.preview_df)}\n")
            f.write(f"Character Limit: {self.char_limit} per document\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Get group column name
            group_col = [col for col in self.preview_df.columns 
                        if col in ['doc_id', 'document_id', 'file']][0]
            
            # Write each document
            for idx, row in self.preview_df.iterrows():
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"DOCUMENT {idx + 1}: {row[group_col]}\n")
                f.write("=" * 80 + "\n")
                
                # Write metadata
                for col in self.preview_df.columns:
                    if col not in [group_col, 'preview_text', 'preview_chars']:
                        f.write(f"{col}: {row[col]}\n")
                
                f.write("\n" + "-" * 80 + "\n")
                f.write("PREVIEW TEXT:\n")
                f.write("-" * 80 + "\n")
                f.write(row['preview_text'])
                f.write("\n")
                
                if row['is_truncated']:
                    f.write("\n[...TRUNCATED...]")
                f.write("\n")
        
        logging.info(f"✓ Saved to: {output_path}")
        logging.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_path
    
    def run(self, formats: List[str] = None) -> List[Path]:
        """
        Run the complete preview generation pipeline.
        
        Parameters:
        -----------
        formats : List[str], optional
            Output formats to generate. Options: 'csv', 'html', 'txt'
            Default: ['csv', 'html']
        
        Returns:
        --------
        List[Path] : Paths to generated files
        """
        if formats is None:
            formats = ['csv', 'html']
        
        # Load data
        self.load_data()
        
        # Create previews
        self.create_previews()
        
        # Save in requested formats
        output_files = []
        
        if 'csv' in formats:
            output_files.append(self.save_csv())
        
        if 'html' in formats:
            output_files.append(self.save_html())
        
        if 'txt' in formats:
            output_files.append(self.save_text())
        
        # Final summary
        logging.info("\n" + "=" * 80)
        logging.info("PREVIEW GENERATION COMPLETE")
        logging.info("=" * 80)
        logging.info(f"\n✓ Generated {len(output_files)} output file(s):")
        for path in output_files:
            logging.info(f"  • {path}")
        
        return output_files


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate document previews for quality inspection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (creates CSV and HTML)
    python document_preview_generator.py --input sentences_lemmatized.parquet --output preview
    
    # Custom character limit
    python document_preview_generator.py --input data.parquet --output preview --chars 500
    
    # Generate all formats
    python document_preview_generator.py --input data.parquet --output preview --format csv html txt
    
    # Specify text column name
    python document_preview_generator.py --input data.parquet --output preview --text-column lemmatized_text

Output Files:
    - CSV: Spreadsheet format for filtering/sorting
    - HTML: Formatted table for visual inspection in browser
    - TXT: Plain text format with one document per section
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input parquet file with processed documents'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output file prefix (without extension)'
    )
    
    parser.add_argument(
        '--chars',
        type=int,
        default=1000,
        help='Number of characters to show per document (default: 1000)'
    )
    
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of column containing text (default: text)'
    )
    
    parser.add_argument(
        '--format',
        nargs='+',
        choices=['csv', 'html', 'txt'],
        default=['csv', 'html'],
        help='Output formats to generate (default: csv html)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        generator = DocumentPreviewGenerator(
            input_path=args.input,
            output_prefix=args.output,
            char_limit=args.chars,
            text_column=args.text_column
        )
        
        generator.run(formats=args.format)
        return 0
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return 1
    
    except ValueError as e:
        logging.error(f"Invalid input: {e}")
        return 1
    
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
        return 130
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
