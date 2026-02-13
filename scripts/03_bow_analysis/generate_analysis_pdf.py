#!/usr/bin/env python3
"""
Generate combined PDF report for Risk Mapping Analysis.

Combines persistence and clustering analysis outputs into a single PDF
with title pages, section headers, and all visualizations.

Usage:
    python generate_analysis_pdf.py [--output PATH]
"""

import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
PERSISTENCE_DIR = RESULTS_DIR / "persistence"
CLUSTERING_DIR = RESULTS_DIR / "clustering"
DEFAULT_OUTPUT = RESULTS_DIR / "risk_mapping_analysis_outputs.pdf"

# Wave labels for clustering outputs
WAVE_LABELS = {
    0: "Wave 0 (Pre-2015)",
    1: "Wave 1 (2015-2018)",
    2: "Wave 2 (2019-2022)",
    3: "Wave 3 (2023+)"
}


# ============================================================================
# Helper functions
# ============================================================================

def add_title_page(pdf: PdfPages, title: str, subtitle: str = "") -> None:
    """Add a title/section page to the PDF."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.55, title, transform=ax.transAxes,
            fontsize=28, fontweight='bold', ha='center', va='center')

    # Subtitle
    if subtitle:
        ax.text(0.5, 0.42, subtitle, transform=ax.transAxes,
                fontsize=18, ha='center', va='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_text_page(pdf: PdfPages, text: str, title: str = "") -> None:
    """Add a text-only page to the PDF."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    y_start = 0.95

    if title:
        ax.text(0.05, y_start, title, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', family='monospace')
        y_start -= 0.08

    ax.text(0.05, y_start, text, transform=ax.transAxes,
            fontsize=9, va='top', family='monospace',
            linespacing=1.3)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, title: str = "") -> None:
    """Add an image as a full page to the PDF."""
    if not image_path.exists():
        print(f"  Warning: {image_path} not found, skipping")
        return

    img = Image.open(image_path)
    img_array = np.array(img)

    # Calculate figure size to fit image while maintaining aspect ratio
    dpi = 100
    max_width, max_height = 11, 8.5  # Letter landscape

    img_width = img.width / dpi
    img_height = img.height / dpi

    # Account for title space
    available_height = max_height - (0.8 if title else 0)

    scale = min(max_width / img_width, available_height / img_height, 1.0)
    fig_width = max(img_width * scale, 8)
    fig_height = img_height * scale + (0.8 if title else 0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    ax.imshow(img_array)
    ax.axis('off')

    pdf.savefig(fig, bbox_inches='tight', dpi=150)
    plt.close(fig)


def read_report(path: Path) -> str:
    """Read a text report file."""
    if path.exists():
        return path.read_text(encoding='utf-8')
    return f"Report not found: {path}"


# ============================================================================
# Main PDF generation
# ============================================================================

def generate_pdf(output_path: Path) -> None:
    """Generate the complete analysis PDF."""

    print(f"Generating PDF: {output_path}")

    with PdfPages(output_path) as pdf:

        # ====================================================================
        # Title page
        # ====================================================================
        add_title_page(
            pdf,
            "Risk Mapping Analysis",
            "Persistence, Diffusion & Clustering\n\nOutput Summary"
        )

        # ====================================================================
        # PART 1: PERSISTENCE ANALYSIS
        # ====================================================================
        add_title_page(pdf, "PART 1: RISK PERSISTENCE ANALYSIS")

        # Persistence report text
        report_text = read_report(PERSISTENCE_DIR / "persistence_report.txt")
        # Remove header lines for cleaner display
        lines = report_text.split('\n')
        if lines and '===' in lines[0]:
            lines = lines[3:]  # Skip header
        add_text_page(pdf, '\n'.join(lines), "Persistence Analysis Summary")

        # Persistence visualizations
        persistence_images = [
            ("persistence_by_actor.png", "Mean persistence rate by actor type and period"),
            ("jaccard_by_actor.png", "Jaccard similarity distribution by actor type"),
            ("persistence_heatmap.png", "Term persistence heatmap (all actors, consecutive waves)"),
            ("persistence_heatmap_kommun.png", "Term persistence heatmap — Municipalities (W0→W1, W1→W2, W2→W3)"),
            ("persistence_heatmap_kommun_w1_w3.png", "Term persistence heatmap — Municipalities (W1→W3 direct)"),
            ("persistence_heatmap_year_länsstyrelse.png", "Term persistence heatmap — Prefectures (year-by-year)"),
            ("persistence_heatmap_year_MCF.png", "Term persistence heatmap — MCF (year-by-year)"),
            ("dropout_ranking.png", "Top 20 terms by dropout frequency"),
            ("adopt_ranking.png", "Top 20 terms by adoption frequency"),
        ]

        for filename, title in persistence_images:
            img_path = PERSISTENCE_DIR / filename
            print(f"  Adding: {filename}")
            add_image_page(pdf, img_path, title)

        # ====================================================================
        # PART 2: CLUSTERING ANALYSIS
        # ====================================================================
        add_title_page(pdf, "PART 2: RISK CLUSTERING ANALYSIS")

        # Clustering report text
        report_text = read_report(CLUSTERING_DIR / "clustering_report.txt")
        # Split into pages if too long
        lines = report_text.split('\n')
        if lines and '===' in lines[0]:
            lines = lines[3:]  # Skip header

        # Split report into chunks for multiple pages
        chunk_size = 55  # lines per page
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i+chunk_size])
            page_title = "Clustering Analysis Summary" if i == 0 else "Clustering Analysis Summary (cont.)"
            add_text_page(pdf, chunk, page_title)

        # Clustering visualizations by wave
        for wave in range(4):
            wave_label = WAVE_LABELS[wave]

            # Section header for wave
            add_title_page(pdf, f"Clustering: {wave_label}")

            wave_images = [
                (f"elbow_{wave}.png", f"Elbow curve — {wave_label}"),
                (f"dendrogram_{wave}.png", f"Hierarchical dendrogram — {wave_label}"),
                (f"pca_scatter_{wave}.png", f"PCA scatter plot — {wave_label}"),
                (f"centroid_heatmap_{wave}.png", f"Cluster centroid heatmap — {wave_label}"),
                (f"actor_distribution_{wave}.png", f"Actor distribution by cluster — {wave_label}"),
            ]

            for filename, title in wave_images:
                img_path = CLUSTERING_DIR / filename
                print(f"  Adding: {filename}")
                add_image_page(pdf, img_path, title)

        # Transition matrices
        add_title_page(pdf, "Cluster Transitions Between Waves")

        transition_images = [
            ("transition_matrix_0_1.png", "Cluster transitions: Wave 0 → Wave 1"),
            ("transition_matrix_1_2.png", "Cluster transitions: Wave 1 → Wave 2"),
            ("transition_matrix_2_3.png", "Cluster transitions: Wave 2 → Wave 3"),
        ]

        for filename, title in transition_images:
            img_path = CLUSTERING_DIR / filename
            print(f"  Adding: {filename}")
            add_image_page(pdf, img_path, title)

        # ====================================================================
        # Metadata page
        # ====================================================================
        metadata = f"""
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Source directories:
  Persistence: {PERSISTENCE_DIR}
  Clustering: {CLUSTERING_DIR}

This PDF was automatically generated by:
  scripts/03_bow_analysis/generate_analysis_pdf.py
"""
        add_text_page(pdf, metadata, "Report Metadata")

    print(f"Done! PDF saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate combined PDF report for Risk Mapping Analysis"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT})"
    )

    args = parser.parse_args()
    generate_pdf(args.output)


if __name__ == "__main__":
    main()
