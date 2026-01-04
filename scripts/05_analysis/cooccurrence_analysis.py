#!/usr/bin/env python3
"""
Co-occurrence Analysis and Chi-Square Tests for RSA Documents

This script performs statistical analysis of concept co-occurrence in Swedish 
municipal risk analysis documents. It tests whether political effects 
(efficiency, accountability, equality, complexity) and institutional actors 
(kommun, stat, lÃ¤nsstyrelse, etc.) appear together more often than expected 
by chance.

Key Features:
- Chi-square tests for independence
- Effect sizes (CramÃ©r's V)
- Temporal analysis (change over time)
- Actor-specific analysis (separating agency subcategories)
- Visualization of results

Usage:
    python cooccurrence_analysis.py --input sentence_vectors_with_metadata.parquet
    
Author: Swedish Risk Analysis Text-as-Data Project
Date: 2024-12-31
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Seed terms for actor identification
# These will be used to split the "agency" category into specific actors
ACTOR_SEED_TERMS = {
    'kommun': ['kommun', 'kommunal', 'kommuner', 'kommunen'],
    'stat': ['stat', 'staten', 'statlig', 'statliga'],
    'lÃ¤nsstyrelse': ['lÃ¤nsstyrelse', 'lÃ¤nsstyrelsen', 'lÃ¤nsstyrelser'],
    'region': ['region', 'regionen', 'regional', 'regionala'],
    'nÃ¤ringsliv': ['nÃ¤ringsliv', 'nÃ¤ringslivet', 'fÃ¶retag', 'fÃ¶retagen'],
    'civilsamhÃ¤lle': ['civilsamhÃ¤lle', 'civilsamhÃ¤llet', 'frivillig', 'frivilliga'],
    'fÃ¶rening': ['fÃ¶rening', 'fÃ¶reningen', 'fÃ¶reningar']
}

# Effect categories (non-actor categories)
EFFECT_CATEGORIES = ['risk', 'accountability', 'complexity', 'efficiency', 'equality']

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """
    Calculate CramÃ©r's V effect size for chi-square test.
    
    CramÃ©r's V measures the strength of association between two categorical 
    variables, ranging from 0 (no association) to 1 (perfect association).
    
    Parameters:
    -----------
    chi2 : float
        Chi-square statistic
    n : int
        Sample size
    r : int
        Number of rows in contingency table
    c : int
        Number of columns in contingency table
    
    Returns:
    --------
    float : CramÃ©r's V (0 to 1)
    
    Interpretation:
    ---------------
    0.00 - 0.10: Negligible association
    0.10 - 0.30: Weak association
    0.30 - 0.50: Moderate association
    0.50+      : Strong association
    """
    # Handle edge cases
    min_dim = min(r, c) - 1
    if min_dim <= 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def interpret_cramers_v(v: float) -> str:
    """Provide interpretation of CramÃ©r's V effect size."""
    if v < 0.10:
        return "negligible"
    elif v < 0.30:
        return "weak"
    elif v < 0.50:
        return "moderate"
    else:
        return "strong"


# =============================================================================
# DATA PREPARATION
# =============================================================================

class SentenceDataProcessor:
    """
    Process sentence-level data to prepare for co-occurrence analysis.
    
    This class handles:
    1. Loading sentence data with category assignments
    2. Identifying specific actors within the "agency" category
    3. Creating binary indicators for each concept
    4. Filtering and grouping data for analysis
    """
    
    def __init__(self, data_path: Path):
        """
        Initialize processor with data file path.
        
        Parameters:
        -----------
        data_path : Path
            Path to parquet file with sentence vectors and metadata
        """
        self.data_path = data_path
        self.df = None
        self.df_binary = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load sentence-level data.
        
        Returns:
        --------
        pd.DataFrame with columns:
            - doc_id: Document identifier
            - municipality: Municipality name
            - year: Publication year
            - maskad: Redaction status (boolean)
            - sentence_id: Sentence number within document
            - category: Concept category
            - target_terms: Terms found in sentence
            - sentence_text: Lemmatized sentence text
            - word_count: Number of words
        """
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        print(f"Loaded {len(self.df)} category-sentence pairs")
        print(f"Unique sentences: {self.df['sentence_id'].nunique()}")
        print(f"Categories: {self.df['category'].unique()}")
        return self.df
    
    def identify_actors(self, sentence_text: str) -> List[str]:
        """
        Identify which specific actors appear in a sentence.
        
        This function checks if any actor seed terms appear as standalone 
        words in the sentence (using word boundaries to avoid substring matches).
        
        Parameters:
        -----------
        sentence_text : str
            Lemmatized sentence text
            
        Returns:
        --------
        List[str] : Actor names found in sentence
        
        Example:
        --------
        >>> identify_actors("kommun ansvar risk hantering")
        ['kommun']
        >>> identify_actors("stat lÃ¤nsstyrelse samverkan")
        ['stat', 'lÃ¤nsstyrelse']
        """
        actors_found = []
        words = set(sentence_text.split())
        
        for actor, seed_terms in ACTOR_SEED_TERMS.items():
            # Check if any seed term appears as a standalone word
            if any(term in words for term in seed_terms):
                actors_found.append(actor)
        
        return actors_found
    
    def create_binary_indicators(self) -> pd.DataFrame:
        """
        Create binary indicators for each concept at sentence level.
        
        This transforms the data from category-sentence pairs (where a sentence 
        can appear multiple times with different categories) into sentence-level 
        data with binary columns for each concept.
        
        Returns:
        --------
        pd.DataFrame with columns:
            - doc_id, municipality, year, maskad, sentence_id, sentence_text, word_count
            - has_resilience: 1 if sentence contains resilience terms, 0 otherwise
            - has_risk: 1 if sentence contains risk terms, 0 otherwise
            - ... (one column per concept)
            
        Structure:
        ----------
        Before (category-sentence pairs):
            sentence_id  category     sentence_text
            1            resilience   "kommun resiliens..."
            1            risk         "kommun resiliens..."
            2            risk         "risk hantering..."
            
        After (sentence-level with indicators):
            sentence_id  sentence_text         has_resilience  has_risk  has_kommun
            1            "kommun resiliens..." 1               1         1
            2            "risk hantering..."   0               1         0
        """
        print("\nCreating binary indicators...")
        
        # Get unique sentences with metadata
        sentence_cols = ['doc_id', 'municipality', 'year', 'maskad', 
                        'sentence_id', 'sentence_text', 'word_count']
        df_sentences = self.df[sentence_cols].drop_duplicates()
        
        # Identify actors in each sentence
        print("Identifying actors in sentences...")
        df_sentences['actors'] = df_sentences['sentence_text'].apply(
            self.identify_actors
        )
        
        # Create binary indicators for effect categories
        print("Creating effect category indicators...")
        for category in EFFECT_CATEGORIES:
            indicator_col = f'has_{category}'
            # Get sentences that have this category
            sentences_with_cat = set(
                self.df[self.df['category'] == category]['sentence_id']
            )
            df_sentences[indicator_col] = df_sentences['sentence_id'].isin(
                sentences_with_cat
            ).astype(int)
        
        # Create binary indicators for each actor
        print("Creating actor indicators...")
        for actor in ACTOR_SEED_TERMS.keys():
            indicator_col = f'has_{actor}'
            df_sentences[indicator_col] = df_sentences['actors'].apply(
                lambda x: 1 if actor in x else 0
            )
        
        # Drop the actors list column (no longer needed)
        df_sentences = df_sentences.drop('actors', axis=1)
        
        self.df_binary = df_sentences
        
        # Print summary statistics
        print(f"\nBinary indicators created for {len(df_sentences)} unique sentences")
        print("\nConcept frequencies:")
        
        # Effect frequencies
        for category in EFFECT_CATEGORIES:
            count = df_sentences[f'has_{category}'].sum()
            pct = count / len(df_sentences) * 100
            print(f"  {category:15s}: {count:5d} sentences ({pct:5.2f}%)")
        
        # Actor frequencies
        print("\nActor frequencies:")
        for actor in ACTOR_SEED_TERMS.keys():
            count = df_sentences[f'has_{actor}'].sum()
            pct = count / len(df_sentences) * 100
            print(f"  {actor:15s}: {count:5d} sentences ({pct:5.2f}%)")
        
        return df_sentences


# =============================================================================
# CHI-SQUARE ANALYSIS
# =============================================================================

class CooccurrenceAnalyzer:
    """
    Perform chi-square tests of independence for concept co-occurrence.
    
    This class tests whether pairs of concepts (effects and/or actors) appear 
    together in sentences more often than expected by chance.
    """
    
    def __init__(self, df_binary: pd.DataFrame):
        """
        Initialize analyzer with binary indicator data.
        
        Parameters:
        -----------
        df_binary : pd.DataFrame
            Sentence-level data with binary indicators for each concept
        """
        self.df = df_binary
        self.results = []
        
    def test_pair(
        self, 
        concept1: str, 
        concept2: str,
        subset: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Test independence between two concepts using chi-square test.
        
        Parameters:
        -----------
        concept1 : str
            Name of first concept (e.g., 'resilience', 'kommun')
        concept2 : str
            Name of second concept
        subset : pd.DataFrame, optional
            Subset of data to analyze (e.g., specific year range)
            If None, uses full dataset
            
        Returns:
        --------
        dict : Test results including:
            - concept1, concept2: Concept names
            - chi2: Chi-square statistic
            - p_value: P-value
            - cramers_v: Effect size
            - effect_strength: Interpretation of effect size
            - observed: 2x2 contingency table (observed counts)
            - expected: 2x2 contingency table (expected counts under independence)
            - n_total: Sample size
            - n_both: Sentences with both concepts
            - n_c1_only: Sentences with concept1 only
            - n_c2_only: Sentences with concept2 only
            - n_neither: Sentences with neither concept
        """
        if subset is not None:
            data = subset
        else:
            data = self.df
        
        col1 = f'has_{concept1}'
        col2 = f'has_{concept2}'
        
        # Create contingency table
        contingency = pd.crosstab(data[col1], data[col2])
        
        # Skip if table is degenerate (one concept has zero occurrences)
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            logging.warning(
                f"Skipping {concept1} × {concept2}: degenerate table "
                f"(shape: {contingency.shape})"
            )
            return None
        
        # Run chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Calculate effect size
        n = len(data)
        v = cramers_v(chi2, n, *contingency.shape)
        
        # Extract counts
        n_both = contingency.loc[1, 1] if (1 in contingency.index and 1 in contingency.columns) else 0
        n_c1_only = contingency.loc[1, 0] if (1 in contingency.index and 0 in contingency.columns) else 0
        n_c2_only = contingency.loc[0, 1] if (0 in contingency.index and 1 in contingency.columns) else 0
        n_neither = contingency.loc[0, 0] if (0 in contingency.index and 0 in contingency.columns) else 0
        
        return {
            'concept1': concept1,
            'concept2': concept2,
            'chi2': chi2,
            'p_value': p_value,
            'cramers_v': v,
            'effect_strength': interpret_cramers_v(v),
            'observed': contingency,
            'expected': expected,
            'n_total': n,
            'n_both': n_both,
            'n_c1_only': n_c1_only,
            'n_c2_only': n_c2_only,
            'n_neither': n_neither
        }
    
    def test_effect_cooccurrence(self) -> pd.DataFrame:
        """
        Test co-occurrence between all pairs of effect categories.
        
        This tests which political effects (efficiency, accountability, etc.) 
        tend to appear together in the same sentences.
        
        Returns:
        --------
        pd.DataFrame : Results for all effect pairs, sorted by chi-square value
        """
        print("\n" + "="*80)
        print("TESTING EFFECT CO-OCCURRENCE")
        print("="*80)
        
        results = []
        
        # Test all pairs of effects
        for i, effect1 in enumerate(EFFECT_CATEGORIES):
            for effect2 in EFFECT_CATEGORIES[i+1:]:
                result = self.test_pair(effect1, effect2)
                if result is None:
                    continue  # Skip degenerate tables
                results.append({
                    'concept1': result['concept1'],
                    'concept2': result['concept2'],
                    'chi2': result['chi2'],
                    'p_value': result['p_value'],
                    'cramers_v': result['cramers_v'],
                    'effect_strength': result['effect_strength'],
                    'n_both': result['n_both'],
                    'expected_both': result['expected'][1, 1] if result['expected'].shape == (2, 2) else 0,
                    'ratio': result['n_both'] / result['expected'][1, 1] if (
                        result['expected'].shape == (2, 2) and result['expected'][1, 1] > 0
                    ) else np.nan
                })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('chi2', ascending=False)
        
        # Print summary
        print(f"\nTested {len(df_results)} effect pairs")
        print("\nTop 5 strongest associations:")
        print(df_results.head().to_string(index=False))
        
        self.results.extend(results)
        return df_results
    
    def test_effect_actor_association(self) -> pd.DataFrame:
        """
        Test association between effects and actors.
        
        This tests which institutional actors are discussed in connection with 
        which political effects (e.g., is "kommun" discussed more with "efficiency"?).
        
        Returns:
        --------
        pd.DataFrame : Results for all effect-actor pairs, sorted by chi-square value
        """
        print("\n" + "="*80)
        print("TESTING EFFECT-ACTOR ASSOCIATIONS")
        print("="*80)
        
        results = []
        
        # Test each effect with each actor
        for effect in EFFECT_CATEGORIES:
            for actor in ACTOR_SEED_TERMS.keys():
                result = self.test_pair(effect, actor)
                if result is None:
                    continue  # Skip degenerate tables
                results.append({
                    'effect': result['concept1'],
                    'actor': result['concept2'],
                    'chi2': result['chi2'],
                    'p_value': result['p_value'],
                    'cramers_v': result['cramers_v'],
                    'effect_strength': result['effect_strength'],
                    'n_both': result['n_both'],
                    'expected_both': result['expected'][1, 1] if result['expected'].shape == (2, 2) else 0,
                    'ratio': result['n_both'] / result['expected'][1, 1] if (
                        result['expected'].shape == (2, 2) and result['expected'][1, 1] > 0
                    ) else np.nan
                })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('chi2', ascending=False)
        
        # Print summary
        print(f"\nTested {len(df_results)} effect-actor pairs")
        print("\nTop 10 strongest associations:")
        print(df_results.head(10).to_string(index=False))
        
        self.results.extend(results)
        return df_results
    
    def test_actor_cooccurrence(self) -> pd.DataFrame:
        """
        Test co-occurrence between pairs of actors.
        
        This tests which institutional actors are discussed together 
        (e.g., kommun and lÃ¤nsstyrelse appearing in same sentences).
        
        Returns:
        --------
        pd.DataFrame : Results for all actor pairs, sorted by chi-square value
        """
        print("\n" + "="*80)
        print("TESTING ACTOR CO-OCCURRENCE")
        print("="*80)
        
        results = []
        actors = list(ACTOR_SEED_TERMS.keys())
        
        # Test all pairs of actors
        for i, actor1 in enumerate(actors):
            for actor2 in actors[i+1:]:
                result = self.test_pair(actor1, actor2)
                if result is None:
                    continue  # Skip degenerate tables
                results.append({
                    'actor1': result['concept1'],
                    'actor2': result['concept2'],
                    'chi2': result['chi2'],
                    'p_value': result['p_value'],
                    'cramers_v': result['cramers_v'],
                    'effect_strength': result['effect_strength'],
                    'n_both': result['n_both'],
                    'expected_both': result['expected'][1, 1] if result['expected'].shape == (2, 2) else 0,
                    'ratio': result['n_both'] / result['expected'][1, 1] if (
                        result['expected'].shape == (2, 2) and result['expected'][1, 1] > 0
                    ) else np.nan
                })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('chi2', ascending=False)
        
        # Print summary
        print(f"\nTested {len(df_results)} actor pairs")
        print("\nTop 10 strongest associations:")
        print(df_results.head(10).to_string(index=False))
        
        self.results.extend(results)
        return df_results


# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

class TemporalAnalyzer:
    """
    Analyze how co-occurrence patterns change over time.
    
    Tests hypothesis H3: Complexity increases over time.
    Also examines whether effect-actor associations change across time periods.
    """
    
    def __init__(self, df_binary: pd.DataFrame):
        """
        Initialize temporal analyzer.
        
        Parameters:
        -----------
        df_binary : pd.DataFrame
            Sentence-level data with year information
        """
        self.df = df_binary
        
    def test_concept_over_time(
        self, 
        concept: str,
        periods: List[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """
        Test if concept frequency changes across time periods.
        
        Parameters:
        -----------
        concept : str
            Concept name (e.g., 'complexity', 'efficiency')
        periods : List[Tuple[int, int]], optional
            List of (start_year, end_year) tuples defining periods
            Default: [(2011, 2015), (2016, 2019), (2020, 2024)]
            
        Returns:
        --------
        pd.DataFrame : Frequency of concept by period
        """
        if periods is None:
            periods = [(2011, 2015), (2016, 2019), (2020, 2024)]
        
        col = f'has_{concept}'
        results = []
        
        for start, end in periods:
            mask = (self.df['year'] >= start) & (self.df['year'] <= end)
            subset = self.df[mask]
            
            if len(subset) == 0:
                continue
                
            count = subset[col].sum()
            total = len(subset)
            pct = count / total * 100
            
            results.append({
                'period': f'{start}-{end}',
                'concept': concept,
                'count': count,
                'total': total,
                'percentage': pct
            })
        
        return pd.DataFrame(results)
    
    def compare_periods(
        self,
        concept1: str,
        concept2: str,
        period1: Tuple[int, int],
        period2: Tuple[int, int]
    ) -> Dict:
        """
        Compare co-occurrence between two time periods.
        
        Tests whether the association between two concepts is different 
        in period1 vs. period2.
        
        Parameters:
        -----------
        concept1, concept2 : str
            Concepts to test
        period1, period2 : Tuple[int, int]
            Time periods as (start_year, end_year)
            
        Returns:
        --------
        dict : Comparison results including chi-square tests for each period
        """
        # Get data for each period
        mask1 = (self.df['year'] >= period1[0]) & (self.df['year'] <= period1[1])
        mask2 = (self.df['year'] >= period2[0]) & (self.df['year'] <= period2[1])
        
        subset1 = self.df[mask1]
        subset2 = self.df[mask2]
        
        # Test co-occurrence in each period
        analyzer1 = CooccurrenceAnalyzer(subset1)
        analyzer2 = CooccurrenceAnalyzer(subset2)
        
        result1 = analyzer1.test_pair(concept1, concept2)
        result2 = analyzer2.test_pair(concept1, concept2)
        
        return {
            'concept1': concept1,
            'concept2': concept2,
            'period1': f'{period1[0]}-{period1[1]}',
            'period2': f'{period2[0]}-{period2[1]}',
            'period1_chi2': result1['chi2'],
            'period1_p': result1['p_value'],
            'period1_v': result1['cramers_v'],
            'period1_n_both': result1['n_both'],
            'period1_ratio': result1['n_both'] / result1['expected'][1,1] if result1['expected'][1,1] > 0 else np.nan,
            'period2_chi2': result2['chi2'],
            'period2_p': result2['p_value'],
            'period2_v': result2['cramers_v'],
            'period2_n_both': result2['n_both'],
            'period2_ratio': result2['n_both'] / result2['expected'][1,1] if result2['expected'][1,1] > 0 else np.nan
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

class ResultsVisualizer:
    """
    Create visualizations of co-occurrence analysis results.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save plots
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_effect_frequencies(self, df_binary: pd.DataFrame):
        """
        Plot bar chart of effect frequencies.
        
        Parameters:
        -----------
        df_binary : pd.DataFrame
            Sentence-level data with binary indicators
        """
        # Calculate frequencies
        frequencies = []
        for effect in EFFECT_CATEGORIES:
            col = f'has_{effect}'
            count = df_binary[col].sum()
            pct = count / len(df_binary) * 100
            frequencies.append({
                'effect': effect.capitalize(),
                'count': count,
                'percentage': pct
            })
        
        df_freq = pd.DataFrame(frequencies)
        df_freq = df_freq.sort_values('percentage', ascending=False)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df_freq['effect'], df_freq['percentage'], 
                     color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Political Effect', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage of Sentences', fontsize=12, fontweight='bold')
        ax.set_title('Frequency of Political Effects in RSA Documents', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, max(df_freq['percentage']) * 1.15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'effect_frequencies.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {self.output_dir / 'effect_frequencies.png'}")
    
    def plot_cooccurrence_heatmap(
        self, 
        df_results: pd.DataFrame,
        analysis_type: str = 'effect-actor'
    ):
        """
        Plot heatmap of chi-square values or CramÃ©r's V.
        
        Parameters:
        -----------
        df_results : pd.DataFrame
            Results from co-occurrence analysis
        analysis_type : str
            Type of analysis: 'effect-effect', 'effect-actor', or 'actor-actor'
        """
        # Determine column names based on analysis type
        if analysis_type == 'effect-actor':
            row_col, col_col = 'effect', 'actor'
            title = 'Effect-Actor Associations'
        elif analysis_type == 'effect-effect':
            row_col, col_col = 'concept1', 'concept2'
            title = 'Effect Co-occurrence'
        elif analysis_type == 'actor-actor':
            row_col, col_col = 'actor1', 'actor2'
            title = 'Actor Co-occurrence'
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")
        
        # Create pivot table
        pivot = df_results.pivot(index=row_col, columns=col_col, values='cramers_v')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': "CramÃ©r's V"}, ax=ax,
                   vmin=0, vmax=0.5, square=True)
        
        ax.set_title(f"{title}\n(CramÃ©r's V Effect Size)", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(col_col.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel(row_col.capitalize(), fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filename = f'{analysis_type.replace("-", "_")}_heatmap.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {self.output_dir / filename}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Main pipeline for co-occurrence analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Chi-square co-occurrence analysis for RSA documents'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('sentence_vectors_with_metadata.parquet'),
        help='Input parquet file with sentence data'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('cooccurrence_results'),
        help='Output directory for results and plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("CO-OCCURRENCE ANALYSIS FOR RSA DOCUMENTS")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print()
    
    # ==========================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ==========================================================================
    
    processor = SentenceDataProcessor(args.input)
    processor.load_data()
    df_binary = processor.create_binary_indicators()
    
    # Save binary indicators for inspection
    binary_output = args.output_dir / 'sentence_binary_indicators.parquet'
    df_binary.to_parquet(binary_output, index=False)
    print(f"\nSaved binary indicators: {binary_output}")
    
    # ==========================================================================
    # STEP 2: CHI-SQUARE TESTS
    # ==========================================================================
    
    analyzer = CooccurrenceAnalyzer(df_binary)
    
    # Test effect co-occurrence
    df_effect_cooc = analyzer.test_effect_cooccurrence()
    df_effect_cooc.to_csv(
        args.output_dir / 'effect_cooccurrence.csv', 
        index=False
    )
    
    # Test effect-actor associations
    df_effect_actor = analyzer.test_effect_actor_association()
    df_effect_actor.to_csv(
        args.output_dir / 'effect_actor_associations.csv',
        index=False
    )
    
    # Test actor co-occurrence
    df_actor_cooc = analyzer.test_actor_cooccurrence()
    df_actor_cooc.to_csv(
        args.output_dir / 'actor_cooccurrence.csv',
        index=False
    )
    
    # ==========================================================================
    # STEP 3: TEMPORAL ANALYSIS
    # ==========================================================================
    
    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS")
    print("="*80)
    
    temporal = TemporalAnalyzer(df_binary)
    
    # Test complexity over time (H3)
    df_complexity_time = temporal.test_concept_over_time('complexity')
    print("\nComplexity frequency over time:")
    print(df_complexity_time.to_string(index=False))
    
    # Test all effects over time
    temporal_results = []
    for effect in EFFECT_CATEGORIES:
        df_temp = temporal.test_concept_over_time(effect)
        temporal_results.append(df_temp)
    
    df_temporal_all = pd.concat(temporal_results, ignore_index=True)
    df_temporal_all.to_csv(
        args.output_dir / 'temporal_frequencies.csv',
        index=False
    )
    
    # ==========================================================================
    # STEP 4: VISUALIZATION
    # ==========================================================================
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    visualizer = ResultsVisualizer(args.output_dir)
    
    # Plot effect frequencies
    visualizer.plot_effect_frequencies(df_binary)
    
    # Plot heatmaps
    visualizer.plot_cooccurrence_heatmap(df_effect_actor, 'effect-actor')
    
    # ==========================================================================
    # STEP 5: SUMMARY REPORT
    # ==========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nOutput files:")
    print(f"  - sentence_binary_indicators.parquet")
    print(f"  - effect_cooccurrence.csv")
    print(f"  - effect_actor_associations.csv")
    print(f"  - actor_cooccurrence.csv")
    print(f"  - temporal_frequencies.csv")
    print(f"  - effect_frequencies.png")
    print(f"  - effect_actor_heatmap.png")
    print("\nNext steps:")
    print("  1. Review CSV files for detailed results")
    print("  2. Examine plots for visual patterns")
    print("  3. Run correspondence analysis for 2D mapping")
    print("  4. Conduct deeper temporal comparisons")


if __name__ == '__main__':
    main()
