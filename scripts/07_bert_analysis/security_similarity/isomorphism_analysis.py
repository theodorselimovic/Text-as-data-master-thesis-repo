#!/usr/bin/env python3
"""
Isomorphism Analysis: Measuring Municipal Copying of Security Risk Framing

Computes semantic similarity between municipal RSA documents and reference documents
(MSB, prefectures) to measure institutional isomorphism in security risk framing.

Uses Swedish Sentence-BERT embeddings with two similarity measures:
1. Max-Match Averaging - captures sentence-level "borrowing"
2. Earth Mover's Distance (EMD) - captures distributional similarity

Two baselines control for confounds:
1. Within-document: different risk category in same RSA
2. Cross-municipality: same risk category in different municipality

Usage:
    python isomorphism_analysis.py \
        --input data/processed/bert_corpus.parquet \
        --output results/02_bert_analysis/security_similarity/ \
        --verbose

    # Run with cached embeddings
    python isomorphism_analysis.py \
        --input data/processed/bert_corpus.parquet \
        --output results/02_bert_analysis/security_similarity/ \
        --embeddings data/processed/sbert_embeddings_sample.npz

Requirements:
    pip install pandas pyarrow numpy torch sentence-transformers POT matplotlib seaborn tqdm
"""

import argparse
import logging
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from tqdm import tqdm

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Security risk categories for isomorphism analysis
SECURITY_CATEGORIES = ["cyber_hot", "antagonistiska_hot"]

# Comparison risk categories (non-security) for baseline comparison
# Using common, well-populated categories
COMPARISON_CATEGORIES = ["naturhot", "teknisk_infrastruktur", "biologiska_hot"]

# Minimum sentences required for a text unit to be included
MIN_SENTENCES = 5

# Default batch size for embedding extraction
DEFAULT_BATCH_SIZES = {"cuda": 32, "mps": 16, "cpu": 8}

# Actor colors (consistent with project conventions)
ACTOR_COLORS = {
    "kommun": "#e41a1c",  # Red
    "lansstyrelse": "#377eb8",  # Blue
    "MCF": "#4daf4a",  # Green
}

# Wave mapping
def map_year_to_wave(year: int) -> int:
    """Map year to analysis wave."""
    if year < 2015:
        return 0
    elif year <= 2018:
        return 1
    elif year <= 2022:
        return 2
    else:
        return 3


# =============================================================================
# RISK DICTIONARY
# =============================================================================

# Import centralized dictionary from scripts/dictionaries/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dictionaries import get_legacy_risk_dictionary

RISK_DICTIONARY = get_legacy_risk_dictionary(include_extended=False)

# =============================================================================
# MUNICIPALITY TO LÄN MAPPING
# =============================================================================

MUNICIPALITY_TO_LAN = {
    # 1 - Stockholm
    "botkyrka": 1, "danderyd": 1, "ekerö": 1, "haninge": 1, "huddinge": 1,
    "järfälla": 1, "lidingö": 1, "nacka": 1, "norrtälje": 1, "nykvarn": 1,
    "nynäshamn": 1, "salem": 1, "sigtuna": 1, "sollentuna": 1, "solna": 1,
    "stockholm": 1, "sundbyberg": 1, "södertälje": 1, "tyresö": 1, "täby": 1,
    "upplands-bro": 1, "upplands väsby": 1, "vallentuna": 1, "vaxholm": 1,
    "värmdö": 1, "österåker": 1,
    # 2 - Uppsala
    "enköping": 2, "heby": 2, "håbo": 2, "knivsta": 2, "tierp": 2,
    "uppsala": 2, "älvkarleby": 2, "östhammar": 2,
    # 3 - Södermanland
    "eskilstuna": 3, "flen": 3, "gnesta": 3, "katrineholm": 3, "nyköping": 3,
    "oxelösund": 3, "strängnäs": 3, "trosa": 3, "vingåker": 3,
    # 4 - Östergötland
    "boxholm": 4, "finspång": 4, "kinda": 4, "linköping": 4, "mjölby": 4,
    "motala": 4, "norrköping": 4, "söderköping": 4, "vadstena": 4,
    "valdemarsvik": 4, "ydre": 4, "åtvidaberg": 4, "ödeshög": 4,
    # 5 - Jönköping
    "aneby": 5, "eksjö": 5, "gislaved": 5, "gnosjö": 5, "habo": 5,
    "jönköping": 5, "mullsjö": 5, "nässjö": 5, "sävsjö": 5, "tranås": 5,
    "vaggeryd": 5, "vetlanda": 5, "värnamo": 5,
    # 6 - Kronoberg
    "alvesta": 6, "lessebo": 6, "ljungby": 6, "markaryd": 6, "tingsryd": 6,
    "uppvidinge": 6, "växjö": 6, "älmhult": 6,
    # 7 - Kalmar
    "borgholm": 7, "emmaboda": 7, "hultsfred": 7, "högsby": 7, "kalmar": 7,
    "mönsterås": 7, "mörbylånga": 7, "nybro": 7, "oskarshamn": 7, "torsås": 7,
    "vimmerby": 7, "västervik": 7,
    # 8 - Gotland
    "gotland": 8,
    # 9 - Blekinge
    "karlshamn": 9, "karlskrona": 9, "olofström": 9, "ronneby": 9, "sölvesborg": 9,
    # 10 - Skåne
    "bjuv": 10, "bromölla": 10, "burlöv": 10, "båstad": 10, "eslöv": 10,
    "helsingborg": 10, "hässleholm": 10, "höganäs": 10, "hörby": 10, "höör": 10,
    "klippan": 10, "kristianstad": 10, "kävlinge": 10, "landskrona": 10,
    "lomma": 10, "lund": 10, "malmö": 10, "osby": 10, "perstorp": 10,
    "simrishamn": 10, "sjöbo": 10, "skurup": 10, "staffanstorp": 10, "svalöv": 10,
    "svedala": 10, "tomelilla": 10, "trelleborg": 10, "vellinge": 10, "ystad": 10,
    "åstorp": 10, "ängelholm": 10, "örkelljunga": 10, "östra göinge": 10,
    # 11 - Halland
    "falkenberg": 11, "halmstad": 11, "hylte": 11, "kungsbacka": 11,
    "laholm": 11, "varberg": 11,
    # 12 - Västra Götaland
    "ale": 12, "alingsås": 12, "bengtsfors": 12, "bollebygd": 12, "borås": 12,
    "dals-ed": 12, "essunga": 12, "falköping": 12, "färgelanda": 12,
    "grästorp": 12, "gullspång": 12, "göteborg": 12, "götene": 12,
    "herrljunga": 12, "hjo": 12, "härryda": 12, "karlsborg": 12, "kungälv": 12,
    "lerum": 12, "lidköping": 12, "lilla edet": 12, "lysekil": 12,
    "mariestad": 12, "mark": 12, "mellerud": 12, "munkedal": 12, "mölndal": 12,
    "orust": 12, "partille": 12, "skara": 12, "skövde": 12, "sotenäs": 12,
    "stenungsund": 12, "strömstad": 12, "svenljunga": 12, "tanum": 12,
    "tibro": 12, "tidaholm": 12, "tjörn": 12, "tranemo": 12, "trollhättan": 12,
    "töreboda": 12, "uddevalla": 12, "ulricehamn": 12, "vara": 12,
    "vårgårda": 12, "vänersborg": 12, "åmål": 12, "öckerö": 12,
    # 13 - Värmland
    "arvika": 13, "eda": 13, "filipstad": 13, "forshaga": 13, "grums": 13,
    "hagfors": 13, "hammarö": 13, "karlstad": 13, "kil": 13, "kristinehamn": 13,
    "munkfors": 13, "storfors": 13, "sunne": 13, "säffle": 13, "torsby": 13,
    "årjäng": 13,
    # 14 - Örebro
    "askersund": 14, "degerfors": 14, "hallsberg": 14, "hällefors": 14,
    "karlskoga": 14, "kumla": 14, "laxå": 14, "lekeberg": 14, "lindesberg": 14,
    "ljusnarsberg": 14, "nora": 14, "örebro": 14,
    # 15 - Västmanland
    "arboga": 15, "fagersta": 15, "hallstahammar": 15, "kungsör": 15,
    "köping": 15, "norberg": 15, "sala": 15, "skinnskatteberg": 15,
    "surahammar": 15, "västerås": 15,
    # 16 - Dalarna
    "avesta": 16, "borlänge": 16, "falun": 16, "gagnef": 16, "hedemora": 16,
    "leksand": 16, "ludvika": 16, "malung-sälen": 16, "mora": 16, "orsa": 16,
    "rättvik": 16, "smedjebacken": 16, "säter": 16, "vansbro": 16,
    "älvdalen": 16,
    # 17 - Gävleborg
    "bollnäs": 17, "gävle": 17, "hofors": 17, "hudiksvall": 17, "ljusdal": 17,
    "nordanstig": 17, "ockelbo": 17, "ovanåker": 17, "sandviken": 17,
    "söderhamn": 17,
    # 18 - Västernorrland
    "härnösand": 18, "kramfors": 18, "sollefteå": 18, "sundsvall": 18,
    "timrå": 18, "ånge": 18, "örnsköldsvik": 18,
    # 19 - Jämtland
    "berg": 19, "bräcke": 19, "härjedalen": 19, "krokom": 19, "ragunda": 19,
    "strömsund": 19, "åre": 19, "östersund": 19,
    # 20 - Västerbotten
    "bjurholm": 20, "dorotea": 20, "lycksele": 20, "malå": 20, "nordmaling": 20,
    "norsjö": 20, "robertsfors": 20, "skellefteå": 20, "sorsele": 20,
    "storuman": 20, "umeå": 20, "vilhelmina": 20, "vindeln": 20, "vännäs": 20,
    "åsele": 20,
    # 21 - Norrbotten
    "arjeplog": 21, "arvidsjaur": 21, "boden": 21, "gällivare": 21,
    "haparanda": 21, "jokkmokk": 21, "kalix": 21, "kiruna": 21, "luleå": 21,
    "pajala": 21, "piteå": 21, "älvsbyn": 21, "överkalix": 21, "övertorneå": 21,
}

LAN_CODES = {
    1: "Stockholm", 2: "Uppsala", 3: "Södermanland", 4: "Östergötland",
    5: "Jönköping", 6: "Kronoberg", 7: "Kalmar", 8: "Gotland", 9: "Blekinge",
    10: "Skåne", 11: "Halland", 12: "Västra Götaland", 13: "Värmland",
    14: "Örebro", 15: "Västmanland", 16: "Dalarna", 17: "Gävleborg",
    18: "Västernorrland", 19: "Jämtland", 20: "Västerbotten", 21: "Norrbotten",
}


# =============================================================================
# DEVICE MANAGER (copied from mechanism_classifier.py)
# =============================================================================

class DeviceManager:
    """Manages device detection and selection for PyTorch inference."""

    def __init__(self, requested_device: str = "auto"):
        self.requested_device = requested_device
        self.device = self._detect_device()
        self.device_type = self._get_device_type()

    def _detect_device(self) -> torch.device:
        if self.requested_device != "auto":
            if self.requested_device == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif (
                self.requested_device == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            elif self.requested_device == "cpu":
                return torch.device("cpu")

        # Auto-detect: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_device_type(self) -> str:
        device_str = str(self.device)
        if "cuda" in device_str:
            return "cuda"
        elif "mps" in device_str:
            return "mps"
        return "cpu"

    def get_recommended_batch_size(self) -> int:
        return DEFAULT_BATCH_SIZES.get(self.device_type, 8)

    def get_device_info(self) -> Dict[str, str]:
        info = {
            "device": str(self.device),
            "device_type": self.device_type,
            "torch_version": torch.__version__,
        }
        if self.device_type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
        elif self.device_type == "mps":
            info["gpu_name"] = "Apple Silicon (MPS)"
        return info


# =============================================================================
# RISK TERM DETECTION
# =============================================================================

def build_risk_patterns() -> Dict[str, re.Pattern]:
    """Build compiled regex patterns for each risk category."""
    patterns = {}
    for category, terms in RISK_DICTIONARY.items():
        # Sort by length (longest first) to avoid partial matches
        sorted_terms = sorted(terms, key=len, reverse=True)
        # Escape special regex characters and add word boundaries
        escaped = [re.escape(t) for t in sorted_terms]
        pattern_str = r"\b(" + "|".join(escaped) + r")\b"
        patterns[category] = re.compile(pattern_str, re.IGNORECASE)
    return patterns


def count_risk_terms(text: str, patterns: Dict[str, re.Pattern]) -> Dict[str, int]:
    """Count risk terms per category in text."""
    counts = {}
    for category, pattern in patterns.items():
        matches = pattern.findall(text)
        counts[category] = len(matches)
    return counts


# =============================================================================
# PARAGRAPH TAGGING
# =============================================================================

def tag_paragraphs_by_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each paragraph with its dominant risk category.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level corpus with columns: doc_id, paragraph_id, sentence_text

    Returns
    -------
    pd.DataFrame
        With added column: paragraph_risk_category
    """
    logger.info("Tagging paragraphs by risk category...")
    patterns = build_risk_patterns()

    # Aggregate sentences to paragraphs
    para_texts = (
        df.groupby(["doc_id", "paragraph_id"])["sentence_text"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )
    para_texts.columns = ["doc_id", "paragraph_id", "paragraph_text"]

    # Count risk terms per paragraph
    risk_counts = []
    for _, row in tqdm(para_texts.iterrows(), total=len(para_texts), desc="Counting risk terms"):
        counts = count_risk_terms(row["paragraph_text"], patterns)
        counts["doc_id"] = row["doc_id"]
        counts["paragraph_id"] = row["paragraph_id"]
        risk_counts.append(counts)

    risk_df = pd.DataFrame(risk_counts)

    # Determine dominant category (excluding generic categories)
    risk_cols = [c for c in RISK_DICTIONARY.keys() if c not in ["riskfamilj", "legitimitetsrisker"]]
    risk_df["dominant_category"] = risk_df[risk_cols].idxmax(axis=1)
    risk_df["max_count"] = risk_df[risk_cols].max(axis=1)

    # Only assign category if at least 1 term present
    risk_df.loc[risk_df["max_count"] == 0, "dominant_category"] = None

    # Merge back to sentence level
    df = df.merge(
        risk_df[["doc_id", "paragraph_id", "dominant_category"]],
        on=["doc_id", "paragraph_id"],
        how="left",
    )
    df = df.rename(columns={"dominant_category": "paragraph_risk_category"})

    # Log statistics
    tagged = df["paragraph_risk_category"].notna().sum()
    total = len(df)
    logger.info(f"  Tagged {tagged:,} / {total:,} sentences ({100*tagged/total:.1f}%)")

    return df


# =============================================================================
# TEXT UNIT EXTRACTION
# =============================================================================

def extract_text_units(
    df: pd.DataFrame,
    categories: List[str],
    min_sentences: int = MIN_SENTENCES,
) -> Dict[Tuple[str, int, str], pd.DataFrame]:
    """
    Extract text units (all paragraphs of a risk category in a document).

    Parameters
    ----------
    df : pd.DataFrame
        Tagged corpus
    categories : List[str]
        Risk categories to extract
    min_sentences : int
        Minimum sentences required

    Returns
    -------
    Dict[(doc_id, year, category), DataFrame]
        Mapping of text unit key to sentences
    """
    logger.info(f"Extracting text units for categories: {categories}")

    text_units = {}
    for category in categories:
        cat_df = df[df["paragraph_risk_category"] == category]

        # Group by document
        for doc_id, doc_group in cat_df.groupby("doc_id"):
            if len(doc_group) >= min_sentences:
                # Get year from first row
                year = int(doc_group["year"].iloc[0]) if "year" in doc_group.columns else 0
                text_units[(doc_id, year, category)] = doc_group.copy()

    logger.info(f"  Extracted {len(text_units)} text units")
    return text_units


# =============================================================================
# REFERENCE MATCHING
# =============================================================================

def get_municipality_name(doc_id: str) -> Optional[str]:
    """Extract municipality name from doc_id."""
    # Pattern: "RSA [Municipality] [Year] [Maskad].pdf"
    match = re.match(r"RSA\s+(.+?)\s+\d{4}", doc_id, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return None


def get_lan_for_municipality(municipality: str) -> Optional[int]:
    """Get län code for municipality."""
    muni_norm = unicodedata.normalize("NFC", municipality.lower().strip())

    if muni_norm in MUNICIPALITY_TO_LAN:
        return MUNICIPALITY_TO_LAN[muni_norm]

    # Try without "kommun" suffix
    muni_clean = muni_norm.replace(" kommun", "").replace("kommun", "").strip()
    if muni_clean in MUNICIPALITY_TO_LAN:
        return MUNICIPALITY_TO_LAN[muni_clean]

    return None


def find_nearest_year(target_year: int, available_years: List[int]) -> Optional[int]:
    """Find nearest available year."""
    if not available_years:
        return None
    return min(available_years, key=lambda y: abs(y - target_year))


def match_references(
    muni_units: Dict[Tuple[str, int, str], pd.DataFrame],
    all_units: Dict[Tuple[str, int, str], pd.DataFrame],
    df: pd.DataFrame,
) -> List[Dict]:
    """
    Match municipality text units to MSB and prefecture references.

    Returns list of comparison dicts with keys:
        muni_key, muni_df, msb_key, msb_df, pref_key, pref_df
    """
    logger.info("Matching municipality units to references...")

    # Index available reference units by actor, category, year
    msb_units = {}
    pref_units = {}

    for (doc_id, year, category), unit_df in all_units.items():
        actor = unit_df["actor_type"].iloc[0] if "actor_type" in unit_df.columns else None

        if actor == "MCF":
            if category not in msb_units:
                msb_units[category] = {}
            msb_units[category][year] = (doc_id, year, category)
        elif actor == "lansstyrelse":
            # Extract län from doc_id
            lan_match = None
            for lan_id, lan_name in LAN_CODES.items():
                if lan_name.lower() in doc_id.lower():
                    lan_match = lan_id
                    break
            if lan_match:
                if (category, lan_match) not in pref_units:
                    pref_units[(category, lan_match)] = {}
                pref_units[(category, lan_match)][year] = (doc_id, year, category)

    # Match each municipality unit
    comparisons = []
    for muni_key, muni_df in muni_units.items():
        doc_id, year, category = muni_key

        # Get municipality's län
        muni_name = get_municipality_name(doc_id)
        lan_id = get_lan_for_municipality(muni_name) if muni_name else None

        comparison = {
            "muni_key": muni_key,
            "muni_df": muni_df,
            "municipality": muni_name,
            "year": year,
            "category": category,
            "lan_id": lan_id,
            "msb_key": None,
            "msb_df": None,
            "pref_key": None,
            "pref_df": None,
        }

        # Find MSB reference
        if category in msb_units:
            available_years = list(msb_units[category].keys())
            nearest_year = find_nearest_year(year, available_years)
            if nearest_year:
                ref_key = msb_units[category][nearest_year]
                comparison["msb_key"] = ref_key
                comparison["msb_df"] = all_units[ref_key]

        # Find prefecture reference
        if lan_id and (category, lan_id) in pref_units:
            available_years = list(pref_units[(category, lan_id)].keys())
            nearest_year = find_nearest_year(year, available_years)
            if nearest_year:
                ref_key = pref_units[(category, lan_id)][nearest_year]
                comparison["pref_key"] = ref_key
                comparison["pref_df"] = all_units[ref_key]

        comparisons.append(comparison)

    # Log statistics
    has_msb = sum(1 for c in comparisons if c["msb_df"] is not None)
    has_pref = sum(1 for c in comparisons if c["pref_df"] is not None)
    logger.info(f"  Matched {has_msb} with MSB, {has_pref} with prefecture")

    return comparisons


# =============================================================================
# BASELINE SAMPLING
# =============================================================================

def sample_within_doc_baseline(
    df: pd.DataFrame,
    doc_id: str,
    exclude_category: str,
    rng: np.random.Generator,
) -> Optional[pd.DataFrame]:
    """Sample one paragraph from different risk category in same document."""
    doc_df = df[df["doc_id"] == doc_id]
    other_cats = doc_df[
        (doc_df["paragraph_risk_category"].notna()) &
        (doc_df["paragraph_risk_category"] != exclude_category)
    ]

    if len(other_cats) < MIN_SENTENCES:
        return None

    # Sample one paragraph
    available_paras = other_cats["paragraph_id"].unique()
    if len(available_paras) == 0:
        return None

    sampled_para = rng.choice(available_paras)
    return other_cats[other_cats["paragraph_id"] == sampled_para]


def sample_cross_muni_baseline(
    text_units: Dict[Tuple[str, int, str], pd.DataFrame],
    exclude_doc_id: str,
    category: str,
    rng: np.random.Generator,
) -> Optional[pd.DataFrame]:
    """Sample one text unit from different municipality, same category."""
    candidates = [
        (key, unit_df) for key, unit_df in text_units.items()
        if key[2] == category and key[0] != exclude_doc_id
    ]

    if not candidates:
        return None

    idx = rng.integers(0, len(candidates))
    return candidates[idx][1]


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

class SentenceBERTExtractor:
    """Extract sentence embeddings using Swedish Sentence-BERT."""

    def __init__(
        self,
        model_name: str = "KBLab/sentence-bert-swedish-cased",
        device: str = "auto",
    ):
        from sentence_transformers import SentenceTransformer

        self.device_manager = DeviceManager(device)
        logger.info(f"Loading SBERT model: {model_name}")
        logger.info(f"  Device: {self.device_manager.get_device_info()}")

        self.model = SentenceTransformer(model_name)
        self.model.to(self.device_manager.device)
        self.batch_size = self.device_manager.get_recommended_batch_size()

    def extract_embeddings(
        self,
        sentences: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract embeddings for sentences."""
        if not sentences:
            return np.array([])

        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings


# =============================================================================
# SIMILARITY COMPUTATION
# =============================================================================

def compute_max_match_similarity(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    return_details: bool = False,
) -> float:
    """
    Compute max-match averaged similarity.

    For each sentence in A, find max cosine similarity to any sentence in B.
    Return average of those maxima.

    If return_details=True, returns (score, max_indices, max_sims) for pair extraction.
    """
    if len(embeddings_a) == 0 or len(embeddings_b) == 0:
        if return_details:
            return np.nan, np.array([]), np.array([])
        return np.nan

    # Normalize for cosine similarity
    a_norm = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    b_norm = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Compute similarity matrix
    sim_matrix = a_norm @ b_norm.T

    # Max per row and indices
    max_sims = sim_matrix.max(axis=1)
    max_indices = sim_matrix.argmax(axis=1)

    if return_details:
        return float(max_sims.mean()), max_indices, max_sims

    return float(max_sims.mean())


def compute_emd_distance(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> float:
    """
    Compute Earth Mover's Distance between embedding distributions.

    Uses POT library for optimal transport.
    """
    if len(embeddings_a) == 0 or len(embeddings_b) == 0:
        return np.nan

    try:
        import ot
    except ImportError:
        logger.warning("POT not installed, skipping EMD computation")
        return np.nan

    n_a, n_b = len(embeddings_a), len(embeddings_b)

    # Uniform weights
    weights_a = np.ones(n_a) / n_a
    weights_b = np.ones(n_b) / n_b

    # Cost matrix (cosine distance)
    a_norm = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    b_norm = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)
    cost_matrix = 1 - (a_norm @ b_norm.T)

    # Compute EMD
    emd = ot.emd2(weights_a, weights_b, cost_matrix)
    return float(emd)


def extract_example_pairs(
    comparisons: List[Dict],
    embeddings: np.ndarray,
    sent_to_idx: Dict[str, int],
    n_high: int = 20,
    n_low: int = 20,
) -> pd.DataFrame:
    """
    Extract example sentence pairs with high and low similarity scores.

    Parameters
    ----------
    comparisons : List[Dict]
        Comparison dicts with muni_df, msb_df, etc.
    embeddings : np.ndarray
        All sentence embeddings
    sent_to_idx : Dict[str, int]
        Sentence text to embedding index mapping
    n_high : int
        Number of high-similarity pairs to extract
    n_low : int
        Number of low-similarity pairs to extract

    Returns
    -------
    pd.DataFrame
        Example pairs with columns: muni_sentence, ref_sentence, similarity, pair_type, reference_type
    """
    logger.info("Extracting example sentence pairs...")

    all_pairs = []

    for comp in comparisons:
        muni_df = comp["muni_df"]
        muni_sents = [str(s) for s in muni_df["sentence_text"]]

        # Get municipality embeddings
        muni_indices = [sent_to_idx.get(s) for s in muni_sents if s in sent_to_idx]
        if not muni_indices:
            continue
        muni_emb = embeddings[muni_indices]

        # Check MSB reference
        if comp["msb_df"] is not None:
            ref_df = comp["msb_df"]
            ref_sents = [str(s) for s in ref_df["sentence_text"]]
            ref_indices = [sent_to_idx.get(s) for s in ref_sents if s in sent_to_idx]

            if ref_indices:
                ref_emb = embeddings[ref_indices]
                _, max_idx, max_sims = compute_max_match_similarity(muni_emb, ref_emb, return_details=True)

                for i, (muni_sent, ref_idx, sim) in enumerate(zip(muni_sents, max_idx, max_sims)):
                    if i < len(muni_indices) and ref_idx < len(ref_sents):
                        all_pairs.append({
                            "municipality": comp["municipality"],
                            "year": comp["year"],
                            "risk_category": comp["category"],
                            "risk_type": comp["risk_type"],
                            "muni_sentence": muni_sent,
                            "ref_sentence": ref_sents[ref_idx],
                            "similarity": float(sim),
                            "reference_type": "MSB",
                        })

    if not all_pairs:
        return pd.DataFrame()

    pairs_df = pd.DataFrame(all_pairs)

    # Sort and sample high/low pairs
    pairs_df = pairs_df.sort_values("similarity", ascending=False)

    high_pairs = pairs_df.head(n_high).copy()
    high_pairs["pair_type"] = "high"

    low_pairs = pairs_df.tail(n_low).copy()
    low_pairs["pair_type"] = "low"

    # Also sample some medium pairs
    n_total = len(pairs_df)
    if n_total > n_high + n_low:
        mid_start = n_total // 2 - 10
        mid_pairs = pairs_df.iloc[mid_start:mid_start + 20].copy()
        mid_pairs["pair_type"] = "medium"
        result = pd.concat([high_pairs, mid_pairs, low_pairs], ignore_index=True)
    else:
        result = pd.concat([high_pairs, low_pairs], ignore_index=True)

    return result


# =============================================================================
# EFFECT SIZE COMPUTATION
# =============================================================================

def compute_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two distributions.

    d = (mean_a - mean_b) / pooled_SD

    Interpretation (conventional thresholds):
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std < 1e-10:
        return np.nan

    return (np.mean(a) - np.mean(b)) / pooled_std


def compute_probability_of_superiority(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute probability of superiority (common language effect size).

    P(A > B) = proportion of all pairwise comparisons where a value from A exceeds B.

    Interpretation:
        0.50: no difference (chance)
        0.56: small effect
        0.64: medium effect
        0.71: large effect

    Note: For large arrays, uses sampling to avoid O(n²) computation.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    # For large arrays, sample to avoid O(n²)
    max_comparisons = 100000
    n_comparisons = len(a) * len(b)

    if n_comparisons <= max_comparisons:
        # Exact computation
        count_superior = 0
        count_ties = 0
        for val_a in a:
            count_superior += np.sum(val_a > b)
            count_ties += np.sum(val_a == b)
        # Ties count as 0.5
        return (count_superior + 0.5 * count_ties) / n_comparisons
    else:
        # Sample-based approximation
        rng = np.random.default_rng(42)
        n_samples = max_comparisons
        idx_a = rng.integers(0, len(a), size=n_samples)
        idx_b = rng.integers(0, len(b), size=n_samples)
        samples_a = a[idx_a]
        samples_b = b[idx_b]
        return np.mean(samples_a > samples_b) + 0.5 * np.mean(samples_a == samples_b)


def compute_effect_sizes(results_df: pd.DataFrame) -> Dict:
    """
    Compute effect sizes for key distribution comparisons.

    Comparisons:
    1. Target vs baseline (MSB/Prefecture vs within-doc)
    2. Security vs other risks (for each similarity metric)

    Returns dict with effect sizes and interpretations.
    """
    effect_sizes = {}

    # 1. Target vs baseline comparisons
    comparisons = [
        ("msb_max_match", "within_doc_max_match", "MSB vs Within-doc baseline (max-match)"),
        ("prefecture_max_match", "within_doc_max_match", "Prefecture vs Within-doc baseline (max-match)"),
        ("msb_emd", "within_doc_emd", "MSB vs Within-doc baseline (EMD)"),
        ("prefecture_emd", "within_doc_emd", "Prefecture vs Within-doc baseline (EMD)"),
    ]

    for col_a, col_b, label in comparisons:
        if col_a in results_df.columns and col_b in results_df.columns:
            a = results_df[col_a].dropna().values
            b = results_df[col_b].dropna().values

            if len(a) > 0 and len(b) > 0:
                d = compute_cohens_d(a, b)
                ps = compute_probability_of_superiority(a, b)
                # Mann-Whitney U test
                stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')

                effect_sizes[label] = {
                    "cohens_d": d,
                    "prob_superiority": ps,
                    "mann_whitney_p": p_value,
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_a": np.mean(a),
                    "mean_b": np.mean(b),
                }

    # 2. Security vs other risks
    # Theory predicts: security risks show HIGHER similarity to MSB (more copying)
    # For max-match: higher = more similar, so test security > other
    # For EMD: lower = more similar, so test security < other
    if "risk_type" in results_df.columns:
        security = results_df[results_df["risk_type"] == "security"]
        other = results_df[results_df["risk_type"] == "other"]

        metrics = ["msb_max_match", "prefecture_max_match", "msb_emd", "prefecture_emd"]

        for metric in metrics:
            if metric in results_df.columns:
                a = security[metric].dropna().values
                b = other[metric].dropna().values

                if len(a) > 0 and len(b) > 0:
                    d = compute_cohens_d(a, b)
                    ps = compute_probability_of_superiority(a, b)

                    # One-tailed Mann-Whitney U test (directional hypothesis)
                    # max-match: security > other (greater similarity)
                    # EMD: security < other (lower distance = greater similarity)
                    if "emd" in metric:
                        alt = "less"  # security < other means more similar
                    else:
                        alt = "greater"  # security > other means more similar

                    stat, p_value = stats.mannwhitneyu(a, b, alternative=alt)

                    label = f"Security vs Other ({metric})"
                    effect_sizes[label] = {
                        "cohens_d": d,
                        "prob_superiority": ps,
                        "mann_whitney_p": p_value,
                        "test_direction": alt,
                        "n_security": len(a),
                        "n_other": len(b),
                        "mean_security": np.mean(a),
                        "mean_other": np.mean(b),
                    }

    return effect_sizes


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    if np.isnan(d):
        return "n/a"
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def interpret_prob_superiority(ps: float) -> str:
    """Interpret probability of superiority."""
    if np.isnan(ps):
        return "n/a"
    if ps < 0.56:
        return "negligible"
    elif ps < 0.64:
        return "small"
    elif ps < 0.71:
        return "medium"
    else:
        return "large"


def save_effect_sizes(effect_sizes: Dict, output_dir: Path):
    """Save effect sizes to CSV and text report."""

    # Save as CSV
    rows = []
    for label, values in effect_sizes.items():
        row = {"comparison": label}
        row.update(values)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "effect_sizes.csv", index=False)

    # Save as text report
    with open(output_dir / "effect_sizes_report.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EFFECT SIZE REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("Effect sizes and significance tests for distribution comparisons.\n\n")

        f.write("Cohen's d interpretation:\n")
        f.write("  |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, ≥0.8: large\n\n")

        f.write("Probability of Superiority (PS) interpretation:\n")
        f.write("  PS = probability that a random value from A exceeds one from B\n")
        f.write("  0.50: no difference, 0.56: small, 0.64: medium, 0.71: large\n\n")

        f.write("Mann-Whitney U test:\n")
        f.write("  Non-parametric test for difference in distributions\n")
        f.write("  p < 0.05: significant, p < 0.01: highly significant\n\n")

        f.write("-" * 70 + "\n\n")

        for label, values in effect_sizes.items():
            f.write(f"{label}\n")
            f.write("-" * len(label) + "\n")

            d = values.get("cohens_d", np.nan)
            ps = values.get("prob_superiority", np.nan)
            p_val = values.get("mann_whitney_p", np.nan)

            f.write(f"  Cohen's d:               {d:+.3f} ({interpret_cohens_d(d)})\n")
            f.write(f"  Prob. of Superiority:    {ps:.3f} ({interpret_prob_superiority(ps)})\n")
            if not np.isnan(p_val):
                test_dir = values.get("test_direction", "two-sided")
                tail_str = " (one-tailed)" if test_dir in ["greater", "less"] else ""
                if p_val < 0.001:
                    f.write(f"  Mann-Whitney p:          <0.001 ***{tail_str}\n")
                elif p_val < 0.01:
                    f.write(f"  Mann-Whitney p:          {p_val:.3f} **{tail_str}\n")
                elif p_val < 0.05:
                    f.write(f"  Mann-Whitney p:          {p_val:.3f} *{tail_str}\n")
                else:
                    f.write(f"  Mann-Whitney p:          {p_val:.3f} (ns){tail_str}\n")

            # Report sample sizes and means
            if "n_a" in values:
                f.write(f"  N (group A):             {values['n_a']}\n")
                f.write(f"  N (group B):             {values['n_b']}\n")
                f.write(f"  Mean A:                  {values['mean_a']:.3f}\n")
                f.write(f"  Mean B:                  {values['mean_b']:.3f}\n")
            elif "n_security" in values:
                f.write(f"  N (security):            {values['n_security']}\n")
                f.write(f"  N (other):               {values['n_other']}\n")
                f.write(f"  Mean (security):         {values['mean_security']:.3f}\n")
                f.write(f"  Mean (other):            {values['mean_other']:.3f}\n")

            f.write("\n")

    logger.info(f"Saved effect sizes to {output_dir / 'effect_sizes.csv'}")
    logger.info(f"Saved effect size report to {output_dir / 'effect_sizes_report.txt'}")


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_isomorphism_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    cache_embeddings: bool = True,
) -> pd.DataFrame:
    """
    Run full isomorphism analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Sentence-level corpus
    output_dir : Path
        Output directory
    seed : int
        Random seed for baseline sampling
    cache_embeddings : bool
        Whether to cache embeddings to disk

    Returns
    -------
    pd.DataFrame
        Results with isomorphism scores
    """
    rng = np.random.default_rng(seed)

    # Step 1: Tag paragraphs
    df = tag_paragraphs_by_risk(df)

    # Step 2: Extract text units
    all_units = extract_text_units(df, list(RISK_DICTIONARY.keys()))

    # Filter to municipality units for both security and comparison categories
    all_target_categories = SECURITY_CATEGORIES + COMPARISON_CATEGORIES
    muni_units = {
        k: v for k, v in all_units.items()
        if k[2] in all_target_categories
        and v["actor_type"].iloc[0] == "kommun"
    }

    # Separate for logging
    security_units = {k: v for k, v in muni_units.items() if k[2] in SECURITY_CATEGORIES}
    comparison_units = {k: v for k, v in muni_units.items() if k[2] in COMPARISON_CATEGORIES}
    logger.info(f"Found {len(security_units)} municipality security risk units")
    logger.info(f"Found {len(comparison_units)} municipality comparison risk units")

    # Step 3: Match references
    comparisons = match_references(muni_units, all_units, df)

    # Tag each comparison with risk_type
    for comp in comparisons:
        comp["risk_type"] = "security" if comp["category"] in SECURITY_CATEGORIES else "other"

    # Step 4: Collect sentences to embed
    sentences_to_embed = set()
    for comp in comparisons:
        # Municipality sentences
        for sent in comp["muni_df"]["sentence_text"]:
            sentences_to_embed.add(str(sent))
        # MSB sentences
        if comp["msb_df"] is not None:
            for sent in comp["msb_df"]["sentence_text"]:
                sentences_to_embed.add(str(sent))
        # Prefecture sentences
        if comp["pref_df"] is not None:
            for sent in comp["pref_df"]["sentence_text"]:
                sentences_to_embed.add(str(sent))

    # Add baseline sentences
    for comp in comparisons:
        # Within-doc baseline
        baseline_df = sample_within_doc_baseline(
            df, comp["muni_key"][0], comp["category"], rng
        )
        comp["within_doc_df"] = baseline_df
        if baseline_df is not None:
            for sent in baseline_df["sentence_text"]:
                sentences_to_embed.add(str(sent))

        # Cross-muni baseline
        cross_df = sample_cross_muni_baseline(
            muni_units, comp["muni_key"][0], comp["category"], rng
        )
        comp["cross_muni_df"] = cross_df
        if cross_df is not None:
            for sent in cross_df["sentence_text"]:
                sentences_to_embed.add(str(sent))

    sentences_list = list(sentences_to_embed)
    logger.info(f"Total sentences to embed: {len(sentences_list)}")

    # Step 5: Extract embeddings
    extractor = SentenceBERTExtractor()
    embeddings = extractor.extract_embeddings(sentences_list)

    # Create sentence -> embedding lookup
    sent_to_idx = {s: i for i, s in enumerate(sentences_list)}

    def get_embeddings_for_df(sent_df: pd.DataFrame) -> np.ndarray:
        if sent_df is None or len(sent_df) == 0:
            return np.array([])
        indices = [sent_to_idx[str(s)] for s in sent_df["sentence_text"] if str(s) in sent_to_idx]
        if not indices:
            return np.array([])
        return embeddings[indices]

    # Step 6: Compute similarities
    logger.info("Computing similarity measures...")
    results = []

    for comp in tqdm(comparisons, desc="Computing similarities"):
        muni_emb = get_embeddings_for_df(comp["muni_df"])

        result = {
            "municipality": comp["municipality"],
            "year": comp["year"],
            "risk_category": comp["category"],
            "risk_type": comp["risk_type"],  # "security" or "other"
            "lan_id": comp["lan_id"],
            "n_sentences": len(muni_emb),
            "wave": map_year_to_wave(comp["year"]),
        }

        # MSB comparison
        if comp["msb_df"] is not None:
            msb_emb = get_embeddings_for_df(comp["msb_df"])
            result["msb_max_match"] = compute_max_match_similarity(muni_emb, msb_emb)
            result["msb_emd"] = compute_emd_distance(muni_emb, msb_emb)
        else:
            result["msb_max_match"] = np.nan
            result["msb_emd"] = np.nan

        # Prefecture comparison
        if comp["pref_df"] is not None:
            pref_emb = get_embeddings_for_df(comp["pref_df"])
            result["prefecture_max_match"] = compute_max_match_similarity(muni_emb, pref_emb)
            result["prefecture_emd"] = compute_emd_distance(muni_emb, pref_emb)
        else:
            result["prefecture_max_match"] = np.nan
            result["prefecture_emd"] = np.nan

        # Within-doc baseline
        if comp["within_doc_df"] is not None:
            within_emb = get_embeddings_for_df(comp["within_doc_df"])
            result["within_doc_max_match"] = compute_max_match_similarity(muni_emb, within_emb)
            result["within_doc_emd"] = compute_emd_distance(muni_emb, within_emb)
        else:
            result["within_doc_max_match"] = np.nan
            result["within_doc_emd"] = np.nan

        # Cross-muni baseline
        if comp["cross_muni_df"] is not None:
            cross_emb = get_embeddings_for_df(comp["cross_muni_df"])
            result["cross_muni_max_match"] = compute_max_match_similarity(muni_emb, cross_emb)
            result["cross_muni_emd"] = compute_emd_distance(muni_emb, cross_emb)
        else:
            result["cross_muni_max_match"] = np.nan
            result["cross_muni_emd"] = np.nan

        # Compute isomorphism indices
        if not np.isnan(result["msb_max_match"]) and not np.isnan(result["within_doc_max_match"]):
            baseline = result["within_doc_max_match"]
            if baseline < 1:
                result["isomorphism_index_msb"] = (result["msb_max_match"] - baseline) / (1 - baseline)
            else:
                result["isomorphism_index_msb"] = np.nan
        else:
            result["isomorphism_index_msb"] = np.nan

        if not np.isnan(result["prefecture_max_match"]) and not np.isnan(result["within_doc_max_match"]):
            baseline = result["within_doc_max_match"]
            if baseline < 1:
                result["isomorphism_index_prefecture"] = (result["prefecture_max_match"] - baseline) / (1 - baseline)
            else:
                result["isomorphism_index_prefecture"] = np.nan
        else:
            result["isomorphism_index_prefecture"] = np.nan

        results.append(result)

    results_df = pd.DataFrame(results)

    # Save results
    output_path = output_dir / "isomorphism_scores.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    # Extract and save example sentence pairs for verification
    example_pairs = extract_example_pairs(comparisons, embeddings, sent_to_idx)
    if len(example_pairs) > 0:
        pairs_path = output_dir / "example_sentence_pairs.csv"
        example_pairs.to_csv(pairs_path, index=False)
        logger.info(f"Saved {len(example_pairs)} example pairs to {pairs_path}")

    return results_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    logger.info("Creating visualizations...")

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Box plot: similarity by risk category
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MSB comparison
    ax1 = axes[0]
    valid_msb = results_df[results_df["msb_max_match"].notna()]
    if len(valid_msb) > 0:
        sns.boxplot(
            data=valid_msb,
            x="risk_category",
            y="msb_max_match",
            ax=ax1,
            palette="Set2",
        )
        ax1.set_title("Municipality → MSB Similarity")
        ax1.set_xlabel("Risk Category")
        ax1.set_ylabel("Max-Match Similarity")
        ax1.set_ylim(0, 1)

    # Prefecture comparison
    ax2 = axes[1]
    valid_pref = results_df[results_df["prefecture_max_match"].notna()]
    if len(valid_pref) > 0:
        sns.boxplot(
            data=valid_pref,
            x="risk_category",
            y="prefecture_max_match",
            ax=ax2,
            palette="Set2",
        )
        ax2.set_title("Municipality → Prefecture Similarity")
        ax2.set_xlabel("Risk Category")
        ax2.set_ylabel("Max-Match Similarity")
        ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "similarity_by_category.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Violin plot: similarity distributions (MSB, Prefecture, Within-doc baseline)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Reshape data for violin plot
    plot_data = []
    for _, row in results_df.iterrows():
        if pd.notna(row.get("msb_max_match")):
            plot_data.append({"Comparison": "→ MSB", "Max-Match Similarity": row["msb_max_match"]})
        if pd.notna(row.get("prefecture_max_match")):
            plot_data.append({"Comparison": "→ Prefecture", "Max-Match Similarity": row["prefecture_max_match"]})
        if pd.notna(row.get("within_doc_max_match")):
            plot_data.append({"Comparison": "Within-doc\n(baseline)", "Max-Match Similarity": row["within_doc_max_match"]})

    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        order = ["→ MSB", "→ Prefecture", "Within-doc\n(baseline)"]
        palette = {"→ MSB": "#4daf4a", "→ Prefecture": "#377eb8", "Within-doc\n(baseline)": "#999999"}

        sns.violinplot(
            data=plot_df,
            x="Comparison",
            y="Max-Match Similarity",
            hue="Comparison",
            ax=ax,
            order=order,
            hue_order=order,
            palette=palette,
            inner="box",
            legend=False,
        )
        ax.set_title("Similarity Distributions by Comparison Type")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "similarity_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2b. Violin plot: EMD distributions (inverted so higher = more similar)
    fig, ax = plt.subplots(figsize=(12, 6))

    emd_data = []
    for _, row in results_df.iterrows():
        if pd.notna(row.get("msb_emd")):
            # Invert: 1 - EMD so higher = more similar
            emd_data.append({"Comparison": "→ MSB", "EMD Similarity (1 - EMD)": 1 - row["msb_emd"]})
        if pd.notna(row.get("prefecture_emd")):
            emd_data.append({"Comparison": "→ Prefecture", "EMD Similarity (1 - EMD)": 1 - row["prefecture_emd"]})
        if pd.notna(row.get("within_doc_emd")):
            emd_data.append({"Comparison": "Within-doc\n(baseline)", "EMD Similarity (1 - EMD)": 1 - row["within_doc_emd"]})

    if emd_data:
        emd_df = pd.DataFrame(emd_data)
        order = ["→ MSB", "→ Prefecture", "Within-doc\n(baseline)"]
        palette = {"→ MSB": "#4daf4a", "→ Prefecture": "#377eb8", "Within-doc\n(baseline)": "#999999"}

        sns.violinplot(
            data=emd_df,
            x="Comparison",
            y="EMD Similarity (1 - EMD)",
            hue="Comparison",
            ax=ax,
            order=order,
            hue_order=order,
            palette=palette,
            inner="box",
            legend=False,
        )
        ax.set_title("EMD Similarity Distributions (Higher = More Similar)")
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(output_dir / "emd_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2c. Bar chart: target vs baseline means (keep for reference)
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["msb_max_match", "prefecture_max_match", "within_doc_max_match", "cross_muni_max_match"]
    labels = ["→ MSB", "→ Prefecture", "Within-doc\n(baseline)", "Cross-muni\n(baseline)"]

    means = [results_df[m].mean() for m in metrics]
    stds = [results_df[m].std() for m in metrics]

    colors = ["#4daf4a", "#377eb8", "#999999", "#999999"]
    x_pos = range(len(metrics))

    bars = ax.bar(x_pos, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Max-Match Similarity")
    ax.set_title("Target vs Baseline Similarity (Means)")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "target_vs_baseline.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Security vs Other risk types comparison
    if "risk_type" in results_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # MSB isomorphism by risk type
        ax1 = axes[0]
        valid_msb = results_df[results_df["isomorphism_index_msb"].notna()]
        if len(valid_msb) > 0:
            sns.boxplot(
                data=valid_msb,
                x="risk_type",
                y="isomorphism_index_msb",
                ax=ax1,
                palette={"security": "#e41a1c", "other": "#377eb8"},
                order=["security", "other"],
            )
            ax1.set_title("Isomorphism with MSB: Security vs Other Risks")
            ax1.set_xlabel("Risk Type")
            ax1.set_ylabel("Isomorphism Index (MSB)")
            ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Prefecture isomorphism by risk type
        ax2 = axes[1]
        valid_pref = results_df[results_df["isomorphism_index_prefecture"].notna()]
        if len(valid_pref) > 0:
            sns.boxplot(
                data=valid_pref,
                x="risk_type",
                y="isomorphism_index_prefecture",
                ax=ax2,
                palette={"security": "#e41a1c", "other": "#377eb8"},
                order=["security", "other"],
            )
            ax2.set_title("Isomorphism with Prefecture: Security vs Other Risks")
            ax2.set_xlabel("Risk Type")
            ax2.set_ylabel("Isomorphism Index (Prefecture)")
            ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / "security_vs_other.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3b. Security vs Other: 2x2 violin plot with Mann-Whitney p-values
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics_config = [
            ("msb_max_match", "Max-Match Similarity to MSB", axes[0, 0]),
            ("prefecture_max_match", "Max-Match Similarity to Prefecture", axes[0, 1]),
            ("msb_emd", "EMD Similarity to MSB", axes[1, 0]),
            ("prefecture_emd", "EMD Similarity to Prefecture", axes[1, 1]),
        ]

        for metric, title, ax in metrics_config:
            if metric not in results_df.columns:
                continue

            valid_data = results_df[results_df[metric].notna()]
            if len(valid_data) == 0:
                continue

            security_vals = valid_data[valid_data["risk_type"] == "security"][metric].values
            other_vals = valid_data[valid_data["risk_type"] == "other"][metric].values

            # For EMD, invert so higher = more similar
            if "emd" in metric:
                y_col = f"{metric}_inv"
                valid_data = valid_data.copy()
                valid_data[y_col] = 1 - valid_data[metric]
                ylabel = "EMD Similarity (higher = more similar)"
                security_vals_test = 1 - security_vals
                other_vals_test = 1 - other_vals
            else:
                y_col = metric
                ylabel = "Max-Match (higher = more similar)"
                security_vals_test = security_vals
                other_vals_test = other_vals

            # One-tailed Mann-Whitney U test (directional hypothesis)
            # Theory: security risks show higher similarity (more copying from MSB)
            # For max-match (higher = more similar): test security > other
            # For EMD (inverted, so higher = more similar): test security > other
            if len(security_vals) > 0 and len(other_vals) > 0:
                # After inversion, both metrics: higher = more similar
                # So we test security > other with alternative='greater'
                stat, p_val = stats.mannwhitneyu(security_vals_test, other_vals_test, alternative='greater')
                if p_val < 0.001:
                    p_str = "p < 0.001 ***"
                elif p_val < 0.01:
                    p_str = f"p = {p_val:.3f} **"
                elif p_val < 0.05:
                    p_str = f"p = {p_val:.2f} *"
                else:
                    p_str = f"p = {p_val:.2f} (ns)"
                p_str += " (one-tailed)"
            else:
                p_str = ""

            sns.violinplot(
                data=valid_data,
                x="risk_type",
                y=y_col,
                ax=ax,
                palette={"security": "#e41a1c", "other": "#377eb8"},
                order=["security", "other"],
                inner="box",
            )
            ax.set_title(f"{title}\n{p_str}")
            ax.set_xlabel("Risk Type")
            ax.set_ylabel(ylabel)

        plt.tight_layout()
        plt.savefig(output_dir / "similarity_measures.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4. Temporal trends
    fig, ax = plt.subplots(figsize=(10, 6))

    wave_means = results_df.groupby("wave")[["msb_max_match", "prefecture_max_match"]].mean()

    # Wave labels mapping (wave number -> display label)
    WAVE_LABELS = {0: "Pre-2015", 1: "2015-18", 2: "2019-22", 3: "2023+"}

    if len(wave_means) > 1:
        wave_means.plot(ax=ax, marker="o", linewidth=2)
        ax.set_xlabel("Wave")
        ax.set_ylabel("Max-Match Similarity")
        ax.set_title("Isomorphism Trends Over Time")
        ax.legend(["→ MSB", "→ Prefecture"])
        ax.set_xticks(wave_means.index)
        ax.set_xticklabels([WAVE_LABELS.get(w, str(w)) for w in wave_means.index])

    plt.tight_layout()
    plt.savefig(output_dir / "temporal_trends.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualizations to {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure institutional isomorphism in RSA security risk framing"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/bert_corpus.parquet"),
        help="Input corpus parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/02_bert_analysis/security_similarity"),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2015,
        help="Minimum year to include (default: 2015)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load corpus
    logger.info(f"Loading corpus from {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"  Loaded {len(df):,} sentences from {df['doc_id'].nunique()} documents")

    # Filter by year
    if args.min_year:
        n_before = len(df)
        df = df[df["year"].astype(int) >= args.min_year]
        logger.info(f"  Filtered to >= {args.min_year}: {len(df):,} sentences ({n_before - len(df):,} removed)")

    # Run analysis
    results_df = run_isomorphism_analysis(df, args.output, seed=args.seed)

    # Create visualizations
    create_visualizations(results_df, args.output)

    # Compute and save effect sizes
    logger.info("\nComputing effect sizes...")
    effect_sizes = compute_effect_sizes(results_df)
    save_effect_sizes(effect_sizes, args.output)

    # Print summary
    logger.info("\n=== Summary ===")
    logger.info(f"Total comparisons: {len(results_df)}")
    logger.info(f"With MSB match: {results_df['msb_max_match'].notna().sum()}")
    logger.info(f"With prefecture match: {results_df['prefecture_max_match'].notna().sum()}")

    logger.info("\nMean similarities (all):")
    logger.info(f"  → MSB: {results_df['msb_max_match'].mean():.3f}")
    logger.info(f"  → Prefecture: {results_df['prefecture_max_match'].mean():.3f}")
    logger.info(f"  Within-doc baseline: {results_df['within_doc_max_match'].mean():.3f}")
    logger.info(f"  Cross-muni baseline: {results_df['cross_muni_max_match'].mean():.3f}")

    logger.info("\nIsomorphism indices (all):")
    logger.info(f"  MSB: {results_df['isomorphism_index_msb'].mean():.3f}")
    logger.info(f"  Prefecture: {results_df['isomorphism_index_prefecture'].mean():.3f}")

    # Breakdown by risk type
    if "risk_type" in results_df.columns:
        logger.info("\n=== Security vs Other Risks ===")
        for risk_type in ["security", "other"]:
            subset = results_df[results_df["risk_type"] == risk_type]
            logger.info(f"\n{risk_type.upper()} risks (n={len(subset)}):")
            logger.info(f"  → MSB similarity: {subset['msb_max_match'].mean():.3f}")
            logger.info(f"  → Prefecture similarity: {subset['prefecture_max_match'].mean():.3f}")
            logger.info(f"  Isomorphism index (MSB): {subset['isomorphism_index_msb'].mean():.3f}")
            logger.info(f"  Isomorphism index (Prefecture): {subset['isomorphism_index_prefecture'].mean():.3f}")

    # Effect sizes summary
    logger.info("\n=== Effect Sizes (Distribution Comparisons) ===")
    for label, values in effect_sizes.items():
        d = values.get("cohens_d", np.nan)
        ps = values.get("prob_superiority", np.nan)
        logger.info(f"\n{label}:")
        logger.info(f"  Cohen's d: {d:+.3f} ({interpret_cohens_d(d)})")
        logger.info(f"  P(A > B):  {ps:.3f} ({interpret_prob_superiority(ps)})")


if __name__ == "__main__":
    main()
