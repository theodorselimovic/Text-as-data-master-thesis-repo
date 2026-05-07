#!/usr/bin/env python3
"""
Named Entity Recognition (NER) Extraction Script

Extracts named entities from the RSA corpus using the Swedish BERT NER model
from KBLab (Royal Library of Sweden). Identifies 5 entity types:
    - TME: Time expressions
    - PRS: Personal names
    - LOC: Locations (geographic entities)
    - EVN: Events
    - ORG: Organizations

Designed to work on both M1 MacBook (MPS backend) and Google Colab (CUDA).

Input Format:
    Parquet file with columns:
    - sentence_text: Text to process
    - doc_id: Document identifier
    - (optional) paragraph_id, actor_type, year, wave

Output Format:
    1. entities.csv - All extracted entities with positions and confidence
    2. entities_by_sentence.csv - Entity counts per sentence
    3. entities_by_document.csv - Entity counts per document with metadata
    4. ner_report.json - Summary statistics and processing metadata

Usage:
    # Basic usage (auto-detect device)
    python ner_extraction.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/05_ner/

    # Specify device and batch size
    python ner_extraction.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/05_ner/ \\
        --device mps \\
        --batch-size 16

    # Filter to specific entity types
    python ner_extraction.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/05_ner/ \\
        --entity-types LOC ORG

    # Test on small sample
    python ner_extraction.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/05_ner/ \\
        --max-sentences 1000 \\
        --verbose

Requirements:
    pip install transformers torch pandas pyarrow

Model:
    KBLab/bert-base-swedish-cased-ner (Hugging Face)

Author: Swedish Risk Analysis Text-as-Data Project
Version: 1.0
Date: 2025-02-24
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration
MODEL_NAME = "KBLab/bert-base-swedish-cased-ner"

# Entity types supported by the model
ENTITY_TYPES = ["TME", "PRS", "LOC", "EVN", "ORG"]

# Entity type descriptions
ENTITY_DESCRIPTIONS = {
    "TME": "Time expressions (dates, periods, times)",
    "PRS": "Personal names (people)",
    "LOC": "Locations (geographic entities, places)",
    "EVN": "Events (named events, incidents)",
    "ORG": "Organizations (companies, agencies, institutions)",
}

# Default batch sizes per device type (tuned for memory efficiency)
DEFAULT_BATCH_SIZES = {
    "cuda": 32,
    "mps": 16,
    "cpu": 8,
}

# Checkpoint frequency (save progress every N sentences)
CHECKPOINT_FREQUENCY = 5000

# Minimum confidence threshold for entity extraction
DEFAULT_MIN_CONFIDENCE = 0.5


# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Entity:
    """Represents a single extracted named entity."""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    sentence_idx: int
    doc_id: str


@dataclass
class ProcessingStats:
    """Statistics for the NER processing run."""
    total_sentences: int = 0
    sentences_processed: int = 0
    total_entities: int = 0
    entities_by_type: Dict[str, int] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    device_used: str = ""
    batch_size: int = 0
    errors: int = 0
    oom_recoveries: int = 0


# =============================================================================
# DEVICE MANAGER
# =============================================================================

class DeviceManager:
    """Manages device detection and selection for PyTorch inference.

    Supports CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU fallback.
    """

    def __init__(self, requested_device: str = "auto"):
        """Initialize device manager.

        Parameters
        ----------
        requested_device : str
            Device to use: "auto", "cuda", "mps", or "cpu".
        """
        import torch
        self.torch = torch
        self.requested_device = requested_device
        self.device = self._detect_device()
        self.device_type = self._get_device_type()

    def _detect_device(self) -> "torch.device":
        """Detect the best available device."""
        torch = self.torch

        if self.requested_device != "auto":
            if self.requested_device == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif self.requested_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            elif self.requested_device == "cpu":
                return torch.device("cpu")
            else:
                logger.warning(
                    f"Requested device '{self.requested_device}' not available, "
                    "falling back to auto-detection"
                )

        # Auto-detect: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_device_type(self) -> str:
        """Get device type as string."""
        device_str = str(self.device)
        if "cuda" in device_str:
            return "cuda"
        elif "mps" in device_str:
            return "mps"
        else:
            return "cpu"

    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size for current device."""
        return DEFAULT_BATCH_SIZES.get(self.device_type, 8)

    def get_device_info(self) -> Dict[str, str]:
        """Get device information for reporting."""
        torch = self.torch
        info = {
            "device": str(self.device),
            "device_type": self.device_type,
            "torch_version": torch.__version__,
        }

        if self.device_type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
        elif self.device_type == "mps":
            info["gpu_name"] = "Apple Silicon (MPS)"

        return info


# =============================================================================
# NER EXTRACTOR
# =============================================================================

class NERExtractor:
    """Extracts named entities using Swedish BERT NER model.

    Handles model loading, batched inference, and subword token merging.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        entity_types: Optional[List[str]] = None,
    ):
        """Initialize NER extractor.

        Parameters
        ----------
        device_manager : DeviceManager
            Device manager for hardware acceleration.
        min_confidence : float
            Minimum confidence score to keep entities (0.0-1.0).
        entity_types : list of str, optional
            Entity types to extract. None = all types.
        """
        self.device_manager = device_manager
        self.min_confidence = min_confidence
        self.entity_types = set(entity_types) if entity_types else set(ENTITY_TYPES)

        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the NER model and tokenizer."""
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Device: {self.device_manager.device}")

        start_time = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device_manager.device)
        self.model.eval()

        # Create pipeline
        self.pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device_manager.device,
            aggregation_strategy="simple",  # Merges subword tokens
        )

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.1f}s")
        self._loaded = True

    def extract_entities(
        self,
        text: str,
        sentence_idx: int,
        doc_id: str,
    ) -> List[Entity]:
        """Extract entities from a single sentence.

        Parameters
        ----------
        text : str
            Input text.
        sentence_idx : int
            Sentence index for tracking.
        doc_id : str
            Document ID for tracking.

        Returns
        -------
        list of Entity
            Extracted entities.
        """
        if not self._loaded:
            self.load_model()

        if not text or not text.strip():
            return []

        try:
            # Run NER pipeline
            results = self.pipeline(text)

            entities = []
            for result in results:
                # Parse entity type (format: B-LOC, I-LOC, etc.)
                entity_group = result.get("entity_group", result.get("entity", ""))

                # Handle different label formats
                if "-" in entity_group:
                    entity_type = entity_group.split("-")[-1]
                else:
                    entity_type = entity_group

                # Filter by entity type
                if entity_type not in self.entity_types:
                    continue

                # Filter by confidence
                confidence = result.get("score", 0.0)
                if confidence < self.min_confidence:
                    continue

                entity = Entity(
                    text=result.get("word", "").strip(),
                    entity_type=entity_type,
                    start=result.get("start", 0),
                    end=result.get("end", 0),
                    confidence=confidence,
                    sentence_idx=sentence_idx,
                    doc_id=doc_id,
                )
                entities.append(entity)

            return entities

        except Exception as e:
            logger.debug(f"Error processing text: {e}")
            return []

    def extract_batch(
        self,
        texts: List[str],
        sentence_indices: List[int],
        doc_ids: List[str],
    ) -> List[Entity]:
        """Extract entities from a batch of sentences.

        Parameters
        ----------
        texts : list of str
            Input texts.
        sentence_indices : list of int
            Sentence indices for tracking.
        doc_ids : list of str
            Document IDs for tracking.

        Returns
        -------
        list of Entity
            Extracted entities from all texts.
        """
        if not self._loaded:
            self.load_model()

        all_entities = []

        # Process batch through pipeline
        try:
            # Filter empty texts
            valid_data = [
                (text, idx, doc_id)
                for text, idx, doc_id in zip(texts, sentence_indices, doc_ids)
                if text and text.strip()
            ]

            if not valid_data:
                return []

            valid_texts = [d[0] for d in valid_data]

            # Run batched inference
            batch_results = self.pipeline(valid_texts)

            # Process results
            for i, results in enumerate(batch_results):
                text, sentence_idx, doc_id = valid_data[i]

                # Handle single result (not a list)
                if isinstance(results, dict):
                    results = [results]

                for result in results:
                    entity_group = result.get("entity_group", result.get("entity", ""))

                    if "-" in entity_group:
                        entity_type = entity_group.split("-")[-1]
                    else:
                        entity_type = entity_group

                    if entity_type not in self.entity_types:
                        continue

                    confidence = result.get("score", 0.0)
                    if confidence < self.min_confidence:
                        continue

                    entity = Entity(
                        text=result.get("word", "").strip(),
                        entity_type=entity_type,
                        start=result.get("start", 0),
                        end=result.get("end", 0),
                        confidence=confidence,
                        sentence_idx=sentence_idx,
                        doc_id=doc_id,
                    )
                    all_entities.append(entity)

        except Exception as e:
            logger.warning(f"Batch processing error: {e}")
            # Fall back to individual processing
            for text, idx, doc_id in zip(texts, sentence_indices, doc_ids):
                entities = self.extract_entities(text, idx, doc_id)
                all_entities.extend(entities)

        return all_entities


# =============================================================================
# NER PROCESSOR (ORCHESTRATOR)
# =============================================================================

class NERProcessor:
    """Main orchestrator for NER extraction pipeline.

    Handles data loading, batched processing, checkpointing, and output.
    """

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        device: str = "auto",
        batch_size: Optional[int] = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        entity_types: Optional[List[str]] = None,
        max_sentences: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        """Initialize NER processor.

        Parameters
        ----------
        input_path : Path
            Path to input parquet file.
        output_dir : Path
            Output directory for results.
        device : str
            Device to use: "auto", "cuda", "mps", or "cpu".
        batch_size : int, optional
            Batch size for inference. None = auto-detect.
        min_confidence : float
            Minimum confidence threshold.
        entity_types : list of str, optional
            Entity types to extract.
        max_sentences : int, optional
            Maximum sentences to process (for testing).
        checkpoint_dir : Path, optional
            Directory for checkpoints.
        verbose : bool
            Enable verbose logging.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.max_sentences = max_sentences
        self.checkpoint_dir = checkpoint_dir or output_dir / "checkpoints"
        self.verbose = verbose

        # Initialize device manager
        self.device_manager = DeviceManager(device)

        # Set batch size
        self.batch_size = batch_size or self.device_manager.get_recommended_batch_size()

        # Initialize extractor
        self.extractor = NERExtractor(
            device_manager=self.device_manager,
            min_confidence=min_confidence,
            entity_types=entity_types,
        )

        # Processing state
        self.df_input = None
        self.entities: List[Entity] = []
        self.stats = ProcessingStats()
        self._checkpoint_path: Optional[Path] = None

    def load_data(self) -> pd.DataFrame:
        """Load input parquet file."""
        logger.info("=" * 70)
        logger.info("LOADING INPUT DATA")
        logger.info("=" * 70)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        df = pd.read_parquet(self.input_path)

        # Validate required columns
        if "sentence_text" not in df.columns:
            raise ValueError("Input must have 'sentence_text' column")

        # Handle doc_id column naming
        if "doc_id" not in df.columns and "file" in df.columns:
            df["doc_id"] = df["file"]
        elif "doc_id" not in df.columns:
            df["doc_id"] = "unknown"

        logger.info(f"Loaded {len(df):,} sentences from {df['doc_id'].nunique():,} documents")
        logger.info(f"Columns: {list(df.columns)}")

        # Limit for testing
        if self.max_sentences and len(df) > self.max_sentences:
            logger.info(f"Limiting to {self.max_sentences:,} sentences (--max-sentences)")
            df = df.head(self.max_sentences).copy()

        self.df_input = df
        self.stats.total_sentences = len(df)
        return df

    def _check_resume(self) -> Tuple[int, List[Entity]]:
        """Check for existing checkpoint to resume from.

        Returns
        -------
        tuple of (start_idx, entities)
            Starting index and previously extracted entities.
        """
        checkpoint_file = self.checkpoint_dir / "ner_checkpoint.json"
        entities_file = self.checkpoint_dir / "ner_entities_partial.csv"

        if not checkpoint_file.exists() or not entities_file.exists():
            return 0, []

        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)

            processed = checkpoint.get("sentences_processed", 0)

            # Load partial entities
            entities_df = pd.read_csv(entities_file)
            entities = [
                Entity(
                    text=row["text"],
                    entity_type=row["entity_type"],
                    start=row["start"],
                    end=row["end"],
                    confidence=row["confidence"],
                    sentence_idx=row["sentence_idx"],
                    doc_id=row["doc_id"],
                )
                for _, row in entities_df.iterrows()
            ]

            logger.info(f"Resuming from checkpoint: {processed:,} sentences processed")
            return processed, entities

        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return 0, []

    def _save_checkpoint(self, processed: int, entities: List[Entity]) -> None:
        """Save checkpoint for resume capability."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save progress marker
        checkpoint_file = self.checkpoint_dir / "ner_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump({"sentences_processed": processed}, f)

        # Save partial entities
        entities_file = self.checkpoint_dir / "ner_entities_partial.csv"
        entities_df = pd.DataFrame([asdict(e) for e in entities])
        entities_df.to_csv(entities_file, index=False)

        logger.debug(f"Checkpoint saved: {processed:,} sentences")

    def _clear_checkpoint(self) -> None:
        """Clear checkpoint files after successful completion."""
        checkpoint_file = self.checkpoint_dir / "ner_checkpoint.json"
        entities_file = self.checkpoint_dir / "ner_entities_partial.csv"

        for f in [checkpoint_file, entities_file]:
            if f.exists():
                f.unlink()

    def process(self) -> List[Entity]:
        """Run NER extraction on all sentences."""
        logger.info("=" * 70)
        logger.info("NER EXTRACTION")
        logger.info("=" * 70)

        device_info = self.device_manager.get_device_info()
        logger.info(f"Device: {device_info['device']} ({device_info.get('gpu_name', 'CPU')})")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Entity types: {sorted(self.extractor.entity_types)}")
        logger.info(f"Min confidence: {self.extractor.min_confidence}")

        # Load model
        self.extractor.load_model()

        # Check for resume
        start_idx, self.entities = self._check_resume()

        # Prepare data
        texts = self.df_input["sentence_text"].tolist()
        doc_ids = self.df_input["doc_id"].tolist()

        # Create sentence indices
        if "sentence_id" in self.df_input.columns:
            sentence_indices = self.df_input["sentence_id"].tolist()
        else:
            sentence_indices = list(range(len(texts)))

        # Process in batches
        start_time = time.time()
        total = len(texts)
        current_batch_size = self.batch_size

        for i in range(start_idx, total, current_batch_size):
            batch_end = min(i + current_batch_size, total)

            batch_texts = texts[i:batch_end]
            batch_indices = sentence_indices[i:batch_end]
            batch_docs = doc_ids[i:batch_end]

            try:
                batch_entities = self.extractor.extract_batch(
                    batch_texts, batch_indices, batch_docs
                )
                self.entities.extend(batch_entities)
                self.stats.sentences_processed = batch_end

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.stats.oom_recoveries += 1
                    # Reduce batch size and retry
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(
                        f"OOM error - reducing batch size to {current_batch_size}"
                    )

                    # Clear cache
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    continue
                else:
                    self.stats.errors += 1
                    logger.error(f"Error processing batch {i}: {e}")
                    continue

            except Exception as e:
                self.stats.errors += 1
                logger.error(f"Error processing batch {i}: {e}")
                continue

            # Progress reporting
            if (batch_end % 1000 == 0) or batch_end == total:
                elapsed = time.time() - start_time
                rate = batch_end / elapsed if elapsed > 0 else 0
                eta = (total - batch_end) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {batch_end:,}/{total:,} sentences "
                    f"({batch_end/total:.1%}) | "
                    f"{len(self.entities):,} entities | "
                    f"ETA: {eta/60:.1f} min"
                )

            # Checkpoint
            if batch_end % CHECKPOINT_FREQUENCY == 0:
                self._save_checkpoint(batch_end, self.entities)

        # Update stats
        elapsed = time.time() - start_time
        self.stats.processing_time_seconds = elapsed
        self.stats.total_entities = len(self.entities)
        self.stats.device_used = device_info["device"]
        self.stats.batch_size = self.batch_size

        # Count entities by type
        for entity in self.entities:
            entity_type = entity.entity_type
            self.stats.entities_by_type[entity_type] = (
                self.stats.entities_by_type.get(entity_type, 0) + 1
            )

        # Clear checkpoint on success
        self._clear_checkpoint()

        logger.info(f"\nExtraction complete in {elapsed:.1f}s")
        logger.info(f"Total entities: {len(self.entities):,}")

        return self.entities

    def create_entity_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with all extracted entities."""
        if not self.entities:
            return pd.DataFrame(columns=[
                "text", "entity_type", "start", "end", "confidence",
                "sentence_idx", "doc_id"
            ])

        return pd.DataFrame([asdict(e) for e in self.entities])

    def aggregate_by_sentence(self) -> pd.DataFrame:
        """Aggregate entity counts by sentence."""
        if not self.entities:
            return pd.DataFrame()

        entity_df = self.create_entity_dataframe()

        # Count entities per sentence per type
        counts = entity_df.groupby(["doc_id", "sentence_idx", "entity_type"]).size()
        counts = counts.unstack(fill_value=0).reset_index()

        # Ensure all entity types are present
        for etype in ENTITY_TYPES:
            if etype not in counts.columns:
                counts[etype] = 0

        # Add total count
        counts["total_entities"] = counts[ENTITY_TYPES].sum(axis=1)

        # Merge with original sentence data
        if self.df_input is not None:
            # Create sentence key for merging
            if "sentence_id" in self.df_input.columns:
                merge_df = self.df_input.copy()
                merge_df["sentence_idx"] = merge_df["sentence_id"]
            else:
                merge_df = self.df_input.copy()
                merge_df["sentence_idx"] = range(len(merge_df))

            counts = counts.merge(
                merge_df[["doc_id", "sentence_idx", "sentence_text"]],
                on=["doc_id", "sentence_idx"],
                how="left",
            )

        return counts

    def aggregate_by_document(self) -> pd.DataFrame:
        """Aggregate entity counts by document."""
        if not self.entities:
            return pd.DataFrame()

        entity_df = self.create_entity_dataframe()

        # Count entities per document per type
        counts = entity_df.groupby(["doc_id", "entity_type"]).size()
        counts = counts.unstack(fill_value=0).reset_index()

        # Ensure all entity types are present
        for etype in ENTITY_TYPES:
            if etype not in counts.columns:
                counts[etype] = 0

        # Add total count
        counts["total_entities"] = counts[ENTITY_TYPES].sum(axis=1)

        # Add unique entity counts
        unique_counts = entity_df.groupby(["doc_id", "entity_type"])["text"].nunique()
        unique_counts = unique_counts.unstack(fill_value=0).reset_index()
        unique_counts.columns = ["doc_id"] + [f"{c}_unique" for c in unique_counts.columns[1:]]

        counts = counts.merge(unique_counts, on="doc_id", how="left")

        # Merge with document metadata
        if self.df_input is not None:
            doc_meta = self.df_input.drop_duplicates("doc_id")[
                ["doc_id"] +
                [c for c in ["actor_type", "year", "wave", "municipality", "entity"]
                 if c in self.df_input.columns]
            ]
            counts = counts.merge(doc_meta, on="doc_id", how="left")

        return counts

    def generate_report(self) -> Dict:
        """Generate JSON report with statistics and metadata."""
        entity_df = self.create_entity_dataframe()

        # Top entities by type
        top_entities = {}
        for etype in ENTITY_TYPES:
            type_entities = entity_df[entity_df["entity_type"] == etype]
            if len(type_entities) > 0:
                top = type_entities["text"].value_counts().head(20).to_dict()
                top_entities[etype] = top

        report = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "input_file": str(self.input_path),
                "output_dir": str(self.output_dir),
                "model": MODEL_NAME,
                "device": self.stats.device_used,
                "batch_size": self.stats.batch_size,
                "min_confidence": self.extractor.min_confidence,
                "entity_types_requested": sorted(self.extractor.entity_types),
            },
            "statistics": {
                "total_sentences": self.stats.total_sentences,
                "sentences_processed": self.stats.sentences_processed,
                "total_entities": self.stats.total_entities,
                "entities_by_type": self.stats.entities_by_type,
                "unique_entities_by_type": {
                    etype: int(entity_df[entity_df["entity_type"] == etype]["text"].nunique())
                    for etype in ENTITY_TYPES
                    if etype in entity_df["entity_type"].values
                },
                "processing_time_seconds": round(self.stats.processing_time_seconds, 1),
                "sentences_per_second": round(
                    self.stats.sentences_processed / max(self.stats.processing_time_seconds, 0.1), 1
                ),
                "errors": self.stats.errors,
                "oom_recoveries": self.stats.oom_recoveries,
            },
            "top_entities": top_entities,
            "entity_type_descriptions": ENTITY_DESCRIPTIONS,
        }

        return report

    def save_outputs(self) -> None:
        """Save all output files."""
        logger.info("=" * 70)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 70)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. All entities
        entity_df = self.create_entity_dataframe()
        entities_path = self.output_dir / "entities.csv"
        entity_df.to_csv(entities_path, index=False)
        logger.info(f"Saved: {entities_path} ({len(entity_df):,} entities)")

        # 2. Entities by sentence
        sentence_df = self.aggregate_by_sentence()
        sentence_path = self.output_dir / "entities_by_sentence.csv"
        sentence_df.to_csv(sentence_path, index=False)
        logger.info(f"Saved: {sentence_path} ({len(sentence_df):,} rows)")

        # 3. Entities by document
        doc_df = self.aggregate_by_document()
        doc_path = self.output_dir / "entities_by_document.csv"
        doc_df.to_csv(doc_path, index=False)
        logger.info(f"Saved: {doc_path} ({len(doc_df):,} documents)")

        # 4. Report
        report = self.generate_report()
        report_path = self.output_dir / "ner_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved: {report_path}")

    def print_summary(self) -> None:
        """Print summary of extraction results."""
        logger.info("=" * 70)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\nSentences processed: {self.stats.sentences_processed:,}")
        logger.info(f"Total entities extracted: {self.stats.total_entities:,}")
        logger.info(f"Processing time: {self.stats.processing_time_seconds:.1f}s")

        logger.info("\nEntities by type:")
        for etype in ENTITY_TYPES:
            count = self.stats.entities_by_type.get(etype, 0)
            desc = ENTITY_DESCRIPTIONS.get(etype, "")
            logger.info(f"  {etype}: {count:,} ({desc})")

        if self.stats.errors > 0:
            logger.info(f"\nErrors encountered: {self.stats.errors}")

        if self.stats.oom_recoveries > 0:
            logger.info(f"OOM recoveries: {self.stats.oom_recoveries}")

        # Print example entities
        if self.entities:
            logger.info("\nExample entities:")
            for etype in ENTITY_TYPES:
                type_entities = [e for e in self.entities if e.entity_type == etype]
                if type_entities:
                    sample = type_entities[:3]
                    examples = ", ".join(f'"{e.text}"' for e in sample)
                    logger.info(f"  {etype}: {examples}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    input_path: Path,
    output_dir: Path,
    device: str = "auto",
    batch_size: Optional[int] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    entity_types: Optional[List[str]] = None,
    max_sentences: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Run the complete NER extraction pipeline.

    Parameters
    ----------
    input_path : Path
        Path to input parquet file.
    output_dir : Path
        Output directory for results.
    device : str
        Device to use: "auto", "cuda", "mps", or "cpu".
    batch_size : int, optional
        Batch size for inference.
    min_confidence : float
        Minimum confidence threshold.
    entity_types : list of str, optional
        Entity types to extract.
    max_sentences : int, optional
        Maximum sentences to process.
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    dict
        Processing report.
    """
    setup_logging(verbose)

    logger.info("=" * 70)
    logger.info("NER EXTRACTION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info("")

    processor = NERProcessor(
        input_path=input_path,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        min_confidence=min_confidence,
        entity_types=entity_types,
        max_sentences=max_sentences,
        verbose=verbose,
    )

    processor.load_data()
    processor.process()
    processor.save_outputs()
    processor.print_summary()

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nFiles generated:")
    logger.info("  - entities.csv: All extracted entities")
    logger.info("  - entities_by_sentence.csv: Entity counts per sentence")
    logger.info("  - entities_by_document.csv: Entity counts per document")
    logger.info("  - ner_report.json: Summary statistics")

    return processor.generate_report()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract named entities from RSA corpus using Swedish BERT NER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Entity Types:
    TME: Time expressions (dates, periods, times)
    PRS: Personal names (people)
    LOC: Locations (geographic entities, places)
    EVN: Events (named events, incidents)
    ORG: Organizations (companies, agencies, institutions)

Examples:
    # Basic usage (auto-detect device)
    python ner_extraction.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/05_ner/

    # Run on M1 Mac with MPS
    python ner_extraction.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/05_ner/ \\
        --device mps \\
        --batch-size 16

    # Extract only locations and organizations
    python ner_extraction.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/05_ner/ \\
        --entity-types LOC ORG

    # Test on small sample
    python ner_extraction.py \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --output results/05_ner/ \\
        --max-sentences 1000 \\
        --verbose

Model:
    KBLab/bert-base-swedish-cased-ner (Royal Library of Sweden)
    https://huggingface.co/KBLab/bert-base-swedish-cased-ner
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet file with sentence_text column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/05_ner"),
        help="Output directory (default: results/05_ner/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device for inference: auto, cuda, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: auto-detect based on device)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Minimum confidence score (0.0-1.0, default: {DEFAULT_MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        nargs="+",
        choices=ENTITY_TYPES,
        default=None,
        help=f"Entity types to extract (default: all). Options: {', '.join(ENTITY_TYPES)}",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum sentences to process (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def main() -> int:
    """Main entry point for command-line execution."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            device=args.device,
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
            entity_types=args.entity_types,
            max_sentences=args.max_sentences,
            verbose=args.verbose,
        )
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except ImportError as e:
        print(f"Error: Missing dependency - {e}", file=sys.stderr)
        print("\nInstall required packages:", file=sys.stderr)
        print("  pip install transformers torch pandas pyarrow", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logging.exception("Unexpected error during processing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
