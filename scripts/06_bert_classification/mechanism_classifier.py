#!/usr/bin/env python3
"""
BERT-based Mechanism Classifier for RSA Corpus

Fine-tunes a Swedish BERT model to classify theoretical mechanisms in RSA
sentences. Implements multi-label classification for four mechanisms:
    - mechanism_legitimacy_offload: Offloading responsibility to another actor
      or claiming limited capabilities (Borraz, 2008)
    - mechanism_legitimacy_lowrisk: Qualifying risk as too low to demand public
      action (Borraz, 2008)
    - mechanism_functional: Functional aptness for handling social risks (Paul, 2021)
    - mechanism_complexity: Complexity empowerment of local actors

The classifier supports three modes:
    1. train: Fine-tune BERT on hand-coded training data
    2. evaluate: Evaluate on held-out test set with calibrated thresholds
    3. predict: Apply trained model to full corpus

Input Format (train/evaluate):
    Excel (.xlsx) or CSV file with columns:
    - sentence_text: Text to classify
    - split: 'train' or 'test' (for combined input file)
    - mechanism_legitimacy_offload: 1 = present, empty = absent (recoded to 0)
    - mechanism_legitimacy_lowrisk: 1 = present, empty = absent (recoded to 0)
    - mechanism_functional: 1 = present, empty = absent (recoded to 0)
    - mechanism_complexity: 1 = present, empty = absent (recoded to 0)
    - Metadata: doc_id, actor_type, year, wave, sentence_id (preserved)

Output Format:
    - Training: Model checkpoint, training history, thresholds
    - Evaluation: Per-mechanism metrics, confusion matrices
    - Prediction: CSV with probability scores and predicted labels

Usage:
    # Train (with combined Excel file containing train/test split)
    python mechanism_classifier.py --mode train \\
        --data results/00_data_preparation/sampling/sample_full.xlsx \\
        --output results/02_bert_analysis/classification/ \\
        --model-dir models/mechanism_classifier/ \\
        --epochs 5 --learning-rate 2e-5

    # Evaluate
    python mechanism_classifier.py --mode evaluate \\
        --data results/00_data_preparation/sampling/sample_full.xlsx \\
        --model-dir models/mechanism_classifier/ \\
        --output results/02_bert_analysis/classification/ \\
        --calibrate-thresholds

    # Predict on full corpus
    python mechanism_classifier.py --mode predict \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --model-dir models/mechanism_classifier/ \\
        --output results/02_bert_analysis/classification/

Requirements:
    pip install transformers torch pandas pyarrow scikit-learn openpyxl

Model:
    KBLab/bert-base-swedish-cased (Royal Library of Sweden)

Author: Swedish Risk Analysis Text-as-Data Project
Version: 1.0
Date: 2025-02-25
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration
MODEL_NAME = "KBLab/bert-base-swedish-cased"

# Mechanism labels (default set - can be overridden via CLI)
MECHANISM_LABELS = [
    "mechanism_legitimacy_offload",
    "mechanism_legitimacy_lowrisk",
    "mechanism_functional",
    "mechanism_complexity",
]

# Label descriptions for documentation
MECHANISM_DESCRIPTIONS = {
    "mechanism_legitimacy_offload": (
        "Offloading responsibility to another actor or claiming limited "
        "capabilities (Borraz, 2008)"
    ),
    "mechanism_legitimacy_lowrisk": (
        "Qualifying risk as too low to demand public action (Borraz, 2008)"
    ),
    "mechanism_functional": (
        "Functional aptness - the instrument is genuinely apt for handling "
        "social risks (Paul, 2021)"
    ),
    "mechanism_complexity": "Complexity empowerment of local actors",
}

# Default batch sizes per device type
DEFAULT_BATCH_SIZES = {
    "cuda": 16,
    "mps": 8,
    "cpu": 4,
}

# Default hyperparameters
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_LENGTH = 512
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_THRESHOLD = 0.5

# Checkpoint frequency
CHECKPOINT_FREQUENCY = 100


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
class TrainingConfig:
    """Configuration for model training."""
    model_name: str = MODEL_NAME
    max_length: int = DEFAULT_MAX_LENGTH
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    use_class_weights: bool = True
    seed: int = 42


@dataclass
class TrainingStats:
    """Statistics from training run."""
    total_samples: int = 0
    train_samples: int = 0
    test_samples: int = 0
    epochs_completed: int = 0
    best_epoch: int = 0
    best_f1_macro: float = 0.0
    training_time_seconds: float = 0.0
    device_used: str = ""
    class_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single mechanism."""
    label: str
    threshold: float
    precision: float
    recall: float
    f1: float
    support_positive: int
    support_negative: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


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
        self.requested_device = requested_device
        self.device = self._detect_device()
        self.device_type = self._get_device_type()

    def _detect_device(self) -> torch.device:
        """Detect the best available device."""
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
        return DEFAULT_BATCH_SIZES.get(self.device_type, 4)

    def get_device_info(self) -> Dict[str, str]:
        """Get device information for reporting."""
        info = {
            "device": str(self.device),
            "device_type": self.device_type,
            "torch_version": torch.__version__,
        }

        if self.device_type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = (
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
            )
        elif self.device_type == "mps":
            info["gpu_name"] = "Apple Silicon (MPS)"

        return info


# =============================================================================
# DATASET
# =============================================================================

class MechanismDataset(Dataset):
    """PyTorch Dataset for mechanism classification.

    Handles tokenization and label encoding for multi-label classification.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        tokenizer=None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        """Initialize dataset.

        Parameters
        ----------
        texts : list of str
            Input sentences.
        labels : np.ndarray, optional
            Binary label matrix of shape (n_samples, n_labels).
        tokenizer : PreTrainedTokenizer
            Hugging Face tokenizer.
        max_length : int
            Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(
                self.labels[idx], dtype=torch.float32
            )

        return item


# =============================================================================
# DATA LOADING
# =============================================================================

class DataLoader:
    """Load and preprocess training/test data."""

    def __init__(self, labels: List[str] = None):
        """Initialize data loader.

        Parameters
        ----------
        labels : list of str
            Label column names.
        """
        self.labels = labels or MECHANISM_LABELS

    def load_labeled_data(
        self, file_path: Path, split: str = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load labeled data from Excel or CSV.

        Recodes empty cells as 0 for all label columns.

        Parameters
        ----------
        file_path : Path
            Path to Excel (.xlsx) or CSV file.
        split : str, optional
            If provided, filter to rows where 'split' column matches this value.
            Use 'train' or 'test' for combined files with split column.

        Returns
        -------
        tuple of (DataFrame, ndarray)
            (Full dataframe, binary label matrix)
        """
        logger.info(f"Loading labeled data from: {file_path}")

        # Load based on file extension
        suffix = file_path.suffix.lower()
        if suffix == '.xlsx':
            df = pd.read_excel(file_path, engine='openpyxl')
        elif suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .xlsx or .csv")

        # Filter by split if requested
        if split is not None:
            if 'split' not in df.columns:
                raise ValueError(
                    f"No 'split' column found. Cannot filter by split='{split}'"
                )
            df = df[df['split'] == split].copy()
            logger.info(f"  Filtered to split='{split}': {len(df)} rows")

        # Validate required columns
        if "sentence_text" not in df.columns:
            raise ValueError(
                f"Missing 'sentence_text' column. Found: {list(df.columns)}"
            )

        # Check for label columns
        missing_labels = [l for l in self.labels if l not in df.columns]
        if missing_labels:
            raise ValueError(
                f"Missing label columns: {missing_labels}. Found: {list(df.columns)}"
            )

        # Recode empty cells as 0
        for label in self.labels:
            df[label] = df[label].fillna(0).astype(int)
            # Validate binary values
            unique_values = set(df[label].unique())
            if not unique_values.issubset({0, 1}):
                raise ValueError(
                    f"Label '{label}' contains non-binary values: {unique_values}"
                )

        # Create label matrix
        label_matrix = df[self.labels].values.astype(np.float32)

        logger.info(f"  Loaded {len(df)} samples")
        logger.info(f"  Labels: {self.labels}")

        # Report class distribution
        for i, label in enumerate(self.labels):
            pos_count = int(label_matrix[:, i].sum())
            neg_count = len(df) - pos_count
            logger.info(
                f"  {label}: {pos_count} positive ({pos_count/len(df):.1%}), "
                f"{neg_count} negative"
            )

        return df, label_matrix

    def load_prediction_data(self, file_path: Path) -> pd.DataFrame:
        """Load data for prediction (no labels required).

        Parameters
        ----------
        file_path : Path
            Path to parquet or CSV file.

        Returns
        -------
        DataFrame
            Dataframe with sentence_text column.
        """
        logger.info(f"Loading prediction data from: {file_path}")

        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        # Validate required columns
        if "sentence_text" not in df.columns:
            raise ValueError(
                f"Missing 'sentence_text' column. Found: {list(df.columns)}"
            )

        logger.info(f"  Loaded {len(df):,} samples")

        return df

    def compute_class_weights(
        self, label_matrix: np.ndarray
    ) -> torch.Tensor:
        """Compute inverse frequency class weights for BCEWithLogitsLoss.

        Parameters
        ----------
        label_matrix : np.ndarray
            Binary label matrix of shape (n_samples, n_labels).

        Returns
        -------
        torch.Tensor
            Positive class weights for each label.
        """
        n_samples = label_matrix.shape[0]
        pos_counts = label_matrix.sum(axis=0)
        neg_counts = n_samples - pos_counts

        # Inverse frequency weighting: weight = neg_count / pos_count
        # Avoid division by zero
        weights = np.where(
            pos_counts > 0,
            neg_counts / pos_counts,
            1.0
        )

        logger.info("Class weights (pos_weight for BCEWithLogitsLoss):")
        for i, label in enumerate(self.labels):
            logger.info(f"  {label}: {weights[i]:.2f}")

        return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# WEIGHTED BCE TRAINER
# =============================================================================

class WeightedBCETrainer:
    """Custom trainer with weighted BCE loss for class imbalance."""

    def __init__(
        self,
        model,
        tokenizer,
        config: TrainingConfig,
        device_manager: DeviceManager,
        pos_weights: Optional[torch.Tensor] = None,
    ):
        """Initialize trainer.

        Parameters
        ----------
        model : PreTrainedModel
            Hugging Face model.
        tokenizer : PreTrainedTokenizer
            Hugging Face tokenizer.
        config : TrainingConfig
            Training configuration.
        device_manager : DeviceManager
            Device manager.
        pos_weights : torch.Tensor, optional
            Positive class weights for weighted BCE.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device_manager = device_manager
        self.device = device_manager.device

        # Move model to device
        self.model.to(self.device)

        # Setup loss function
        if pos_weights is not None:
            self.pos_weights = pos_weights.to(self.device)
        else:
            self.pos_weights = None

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
        self.best_model_state = None
        self.best_f1 = 0.0
        self.best_epoch = 0

    def _compute_loss(
        self,
        outputs,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted BCE loss."""
        logits = outputs.logits

        if self.pos_weights is not None:
            loss_fn = BCEWithLogitsLoss(pos_weight=self.pos_weights)
        else:
            loss_fn = BCEWithLogitsLoss()

        return loss_fn(logits, labels)

    def _setup_optimizer(
        self,
        num_training_steps: int,
    ) -> None:
        """Setup optimizer and learning rate scheduler."""
        from transformers import get_linear_schedule_with_warmup

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        logger.info(f"Warmup steps: {num_warmup_steps}")
        logger.info(f"Total training steps: {num_training_steps}")

    def train(
        self,
        train_dataset: MechanismDataset,
        eval_dataset: Optional[MechanismDataset] = None,
    ) -> Dict:
        """Train the model.

        Parameters
        ----------
        train_dataset : MechanismDataset
            Training dataset.
        eval_dataset : MechanismDataset, optional
            Evaluation dataset for validation.

        Returns
        -------
        dict
            Training history and best metrics.
        """
        from torch.utils.data import DataLoader

        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Calculate total steps
        num_training_steps = (
            len(train_loader)
            * self.config.epochs
            // self.config.gradient_accumulation_steps
        )

        # Setup optimizer
        self._setup_optimizer(num_training_steps)

        logger.info("=" * 70)
        logger.info("TRAINING")
        logger.info("=" * 70)
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Epochs: {self.config.epochs}")

        self.model.train()
        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            logger.info("-" * 40)

            for step, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Compute loss
                loss = self._compute_loss(outputs, labels)
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    epoch_steps += 1

                # Progress logging
                if (step + 1) % 50 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    logger.info(
                        f"  Step {step + 1}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f}"
                    )

            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"  Epoch loss: {avg_epoch_loss:.4f}")

            # Evaluation
            if eval_dataset is not None:
                eval_metrics = self.evaluate(eval_dataset)
                f1_macro = eval_metrics["f1_macro"]

                logger.info(f"  Validation F1 (macro): {f1_macro:.4f}")

                # Track best model
                if f1_macro > self.best_f1:
                    self.best_f1 = f1_macro
                    self.best_epoch = epoch + 1
                    self.best_model_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    logger.info(f"  New best model (F1={f1_macro:.4f})")

                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_epoch_loss,
                    "eval_f1_macro": f1_macro,
                    "eval_metrics": eval_metrics,
                })
            else:
                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_epoch_loss,
                })

        training_time = time.time() - start_time
        logger.info(f"\nTraining complete in {training_time:.1f}s")
        logger.info(f"Best epoch: {self.best_epoch} (F1={self.best_f1:.4f})")

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model checkpoint")

        return {
            "training_history": self.training_history,
            "best_epoch": self.best_epoch,
            "best_f1": self.best_f1,
            "training_time_seconds": training_time,
        }

    def evaluate(
        self,
        dataset: MechanismDataset,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Evaluate model on dataset.

        Parameters
        ----------
        dataset : MechanismDataset
            Evaluation dataset.
        thresholds : dict, optional
            Per-label classification thresholds.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        from torch.utils.data import DataLoader
        from sklearn.metrics import (
            precision_recall_fscore_support,
            accuracy_score,
        )

        if thresholds is None:
            thresholds = {label: DEFAULT_THRESHOLD for label in MECHANISM_LABELS}

        self.model.eval()

        eval_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size * 2,  # Can use larger batch for eval
            shuffle=False,
        )

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels)

        # Concatenate results
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0).numpy()

        # Convert to probabilities
        probs = torch.sigmoid(logits).numpy()

        # Apply thresholds
        threshold_values = np.array([
            thresholds[label] for label in MECHANISM_LABELS
        ])
        predictions = (probs >= threshold_values).astype(int)

        # Compute metrics per label
        metrics_per_label = {}
        f1_scores = []

        for i, label in enumerate(MECHANISM_LABELS):
            y_true = labels[:, i]
            y_pred = predictions[:, i]

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )

            # Confusion matrix elements
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())

            metrics_per_label[label] = {
                "threshold": thresholds[label],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support_positive": int(y_true.sum()),
                "support_negative": int((1 - y_true).sum()),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            }
            f1_scores.append(f1)

        # Aggregated metrics
        f1_macro = np.mean(f1_scores)
        f1_micro_num = sum(m["true_positives"] for m in metrics_per_label.values())
        f1_micro_denom = sum(
            m["true_positives"] + 0.5 * (m["false_positives"] + m["false_negatives"])
            for m in metrics_per_label.values()
        )
        f1_micro = f1_micro_num / f1_micro_denom if f1_micro_denom > 0 else 0.0

        # Exact match ratio (both labels correct)
        exact_match = (predictions == labels).all(axis=1).mean()

        # Hamming loss
        hamming_loss = (predictions != labels).mean()

        self.model.train()

        return {
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "exact_match_ratio": float(exact_match),
            "hamming_loss": float(hamming_loss),
            "per_label": metrics_per_label,
            "thresholds": thresholds,
        }

    def calibrate_thresholds(
        self,
        dataset: MechanismDataset,
        metric: str = "f1",
    ) -> Dict[str, float]:
        """Find optimal classification thresholds per label.

        Parameters
        ----------
        dataset : MechanismDataset
            Dataset for threshold calibration.
        metric : str
            Metric to optimize ("f1", "precision", "recall").

        Returns
        -------
        dict
            Optimal threshold per label.
        """
        from torch.utils.data import DataLoader
        from sklearn.metrics import precision_recall_fscore_support

        logger.info("Calibrating classification thresholds...")

        self.model.eval()

        eval_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
        )

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0).numpy()
        probs = torch.sigmoid(logits).numpy()

        # Search for optimal threshold per label
        thresholds = {}
        threshold_candidates = np.arange(0.1, 0.9, 0.05)

        for i, label in enumerate(MECHANISM_LABELS):
            y_true = labels[:, i]
            y_probs = probs[:, i]

            best_threshold = 0.5
            best_score = 0.0

            for thresh in threshold_candidates:
                y_pred = (y_probs >= thresh).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )

                if metric == "f1":
                    score = f1
                elif metric == "precision":
                    score = precision
                elif metric == "recall":
                    score = recall
                else:
                    score = f1

                if score > best_score:
                    best_score = score
                    best_threshold = thresh

            thresholds[label] = float(best_threshold)
            logger.info(
                f"  {label}: threshold={best_threshold:.2f} "
                f"({metric}={best_score:.3f})"
            )

        self.model.train()
        return thresholds

    def predict(
        self,
        texts: List[str],
        thresholds: Optional[Dict[str, float]] = None,
        batch_size: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for texts.

        Parameters
        ----------
        texts : list of str
            Input texts.
        thresholds : dict, optional
            Per-label classification thresholds.
        batch_size : int
            Batch size for inference.

        Returns
        -------
        tuple of (ndarray, ndarray)
            (Probability matrix, prediction matrix)
        """
        from torch.utils.data import DataLoader

        if thresholds is None:
            thresholds = {label: DEFAULT_THRESHOLD for label in MECHANISM_LABELS}

        self.model.eval()

        # Create dataset
        dataset = MechanismDataset(
            texts=texts,
            labels=None,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        all_logits = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                all_logits.append(outputs.logits.cpu())

        logits = torch.cat(all_logits, dim=0)
        probs = torch.sigmoid(logits).numpy()

        # Apply thresholds
        threshold_values = np.array([
            thresholds[label] for label in MECHANISM_LABELS
        ])
        predictions = (probs >= threshold_values).astype(int)

        self.model.train()
        return probs, predictions


# =============================================================================
# MECHANISM PIPELINE
# =============================================================================

class MechanismPipeline:
    """Main orchestrator for mechanism classification pipeline."""

    def __init__(
        self,
        model_dir: Path,
        output_dir: Path,
        device: str = "auto",
        verbose: bool = False,
    ):
        """Initialize pipeline.

        Parameters
        ----------
        model_dir : Path
            Directory to save/load model.
        output_dir : Path
            Directory for outputs.
        device : str
            Device to use.
        verbose : bool
            Enable verbose logging.
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.device_manager = DeviceManager(device)
        self.verbose = verbose

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.thresholds = None
        self.data_loader = DataLoader(labels=MECHANISM_LABELS)

    def load_model(self, for_training: bool = False) -> None:
        """Load or initialize model.

        Parameters
        ----------
        for_training : bool
            If True, initialize fresh model for training.
            If False, load trained model from model_dir.
        """
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        if for_training:
            logger.info(f"Initializing model: {MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=len(MECHANISM_LABELS),
                problem_type="multi_label_classification",
            )
        else:
            logger.info(f"Loading model from: {self.model_dir}")
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {self.model_dir}"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            )

            # Load thresholds if available
            threshold_path = self.model_dir / "thresholds.json"
            if threshold_path.exists():
                with open(threshold_path, "r") as f:
                    self.thresholds = json.load(f)
                logger.info(f"Loaded thresholds: {self.thresholds}")

        logger.info(f"Model loaded on device: {self.device_manager.device}")

    def save_model(self) -> None:
        """Save model and tokenizer to model_dir."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

        logger.info(f"Model saved to: {self.model_dir}")

        # Save thresholds
        if self.thresholds is not None:
            threshold_path = self.model_dir / "thresholds.json"
            with open(threshold_path, "w") as f:
                json.dump(self.thresholds, f, indent=2)
            logger.info(f"Thresholds saved to: {threshold_path}")

    def train(
        self,
        train_data_path: Path = None,
        test_data_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
        config: Optional[TrainingConfig] = None,
    ) -> Dict:
        """Train model on labeled data.

        Parameters
        ----------
        train_data_path : Path, optional
            Path to training CSV/Excel.
        test_data_path : Path, optional
            Path to test CSV/Excel for validation.
        data_path : Path, optional
            Path to combined Excel/CSV with 'split' column. If provided,
            train_data_path and test_data_path are ignored.
        config : TrainingConfig, optional
            Training configuration.

        Returns
        -------
        dict
            Training results.
        """
        if config is None:
            config = TrainingConfig()
            config.batch_size = self.device_manager.get_recommended_batch_size()

        logger.info("=" * 70)
        logger.info("TRAINING MODE")
        logger.info("=" * 70)

        # Load data - either from combined file or separate files
        if data_path is not None:
            train_df, train_labels = self.data_loader.load_labeled_data(
                data_path, split='train'
            )
            test_df, test_labels = self.data_loader.load_labeled_data(
                data_path, split='test'
            )
        else:
            train_df, train_labels = self.data_loader.load_labeled_data(
                train_data_path
            )
            test_df = None
            test_labels = None
            if test_data_path is not None:
                test_df, test_labels = self.data_loader.load_labeled_data(
                    test_data_path
                )

        # Compute class weights
        pos_weights = None
        if config.use_class_weights:
            pos_weights = self.data_loader.compute_class_weights(train_labels)

        # Load model
        self.load_model(for_training=True)

        # Create datasets
        train_dataset = MechanismDataset(
            texts=train_df["sentence_text"].tolist(),
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=config.max_length,
        )

        eval_dataset = None
        if test_df is not None:
            eval_dataset = MechanismDataset(
                texts=test_df["sentence_text"].tolist(),
                labels=test_labels,
                tokenizer=self.tokenizer,
                max_length=config.max_length,
            )

        # Initialize trainer
        self.trainer = WeightedBCETrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=config,
            device_manager=self.device_manager,
            pos_weights=pos_weights,
        )

        # Train
        train_results = self.trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Calibrate thresholds on test set
        if eval_dataset is not None:
            self.thresholds = self.trainer.calibrate_thresholds(eval_dataset)
        else:
            self.thresholds = {
                label: DEFAULT_THRESHOLD for label in MECHANISM_LABELS
            }

        # Save model
        self.save_model()

        # Save training report
        self._save_training_report(train_results, config)

        return train_results

    def _save_training_report(
        self, train_results: Dict, config: TrainingConfig
    ) -> None:
        """Save training report and history."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training report
        report = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "model_name": config.model_name,
                "model_dir": str(self.model_dir),
                "output_dir": str(self.output_dir),
                "device": self.device_manager.get_device_info(),
            },
            "config": asdict(config),
            "results": {
                "best_epoch": train_results["best_epoch"],
                "best_f1_macro": train_results["best_f1"],
                "training_time_seconds": train_results["training_time_seconds"],
            },
            "thresholds": self.thresholds,
            "mechanism_descriptions": MECHANISM_DESCRIPTIONS,
        }

        report_path = self.output_dir / "training_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Training report saved: {report_path}")

        # Training history
        history_df = pd.DataFrame(train_results["training_history"])
        history_path = self.output_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info(f"Training history saved: {history_path}")

    def evaluate(
        self,
        test_data_path: Path = None,
        data_path: Path = None,
        calibrate_thresholds: bool = False,
    ) -> Dict:
        """Evaluate model on test data.

        Parameters
        ----------
        test_data_path : Path, optional
            Path to test CSV/Excel.
        data_path : Path, optional
            Path to combined Excel/CSV with 'split' column. If provided,
            test_data_path is ignored and test split is extracted.
        calibrate_thresholds : bool
            Whether to recalibrate thresholds on test data.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        logger.info("=" * 70)
        logger.info("EVALUATION MODE")
        logger.info("=" * 70)

        # Load model
        self.load_model(for_training=False)

        # Initialize trainer for evaluation
        config = TrainingConfig()
        config.batch_size = self.device_manager.get_recommended_batch_size()

        self.trainer = WeightedBCETrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=config,
            device_manager=self.device_manager,
        )

        # Load data - either from combined file or separate file
        if data_path is not None:
            test_df, test_labels = self.data_loader.load_labeled_data(
                data_path, split='test'
            )
        else:
            test_df, test_labels = self.data_loader.load_labeled_data(
                test_data_path
            )

        test_dataset = MechanismDataset(
            texts=test_df["sentence_text"].tolist(),
            labels=test_labels,
            tokenizer=self.tokenizer,
            max_length=config.max_length,
        )

        # Calibrate thresholds if requested
        if calibrate_thresholds:
            self.thresholds = self.trainer.calibrate_thresholds(test_dataset)
            # Save updated thresholds
            threshold_path = self.model_dir / "thresholds.json"
            with open(threshold_path, "w") as f:
                json.dump(self.thresholds, f, indent=2)

        # Evaluate
        metrics = self.trainer.evaluate(
            test_dataset,
            thresholds=self.thresholds,
        )

        # Save evaluation report
        report_data_path = data_path if data_path is not None else test_data_path
        self._save_evaluation_report(metrics, report_data_path)

        # Print summary
        self._print_evaluation_summary(metrics)

        return metrics

    def _save_evaluation_report(
        self, metrics: Dict, test_data_path: Path
    ) -> None:
        """Save evaluation report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "model_dir": str(self.model_dir),
                "test_data": str(test_data_path),
                "device": self.device_manager.get_device_info(),
            },
            "metrics": metrics,
            "mechanism_descriptions": MECHANISM_DESCRIPTIONS,
        }

        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation report saved: {report_path}")

    def _print_evaluation_summary(self, metrics: Dict) -> None:
        """Print evaluation summary."""
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 70)

        logger.info(f"\nAggregated Metrics:")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (micro): {metrics['f1_micro']:.4f}")
        logger.info(f"  Exact match ratio: {metrics['exact_match_ratio']:.4f}")
        logger.info(f"  Hamming loss: {metrics['hamming_loss']:.4f}")

        logger.info(f"\nPer-Label Metrics:")
        for label, m in metrics["per_label"].items():
            logger.info(f"\n  {label}:")
            logger.info(f"    Threshold: {m['threshold']:.2f}")
            logger.info(f"    Precision: {m['precision']:.4f}")
            logger.info(f"    Recall: {m['recall']:.4f}")
            logger.info(f"    F1: {m['f1']:.4f}")
            logger.info(
                f"    Confusion: TP={m['true_positives']}, "
                f"FP={m['false_positives']}, "
                f"TN={m['true_negatives']}, "
                f"FN={m['false_negatives']}"
            )

    def predict(
        self,
        input_path: Path,
        batch_size: int = 16,
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate predictions for corpus.

        Parameters
        ----------
        input_path : Path
            Path to input parquet/CSV.
        batch_size : int
            Batch size for inference.
        max_samples : int, optional
            Maximum samples to process (for testing).

        Returns
        -------
        DataFrame
            Input data with predictions added.
        """
        logger.info("=" * 70)
        logger.info("PREDICTION MODE")
        logger.info("=" * 70)

        # Load model
        self.load_model(for_training=False)

        # Initialize trainer
        config = TrainingConfig()
        config.batch_size = batch_size

        self.trainer = WeightedBCETrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=config,
            device_manager=self.device_manager,
        )

        # Load data
        df = self.data_loader.load_prediction_data(input_path)

        if max_samples is not None and len(df) > max_samples:
            logger.info(f"Limiting to {max_samples:,} samples")
            df = df.head(max_samples).copy()

        texts = df["sentence_text"].tolist()

        # Get thresholds
        thresholds = self.thresholds or {
            label: DEFAULT_THRESHOLD for label in MECHANISM_LABELS
        }

        logger.info(f"Predicting {len(texts):,} samples...")
        logger.info(f"Thresholds: {thresholds}")

        start_time = time.time()

        # Predict in batches with progress
        all_probs = []
        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            probs, preds = self.trainer.predict(
                batch_texts,
                thresholds=thresholds,
                batch_size=batch_size,
            )

            all_probs.append(probs)
            all_preds.append(preds)

            # Progress
            if (i + batch_size) % 1000 == 0 or i + batch_size >= len(texts):
                progress = min(i + batch_size, len(texts))
                logger.info(f"  Progress: {progress:,}/{len(texts):,}")

        probs = np.vstack(all_probs)
        preds = np.vstack(all_preds)

        elapsed = time.time() - start_time
        logger.info(f"Prediction complete in {elapsed:.1f}s")

        # Add predictions to dataframe
        for i, label in enumerate(MECHANISM_LABELS):
            df[f"prob_{label.replace('mechanism_', '')}"] = probs[:, i]
            df[f"pred_{label.replace('mechanism_', '')}"] = preds[:, i]

        # Save predictions
        self._save_predictions(df, input_path)

        return df

    def _save_predictions(
        self, df: pd.DataFrame, input_path: Path
    ) -> None:
        """Save predictions and report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Predictions CSV
        pred_path = self.output_dir / "predictions.csv"
        df.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved: {pred_path}")

        # Summary statistics
        summary = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "input_file": str(input_path),
                "model_dir": str(self.model_dir),
                "total_samples": len(df),
            },
            "thresholds": self.thresholds,
            "predictions_summary": {},
        }

        for label in MECHANISM_LABELS:
            short_label = label.replace("mechanism_", "")
            pred_col = f"pred_{short_label}"
            prob_col = f"prob_{short_label}"

            if pred_col in df.columns:
                pos_count = int(df[pred_col].sum())
                neg_count = len(df) - pos_count

                summary["predictions_summary"][label] = {
                    "positive": pos_count,
                    "positive_pct": round(pos_count / len(df) * 100, 2),
                    "negative": neg_count,
                    "mean_probability": round(df[prob_col].mean(), 4),
                    "median_probability": round(df[prob_col].median(), 4),
                }

        report_path = self.output_dir / "predictions_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions report saved: {report_path}")

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 70)

        for label, stats in summary["predictions_summary"].items():
            logger.info(f"\n{label}:")
            logger.info(
                f"  Positive: {stats['positive']:,} ({stats['positive_pct']:.1f}%)"
            )
            logger.info(
                f"  Mean probability: {stats['mean_probability']:.4f}"
            )


# =============================================================================
# CLI
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="BERT-based mechanism classifier for RSA sentences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mechanisms:
    mechanism_legitimacy:
        Defining parameters of blame / institutional risk management
        legitimization (Borraz, 2008)

    mechanism_functional:
        Functional aptness for handling social risks (Paul, 2021)

    mechanism_complexity:
        Complexity empowerment of local actors

Examples:
    # Train model (using Excel file with train/test split column)
    python mechanism_classifier.py --mode train \\
        --data results/00_data_preparation/sampling/sample_full.xlsx \\
        --output results/02_bert_analysis/classification/ \\
        --model-dir models/mechanism_classifier/ \\
        --epochs 5 --learning-rate 2e-5

    # Evaluate model
    python mechanism_classifier.py --mode evaluate \\
        --data results/00_data_preparation/sampling/sample_full.xlsx \\
        --model-dir models/mechanism_classifier/ \\
        --output results/02_bert_analysis/classification/ \\
        --calibrate-thresholds

    # Predict on corpus
    python mechanism_classifier.py --mode predict \\
        --input data/processed/bert_corpus_filtered.parquet \\
        --model-dir models/mechanism_classifier/ \\
        --output results/02_bert_analysis/classification/

Model:
    KBLab/bert-base-swedish-cased (Royal Library of Sweden)
    https://huggingface.co/KBLab/bert-base-swedish-cased
        """,
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "predict"],
        required=True,
        help="Operation mode: train, evaluate, or predict",
    )

    # Input/output paths
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to Excel/CSV file with 'split' column (train/test modes)",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        help="Path to training CSV (alternative to --data)",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        help="Path to test CSV (alternative to --data)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input parquet/CSV (required for predict mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/02_bert_analysis/classification"),
        help="Output directory (default: results/02_bert_analysis/classification/)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/mechanism_classifier"),
        help="Model directory (default: models/mechanism_classifier/)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: auto-detect based on device)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse frequency class weighting",
    )
    parser.add_argument(
        "--exclude-mechanisms",
        type=str,
        nargs="+",
        default=[],
        help="Mechanism labels to exclude from training (e.g., mechanism_legitimacy_lowrisk)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Evaluation options
    parser.add_argument(
        "--calibrate-thresholds",
        action="store_true",
        help="Calibrate classification thresholds on test data",
    )

    # Prediction options
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device for inference (default: auto)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.mode == "train":
        # Either --data or --train-data is required
        if args.data is None and args.train_data is None:
            raise ValueError("--data or --train-data is required for train mode")
        if args.data is not None and not args.data.exists():
            raise FileNotFoundError(f"Data file not found: {args.data}")
        if args.train_data is not None and not args.train_data.exists():
            raise FileNotFoundError(f"Train data not found: {args.train_data}")

    elif args.mode == "evaluate":
        # Either --data or --test-data is required
        if args.data is None and args.test_data is None:
            raise ValueError("--data or --test-data is required for evaluate mode")
        if args.data is not None and not args.data.exists():
            raise FileNotFoundError(f"Data file not found: {args.data}")
        if args.test_data is not None and not args.test_data.exists():
            raise FileNotFoundError(f"Test data not found: {args.test_data}")

    elif args.mode == "predict":
        if args.input is None:
            raise ValueError("--input is required for predict mode")
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        validate_args(args)

        # Filter mechanism labels if exclusions specified
        global MECHANISM_LABELS
        if args.exclude_mechanisms:
            excluded = set(args.exclude_mechanisms)
            invalid = excluded - set(MECHANISM_LABELS)
            if invalid:
                raise ValueError(f"Unknown mechanisms to exclude: {invalid}")
            MECHANISM_LABELS = [m for m in MECHANISM_LABELS if m not in excluded]
            logger.info(f"Excluded mechanisms: {args.exclude_mechanisms}")
            logger.info(f"Training with {len(MECHANISM_LABELS)} mechanisms: {MECHANISM_LABELS}")

        # Set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Initialize pipeline
        pipeline = MechanismPipeline(
            model_dir=args.model_dir,
            output_dir=args.output,
            device=args.device,
            verbose=args.verbose,
        )

        if args.mode == "train":
            # Create training config
            config = TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                use_class_weights=not args.no_class_weights,
                seed=args.seed,
            )

            if args.batch_size is not None:
                config.batch_size = args.batch_size
            else:
                config.batch_size = pipeline.device_manager.get_recommended_batch_size()

            # Use combined data file or separate train/test files
            if args.data is not None:
                pipeline.train(
                    data_path=args.data,
                    config=config,
                )
            else:
                pipeline.train(
                    train_data_path=args.train_data,
                    test_data_path=args.test_data,
                    config=config,
                )

        elif args.mode == "evaluate":
            # Use combined data file or separate test file
            if args.data is not None:
                pipeline.evaluate(
                    data_path=args.data,
                    calibrate_thresholds=args.calibrate_thresholds,
                )
            else:
                pipeline.evaluate(
                    test_data_path=args.test_data,
                    calibrate_thresholds=args.calibrate_thresholds,
                )

        elif args.mode == "predict":
            batch_size = args.batch_size
            if batch_size is None:
                batch_size = pipeline.device_manager.get_recommended_batch_size() * 2

            pipeline.predict(
                input_path=args.input,
                batch_size=batch_size,
                max_samples=args.max_samples,
            )

        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE")
        logger.info("=" * 70)

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
        print(
            "  pip install transformers torch pandas pyarrow scikit-learn",
            file=sys.stderr,
        )
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
