"""
Main trainer module for LayoutLMv3 fine-tuning.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator
)

from utils.config import get_config
from .metrics import ModelMetrics
from .loss import WeightedCrossEntropyLoss

logger = logging.getLogger(__name__)


class WeightedLossTrainer(Trainer):
    """Custom Trainer with weighted loss support."""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get num_labels (handle DataParallel wrapper)
        num_labels = model.module.config.num_labels if hasattr(model, 'module') else model.config.num_labels
        
        # Weighted loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


class InvoiceModelTrainer:
    """Main training orchestration class."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config = get_config(config_dir) if config_dir else get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get label mapping
        model_config = self.config.get_model_param('model', {})
        self.id2label = model_config.get('id2label', self._get_default_labels())
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.id2label)
        
        logger.info(f"Trainer initialized for {self.num_labels} labels")
    
    def _get_default_labels(self) -> Dict[int, str]:
        """Default label mapping."""
        return {
            0: "O", 1: "B-invoice_number", 2: "I-invoice_number",
            3: "B-invoice_date", 4: "I-invoice_date", 5: "B-buyer_address",
            6: "I-buyer_address", 7: "B-seller_address", 8: "I-seller_address",
            9: "B-product_description", 10: "I-product_description",
            11: "B-product_quantity", 12: "B-product_unit_price",
            13: "B-product_total_price", 14: "B-payment_total",
            15: "B-payment_sub_total"
        }
    
    def load_model(self) -> LayoutLMv3ForTokenClassification:
        """Load and initialize model."""
        model_name = self.config.get_model_param('model.model_name', 'microsoft/layoutlmv3-base')
        
        logger.info(f"Loading model: {model_name}")
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        model.to(self.device)
        
        return model
    
    def create_training_args(self, output_dir: str = "./model_output") -> TrainingArguments:
        """Create training arguments from config."""
        training_config = self.config.get_model_param('training', {})
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get('num_train_epochs', 15),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 8),
            learning_rate=float(training_config.get('learning_rate', 1e-5)),
            weight_decay=training_config.get('weight_decay', 0.1),
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=True,
            seed=42,
        )
    
    def train(self, dataset_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run complete training pipeline."""
        logger.info("Starting training pipeline...")
        
        # Load datasets
        train_data, val_data, test_data = self._load_datasets(dataset_path)
        
        # Initialize components
        model = self.load_model()
        metrics = ModelMetrics(self.id2label)
        loss_calculator = WeightedCrossEntropyLoss(self.num_labels)
        
        # Create trainer
        trainer = self._create_trainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            metrics=metrics,
            loss_calculator=loss_calculator,
            output_dir=output_dir or "./model_output"
        )
        
        # Train
        train_result = trainer.train()
        
        # Evaluate
        val_results = trainer.evaluate()
        trainer.eval_dataset = test_data
        test_results = trainer.evaluate()
        
        # Save
        self._save_results(trainer, train_result, val_results, test_results, output_dir)
        
        return {
            "train": train_result.metrics,
            "validation": val_results,
            "test": test_results,
            "model_path": output_dir or "./model_output"
        }
    
    def _load_datasets(self, dataset_path: str) -> Tuple:
        """Load datasets from disk."""
        path = Path(dataset_path)
        return (
            load_from_disk(path / "train"),
            load_from_disk(path / "val"),
            load_from_disk(path / "test")
        )
    
    def _create_trainer(self, model, train_dataset, val_dataset, metrics, loss_calculator, output_dir):
        """Create Trainer instance."""
        training_args = self.create_training_args(output_dir)
        class_weights = loss_calculator.get_weights(self.device)
        
        # Create callbacks
        callbacks = []
        early_stopping = self.config.get_model_param('training.early_stopping_patience', 5)
        if early_stopping > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping))
        
        return WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            compute_metrics=metrics.compute_metrics,
            class_weights=class_weights,
            callbacks=callbacks,
        )
    
    def _save_results(self, trainer, train_result, val_results, test_results, output_dir):
        """Save model and metrics."""
        trainer.save_model(output_dir)
        
        metrics = {
            "train": train_result.metrics,
            "validation": val_results,
            "test": test_results
        }
        
        with open(Path(output_dir) / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Model saved to: {output_dir}")