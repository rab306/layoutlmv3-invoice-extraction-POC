"""
Metrics and evaluation functions for invoice extraction model.
"""

import numpy as np
from typing import Dict, List, Tuple
from evaluate import load as load_metric


class ModelMetrics:
    """Handles computation and tracking of model metrics."""
    
    def __init__(self, id2label: Dict[int, str]):
        """
        Initialize metrics calculator.
        
        Args:
            id2label: Mapping from label IDs to label names
        """
        self.id2label = id2label
        self.seqeval_metric = load_metric("seqeval")
    
    def compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute token classification metrics using seqeval.
        
        Args:
            predictions: Model predictions (batch_size, seq_len)
            labels: Ground truth labels (batch_size, seq_len)
            
        Returns:
            Dictionary of metrics
        """
        # Remove padding and special tokens
        true_predictions = []
        true_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            pred_labels = []
            true_label_seq = []
            
            for pred, label in zip(pred_seq, label_seq):
                if label == -100:  # Skip padding
                    continue
                pred_labels.append(self.id2label[pred])
                true_label_seq.append(self.id2label[label])
            
            true_predictions.append(pred_labels)
            true_labels.append(true_label_seq)
        
        # Compute seqeval metrics
        results = self.seqeval_metric.compute(
            predictions=true_predictions,
            references=true_labels
        )
        
        # Flatten metrics for Trainer compatibility
        return self._flatten_metrics(results)
    
    def _flatten_metrics(self, results: Dict) -> Dict[str, float]:
        """
        Flatten nested seqeval results into flat dictionary.
        
        Args:
            results: Nested seqeval results
            
        Returns:
            Flat dictionary of metrics
        """
        report = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
        # Add per-entity metrics
        for key, value in results.items():
            if key not in ["overall_precision", "overall_recall", "overall_f1", "overall_accuracy"]:
                if isinstance(value, dict):
                    report[f"{key}_f1"] = value["f1"]
                else:
                    report[key] = value
        
        return report
    
    def compute_per_entity_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, int]]:
        """
        Compute accuracy for each entity type.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            
        Returns:
            Dictionary with per-entity statistics
        """
        entity_stats = {}
        
        for pred_seq, label_seq in zip(predictions, labels):
            for pred, label in zip(pred_seq, label_seq):
                if label == -100 or label == 0:  # Skip padding and 'O'
                    continue
                
                label_name = self.id2label[label]
                
                if label_name not in entity_stats:
                    entity_stats[label_name] = {"correct": 0, "total": 0}
                
                entity_stats[label_name]["total"] += 1
                
                if pred == label:
                    entity_stats[label_name]["correct"] += 1
        
        return entity_stats