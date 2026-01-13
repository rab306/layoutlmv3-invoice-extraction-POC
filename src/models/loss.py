"""
Custom loss functions for invoice extraction model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class WeightedCrossEntropyLoss:
    """Weighted cross-entropy loss with class balancing."""
    
    def __init__(self, num_labels: int, label_counts: Optional[dict[int, int]] = None):
        """
        Initialize weighted loss.
        
        Args:
            num_labels: Number of output classes
            label_counts: Dictionary of label_id -> count (for weight calculation)
        """
        self.num_labels = num_labels
        self.label_counts = label_counts or self._get_default_counts()
        self.class_weights = self._compute_class_weights()
    
    def _get_default_counts(self) -> dict[int, int]:
        """Get default label counts based on Phase 5 statistics."""
        return {
            0: 220000,   # O
            1: 3500,     # B-invoice_number
            2: 1700,     # I-invoice_number
            3: 5000,     # B-invoice_date
            4: 2500,     # I-invoice_date
            5: 7400,     # B-buyer_address
            6: 14500,    # I-buyer_address
            7: 6000,     # B-seller_address
            8: 14000,    # I-seller_address
            9: 8000,     # B-product_description
            10: 12900,   # I-product_description
            11: 3300,    # B-product_quantity
            12: 2200,    # B-product_unit_price
            13: 2200,    # B-product_total_price
            14: 4200,    # B-payment_total
            15: 4000,    # B-payment_sub_total
        }
    
    def _compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights with square root smoothing.
        
        Returns:
            Tensor of class weights
        """
        total_tokens = sum(self.label_counts.values())
        class_weights = []
        
        for class_id in range(self.num_labels):
            count = self.label_counts.get(class_id, 1)
            # Square root smoothing for balanced weights
            weight = np.sqrt(total_tokens / (self.num_labels * count))
            class_weights.append(weight)
        
        # Normalize
        class_weights = np.array(class_weights)
        class_weights = class_weights / class_weights.sum() * self.num_labels
        
        return torch.FloatTensor(class_weights)
    
    def get_weights(self, device: str = "cpu") -> torch.Tensor:
        """
        Get class weights tensor.
        
        Args:
            device: Target device for tensor
            
        Returns:
            Class weights tensor
        """
        return self.class_weights.to(device)
    
    def create_loss_function(self, device: str = "cpu") -> nn.Module:
        """
        Create PyTorch loss function with computed weights.
        
        Args:
            device: Target device
            
        Returns:
            CrossEntropyLoss instance with weights
        """
        weights = self.get_weights(device)
        return nn.CrossEntropyLoss(weight=weights, ignore_index=-100)