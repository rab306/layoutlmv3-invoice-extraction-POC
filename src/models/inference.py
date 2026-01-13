"""
Model inference for invoice extraction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

from utils.config import get_config

logger = logging.getLogger(__name__)


class InvoiceExtractor:
    """Handles model inference for invoice extraction."""
    
    def __init__(self, model_path: Optional[str] = None, config_dir: Optional[str] = None):
        """
        Initialize invoice extractor.
        
        Args:
            model_path: Path to trained model (uses config if None)
            config_dir: Directory containing config files
        """
        self.config = get_config(config_dir) if config_dir else get_config()
        
        # Get model path
        if model_path is None:
            model_path = self.config.get_data_path(
                'paths.results.best_model',
                'results/checkpoints/layoutlmv3_invoice_best'
            )
        
        self.model_path = Path(model_path)
        
        # Load model configuration
        self.id2label, self.label2id = self._load_label_mapping()
        self.processor = self._initialize_processor()
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"InvoiceExtractor initialized with model: {self.model_path}")
    
    def _load_label_mapping(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Load label mapping from model config or defaults."""
        config_path = self.model_path / "config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'id2label' in config:
                id2label = {int(k): v for k, v in config['id2label'].items()}
                label2id = {v: k for k, v in id2label.items()}
                return id2label, label2id
        
        # Fallback to defaults
        id2label = {
            0: "O", 1: "B-invoice_number", 2: "I-invoice_number",
            3: "B-invoice_date", 4: "I-invoice_date", 5: "B-buyer_address",
            6: "I-buyer_address", 7: "B-seller_address", 8: "I-seller_address",
            9: "B-product_description", 10: "I-product_description",
            11: "B-product_quantity", 12: "B-product_unit_price",
            13: "B-product_total_price", 14: "B-payment_total",
            15: "B-payment_sub_total"
        }
        label2id = {v: k for k, v in id2label.items()}
        
        return id2label, label2id
    
    def _initialize_processor(self) -> LayoutLMv3Processor:
        """Initialize LayoutLMv3 processor."""
        return LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False
        )
    
    def _load_model(self) -> LayoutLMv3ForTokenClassification:
        """Load trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        return LayoutLMv3ForTokenClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True
        )
    
    def extract_from_image(self, image_path: Path, words: List[str], boxes: List[List[int]]) -> Dict[str, Any]:
        """
        Extract entities from invoice image.
        
        Args:
            image_path: Path to invoice image
            words: List of OCR words
            boxes: List of bounding boxes [x0, y0, x1, y1]
            
        Returns:
            Dictionary with extracted entities
        """
        # Load and process image
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Prepare inputs
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
        
        # Extract entities
        entities = self._extract_entities(words, predictions)
        
        return {
            "filename": image_path.name,
            "entities": entities,
            "raw_predictions": predictions.tolist()
        }
    
    def _extract_entities(self, words: List[str], predictions: np.ndarray) -> Dict[str, List[str]]:
        """
        Extract and group entities from predictions using BIO labels
        without altering the label space.

        This function intentionally preserves full BIO labels
        (e.g. "B-seller_address", "I-seller_address") to match the
        notebook behavior exactly. No normalization or aggregation
        to entity-level names is performed.

        Args:
            words: List of OCR words (one per token, same assumption as notebook)
            predictions: Model predictions (label IDs) for each token

        Returns:
            Dictionary mapping BIO labels to lists of extracted text spans
            (e.g. {"B-seller_address": ["ACME Corp"], ...})
        """
        # Initialize entities using full BIO labels (not entity names)
        entities = {label: [] for label in self.label2id.keys() if label != "O"}

        current_entity = None
        current_words = []

        for word, pred_id in zip(words, predictions):
            label = self.id2label[pred_id]

            if label == "O":
                if current_entity:
                    entities[current_entity].append(" ".join(current_words))
                    current_entity = None
                    current_words = []
                continue

            if label.startswith("B-"):
                if current_entity:
                    entities[current_entity].append(" ".join(current_words))

                # Keep full BIO label to match notebook behavior
                current_entity = label
                current_words = [word]

            elif label.startswith("I-"):
                # Continue only if it matches the current B- label
                if current_entity and label.replace("I-", "B-") == current_entity:
                    current_words.append(word)

        # Handle last open entity
        if current_entity:
            entities[current_entity].append(" ".join(current_words))

        return entities

    
    def batch_extract(self, image_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple images.
        
        Args:
            image_data: List of dictionaries with 'image_path', 'words', 'boxes'
            
        Returns:
            List of extraction results
        """
        results = []
        
        for data in image_data:
            try:
                result = self.extract_from_image(
                    Path(data['image_path']),
                    data['words'],
                    data['boxes']
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {data.get('image_path', 'unknown')}: {e}")
                results.append({
                    "filename": data.get('image_path', 'unknown'),
                    "error": str(e),
                    "entities": {}
                })
        
        return results