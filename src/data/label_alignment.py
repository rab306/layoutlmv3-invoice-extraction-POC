"""
Phase 5: Label Alignment
Matches ground truth labels to OCR words and assigns BIO tags.
Now with configuration integration.
"""

import json
import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz, process
from tqdm import tqdm

from utils.config import get_config

logger = logging.getLogger(__name__)


class LabelAligner:
    """Aligns ground truth labels to OCR output using fuzzy matching with config support."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize label aligner with configuration.
        
        Args:
            config_dir: Directory containing config files (uses default if None)
        """
        self.config = get_config(config_dir) if config_dir else get_config()
        
        # Get configuration values with defaults
        self.fuzzy_threshold = self.config.get_data_path(
            'label_alignment.fuzzy_threshold',
            70
        )
        
        self.numeric_threshold = self.config.get_data_path(
            'label_alignment.numeric_threshold',
            55
        )
        
        self.max_sequence_distance = self.config.get_data_path(
            'label_alignment.max_sequence_distance',
            30
        )
        
        self.spatial_max_distance = self.config.get_data_path(
            'label_alignment.spatial_max_distance',
            130
        )
        
        # Define label mapping (from config or default)
        self.LABEL2ID = {
            'O': 0,
            'B-invoice_number': 1,
            'I-invoice_number': 2,
            'B-invoice_date': 3,
            'I-invoice_date': 4,
            'B-buyer_address': 5,
            'I-buyer_address': 6,
            'B-seller_address': 7,
            'I-seller_address': 8,
            'B-product_description': 9,
            'I-product_description': 10,
            'B-product_quantity': 11,
            'B-product_unit_price': 12,
            'B-product_total_price': 13,
            'B-payment_total': 14,
            'B-payment_sub_total': 15
        }
        
        self.ID2LABEL = {v: k for k, v in self.LABEL2ID.items()}
        
        logger.info(f"LabelAligner initialized with config from {config_dir or 'default'}")
        logger.info(f"Fuzzy threshold: {self.fuzzy_threshold}")
        logger.info(f"Numeric threshold: {self.numeric_threshold}")
        logger.info(f"Labels defined: {len(self.LABEL2ID)}")
    
    @staticmethod
    def normalize_string(s: Any) -> str:
        """Normalize string for matching: lowercase, remove extra whitespace."""
        if s is None:
            return ""
        s = str(s).lower().strip()
        s = re.sub(r'\s+', ' ', s)
        return s
    
    @staticmethod
    def normalize_numeric(s: Any) -> str:
        """Normalize numeric strings: remove currency symbols and commas."""
        if s is None:
            return ""
        s = str(s)
        s = re.sub(r'[$€£¥]', '', s)
        s = s.replace(',', '')
        return s.strip()
    
    def fuzzy_match_word(self, target: str, word_list: List[str], threshold: int = None) -> Tuple[Optional[str], float, int]:
        """
        Find best fuzzy match for target in word_list.
        
        Returns: (matched_word, match_score, word_index) or (None, 0, -1)
        """
        if threshold is None:
            threshold = self.fuzzy_threshold
        
        if not word_list or not target:
            return None, 0, -1
        
        result = process.extractOne(
            target,
            word_list,
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )
        
        if result:
            matched_word, score, idx = result
            return matched_word, score, idx
        
        return None, 0, -1
    
    def find_sequence_match(self, target_tokens: List[str], ocr_words: List[str], 
                           ocr_boxes: List[List[int]], threshold: int = None) -> List[int]:
        """
        Find a sequence of OCR words that matches target tokens.
        
        Returns: List of indices in ocr_words, or []
        """
        if threshold is None:
            threshold = self.fuzzy_threshold
        
        if not target_tokens or not ocr_words:
            return []
        
        matched_indices = []
        
        for token in target_tokens:
            norm_token = self.normalize_string(token)
            if not norm_token:
                continue
            
            match, score, idx = self.fuzzy_match_word(norm_token, ocr_words, threshold)
            
            if match and idx not in matched_indices:
                matched_indices.append(idx)
        
        # Check if matched words are roughly sequential
        if len(matched_indices) > 1:
            matched_indices.sort()
            gaps = [matched_indices[i+1] - matched_indices[i] for i in range(len(matched_indices)-1)]
            
            if any(gap > self.max_sequence_distance for gap in gaps):
                boxes = [ocr_boxes[idx] for idx in matched_indices]
                if not self._are_spatially_close(boxes):
                    return []
        
        return matched_indices
    
    def _are_spatially_close(self, boxes: List[List[int]]) -> bool:
        """Check if bounding boxes are spatially close."""
        if len(boxes) < 2:
            return True
        
        max_distance = self.spatial_max_distance
        
        for i in range(len(boxes) - 1):
            box1, box2 = boxes[i], boxes[i+1]
            center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
            center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if distance > max_distance:
                return False
        
        return True
    
    @staticmethod
    def group_by_row(words: List[str], boxes: List[List[int]], threshold: int = 25) -> List[List[Tuple]]:
        """
        Group words into rows based on Y-coordinate.
        
        Returns: List of rows, each containing (word, box, original_index)
        """
        if not words:
            return []
        
        items = [(words[i], boxes[i], i) for i in range(len(words))]
        items.sort(key=lambda x: (x[1][1] + x[1][3]) / 2)
        
        rows = []
        current_row = [items[0]]
        current_y = (items[0][1][1] + items[0][1][3]) / 2
        
        for item in items[1:]:
            y_center = (item[1][1] + item[1][3]) / 2
            
            if abs(y_center - current_y) < threshold:
                current_row.append(item)
            else:
                current_row.sort(key=lambda x: x[1][0])
                rows.append(current_row)
                current_row = [item]
                current_y = y_center
        
        if current_row:
            current_row.sort(key=lambda x: x[1][0])
            rows.append(current_row)
        
        return rows
    
    def match_numeric_field(self, field_value: Any, ocr_words: List[str], field_name: str) -> List[Tuple[int, int]]:
        """
        Special ultra-lenient matching for numeric fields.
        
        Returns: List of (word_index, label_id)
        """
        if field_value is None:
            return []
        
        target = str(field_value).replace('$', '').replace(',', '').replace(' ', '').strip()
        
        if not target:
            return []
        
        norm_words = []
        for word in ocr_words:
            word_clean = word.replace('$', '').replace(',', '').replace(' ', '').strip()
            norm_words.append(word_clean)
        
        # Strategy 1: Exact match
        try:
            idx = norm_words.index(target)
            return [(idx, self.LABEL2ID[f'B-{field_name}'])]
        except ValueError:
            pass
        
        # Strategy 2: Partial substring match
        for idx, word_clean in enumerate(norm_words):
            if len(word_clean) >= 3:
                if target in word_clean or word_clean in target:
                    shorter = min(len(target), len(word_clean))
                    longer = max(len(target), len(word_clean))
                    if shorter / longer >= 0.55:
                        return [(idx, self.LABEL2ID[f'B-{field_name}'])]
        
        # Strategy 3: Digit overlap match
        target_digits = ''.join(c for c in target if c.isdigit())
        
        if len(target_digits) >= 2:
            for idx, word_clean in enumerate(norm_words):
                word_digits = ''.join(c for c in word_clean if c.isdigit())
                
                if len(word_digits) >= 2:
                    common_digits = sum(1 for d in target_digits if d in word_digits)
                    overlap_ratio = common_digits / len(target_digits)
                    
                    if overlap_ratio >= 0.65:
                        return [(idx, self.LABEL2ID[f'B-{field_name}'])]
        
        # Strategy 4: Very lenient fuzzy match
        for idx, word_clean in enumerate(norm_words):
            if len(word_clean) >= 2:
                similarity = fuzz.ratio(target, word_clean)
                
                if similarity >= self.numeric_threshold:
                    return [(idx, self.LABEL2ID[f'B-{field_name}'])]
        
        return []
    
    def match_simple_field(self, field_value: Any, ocr_words: List[str], field_name: str) -> List[Tuple[int, int]]:
        """
        Match a simple single-value field.
        
        Returns: List of (word_index, label_id)
        """
        if field_value is None:
            return []
        
        if 'payment' in field_name:
            return self.match_numeric_field(field_value, ocr_words, field_name)
        
        norm_value = self.normalize_string(field_value)
        norm_words = [self.normalize_string(w) for w in ocr_words]
        
        # Try exact match
        try:
            idx = norm_words.index(norm_value)
            return [(idx, self.LABEL2ID[f'B-{field_name}'])]
        except ValueError:
            pass
        
        # Try fuzzy match
        match, score, idx = self.fuzzy_match_word(norm_value, norm_words, self.fuzzy_threshold)
        
        if match:
            return [(idx, self.LABEL2ID[f'B-{field_name}'])]
        
        # Try splitting the value
        if '.' in str(field_value) or ' ' in str(field_value):
            tokens = re.split(r'[\s\.]', str(field_value))
            tokens = [t for t in tokens if t]
            
            if len(tokens) > 1:
                matches = []
                for token in tokens:
                    norm_token = self.normalize_string(token)
                    match, score, idx = self.fuzzy_match_word(norm_token, norm_words, self.fuzzy_threshold)
                    if match:
                        matches.append(idx)
                
                if matches:
                    i_tag_key = f'I-{field_name}'
                    
                    if i_tag_key in self.LABEL2ID:
                        result = [(matches[0], self.LABEL2ID[f'B-{field_name}'])]
                        for idx in matches[1:]:
                            result.append((idx, self.LABEL2ID[i_tag_key]))
                        return result
                    else:
                        return [(matches[0], self.LABEL2ID[f'B-{field_name}'])]
        
        return []
    
    def match_address_field(self, address_value: Any, ocr_words: List[str], 
                           ocr_boxes: List[List[int]], field_name: str) -> List[Tuple[int, int]]:
        """
        Match a multi-word address field.
        
        Returns: List of (word_index, label_id)
        """
        if address_value is None:
            return []
        
        tokens = str(address_value).split()
        matched_indices = self.find_sequence_match(tokens, ocr_words, ocr_boxes, self.fuzzy_threshold)
        
        if not matched_indices:
            return []
        
        result = [(matched_indices[0], self.LABEL2ID[f'B-{field_name}'])]
        for idx in matched_indices[1:]:
            result.append((idx, self.LABEL2ID[f'I-{field_name}']))
        
        return result
    
    def match_products(self, products: List[Dict], ocr_words: List[str], 
                      ocr_boxes: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Match product line items.
        
        Returns: List of (word_index, label_id)
        """
        if not products:
            return []
        
        results = []
        rows = self.group_by_row(ocr_words, ocr_boxes)
        
        for product in products:
            description = product.get('description')
            quantity = product.get('quantity')
            unit_price = product.get('unit_price')
            total_price = product.get('total_price')
            
            if description:
                desc_tokens = str(description).split()
                desc_matches = self.find_sequence_match(desc_tokens, ocr_words, ocr_boxes, self.fuzzy_threshold)
                
                if desc_matches:
                    results.append((desc_matches[0], self.LABEL2ID['B-product_description']))
                    for idx in desc_matches[1:]:
                        results.append((idx, self.LABEL2ID['I-product_description']))
                    
                    desc_y = (ocr_boxes[desc_matches[0]][1] + ocr_boxes[desc_matches[0]][3]) / 2
                    
                    matching_row = None
                    for row in rows:
                        row_y = (row[0][1][1] + row[0][1][3]) / 2
                        if abs(row_y - desc_y) < 20:
                            matching_row = row
                            break
                    
                    if matching_row:
                        row_words = [item[0] for item in matching_row]
                        row_indices = [item[2] for item in matching_row]
                        
                        if quantity is not None:
                            qty_matches = self.match_numeric_field(quantity, row_words, 'product_quantity')
                            if qty_matches:
                                local_idx = qty_matches[0][0]
                                global_idx = row_indices[local_idx]
                                results.append((global_idx, self.LABEL2ID['B-product_quantity']))
                        
                        if unit_price is not None:
                            price_matches = self.match_numeric_field(unit_price, row_words, 'product_unit_price')
                            if price_matches:
                                local_idx = price_matches[0][0]
                                global_idx = row_indices[local_idx]
                                results.append((global_idx, self.LABEL2ID['B-product_unit_price']))
                        
                        if total_price is not None:
                            total_matches = self.match_numeric_field(total_price, row_words, 'product_total_price')
                            if total_matches:
                                local_idx = total_matches[0][0]
                                global_idx = row_indices[local_idx]
                                results.append((global_idx, self.LABEL2ID['B-product_total_price']))
        
        return results
    
    def label_invoice(self, ground_truth: Dict, ocr_result: Dict) -> Optional[Tuple]:
        """
        Create BIO labels for a single invoice.
        
        Returns: (words, boxes, labels, label_strings) or None if failed
        """
        ocr_words = ocr_result['words']
        ocr_boxes = ocr_result['boxes']
        
        if not ocr_words:
            return None
        
        labels = [0] * len(ocr_words)
        labeled_indices = set()
        
        # Match simple fields
        simple_fields = [
            ('invoice_number', ground_truth.get('invoice_number')),
            ('invoice_date', ground_truth.get('invoice_date')),
            ('payment_total', ground_truth.get('payment_total')),
            ('payment_sub_total', ground_truth.get('payment_sub_total'))
        ]
        
        for field_name, field_value in simple_fields:
            matches = self.match_simple_field(field_value, ocr_words, field_name)
            for idx, label_id in matches:
                if idx not in labeled_indices:
                    labels[idx] = label_id
                    labeled_indices.add(idx)
        
        # Match address fields
        address_fields = [
            ('buyer_address', ground_truth.get('buyer_address')),
            ('seller_address', ground_truth.get('seller_address'))
        ]
        
        for field_name, field_value in address_fields:
            matches = self.match_address_field(field_value, ocr_words, ocr_boxes, field_name)
            for idx, label_id in matches:
                if idx not in labeled_indices:
                    labels[idx] = label_id
                    labeled_indices.add(idx)
        
        # Match products
        products = ground_truth.get('products', [])
        product_matches = self.match_products(products, ocr_words, ocr_boxes)
        for idx, label_id in product_matches:
            if idx not in labeled_indices:
                labels[idx] = label_id
                labeled_indices.add(idx)
        
        label_strings = [self.ID2LABEL[label_id] for label_id in labels]
        
        return (ocr_words, ocr_boxes, labels, label_strings)
    
    def run_alignment(self, ground_truth_path: Optional[str] = None, 
                     ocr_results_path: Optional[str] = None,
                     output_path: Optional[str] = None,
                     max_samples: Optional[int] = None,
                     test_mode: bool = False) -> Dict[str, Any]:
        """
        Run the complete label alignment pipeline.
        
        Returns: Dictionary with alignment results
        """
        # Get paths from config if not specified
        if ground_truth_path is None:
            ground_truth_path = self.config.get_data_path(
                'paths.processed.cleaned_data',
                'data/processed/cleaned_data.json'
            )
        
        if ocr_results_path is None:
            ocr_results_path = self.config.get_data_path(
                'paths.processed.ocr_results',
                'data/processed/ocr_results.json'
            )
        
        if output_path is None:
            output_path = self.config.get_data_path(
                'paths.processed.labeled_dataset',
                'data/processed/labeled_dataset.json'
            )
        
        # In test mode, save to temporary location
        if test_mode:
            output_path = "/tmp/test_labeled_dataset.json"
            max_samples = max_samples or 5
            logger.info(f"TEST MODE: Processing {max_samples} samples")
            print(f"TEST MODE: Processing {max_samples} samples")
            print(f"Output will be saved to: {output_path}")
            print("(Not saved to actual data directory)")
        
        print("\n" + "=" * 80)
        print("PHASE 5: LABEL ALIGNMENT")
        print("=" * 80)
        
        # STEP 1: Define Label Schema
        logger.info("[STEP 1] Defining label schema...")
        print("\n[STEP 1] Defining label schema...")
        print(f"✓ Defined {len(self.LABEL2ID)} labels")
        
        # STEP 2: Load Data
        logger.info("[STEP 2] Loading preprocessed data...")
        print("\n[STEP 2] Loading preprocessed data...")
        
        ground_truth_path_obj = Path(ground_truth_path)
        ocr_results_path_obj = Path(ocr_results_path)
        
        if not ground_truth_path_obj.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path_obj}")
        
        if not ocr_results_path_obj.exists():
            raise FileNotFoundError(f"OCR results file not found: {ocr_results_path_obj}")
        
        with open(ground_truth_path_obj, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        logger.info(f"Loaded {len(ground_truth_data)} ground truth records")
        print(f"✓ Loaded {len(ground_truth_data)} ground truth records")
        
        with open(ocr_results_path_obj, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        logger.info(f"Loaded {len(ocr_data)} OCR records")
        print(f"✓ Loaded {len(ocr_data)} OCR records")
        
        ocr_lookup = {item['filename'].replace('.png', '.json'): item for item in ocr_data}
        logger.info(f"Created OCR lookup dictionary")
        print(f"✓ Created OCR lookup dictionary")
        
        # Limit samples if specified
        if max_samples:
            ground_truth_data = ground_truth_data[:max_samples]
            logger.info(f"Limited to {max_samples} samples for processing")
            print(f"✓ Limited to {max_samples} samples for processing")
        
        # STEP 3: Process All Invoices
        logger.info(f"[STEP 6] Processing {len(ground_truth_data)} invoices...")
        print(f"\n[STEP 6] Processing {len(ground_truth_data)} invoices...")
        
        labeled_dataset = []
        failed_count = 0
        label_stats = defaultdict(int)
        
        for gt_record in tqdm(ground_truth_data, desc="Labeling invoices"):
            filename = gt_record['filename']
            ocr_result = ocr_lookup.get(filename)
            
            if not ocr_result:
                failed_count += 1
                continue
            
            result = self.label_invoice(gt_record, ocr_result)
            
            if result is None:
                failed_count += 1
                continue
            
            words, boxes, labels, label_strings = result
            
            labeled_sample = {
                'filename': filename,
                'words': words,
                'boxes': boxes,
                'labels': labels,
                'label_strings': label_strings
            }
            
            labeled_dataset.append(labeled_sample)
            
            for label in labels:
                label_stats[label] += 1
        
        logger.info(f"Successfully labeled {len(labeled_dataset)} invoices")
        print(f"\n✓ Successfully labeled {len(labeled_dataset)} invoices")
        if failed_count > 0:
            logger.warning(f"Failed to label {failed_count} invoices")
            print(f"⚠ Failed to label {failed_count} invoices")
        
        # STEP 4: Save Labeled Dataset
        logger.info("[STEP 7] Saving labeled dataset...")
        print("\n[STEP 7] Saving labeled dataset...")
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(labeled_dataset, f, indent=2, ensure_ascii=False)
        
        file_size_mb = output_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(labeled_dataset)} records to {output_path}")
        print(f"✓ Saved to: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        
        # STEP 5: Generate Summary
        total_tokens = sum(label_stats.values())
        entity_tokens = total_tokens - label_stats[0]
        entity_percentage = (entity_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        
        print("\n" + "=" * 80)
        print("PHASE 5 COMPLETE - SUMMARY")
        print("=" * 80)
        
        print(f"\n✓ Successfully processed {len(labeled_dataset)} invoices")
        print(f"✓ Total tokens labeled: {total_tokens:,}")
        print(f"✓ Entity tokens: {entity_tokens:,} ({entity_percentage:.1f}%)")
        print(f"✓ Output file: {output_path} ({file_size_mb:.2f} MB)")
        
        if failed_count > 0:
            print(f"⚠ Failed invoices: {failed_count}")
        
        return {
            "labeled_samples": len(labeled_dataset),
            "failed_samples": failed_count,
            "total_tokens": total_tokens,
            "entity_tokens": entity_tokens,
            "entity_percentage": entity_percentage,
            "output_path": str(output_path_obj),
            "label_distribution": dict(label_stats)
        }