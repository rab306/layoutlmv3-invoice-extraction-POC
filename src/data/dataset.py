"""
Phase 6: Dataset Creation for LayoutLMv3
Memory-efficient implementation with chunked processing.
Now with configuration integration.
"""

import json
import gc
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import LayoutLMv3Processor
from datasets import Dataset, concatenate_datasets, load_from_disk

from utils.config import get_config

logger = logging.getLogger(__name__)


class LayoutLMDatasetCreator:
    """Creates LayoutLMv3-compatible datasets from labeled data with config support."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize dataset creator with configuration.
        
        Args:
            config_dir: Directory containing config files (uses default if None)
        """
        self.config = get_config(config_dir) if config_dir else get_config()
        
        # Get configuration values with defaults
        images_dir = self.config.get_data_path('paths.raw.images', 'data/raw/images')
        self.images_dir = Path(images_dir)
        
        self.processor_name = self.config.get_data_path(
            'dataset.processor',
            'microsoft/layoutlmv3-base'
        )
        
        self.max_length = self.config.get_data_path(
            'dataset.max_length',
            512
        )
        
        self.image_size = self.config.get_data_path(
            'dataset.image_size',
            224
        )
        
        # Split ratios
        split_config = self.config.get_data_path(
            'dataset.split',
            {'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1, 'random_state': 42}
        )
        self.train_ratio = split_config.get('train_ratio', 0.8)
        self.val_ratio = split_config.get('val_ratio', 0.1)
        self.test_ratio = split_config.get('test_ratio', 0.1)
        self.random_state = split_config.get('random_state', 42)
        
        # Memory safety chunk sizes
        memory_config = self.config.get_data_path(
            'dataset.memory_safety',
            {'train_chunk_size': 300, 'val_chunk_size': 200, 'test_chunk_size': 200}
        )
        self.train_chunk_size = memory_config.get('train_chunk_size', 300)
        self.val_chunk_size = memory_config.get('val_chunk_size', 200)
        self.test_chunk_size = memory_config.get('test_chunk_size', 200)
        
        self.processor = None
        
        logger.info(f"LayoutLMDatasetCreator initialized with config from {config_dir or 'default'}")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Processor: {self.processor_name}")
        logger.info(f"Max length: {self.max_length}")
        logger.info(f"Split ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that required directories exist."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
    
    def initialize_processor(self) -> None:
        """Initialize LayoutLMv3 processor."""
        logger.info("Initializing LayoutLMv3 processor...")
        self.processor = LayoutLMv3Processor.from_pretrained(
            self.processor_name,
            apply_ocr=False
        )
        logger.info("✓ Processor initialized")
    
    def load_labeled_data(self, labeled_data_path: Optional[str] = None) -> List[Dict]:
        """Load labeled dataset from JSON file."""
        if labeled_data_path is None:
            labeled_data_path = self.config.get_data_path(
                'paths.processed.labeled_dataset',
                'data/processed/labeled_dataset.json'
            )
        
        labeled_data_path_obj = Path(labeled_data_path)
        logger.info(f"Loading labeled dataset from {labeled_data_path_obj}...")
        
        if not labeled_data_path_obj.exists():
            raise FileNotFoundError(f"Labeled dataset not found: {labeled_data_path_obj}")
        
        with open(labeled_data_path_obj, 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)
        
        logger.info(f"✓ Loaded {len(labeled_data)} labeled samples")
        return labeled_data
    
    def split_dataset(self, labeled_data: List[Dict]) -> Tuple[List, List, List]:
        """
        Split dataset into train/val/test sets using configured ratios.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info(f"\nSplitting dataset: {self.train_ratio*100:.0f}% train, "
                   f"{self.val_ratio*100:.0f}% val, {self.test_ratio*100:.0f}% test")
        
        # First split: train vs (val + test)
        train_val_ratio = self.val_ratio + self.test_ratio
        train_data, temp_data = train_test_split(
            labeled_data,
            test_size=train_val_ratio,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Second split: val vs test
        val_test_ratio = self.test_ratio / (self.val_ratio + self.test_ratio) if (self.val_ratio + self.test_ratio) > 0 else 0.5
        val_data, test_data = train_test_split(
            temp_data,
            test_size=val_test_ratio,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Verify sizes
        total = len(labeled_data)
        logger.info(f"✓ Train set: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
        logger.info(f"✓ Val set: {len(val_data)} samples ({len(val_data)/total*100:.1f}%)")
        logger.info(f"✓ Test set: {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")
        
        # Verify no overlap
        train_files = set(s['filename'] for s in train_data)
        val_files = set(s['filename'] for s in val_data)
        test_files = set(s['filename'] for s in test_data)
        
        assert len(train_files & val_files) == 0, "Train/Val overlap!"
        assert len(train_files & test_files) == 0, "Train/Test overlap!"
        assert len(val_files & test_files) == 0, "Val/Test overlap!"
        logger.info("✓ No data leakage detected")
        
        return train_data, val_data, test_data
    
    def _load_image(self, filename: str) -> Image.Image:
        """Load image from filename."""
        img_filename = filename.replace('.json', '.png')
        img_path = self.images_dir / img_filename
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        return image
    
    def _process_single_sample(self, sample: Dict) -> Optional[Dict]:
        """
        Process single sample into LayoutLMv3 format.
        
        Returns:
            Processed sample or None if failed
        """
        try:
            image = self._load_image(sample['filename'])
            words = sample['words']
            boxes = sample['boxes']
            labels = sample['labels']
            
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                word_labels=labels,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            processed = {
                'pixel_values': encoding['pixel_values'].squeeze(0).numpy(),
                'input_ids': encoding['input_ids'].squeeze(0).tolist(),
                'attention_mask': encoding['attention_mask'].squeeze(0).tolist(),
                'bbox': encoding['bbox'].squeeze(0).tolist(),
                'labels': encoding['labels'].squeeze(0).tolist()
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing {sample['filename']}: {str(e)[:100]}...")
            return None
    
    def process_split(self, split_data: List[Dict], split_name: str, 
                     output_dir: Path, chunk_size: int) -> int:
        """
        Process dataset split in memory-safe chunks.
        
        Returns:
            Number of successfully processed samples
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {split_name.upper()} SET")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {len(split_data)}")
        logger.info(f"Chunk size: {chunk_size}")
        
        # Also print for user visibility
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} SET")
        print(f"{'='*60}")
        print(f"Total samples: {len(split_data)}")
        print(f"Chunk size: {chunk_size}")
        
        num_chunks = (len(split_data) + chunk_size - 1) // chunk_size
        logger.info(f"Number of chunks: {num_chunks}")
        print(f"Number of chunks: {num_chunks}")
        
        chunk_datasets = []
        failed_count = 0
        
        for chunk_idx in range(num_chunks):
            logger.info(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ---")
            print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ---")
            
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(split_data))
            chunk_data = split_data[start_idx:end_idx]
            
            logger.info(f"Processing samples {start_idx} to {end_idx-1} ({len(chunk_data)} samples)...")
            print(f"Processing samples {start_idx} to {end_idx-1} ({len(chunk_data)} samples)...")
            
            # Process chunk
            processed_samples = []
            for sample in tqdm(chunk_data, desc=f"Processing {split_name}"):
                processed = self._process_single_sample(sample)
                if processed is not None:
                    processed_samples.append(processed)
                else:
                    failed_count += 1
            
            # Convert chunk to dataset
            if processed_samples:
                logger.info(f"Converting to dataset...")
                print(f"Converting to dataset...")
                
                chunk_dict = {
                    'pixel_values': [s['pixel_values'] for s in processed_samples],
                    'input_ids': [s['input_ids'] for s in processed_samples],
                    'attention_mask': [s['attention_mask'] for s in processed_samples],
                    'bbox': [s['bbox'] for s in processed_samples],
                    'labels': [s['labels'] for s in processed_samples]
                }
                
                chunk_dataset = Dataset.from_dict(chunk_dict)
                chunk_datasets.append(chunk_dataset)
                
                logger.info(f"✓ Chunk {chunk_idx + 1} complete: {len(chunk_dataset)} samples")
                print(f"✓ Chunk {chunk_idx + 1} complete: {len(chunk_dataset)} samples")
                
                # Clear memory
                del processed_samples
                del chunk_dict
                gc.collect()
            else:
                logger.warning(f"Chunk {chunk_idx + 1} produced no valid samples")
                print(f"⚠ Chunk {chunk_idx + 1} produced no valid samples")
        
        # Concatenate all chunks
        if chunk_datasets:
            logger.info(f"\nConcatenating {len(chunk_datasets)} chunks...")
            print(f"\nConcatenating {len(chunk_datasets)} chunks...")
            final_dataset = concatenate_datasets(chunk_datasets)
            
            # Save to disk
            output_path = output_dir / split_name
            logger.info(f"Saving to disk: {output_path}")
            print(f"Saving to disk: {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)
            final_dataset.save_to_disk(str(output_path))
            
            successful = len(split_data) - failed_count
            logger.info(f"\n✓ {split_name.upper()} COMPLETE")
            logger.info(f"  Successful: {successful}/{len(split_data)} samples")
            if failed_count > 0:
                logger.warning(f"  Failed: {failed_count} samples")
            
            print(f"\n✓ {split_name.upper()} COMPLETE")
            print(f"  Successful: {successful}/{len(split_data)} samples")
            if failed_count > 0:
                print(f"  Failed: {failed_count} samples")
            
            # Clear memory before returning
            del chunk_datasets
            del final_dataset
            gc.collect()
            
            return successful
        else:
            logger.error(f"No valid samples for {split_name}")
            print(f"✗ No valid samples for {split_name}")
            return 0
    
    def create_datasets(self, labeled_data_path: Optional[str] = None,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create complete dataset pipeline.
        
        Returns:
            Dictionary with dataset creation results
        """
        # Get output directory from config if not specified
        if output_dir is None:
            output_dir = self.config.get_data_path(
                'paths.datasets.layoutlm_dataset',
                'layoutlm_dataset'
            )
        
        output_dir_path = Path(output_dir)
        
        print("\n" + "=" * 80)
        print("PHASE 6: DATASET CREATION (MEMORY-SAFE VERSION)")
        print("=" * 80)
        
        logger.info("=" * 80)
        logger.info("PHASE 6: DATASET CREATION (MEMORY-SAFE VERSION)")
        logger.info("=" * 80)
        
        # Initialize processor
        self.initialize_processor()
        
        # Load data
        labeled_data = self.load_labeled_data(labeled_data_path)
        
        # Split data
        train_data, val_data, test_data = self.split_dataset(labeled_data)
        
        # Create output directory
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Process splits
        results = {}
        
        # Process train
        logger.info("\n" + "🔵 " * 30)
        print("\n" + "🔵 " * 30)
        train_count = self.process_split(train_data, "train", output_dir_path, self.train_chunk_size)
        results["train_count"] = train_count
        
        # Process val
        logger.info("\n" + "🟢 " * 30)
        print("\n" + "🟢 " * 30)
        val_count = self.process_split(val_data, "val", output_dir_path, self.val_chunk_size)
        results["val_count"] = val_count
        
        # Process test
        logger.info("\n" + "🟡 " * 30)
        print("\n" + "🟡 " * 30)
        test_count = self.process_split(test_data, "test", output_dir_path, self.test_chunk_size)
        results["test_count"] = test_count
        
        # Calculate sizes
        results["sizes"] = self._calculate_sizes(output_dir_path)
        results["output_dir"] = str(output_dir_path)
        
        # Validate datasets
        validation = self._validate_datasets(output_dir_path)
        results.update(validation)
        
        # Generate summary
        self._generate_summary(results)
        
        return results
    
    def _calculate_sizes(self, output_dir: Path) -> Dict[str, float]:
        """Calculate dataset sizes in MB."""
        sizes = {}
        for split in ["train", "val", "test"]:
            split_dir = output_dir / split
            if split_dir.exists():
                total = 0
                for filepath in split_dir.rglob("*"):
                    if filepath.is_file():
                        total += filepath.stat().st_size
                sizes[split] = total / (1024 ** 2)  # MB
            else:
                sizes[split] = 0.0
        return sizes
    
    def _validate_datasets(self, output_dir: Path) -> Dict[str, Any]:
        """Load and validate created datasets."""
        logger.info("\n" + "="*80)
        logger.info("VALIDATING DATASETS")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("VALIDATING DATASETS")
        print("="*80)
        
        validation_results = {}
        
        for split in ["train", "val", "test"]:
            split_dir = output_dir / split
            if split_dir.exists():
                try:
                    dataset = load_from_disk(str(split_dir))
                    validation_results[f"{split}_count"] = len(dataset)
                    
                    # Validate sample
                    if len(dataset) > 0:
                        sample = dataset[0]
                        # Check shapes
                        assert np.array(sample['pixel_values']).shape == (3, self.image_size, self.image_size), \
                            f"Invalid pixel shape in {split}"
                        assert len(sample['input_ids']) == self.max_length, \
                            f"Invalid input_ids length in {split}"
                        assert len(sample['bbox']) == self.max_length, \
                            f"Invalid bbox length in {split}"
                        assert len(sample['labels']) == self.max_length, \
                            f"Invalid labels length in {split}"
                        
                        if split == "train":  # Only print for train
                            logger.info(f"\n--- Sample Validation ---")
                            logger.info(f"Keys: {list(sample.keys())}")
                            logger.info(f"pixel_values shape: {np.array(sample['pixel_values']).shape}")
                            logger.info(f"input_ids length: {len(sample['input_ids'])}")
                            logger.info(f"bbox length: {len(sample['bbox'])}")
                            logger.info(f"labels length: {len(sample['labels'])}")
                            logger.info("✓ All validations passed!")
                            
                            print(f"\n--- Sample Validation ---")
                            print(f"Keys: {list(sample.keys())}")
                            print(f"pixel_values shape: {np.array(sample['pixel_values']).shape}")
                            print(f"input_ids length: {len(sample['input_ids'])}")
                            print(f"bbox length: {len(sample['bbox'])}")
                            print(f"labels length: {len(sample['labels'])}")
                            print("✓ All validations passed!")
                    
                    logger.info(f"✓ {split} loaded: {len(dataset)} samples")
                    print(f"✓ {split} loaded: {len(dataset)} samples")
                    
                except Exception as e:
                    logger.error(f"Error loading {split}: {e}")
                    print(f"✗ Error loading {split}: {e}")
                    validation_results[f"{split}_count"] = 0
            else:
                logger.warning(f"{split} directory not found")
                print(f"⚠ {split} directory not found")
                validation_results[f"{split}_count"] = 0
        
        return validation_results
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """Generate final summary report."""
        logger.info("\n" + "="*80)
        logger.info("PHASE 6 COMPLETE - SUMMARY")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("PHASE 6 COMPLETE - SUMMARY")
        print("="*80)
        
        sizes = results.get("sizes", {})
        total_size = sum(sizes.values())
        
        logger.info(f"\n✓ Datasets created successfully")
        logger.info(f"✓ Output directory: {results.get('output_dir', 'N/A')}")
        logger.info(f"✓ Total size: {total_size:.2f} MB")
        
        print(f"\n✓ Datasets created successfully")
        print(f"✓ Output directory: {results.get('output_dir', 'N/A')}")
        print(f"✓ Total size: {total_size:.2f} MB")
        
        logger.info("\n--- Dataset Statistics ---")
        print("\n--- Dataset Statistics ---")
        for split in ["train", "val", "test"]:
            count = results.get(f"{split}_count", 0)
            size = sizes.get(split, 0.0)
            logger.info(f"{split.capitalize():6}: {count:5} samples, {size:7.2f} MB")
            print(f"{split.capitalize():6}: {count:5} samples, {size:7.2f} MB")
        
        logger.info("\n--- Ready for Kaggle Upload ---")
        print("\n--- Ready for Kaggle Upload ---")
        logger.info(f"Upload folder: {results.get('output_dir', 'N/A')}/")
        print(f"Upload folder: {results.get('output_dir', 'N/A')}/")
        for split in ["train", "val", "test"]:
            size = sizes.get(split, 0.0)
            logger.info(f"  ├── {split}/ ({size:.2f} MB)")
            print(f"  ├── {split}/ ({size:.2f} MB)")
        
        logger.info("\n" + "="*80)
        logger.info("✅ PHASE 6 SUCCESS - Ready for Phase 7!")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("✅ PHASE 6 SUCCESS - Ready for Phase 7!")
        print("="*80)