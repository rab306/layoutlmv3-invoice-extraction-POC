"""
Phase 4: OCR Processing with Tesseract
Production implementation with configuration integration.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pytesseract
from PIL import Image
from tqdm import tqdm

from utils.config import get_config

logger = logging.getLogger(__name__)


class TesseractOCR:
    """OCR processing using Tesseract engine with configuration support."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize OCR processor with configuration.
        
        Args:
            config_dir: Directory containing config files (uses default if None)
        """
        self.config = get_config(config_dir) if config_dir else get_config()
        
        # Get configuration values with defaults
        images_dir = self.config.get_data_path('paths.raw.images', 'data/raw/images')
        self.images_dir = Path(images_dir)
        
        # OCR configuration
        self.tesseract_config = self.config.get_data_path(
            'ocr.tesseract_config',
            '--psm 6 --oem 3'
        )
        
        self.check_alignment_count = self.config.get_data_path(
            'ocr.check_alignment_count',
            10
        )
        
        self.checkpoint_interval = self.config.get_data_path(
            'ocr.checkpoint_interval',
            100
        )
        
        logger.info(f"TesseractOCR initialized with config from {config_dir or 'default'}")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Tesseract config: {self.tesseract_config}")
        
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that images directory exists."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Check for images
        image_extensions = ('.png', '.jpg', '.jpeg')
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(f"*{ext}")))
        
        if not image_files:
            raise ValueError(f"No images found in {self.images_dir}")
        
        logger.info(f"Found {len(image_files)} images in {self.images_dir}")
    
    @staticmethod
    def extract_words_and_boxes(image_path: Path, config: str = '--psm 6 --oem 3') -> Tuple[List[str], List[List[int]]]:
        """
        Extract words and bounding boxes from an image using Tesseract.
        
        Args:
            image_path: Path to the image file
            config: Tesseract configuration string
            
        Returns:
            Tuple of (words, boxes) where boxes are [x0, y0, x1, y1]
        """
        # Load image
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Run Tesseract OCR with bounding box data
        ocr_data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
        
        words = []
        boxes = []
        
        # Extract non-empty words with their bounding boxes
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            word = ocr_data['text'][i].strip()
            
            # Skip empty strings and low confidence results
            if word and int(ocr_data['conf'][i]) > 0:
                # Extract bounding box coordinates
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Convert to [x0, y0, x1, y1] format
                box = [x, y, x + w, x + h]
                
                words.append(word)
                boxes.append(box)
        
        return words, boxes
    
    def test_ocr_on_sample(self, sample_count: int = 1) -> None:
        """
        Test OCR on sample images.
        
        Args:
            sample_count: Number of sample images to test
        """
        logger.info(f"Testing OCR on {sample_count} sample image(s)...")
        
        image_files = sorted([
            f for f in self.images_dir.iterdir() 
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])
        
        for i in range(min(sample_count, len(image_files))):
            img_file = image_files[i]
            
            logger.info(f"Testing on: {img_file.name}")
            
            try:
                test_words, test_boxes = self.extract_words_and_boxes(img_file, self.tesseract_config)
                logger.info(f"OCR successful!")
                logger.info(f"  Words extracted: {len(test_words)}")
                logger.info(f"  Sample words: {test_words[:10]}")
                logger.info(f"  Sample boxes: {test_boxes[:3]}")
                
                # Also print for user visibility
                print(f"✓ OCR test successful on {img_file.name}")
                print(f"  Words extracted: {len(test_words)}")
                if test_words:
                    print(f"  First 5 words: {test_words[:5]}")
                
            except Exception as e:
                logger.error(f"ERROR during OCR test: {e}")
                print(f"✗ ERROR during OCR test: {e}")
                raise
    
    def process_images(
        self, 
        output_path: Optional[str] = None,
        max_images: Optional[int] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Process images with OCR.
        
        Args:
            output_path: Path to save OCR results (uses config if None)
            max_images: Maximum number of images to process (None for all)
            resume: Whether to resume from existing results
            
        Returns:
            Dictionary with processing results
        """
        # Get output path from config if not specified
        if output_path is None:
            output_path = self.config.get_data_path(
                'paths.processed.ocr_results',
                'data/processed/ocr_results.json'
            )
        
        output_path_obj = Path(output_path)
        
        print("\n" + "=" * 80)
        print("PHASE 4: OCR PROCESSING")
        print("=" * 80)
        
        # STEP 1: Setup
        logger.info("[STEP 1] Setting up OCR configuration...")
        print(f"\n[STEP 1] Setting up OCR configuration...")
        print(f"✓ Images directory: {self.images_dir}")
        print(f"✓ Output file: {output_path_obj}")
        print(f"✓ Tesseract config: {self.tesseract_config}")
        
        # Get all image files
        image_files = sorted([
            f.name for f in self.images_dir.iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to process")
        print(f"✓ Found {len(image_files)} images to process")
        
        # STEP 2: Test on sample
        logger.info("[STEP 2] Testing OCR on sample image...")
        print("\n[STEP 2] Testing OCR on sample image...")
        self.test_ocr_on_sample(sample_count=1)
        
        # STEP 3: Load existing results or start fresh
        logger.info("[STEP 3] Setting up processing...")
        print("\n[STEP 3] Setting up processing...")
        ocr_results = []
        
        if resume and output_path_obj.exists():
            logger.info(f"Loading existing results from {output_path_obj}")
            print(f"Loading existing results from {output_path_obj}")
            try:
                with open(output_path_obj, 'r', encoding='utf-8') as f:
                    ocr_results = json.load(f)
                logger.info(f"Loaded {len(ocr_results)} existing results")
                print(f"✓ Loaded {len(ocr_results)} existing results")
            except json.JSONDecodeError as e:
                logger.warning(f"Existing file is corrupted, starting fresh: {e}")
                print(f"⚠ Existing file is corrupted, starting fresh")
                ocr_results = []
        else:
            logger.info("Starting fresh processing")
            print("Starting fresh processing")
            # Ensure output directory exists
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Get list of already processed files
        processed_files = set([result['filename'] for result in ocr_results])
        files_to_process = [f for f in image_files if f not in processed_files]
        
        logger.info(f"Total images: {len(image_files)}")
        logger.info(f"Already processed: {len(processed_files)}")
        logger.info(f"Remaining to process: {len(files_to_process)}")
        
        print(f"\nTotal images: {len(image_files)}")
        print(f"Already processed: {len(processed_files)}")
        print(f"Remaining to process: {len(files_to_process)}")
        
        if not files_to_process:
            logger.info("All images already processed!")
            print("✓ All images already processed!")
            return self._generate_summary(ocr_results, str(output_path_obj), 0)
        
        # STEP 4: Process images
        logger.info(f"[STEP 4] Processing {len(files_to_process)} images...")
        print(f"\n[STEP 4] Processing {len(files_to_process)} images...")
        print("Progress will be saved periodically\n")
        
        start_time = time.time()
        errors = []
        
        for idx, img_file in enumerate(tqdm(files_to_process, desc="Processing images")):
            try:
                img_path = self.images_dir / img_file
                words, boxes = self.extract_words_and_boxes(img_path, self.tesseract_config)
                
                result = {
                    'filename': img_file,
                    'words': words,
                    'boxes': boxes,
                    'num_words': len(words)
                }
                
                ocr_results.append(result)
                
                # Save checkpoint at configured interval
                checkpoint_int = self.checkpoint_interval
                if (idx + 1) % checkpoint_int == 0:
                    self._save_results(ocr_results, output_path_obj)
                    checkpoint_msg = f"Checkpoint saved at {len(ocr_results)} images"
                    logger.info(checkpoint_msg)
                    print(f"\n✓ {checkpoint_msg}")
                    
            except Exception as e:
                error_msg = f"Error processing {img_file}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                continue
        
        elapsed_time = time.time() - start_time
        
        # STEP 5: Save final results
        logger.info("[STEP 5] Saving final OCR results...")
        print("\n[STEP 5] Saving final OCR results...")
        self._save_results(ocr_results, output_path_obj)
        
        # STEP 6: Generate summary
        return self._generate_summary(ocr_results, str(output_path_obj), elapsed_time, errors)
    
    def _save_results(self, results: List[Dict], output_path: Path) -> None:
        """Save OCR results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def _generate_summary(
        self, 
        results: List[Dict], 
        output_file: str,
        elapsed_time: float,
        errors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate processing summary."""
        if errors is None:
            errors = []
        
        logger.info("[STEP 6] Generating OCR summary...")
        print("\n[STEP 6] Generating OCR summary...")
        
        # Basic statistics
        word_counts = [result['num_words'] for result in results]
        
        print(f"\n--- OCR Statistics ---")
        print(f"Total images processed: {len(results)}")
        print(f"Total words extracted: {sum(word_counts)}")
        if results:
            print(f"Average words per image: {sum(word_counts) / len(word_counts):.1f}")
            print(f"Min words per image: {min(word_counts)}")
            print(f"Max words per image: {max(word_counts)}")
        
        # Check for potential issues
        if results:
            low_word_images = [r for r in results if r['num_words'] < 10]
            if low_word_images:
                logger.warning(f"{len(low_word_images)} images have < 10 words")
                print(f"\n⚠ WARNING: {len(low_word_images)} images have < 10 words")
        
        # Show errors if any
        if errors:
            logger.warning(f"Errors encountered: {len(errors)}")
            print(f"\n⚠ Errors encountered: {len(errors)}")
        
        # Sample output
        if results:
            print("\n--- Sample OCR Result ---")
            sample_result = results[0]
            print(f"Filename: {sample_result['filename']}")
            print(f"Number of words: {sample_result['num_words']}")
            print(f"First 5 words: {sample_result['words'][:5]}")
        
        print("\n" + "=" * 80)
        print("PHASE 4 COMPLETE - SUMMARY")
        print("=" * 80)
        
        print(f"\n✓ Successfully processed {len(results)} images")
        if elapsed_time > 0:
            print(f"✓ Processing time: {elapsed_time / 60:.1f} minutes")
        print(f"✓ Output file: {output_file}")
        
        if errors:
            print(f"⚠ Errors: {len(errors)} images failed")
        
        print("\n" + "=" * 80)
        
        return {
            "images_processed": len(results),
            "total_words": sum(word_counts) if results else 0,
            "output_file": output_file,
            "processing_time_seconds": elapsed_time,
            "errors": errors
        }