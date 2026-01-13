#!/usr/bin/env python3
"""
Script to run Phase 4: OCR Processing
Now with configuration support.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from data.ocr import TesseractOCR


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ocr_processing.log')
        ]
    )


def main():
    """Run OCR processing with command line options."""
    parser = argparse.ArgumentParser(description='Run OCR processing on invoice images')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Directory containing configuration files')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - process only 2 images and dont save results')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from existing results')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Override images directory (uses config if not specified)')
    parser.add_argument('--output', type=str, default=None,
                       help='Override output file path (uses config if not specified)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("Running Phase 4: OCR Processing")
    print("=" * 50)
    logger.info(f"Starting OCR processing with config from: {args.config_dir}")
    
    try:
        # Initialize OCR processor with config
        ocr = TesseractOCR(config_dir=args.config_dir)
        
        # Override images directory if specified
        if args.images_dir:
            ocr.images_dir = Path(args.images_dir)
            logger.info(f"Overriding images directory to: {args.images_dir}")
        
        if args.test:
            logger.info("Running in test mode")
            print("TEST MODE: Processing 2 images, results will not be saved")
            
            # Test OCR on 2 samples
            ocr.test_ocr_on_sample(sample_count=2)
            
            # Process but don't save to actual output directory
            test_output = "/tmp/test_ocr_results.json"
            results = ocr.process_images(
                output_path=test_output,
                max_images=2,
                resume=False
            )
            
            logger.info(f"Test completed successfully")
            print(f"\n✓ Test completed successfully")
            print(f"Test output would be saved to: {test_output}")
            print("(Not saved to actual data directory)")
            
        else:
            # Full processing
            results = ocr.process_images(
                output_path=args.output,
                max_images=args.max_images,
                resume=not args.no_resume
            )
            
            logger.info(f"OCR processing completed successfully")
            print(f"\n✓ OCR processing completed")
            print(f"Images processed: {results['images_processed']}")
            print(f"Total words: {results['total_words']}")
            print(f"Output: {results['output_file']}")
            
            if results['errors']:
                logger.warning(f"{len(results['errors'])} errors occurred")
                print(f"⚠ {len(results['errors'])} errors occurred")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during OCR processing: {e}", exc_info=True)
        print(f"\n✗ Error during OCR processing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())