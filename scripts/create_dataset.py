#!/usr/bin/env python3
"""
Script to run Phase 6: Dataset Creation
Memory-efficient LayoutLMv3 dataset creation with config support.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from data.dataset import LayoutLMDatasetCreator


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_creation.log')
        ]
    )


def main():
    """Run dataset creation with command line options."""
    parser = argparse.ArgumentParser(description='Create LayoutLMv3 datasets from labeled data')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Directory containing configuration files')
    parser.add_argument('--labeled-data', type=str, default=None,
                       help='Override labeled dataset file path (uses config if not specified)')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Override images directory (uses config if not specified)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory (uses config if not specified)')
    parser.add_argument('--processor', type=str, default=None,
                       help='Override processor name (uses config if not specified)')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Override maximum sequence length (uses config if not specified)')
    parser.add_argument('--train-ratio', type=float, default=None,
                       help='Override training set ratio (uses config if not specified)')
    parser.add_argument('--val-ratio', type=float, default=None,
                       help='Override validation set ratio (uses config if not specified)')
    parser.add_argument('--test-ratio', type=float, default=None,
                       help='Override test set ratio (uses config if not specified)')
    parser.add_argument('--train-chunk', type=int, default=None,
                       help='Override chunk size for training set (uses config if not specified)')
    parser.add_argument('--val-chunk', type=int, default=None,
                       help='Override chunk size for validation set (uses config if not specified)')
    parser.add_argument('--test-chunk', type=int, default=None,
                       help='Override chunk size for test set (uses config if not specified)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode - process only 30 samples for testing')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("Running Phase 6: Dataset Creation")
    print("=" * 50)
    logger.info(f"Starting dataset creation with config from: {args.config_dir}")
    
    try:
        # Initialize dataset creator with config
        creator = LayoutLMDatasetCreator(config_dir=args.config_dir)
        
        # Override parameters if specified
        if args.images_dir:
            creator.images_dir = Path(args.images_dir)
            logger.info(f"Overriding images directory to: {args.images_dir}")
        
        if args.processor:
            creator.processor_name = args.processor
            logger.info(f"Overriding processor to: {args.processor}")
        
        if args.max_length:
            creator.max_length = args.max_length
            logger.info(f"Overriding max length to: {args.max_length}")
        
        if args.train_ratio is not None:
            creator.train_ratio = args.train_ratio
            logger.info(f"Overriding train ratio to: {args.train_ratio}")
        
        if args.val_ratio is not None:
            creator.val_ratio = args.val_ratio
            logger.info(f"Overriding val ratio to: {args.val_ratio}")
        
        if args.test_ratio is not None:
            creator.test_ratio = args.test_ratio
            logger.info(f"Overriding test ratio to: {args.test_ratio}")
        
        if args.train_chunk:
            creator.train_chunk_size = args.train_chunk
            logger.info(f"Overriding train chunk size to: {args.train_chunk}")
        
        if args.val_chunk:
            creator.val_chunk_size = args.val_chunk
            logger.info(f"Overriding val chunk size to: {args.val_chunk}")
        
        if args.test_chunk:
            creator.test_chunk_size = args.test_chunk
            logger.info(f"Overriding test chunk size to: {args.test_chunk}")
        
        # For test mode, we need to limit data before processing
        if args.test_mode:
            logger.info("Running in test mode")
            print("TEST MODE: Processing limited samples")
            print("Will create temporary test datasets in /tmp/")
            
            # Load and limit data
            labeled_data_path = args.labeled_data or creator.config.get_data_path(
                'paths.processed.labeled_dataset',
                'data/processed/labeled_dataset.json'
            )
            
            with open(labeled_data_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
            
            # Use only 30 samples for testing
            test_data = full_data[:30]
            
            # Create temporary files
            test_output_dir = "/tmp/layoutlm_dataset_test"
            test_labeled_data = "/tmp/test_labeled_data.json"
            
            # Save limited data
            with open(test_labeled_data, 'w') as f:
                json.dump(test_data, f)
            
            # Run with test data
            results = creator.create_datasets(
                labeled_data_path=test_labeled_data,
                output_dir=test_output_dir
            )
            
            logger.info(f"Test completed successfully")
            print(f"\n✓ Test completed successfully")
            print(f"Test output in: {test_output_dir}")
            print("(Not saved to actual output directory)")
            
        else:
            # Full processing
            results = creator.create_datasets(
                labeled_data_path=args.labeled_data,
                output_dir=args.output_dir
            )
            
            logger.info(f"Dataset creation completed successfully")
            print(f"\n✓ Dataset creation completed successfully")
            print(f"Output directory: {results['output_dir']}")
            print(f"Total size: {sum(results['sizes'].values()):.2f} MB")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during dataset creation: {e}", exc_info=True)
        print(f"\n✗ Error during dataset creation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())