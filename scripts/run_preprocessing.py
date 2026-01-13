#!/usr/bin/env python3
"""
Script to run Phase 3: Data Preprocessing
Now with configuration support.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from data.preprocessing import DataPreprocessor


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('preprocessing.log')
        ]
    )


def main():
    """Run Phase 3 preprocessing with command line options."""
    parser = argparse.ArgumentParser(description='Run data preprocessing for invoice data')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Directory containing configuration files')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Override output file path (uses config if not specified)')
    parser.add_argument('--raw-data-dir', type=str, default=None,
                       help='Override raw data directory (uses config if not specified)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("Running Phase 3: Data Preprocessing")
    print("=" * 50)
    logger.info(f"Starting data preprocessing with config from: {args.config_dir}")
    
    try:
        # Initialize preprocessor with config
        preprocessor = DataPreprocessor(config_dir=args.config_dir)
        
        # Override raw data directory if specified
        if args.raw_data_dir:
            preprocessor.raw_data_dir = Path(args.raw_data_dir)
            preprocessor.json_dir = preprocessor.raw_data_dir / "json"
            preprocessor.image_dir = preprocessor.raw_data_dir / "images"
            logger.info(f"Overriding raw data directory to: {args.raw_data_dir}")
        
        # Run preprocessing pipeline
        results = preprocessor.run_pipeline(output_path=args.output_path)
        
        if results.get("success"):
            logger.info(f"Preprocessing completed successfully")
            print(f"\n✓ Successfully processed {results['records_processed']} records")
            print(f"Output: {results['output_path']}")
            
            # Log validation summary if available
            if "validation" in results:
                val = results["validation"]
                logger.info(f"Validation: {val.get('total_records', 0)} records processed")
            
            return 0
        else:
            error_msg = results.get('error', 'Unknown error')
            logger.error(f"Preprocessing failed: {error_msg}")
            print(f"\n✗ Processing failed: {error_msg}")
            return 1
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        print(f"\n✗ Fatal error during preprocessing: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())