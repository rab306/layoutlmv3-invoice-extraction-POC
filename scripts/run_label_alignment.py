#!/usr/bin/env python3
"""
Script to run Phase 5: Label Alignment
Now with configuration support.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from data.label_alignment import LabelAligner


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('label_alignment.log')
        ]
    )


def main():
    """Run label alignment with command line options."""
    parser = argparse.ArgumentParser(description='Run label alignment between ground truth and OCR results')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Directory containing configuration files')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - process only 5 samples and save to /tmp/')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process (None for all)')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Override ground truth file path (uses config if not specified)')
    parser.add_argument('--ocr-results', type=str, default=None,
                       help='Override OCR results file path (uses config if not specified)')
    parser.add_argument('--output', type=str, default=None,
                       help='Override output file path (uses config if not specified)')
    parser.add_argument('--fuzzy-threshold', type=int, default=None,
                       help='Override fuzzy matching threshold (uses config if not specified)')
    parser.add_argument('--numeric-threshold', type=int, default=None,
                       help='Override numeric matching threshold (uses config if not specified)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("Running Phase 5: Label Alignment")
    print("=" * 50)
    logger.info(f"Starting label alignment with config from: {args.config_dir}")
    
    try:
        # Initialize label aligner with config
        aligner = LabelAligner(config_dir=args.config_dir)
        
        # Override thresholds if specified
        if args.fuzzy_threshold is not None:
            aligner.fuzzy_threshold = args.fuzzy_threshold
            logger.info(f"Overriding fuzzy threshold to: {args.fuzzy_threshold}")
        
        if args.numeric_threshold is not None:
            aligner.numeric_threshold = args.numeric_threshold
            logger.info(f"Overriding numeric threshold to: {args.numeric_threshold}")
        
        # Run alignment
        results = aligner.run_alignment(
            ground_truth_path=args.ground_truth,
            ocr_results_path=args.ocr_results,
            output_path=args.output,
            max_samples=args.max_samples,
            test_mode=args.test
        )
        
        logger.info(f"Label alignment completed successfully")
        print(f"\n✓ Label alignment completed successfully")
        print(f"Labeled samples: {results['labeled_samples']}")
        print(f"Failed samples: {results['failed_samples']}")
        print(f"Output: {results['output_path']}")
        
        if results['failed_samples'] > 0:
            logger.warning(f"{results['failed_samples']} samples failed")
            print(f"⚠ {results['failed_samples']} samples failed")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during label alignment: {e}", exc_info=True)
        print(f"\n✗ Error during label alignment: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())