#!/usr/bin/env python3
"""
Script to run model training.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from models.trainer import InvoiceModelTrainer


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def main():
    """Run model training with command line options."""
    parser = argparse.ArgumentParser(description='Train LayoutLMv3 model for invoice extraction')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Directory containing configuration files')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to dataset directory (uses config if not specified)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for model (uses config if not specified)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("Training LayoutLMv3 Model for Invoice Extraction")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = InvoiceModelTrainer(config_dir=args.config_dir)
        
        # Get dataset path from config if not specified
        if args.dataset_path is None:
            from utils.config import get_config
            config = get_config(args.config_dir)
            args.dataset_path = config.get_data_path(
                'paths.datasets.layoutlm_dataset',
                'layoutlm_dataset'
            )
        
        logger.info(f"Dataset path: {args.dataset_path}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Run training
        results = trainer.train(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir
        )
        
        # Print results
        print(f"\n✓ Training completed successfully")
        print(f"Model saved to: {results['model_path']}")
        print(f"Validation F1: {results['validation']['eval_f1']:.4f}")
        print(f"Test F1: {results['test']['eval_f1']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n✗ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())