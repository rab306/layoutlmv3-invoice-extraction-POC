#!/usr/bin/env python3
"""
Script to run model inference on invoices.
"""

import sys
import argparse
import json
import logging
from pathlib import Path

# Add src to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from models.inference import InvoiceExtractor


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('inference.log')
        ]
    )


def main():
    """Run model inference with command line options."""
    parser = argparse.ArgumentParser(description='Run invoice extraction with trained model')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Directory containing configuration files')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model (uses config if not specified)')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to invoice image')
    parser.add_argument('--ocr-results', type=str, required=True,
                       help='Path to OCR results JSON file')
    parser.add_argument('--output', type=str, default='extraction_results.json',
                       help='Output file for extraction results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("Running Invoice Extraction")
    print("=" * 50)
    
    try:
        # Initialize extractor
        extractor = InvoiceExtractor(
            model_path=args.model_path,
            config_dir=args.config_dir
        )
        
        # Load OCR results
        with open(args.ocr_results, 'r') as f:
            ocr_data = json.load(f)
        
        # Find matching OCR result for image
        image_name = Path(args.image_path).name
        ocr_result = None
        
        for item in ocr_data:
            if item['filename'] == image_name:
                ocr_result = item
                break
        
        if not ocr_result:
            raise ValueError(f"No OCR results found for image: {image_name}")
        
        # Run extraction
        result = extractor.extract_from_image(
            image_path=Path(args.image_path),
            words=ocr_result['words'],
            boxes=ocr_result['boxes']
        )
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Print summary
        print(f"\n✓ Extraction completed")
        print(f"Image: {result['filename']}")
        print(f"Entities found:")
        
        for entity_type, values in result['entities'].items():
            if values:
                print(f"  {entity_type}: {values}")
        
        print(f"\nResults saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        print(f"\n✗ Inference failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())