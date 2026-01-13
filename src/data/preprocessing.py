"""
Phase 3: Data Preprocessing
Production implementation of the notebook logic.
Now with configuration integration.
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from utils.config import get_config

logger = logging.getLogger(__name__)

def count_true_nulls(series: pd.Series) -> int:
    """
    Count nulls including empty lists and empty strings.
    
    Args:
        series: Pandas Series to check
        
    Returns:
        Count of true null values
    """
    count = 0
    for val in series:
        if isinstance(val, list):
            if len(val) == 0:
                count += 1
            continue
        
        try:
            if pd.isna(val):
                count += 1
                continue
        except (ValueError, TypeError):
            pass
        
        if isinstance(val, str) and val.strip() == '':
            count += 1
    
    return count


class DataPreprocessor:
    """Handles data loading, cleaning, and validation of invoice data."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the data preprocessor with configuration.
        
        Args:
            config_dir: Directory containing config files (uses default if None)
        """
        self.config = get_config(config_dir) if config_dir else get_config()
        
        # Get configuration values with defaults
        self.raw_data_dir = Path(
            self.config.get_data_path('paths.raw.base_dir', 'data/raw')
        )
        
        # Set default target fields from config or fallback
        self.target_fields = self.config.get_data_path(
            'preprocessing.target_fields',
            [
                'filename',
                'invoice_number',
                'invoice_date',
                'buyer_address',
                'products',
                'seller_address',
                'payment_total',
                'payment_sub_total'
            ]
        )
        
        # Validation fields from config
        self.tier1_fields = self.config.get_data_path(
            'preprocessing.validation.tier1_fields',
            ['invoice_number', 'invoice_date', 'buyer_address', 'products']
        )
        
        self.tier2_fields = self.config.get_data_path(
            'preprocessing.validation.tier2_fields',
            ['seller_address', 'payment_total', 'payment_sub_total']
        )
        
        # Expected null ranges from config
        self.tier2_null_range = self.config.get_data_path(
            'preprocessing.validation.expected_null_ranges',
            {'tier2_min': 800, 'tier2_max': 1200}
        )
        
        # Set up paths
        self.json_dir = self.raw_data_dir / "json"
        self.image_dir = self.raw_data_dir / "images"
        
        logger.info(f"DataPreprocessor initialized with config from {config_dir or 'default'}")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"JSON directory: {self.json_dir}")
        logger.info(f"Image directory: {self.image_dir}")
    
    def _validate_paths(self) -> None:
        """Validate that required directories exist."""
        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON directory not found: {self.json_dir}")
        
        if not self.image_dir.exists():
            logger.warning(f"Image directory not found: {self.image_dir}")
    
    def load_json_files(self) -> pd.DataFrame:
        """
        Load all JSON files from the raw data directory.
        
        Returns:
            DataFrame with flattened JSON data
        """
        logger.info(f"Loading JSON files from {self.json_dir}")
        
        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON directory not found: {self.json_dir}")
        
        json_files = sorted([f for f in os.listdir(self.json_dir) if f.endswith(".json")])
        logger.info(f"Found {len(json_files)} JSON files")
        
        json_data = []
        for f in json_files:
            file_path = self.json_dir / f
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                data["filename"] = f
                json_data.append(data)
        
        # Flatten to DataFrame
        df = pd.json_normalize(json_data, sep="_")
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        return df
    
    def select_target_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only the target fields for processing.
        
        Args:
            df: Raw DataFrame with all fields
            
        Returns:
            DataFrame with only target fields
        """
        logger.info(f"Selecting {len(self.target_fields)} target fields...")
        
        # Verify all fields exist
        missing_fields = [f for f in self.target_fields if f not in df.columns]
        if missing_fields:
            logger.warning(f"Missing fields: {missing_fields}")
        else:
            logger.info("All target fields present")
        
        cleaned_df = df[self.target_fields].copy()
        return cleaned_df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the quality of the cleaned data.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data quality...")
        
        total_records = len(df)
        validation_results = {
            "total_records": total_records,
            "field_coverage": {},
            "tier1_validation": {},
            "tier2_validation": {},
            "issues": []
        }
        
        # Calculate coverage for each field
        logger.info("Calculating field coverage...")
        for field in self.target_fields[1:]:  # Skip filename
            null_count = count_true_nulls(df[field])
            coverage = (total_records - null_count) / total_records * 100
            validation_results["field_coverage"][field] = {
                "null_count": null_count,
                "coverage_percent": coverage
            }
        
        # Validate Tier 1 fields (should have 0 nulls)
        tier2_min = self.tier2_null_range.get('tier2_min', 800)
        tier2_max = self.tier2_null_range.get('tier2_max', 1200)
        
        for field in self.tier1_fields:
            null_count = count_true_nulls(df[field])
            validation_results["tier1_validation"][field] = null_count
            if null_count > 0:
                validation_results["issues"].append(f"{field} has {null_count} nulls (expected 0)")
        
        # Validate Tier 2 fields (should have ~1000 nulls)
        for field in self.tier2_fields:
            null_count = count_true_nulls(df[field])
            validation_results["tier2_validation"][field] = null_count
            if not (tier2_min <= null_count <= tier2_max):
                validation_results["issues"].append(
                    f"{field} has {null_count} nulls (expected {tier2_min}-{tier2_max})"
                )
        
        return validation_results
    
    def validate_products_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the structure of products field.
        
        Args:
            df: DataFrame with products field
            
        Returns:
            Dictionary with products validation results
        """
        logger.info("Validating products structure...")
        
        results = {
            "is_all_lists": False,
            "products_distribution": {},
            "empty_products_count": 0,
            "schema_valid": True,
            "schema_issues": []
        }
        
        # Check products is always a list
        is_list = df['products'].apply(lambda x: isinstance(x, list))
        results["is_all_lists"] = is_list.all()
        
        # Count products per invoice
        df['_num_products'] = df['products'].apply(len)
        results["products_distribution"] = {
            "min": df['_num_products'].min(),
            "max": df['_num_products'].max(),
            "mean": float(df['_num_products'].mean()),
            "median": float(df['_num_products'].median())
        }
        
        # Check for empty product lists
        empty_products = (df['_num_products'] == 0).sum()
        results["empty_products_count"] = empty_products
        
        # Validate product structure (sample up to 100 invoices)
        logger.info("Validating product schema...")
        required_keys = ['description', 'quantity', 'unit_price', 'total_price']
        invalid_products = 0
        
        sample_size = min(100, len(df))
        for products_list in df['products'].sample(sample_size):
            for product in products_list:
                if not isinstance(product, dict):
                    invalid_products += 1
                    continue
                missing_keys = [k for k in required_keys if k not in product]
                if missing_keys:
                    invalid_products += 1
                    results["schema_issues"].append(f"Missing keys: {missing_keys}")
        
        if invalid_products > 0:
            results["schema_valid"] = False
            results["schema_issues_count"] = invalid_products
        
        # Clean up temporary column
        df.drop(columns=['_num_products'], inplace=True)
        
        return results
    
    def check_duplicates_and_alignment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for duplicates and file alignment with images.
        
        Args:
            df: DataFrame with filename field
            
        Returns:
            Dictionary with check results
        """
        logger.info("Checking duplicates and file alignment...")
        
        results = {
            "duplicate_count": 0,
            "image_alignment": {},
            "missing_images": []
        }
        
        # Check for duplicate filenames
        duplicates = df['filename'].duplicated().sum()
        results["duplicate_count"] = duplicates
        
        # Verify image files exist (check first 10)
        check_count = 10  # Could be moved to config
        logger.info(f"Checking file alignment (first {check_count})...")
        for filename in df['filename'].head(check_count):
            img_filename = filename.replace('.json', '.png')
            img_file_path = self.image_dir / img_filename
            exists = os.path.exists(img_file_path)
            results["image_alignment"][img_filename] = exists
            if not exists:
                results["missing_images"].append(img_filename)
        
        return results
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """
        Save cleaned data to JSON file.
        
        Args:
            df: Cleaned DataFrame
            output_path: Path to save the cleaned data (uses config if None)
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = self.config.get_data_path(
                'paths.processed.cleaned_data',
                'data/processed/cleaned_data.json'
            )
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving cleaned data to {output_path_obj}")
        
        # Convert DataFrame to list of dicts
        cleaned_data = df.to_dict('records')
        
        # Save to file
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        # Get file size
        file_size_mb = output_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(cleaned_data)} records, file size: {file_size_mb:.2f} MB")
        
        return str(output_path_obj)
    
    def generate_summary_report(self, df: pd.DataFrame, validation_results: Dict[str, Any]) -> str:
        """
        Generate a summary report of the preprocessing.
        
        Args:
            df: Cleaned DataFrame
            validation_results: Results from validation methods
            
        Returns:
            Formatted summary report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA PREPROCESSING SUMMARY REPORT")
        report_lines.append("=" * 80)
        
        report_lines.append(f"\nTotal invoices processed: {len(df)}")
        report_lines.append(f"Fields per invoice: {len(self.target_fields)} (7 target + 1 metadata)")
        
        report_lines.append("\n--- Fields Included ---")
        for i, field in enumerate(self.target_fields[1:], 1):  # Skip filename
            tier = "Tier 1" if field in self.tier1_fields else "Tier 2"
            coverage = validation_results["field_coverage"][field]["coverage_percent"]
            report_lines.append(f"{i}. {field:<25} ({tier:<6}) Coverage: {coverage:>5.1f}%")
        
        report_lines.append("\n--- Validation Summary ---")
        for field in self.tier1_fields:
            nulls = validation_results["tier1_validation"][field]
            status = "✓" if nulls == 0 else "✗"
            report_lines.append(f"{status} {field:<20}: {nulls} nulls (expected 0)")
        
        tier2_min = self.tier2_null_range.get('tier2_min', 800)
        tier2_max = self.tier2_null_range.get('tier2_max', 1200)
        for field in self.tier2_fields:
            nulls = validation_results["tier2_validation"][field]
            status = "✓" if tier2_min <= nulls <= tier2_max else "⚠"
            expected_range = f"{tier2_min}-{tier2_max}"
            report_lines.append(f"{status} {field:<20}: {nulls} nulls (expected {expected_range})")
        
        if validation_results["issues"]:
            report_lines.append("\n--- Issues Found ---")
            for issue in validation_results["issues"]:
                report_lines.append(f"• {issue}")
        
        return "\n".join(report_lines)
    
    def run_pipeline(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            output_path: Path to save the cleaned data (uses config if None)
            
        Returns:
            Dictionary with processing results
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: DATA PREPROCESSING")
        logger.info("=" * 80)
        
        results = {}
        
        try:
            # Validate paths first
            self._validate_paths()
            
            # Step 1: Load JSON files
            logger.info("[STEP 1] Loading JSON files...")
            raw_df = self.load_json_files()
            
            # Step 2: Select target fields
            logger.info("[STEP 2] Selecting target fields...")
            cleaned_df = self.select_target_fields(raw_df)
            
            # Step 3: Validate data quality
            logger.info("[STEP 3] Validating data quality...")
            validation_results = self.validate_data_quality(cleaned_df)
            results["validation"] = validation_results
            
            # Step 4: Validate products structure
            logger.info("[STEP 4] Validating products structure...")
            products_validation = self.validate_products_structure(cleaned_df)
            results["products_validation"] = products_validation
            
            # Step 5: Check duplicates and alignment
            logger.info("[STEP 5] Checking duplicates and file alignment...")
            alignment_results = self.check_duplicates_and_alignment(cleaned_df)
            results["alignment"] = alignment_results
            
            # Step 6: Save cleaned data
            logger.info("[STEP 6] Saving cleaned data...")
            saved_path = self.save_cleaned_data(cleaned_df, output_path)
            results["output_path"] = saved_path
            
            # Step 7: Generate and print summary
            logger.info("[STEP 7] Generating summary report...")
            summary = self.generate_summary_report(cleaned_df, validation_results)
            
            # Print to console for visibility (keeping some prints for user feedback)
            print(summary)
            
            # Also log the summary
            for line in summary.split('\n'):
                if line.strip():
                    logger.info(line)
            
            logger.info("=" * 80)
            logger.info("PHASE 3 COMPLETE")
            logger.info("=" * 80)
            
            results["success"] = True
            results["records_processed"] = len(cleaned_df)
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            results["success"] = False
            results["error"] = str(e)
            raise
        
        return results


def main():
    """Main function for standalone execution."""
    import sys
    import logging as std_logging
    
    # Set up basic logging
    std_logging.basicConfig(
        level=std_logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get command line arguments
    config_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run preprocessing
    preprocessor = DataPreprocessor(config_dir)
    results = preprocessor.run_pipeline(output_path)
    
    if results.get("success"):
        print(f"\n✓ Preprocessing complete. Output saved to: {results['output_path']}")
        sys.exit(0)
    else:
        print(f"\n✗ Preprocessing failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()