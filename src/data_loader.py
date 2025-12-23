"""
Data Loader for Vietnamese AMR Dataset
Handles parsing, validation, and dataset creation
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class AMRDataLoader:
    """Load and parse Vietnamese AMR data"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        
    def parse_amr_file(self, filepath: Path) -> List[Dict[str, str]]:
        """
        Parse AMR file in VLSP format
        
        Returns:
            List of dicts with 'sentence' and 'amr' keys
        """
        logger.info(f"Parsing AMR file: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by empty lines to separate examples
        examples = []
        current_sentence = None
        current_amr_lines = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_sentence and current_amr_lines:
                    amr_graph = '\n'.join(current_amr_lines)
                    examples.append({
                        'sentence': current_sentence,
                        'amr': amr_graph
                    })
                current_sentence = None
                current_amr_lines = []
                continue
            
            # Parse sentence line
            if line.startswith('#::snt '):
                current_sentence = line.replace('#::snt ', '').strip()
            # Parse AMR lines
            elif current_sentence is not None:
                current_amr_lines.append(line)
        
        # Handle last example
        if current_sentence and current_amr_lines:
            amr_graph = '\n'.join(current_amr_lines)
            examples.append({
                'sentence': current_sentence,
                'amr': amr_graph
            })
        
        logger.info(f"Loaded {len(examples)} examples from {filepath.name}")
        return examples
    
    def parse_test_file(self, filepath: Path) -> List[str]:
        """
        Parse test file containing only sentences
        
        Returns:
            List of sentences
        """
        logger.info(f"Parsing test file: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(sentences)} sentences from {filepath.name}")
        return sentences
    
    def validate_amr_syntax(self, amr: str) -> Tuple[bool, Optional[str]]:
        """
        Basic AMR syntax validation
        
        Returns:
            (is_valid, error_message)
        """
        # Check balanced parentheses
        open_count = amr.count('(')
        close_count = amr.count(')')
        
        if open_count != close_count:
            return False, f"Unbalanced parentheses: {open_count} open, {close_count} close"
        
        # Check for at least one variable/concept pattern
        if not re.search(r'\([a-z0-9]+\s*/\s*[^\s)]+', amr):
            return False, "No valid (variable / concept) pattern found"
        
        return True, None
    
    def load_training_data(
        self, 
        train_files: List[str],
        validation_split: float = 0.05,
        max_samples: Optional[int] = None
    ) -> Tuple[Dataset, Dataset]:
        """
        Load and split training data
        
        Returns:
            (train_dataset, validation_dataset)
        """
        all_examples = []
        
        for train_file in train_files:
            filepath = self.data_dir / train_file
            if not filepath.exists():
                logger.warning(f"Training file not found: {filepath}")
                continue
                
            examples = self.parse_amr_file(filepath)
            all_examples.extend(examples)
        
        if not all_examples:
            raise ValueError("No training data loaded!")
        
        # Filter valid examples
        valid_examples = []
        invalid_count = 0
        
        for ex in all_examples:
            is_valid, error = self.validate_amr_syntax(ex['amr'])
            if is_valid:
                valid_examples.append(ex)
            else:
                invalid_count += 1
                if invalid_count <= 5:  # Log first 5 errors
                    logger.warning(f"Invalid AMR: {error}")
        
        if invalid_count > 0:
            logger.warning(f"Filtered {invalid_count} invalid examples")
        
        logger.info(f"Total valid training examples: {len(valid_examples)}")
        
        # Apply max_samples if specified
        if max_samples and max_samples < len(valid_examples):
            valid_examples = valid_examples[:max_samples]
            logger.info(f"Limited to {max_samples} samples")
        
        # Split train/validation
        if validation_split > 0:
            train_examples, val_examples = train_test_split(
                valid_examples,
                test_size=validation_split,
                random_state=42
            )
            logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} validation")
        else:
            train_examples = valid_examples
            val_examples = []
            logger.info("No validation split")
        
        # Convert to HuggingFace Dataset
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_examples))
        val_dataset = Dataset.from_pandas(pd.DataFrame(val_examples)) if val_examples else None
        
        return train_dataset, val_dataset
    
    def load_test_data(self, test_file: str, ground_truth_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load test data
        
        Returns:
            DataFrame with 'sentence' and optionally 'amr' columns
        """
        filepath = self.data_dir / test_file
        
        if not filepath.exists():
            raise FileNotFoundError(f"Test file not found: {filepath}")
        
        sentences = self.parse_test_file(filepath)
        df = pd.DataFrame({'sentence': sentences})
        
        # Load ground truth if available
        if ground_truth_file:
            gt_filepath = self.data_dir / ground_truth_file
            if gt_filepath.exists():
                gt_examples = self.parse_amr_file(gt_filepath)
                gt_dict = {ex['sentence']: ex['amr'] for ex in gt_examples}
                df['amr'] = df['sentence'].map(gt_dict)
                logger.info(f"Loaded ground truth for {df['amr'].notna().sum()} samples")
            else:
                logger.warning(f"Ground truth file not found: {gt_filepath}")
        
        return df
    
    def save_predictions(
        self,
        predictions: List[Dict],
        output_dir: Path,
        prefix: str = "predictions"
    ) -> Dict[str, Path]:
        """
        Save predictions in multiple formats
        
        Returns:
            Dict of format_name -> filepath
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 1. Full CSV with metadata
        df = pd.DataFrame(predictions)
        csv_path = output_dir / f"{prefix}_full.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        saved_files['csv'] = csv_path
        logger.info(f"Saved full CSV: {csv_path}")
        
        # 2. Submission CSV (sentence + amr only)
        submit_df = df[['sentence', 'graph_amr']].rename(columns={'graph_amr': 'amr'})
        submit_path = output_dir / f"{prefix}_submission.csv"
        submit_df.to_csv(submit_path, index=False, encoding='utf-8')
        saved_files['submission'] = submit_path
        logger.info(f"Saved submission CSV: {submit_path}")
        
        # 3. VLSP format (with #::snt headers)
        vlsp_path = output_dir / f"{prefix}_vlsp.txt"
        with open(vlsp_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(f"#::snt {pred['sentence']}\n")
                f.write(f"{pred['graph_amr']}\n\n")
        saved_files['vlsp'] = vlsp_path
        logger.info(f"Saved VLSP format: {vlsp_path}")
        
        # 4. AMR only (for SMATCH evaluation)
        amr_path = output_dir / f"{prefix}_amr_only.txt"
        with open(amr_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                # Write in standardized format
                f.write(f"# ::snt {pred['sentence']}\n")
                f.write(f"{pred['graph_amr']}\n\n")
        saved_files['amr_only'] = amr_path
        logger.info(f"Saved AMR only: {amr_path}")
        
        return saved_files


def test_data_loader():
    """Test the data loader"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.config import DATA_DIR, DATA_CONFIG
    
    loader = AMRDataLoader(DATA_DIR)
    
    # Test training data
    train_ds, val_ds = loader.load_training_data(
        DATA_CONFIG['train_files'],
        validation_split=0.05
    )
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds) if val_ds else 0}")
    print(f"Sample: {train_ds[0]}")


if __name__ == "__main__":
    test_data_loader()
