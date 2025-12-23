"""
Evaluation Module for Vietnamese AMR Parser
Implements SMATCH scoring and other metrics
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SMATCHEvaluator:
    """
    SMATCH (Semantic Match) scoring for AMR evaluation
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._check_smatch_available()
    
    def _check_smatch_available(self):
        """Check if SMATCH is available"""
        try:
            import smatch
            self.use_python = True
            logger.info("Using Python smatch library")
        except ImportError:
            logger.warning("smatch library not found. Install with: pip install smatch")
            self.use_python = False
    
    def install_smatch(self):
        """Install SMATCH if not available"""
        logger.info("Installing SMATCH...")
        try:
            subprocess.run(
                ["pip", "install", "smatch"],
                check=True,
                capture_output=True
            )
            logger.info("SMATCH installed successfully")
            self._check_smatch_available()
        except Exception as e:
            logger.error(f"Failed to install SMATCH: {e}")
    
    def compute_smatch_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute SMATCH score between predictions and references
        
        Args:
            predictions: List of predicted AMR graphs
            references: List of reference AMR graphs
        
        Returns:
            Dict with precision, recall, f1
        """
        if not self.use_python:
            logger.error("SMATCH not available. Installing...")
            self.install_smatch()
            if not self.use_python:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'error': 'SMATCH not available'}
        
        try:
            import smatch
            
            # Write to temp files
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as pred_file:
                for pred in predictions:
                    pred_file.write(pred + '\n\n')
                pred_path = pred_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as ref_file:
                for ref in references:
                    ref_file.write(ref + '\n\n')
                ref_path = ref_file.name
            
            # Compute SMATCH
            logger.info("Computing SMATCH scores...")
            precision, recall, f1 = smatch.score_amr_pairs(
                pred_path,
                ref_path,
                max_iter=5,
                match_only=False
            )
            
            # Clean up
            Path(pred_path).unlink()
            Path(ref_path).unlink()
            
            result = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            
            logger.info(f"SMATCH Scores - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error computing SMATCH: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'error': str(e)}
    
    def compute_pairwise_smatch(
        self,
        pred: str,
        ref: str
    ) -> float:
        """
        Compute SMATCH for a single pair
        
        Returns:
            F1 score
        """
        try:
            import smatch
            
            # Parse AMRs
            pred_amr = smatch.AMR.parse_AMR_line(pred)
            ref_amr = smatch.AMR.parse_AMR_line(ref)
            
            # Compute score
            score = smatch.get_amr_match(pred_amr, ref_amr)[2]  # F1
            return float(score)
            
        except Exception as e:
            logger.error(f"Error in pairwise SMATCH: {e}")
            return 0.0


class AMREvaluator:
    """
    Complete evaluation suite for AMR parsing
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.smatch_evaluator = SMATCHEvaluator(
            timeout=self.config.get('smatch_timeout', 30)
        )
    
    def evaluate_predictions(
        self,
        predictions: List[Dict],
        ground_truth: Optional[List[str]] = None
    ) -> Dict:
        """
        Comprehensive evaluation of predictions
        
        Args:
            predictions: List of prediction dicts with 'graph_amr'
            ground_truth: Optional list of reference AMRs
        
        Returns:
            Dict with all metrics
        """
        metrics = {}
        
        # Basic validity metrics
        total = len(predictions)
        valid = sum(1 for p in predictions if p.get('is_valid', False))
        
        metrics['total_samples'] = total
        metrics['valid_amrs'] = valid
        metrics['validity_rate'] = valid / total if total > 0 else 0.0
        
        # Average generation time if available
        gen_times = [p.get('generation_time', 0) for p in predictions if 'generation_time' in p]
        if gen_times:
            metrics['avg_generation_time'] = sum(gen_times) / len(gen_times)
            metrics['total_generation_time'] = sum(gen_times)
        
        # Average number of concepts
        concept_counts = [p.get('num_concepts', 0) for p in predictions if 'num_concepts' in p]
        if concept_counts:
            metrics['avg_concepts_per_amr'] = sum(concept_counts) / len(concept_counts)
        
        # SMATCH evaluation if ground truth provided
        if ground_truth is not None and len(ground_truth) == len(predictions):
            logger.info("Computing SMATCH scores against ground truth...")
            
            pred_amrs = [p['graph_amr'] for p in predictions]
            
            smatch_scores = self.smatch_evaluator.compute_smatch_score(
                pred_amrs,
                ground_truth
            )
            
            metrics.update({
                'smatch_precision': smatch_scores['precision'],
                'smatch_recall': smatch_scores['recall'],
                'smatch_f1': smatch_scores['f1']
            })
        
        return metrics
    
    def evaluate_and_save(
        self,
        predictions: List[Dict],
        output_dir: Path,
        ground_truth: Optional[List[str]] = None,
        prefix: str = "evaluation"
    ) -> Dict:
        """
        Evaluate and save results
        
        Returns:
            Metrics dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute metrics
        metrics = self.evaluate_predictions(predictions, ground_truth)
        
        # Save metrics
        metrics_file = output_dir / f"{prefix}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AMR EVALUATION METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Save detailed results
        if ground_truth:
            results_df = pd.DataFrame(predictions)
            results_df['ground_truth'] = ground_truth
            
            # Add pairwise SMATCH if available
            if self.smatch_evaluator.use_python:
                logger.info("Computing pairwise SMATCH scores...")
                results_df['smatch_score'] = [
                    self.smatch_evaluator.compute_pairwise_smatch(
                        pred['graph_amr'], gt
                    )
                    for pred, gt in zip(predictions, ground_truth)
                ]
            
            results_file = output_dir / f"{prefix}_detailed.csv"
            results_df.to_csv(results_file, index=False, encoding='utf-8')
            logger.info(f"Detailed results saved to {results_file}")
        
        return metrics
    
    def compare_models(
        self,
        predictions_dict: Dict[str, List[Dict]],
        ground_truth: List[str],
        output_dir: Path
    ):
        """
        Compare multiple models
        
        Args:
            predictions_dict: Dict of model_name -> predictions
            ground_truth: Reference AMRs
            output_dir: Where to save comparison
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison = {}
        
        for model_name, predictions in predictions_dict.items():
            logger.info(f"Evaluating {model_name}...")
            metrics = self.evaluate_predictions(predictions, ground_truth)
            comparison[model_name] = metrics
        
        # Save comparison
        comparison_df = pd.DataFrame(comparison).T
        comparison_file = output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file)
        
        logger.info(f"Model comparison saved to {comparison_file}")
        logger.info("\nComparison Summary:")
        logger.info(comparison_df.to_string())
        
        return comparison


def test_evaluator():
    """Test the evaluator"""
    # Create sample data
    predictions = [
        {
            'sentence': 'Test sentence 1',
            'graph_amr': '(t / test :ARG0(s / sentence))',
            'is_valid': True,
            'num_concepts': 2
        },
        {
            'sentence': 'Test sentence 2',
            'graph_amr': '(a / another :mod(t / test))',
            'is_valid': True,
            'num_concepts': 2
        }
    ]
    
    ground_truth = [
        '(t / test :ARG0(s / sentence))',
        '(a / another :mod(t / test))'
    ]
    
    evaluator = AMREvaluator()
    metrics = evaluator.evaluate_predictions(predictions, ground_truth)
    
    print("Test Evaluation:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    test_evaluator()
