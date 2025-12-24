"""
MTUP (Multi-Task Unified Prompt) Preprocessor for Vietnamese AMR
Generates training data with two-stage format:
1. AMR without variables
2. AMR with variables (in the same prompt)
"""

import re
import logging
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
from prompt_templates import get_template, format_mtup_example

logger = logging.getLogger(__name__)


class MTUPAMRPreprocessor:
    """
    MTUP Preprocessor for Vietnamese AMR

    Pipeline:
    1. Input: Raw AMR with variables (from dataset)
    2. Generate: AMR without variables (Task 1 output)
    3. Keep: AMR with variables (Task 2 output)
    4. Format: Both in unified prompt template
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.template_name = self.config.get('template_name', 'recommended')
        self.stats = {
            'processed': 0,
            'errors': 0,
            'avg_no_var_length': 0,
            'avg_with_var_length': 0,
        }

    def extract_variable_to_concept_map(self, amr_string: str) -> Dict[str, str]:
        """
        Build mapping: variable → concept
        Handles both (var / concept) and (var/concept) formats
        """
        var_to_concept = {}

        # Pattern matches: (var / concept) or (var/concept)
        pattern = r'\(([a-z0-9]+)\s*/\s*([^\s\)]+)'
        matches = re.findall(pattern, amr_string)

        for var, concept in matches:
            concept = concept.strip()
            # Remove trailing special characters
            concept = re.sub(r'[,;:\s]+$', '', concept)
            var_to_concept[var] = concept

        return var_to_concept

    def remove_variables(self, amr_string: str) -> str:
        """
        Remove variables from AMR: (var / concept) → (concept)
        This creates the "Task 1 output" - AMR without variables

        Important: Variables can include Vietnamese characters (đ, ô, etc.)
        """
        # Remove variable declarations: (var / concept) → (concept)
        # Match any non-whitespace, non-special character for variable names
        # Includes Vietnamese characters: đ, ô, ê, â, ă, ư, ơ, etc.
        cleaned = re.sub(r'\([^\s/:()]+\s*/', r'(', amr_string)
        return cleaned

    def linearize(self, amr_string: str, keep_spaces: bool = False) -> str:
        """
        Convert multi-line AMR to single line
        Preserves structure while making it compact

        Args:
            keep_spaces: If True, keep spaces after colons for readability
        """
        # Join lines
        linear = ' '.join(amr_string.split())

        # Normalize multiple spaces
        linear = re.sub(r'\s+', ' ', linear)

        # Clean up spacing around parentheses
        linear = re.sub(r'\s*\(\s*', '(', linear)
        linear = re.sub(r'\s*\)\s*', ')', linear)

        if not keep_spaces:
            # Remove spaces around colons for compact format
            linear = re.sub(r'\s*:\s*', ':', linear)
        else:
            # Keep one space after colon for readability
            linear = re.sub(r'\s*:\s*', ': ', linear)

        return linear.strip()

    def format_graph(self, amr_string: str, indent: int = 4) -> str:
        """
        Format linearized AMR into indented graph structure
        For "Task 2 output" - AMR with variables
        Simple and clean formatting
        """
        # Just return the original formatting if it's already multi-line
        if '\n' in amr_string:
            return amr_string.strip()

        # Otherwise, keep it linear for simplicity
        return amr_string.strip()

    def clean_amr(self, amr_string: str) -> str:
        """Clean and normalize AMR string"""
        # Remove excessive whitespace
        amr_string = re.sub(r'\s+', ' ', amr_string)

        # Normalize concepts with underscores
        # Keep multi-word concepts as underscored

        return amr_string.strip()

    def preprocess_for_mtup(
        self,
        sentence: str,
        amr_with_vars: str
    ) -> str:
        """
        Main preprocessing for MTUP format

        Args:
            sentence: Vietnamese input sentence
            amr_with_vars: Original AMR with variables (from dataset)

        Returns:
            Formatted MTUP training example (full prompt)
        """
        try:
            self.stats['processed'] += 1

            # Step 1: Generate AMR without variables (for Task 1 output)
            amr_no_vars_graph = self.remove_variables(amr_with_vars)
            amr_no_vars_linear = self.linearize(amr_no_vars_graph)

            # Step 2: Format AMR with variables (for Task 2 output)
            # Can be either linearized or graph format based on config
            if self.config.get('use_graph_format', True):
                # Use graph format (keep original multi-line format)
                amr_with_vars_formatted = self.format_graph(amr_with_vars)
            else:
                # Use linearized format with readable spacing
                amr_with_vars_formatted = self.linearize(amr_with_vars, keep_spaces=True)

            # Step 3: Format using MTUP template
            mtup_example = format_mtup_example(
                sentence=sentence,
                amr_no_vars=amr_no_vars_linear,
                amr_with_vars=amr_with_vars_formatted,
                template_name=self.template_name
            )

            # Update stats
            self.stats['avg_no_var_length'] += len(amr_no_vars_linear)
            self.stats['avg_with_var_length'] += len(amr_with_vars_formatted)

            return mtup_example

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"MTUP preprocessing error: {e}")
            logger.error(f"Sentence: {sentence[:100]}")
            # Fallback: return simple format
            return f"Câu: {sentence}\n\nAMR:\n{amr_with_vars}"

    def get_stats(self) -> Dict:
        """Get preprocessing statistics"""
        stats = self.stats.copy()
        if stats['processed'] > 0:
            stats['avg_no_var_length'] = stats['avg_no_var_length'] / stats['processed']
            stats['avg_with_var_length'] = stats['avg_with_var_length'] / stats['processed']
        return stats

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'processed': 0,
            'errors': 0,
            'avg_no_var_length': 0,
            'avg_with_var_length': 0,
        }


def test_mtup_preprocessor():
    """Test the MTUP preprocessor"""
    preprocessor = MTUPAMRPreprocessor(config={
        'template_name': 'v2_natural',
        'use_graph_format': True
    })

    # Test example
    sentence = "Tôi nhớ lời chủ tịch xã nhắc về vấn đề quan trọng."
    amr = """(n / nhớ
    :pivot(t / tôi)
    :theme(l / lời
        :poss(c / chủ_tịch
            :mod(x / xã))
        :topic(v / vấn_đề
            :mod(q / quan_trọng))))"""

    result = preprocessor.preprocess_for_mtup(sentence, amr)

    print("=" * 80)
    print("MTUP PREPROCESSOR TEST")
    print("=" * 80)
    print("\nInput Sentence:")
    print(sentence)
    print("\nInput AMR (with vars):")
    print(amr)
    print("\n" + "=" * 80)
    print("MTUP FORMATTED OUTPUT:")
    print("=" * 80)
    print(result)
    print("\n" + "=" * 80)
    print("STATS:")
    print("=" * 80)
    stats = preprocessor.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    test_mtup_preprocessor()
