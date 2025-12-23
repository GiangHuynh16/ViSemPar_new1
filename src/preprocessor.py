"""
Improved AMR Preprocessor for Vietnamese
Addresses hallucination and preserves co-references
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ImprovedAMRPreprocessor:
    """
    Enhanced preprocessing that preserves semantic information
    Key improvements:
    1. Preserves co-references by replacing variables with concepts
    2. Normalizes concepts for consistency
    3. Handles multi-word expressions
    4. Fixes malformed AMR structures
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.stats = {
            'processed': 0,
            'errors': 0,
            'variable_replacements': 0,
            'concept_normalizations': 0
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
    
    def replace_variable_references(self, amr_string: str, var_to_concept: Dict[str, str]) -> str:
        """
        Replace variable references with concept references
        This is KEY for preserving co-reference information
        
        Examples:
        - :ARG0 p → :ARG0(person)
        - :location c) → :location(city))
        """
        result = amr_string
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_vars = sorted(var_to_concept.items(), key=lambda x: len(x[0]), reverse=True)
        
        for var, concept in sorted_vars:
            # Pattern 1: :relation variable (followed by space or newline)
            # :ARG0 p → :ARG0(person)
            pattern1 = rf':([a-zA-Z0-9\-_]+)\s+{re.escape(var)}(?=\s|$|\))'
            replacement1 = rf':\1({concept})'
            result = re.sub(pattern1, replacement1, result)
            
            # Pattern 2: standalone variable before closing paren
            # variable) → (concept))
            pattern2 = rf'(?<!\()\s+{re.escape(var)}\s*\)'
            replacement2 = f'({concept})'
            result = re.sub(pattern2, replacement2, result)
            
            # Pattern 3: variable at end of line
            pattern3 = rf'\s+{re.escape(var)}$'
            replacement3 = f'({concept})'
            result = re.sub(pattern3, replacement3, result, flags=re.MULTILINE)
        
        self.stats['variable_replacements'] += len(var_to_concept)
        return result
    
    def remove_variables(self, amr_string: str) -> str:
        """
        Remove variable declarations: (var / concept) → (concept)
        Only removes the variable part, preserves full concept
        """
        # Match only lowercase variable names (a-z0-9)
        cleaned = re.sub(r'\(([a-z0-9]+)\s*/', r'(', amr_string)
        return cleaned
    
    def normalize_concepts(self, amr_string: str) -> str:
        """
        Normalize concept names for consistency
        - Replace spaces with underscores in concepts
        - Ensure proper formatting
        """
        if not self.config.get('normalize_concepts', True):
            return amr_string
        
        def replace_spaces_in_concepts(match):
            concept = match.group(1)
            # Replace spaces with underscores
            concept_normalized = concept.replace(' ', '_')
            # Remove multiple underscores
            concept_normalized = re.sub(r'_+', '_', concept_normalized)
            return f'/{concept_normalized}'
        
        # Match concepts after /
        result = re.sub(
            r'/\s*([^\s:)]+(?:\s+[^\s:)]+)*)',
            replace_spaces_in_concepts,
            amr_string
        )
        
        self.stats['concept_normalizations'] += 1
        return result
    
    def remove_wiki_tags(self, amr_string: str) -> str:
        """Remove wiki tags that aren't needed for training"""
        # Remove :wiki with parentheses
        cleaned = re.sub(r':wiki\s*\([^)]+\)', '', amr_string)
        # Remove :wiki with dash
        cleaned = re.sub(r':wiki\s*-', '', cleaned)
        return cleaned
    
    def linearize(self, amr_string: str) -> str:
        """
        Convert multi-line AMR to single line
        Preserves structure while making it model-friendly
        """
        # Join lines
        linear = ' '.join(amr_string.split())
        
        # Normalize multiple spaces
        linear = re.sub(r'\s+', ' ', linear)
        
        # Clean up spacing around parentheses and colons
        linear = re.sub(r'\s*\(\s*', '(', linear)
        linear = re.sub(r'\s*\)\s*', ')', linear)
        linear = re.sub(r'\s*:\s*', ':', linear)
        
        return linear.strip()
    
    def fix_malformed_structures(self, amr_string: str) -> str:
        """
        Fix common malformed AMR structures
        """
        result = amr_string
        
        # Fix double concepts: (concept1 concept2) → (concept1_concept2)
        result = re.sub(r'\(([^/\s]+)\s+([^/\s)]+)\)', r'(\1_\2)', result)
        
        # Fix missing parentheses around concepts
        # :relation concept) → :relation(concept))
        result = re.sub(r':([a-zA-Z0-9\-_]+)\s+([^(\s)]+)\)', r':\1(\2)', result)
        
        return result
    
    def validate_preprocessed(self, amr_string: str) -> Tuple[bool, str]:
        """
        Validate the preprocessed AMR
        Returns: (is_valid, error_message)
        """
        # Check balanced parentheses
        open_count = amr_string.count('(')
        close_count = amr_string.count(')')
        if open_count != close_count:
            return False, f"Unbalanced parentheses: {open_count} open, {close_count} close"
        
        # Check that no variables remain (all should be replaced)
        if re.search(r':([a-zA-Z0-9\-_]+)\s+[a-z][0-9]*(?=\s|\))', amr_string):
            return False, "Variable references still present"
        
        # Check for at least one concept
        if not re.search(r'\([^\s)]+', amr_string):
            return False, "No concepts found"
        
        return True, ""
    
    def preprocess(self, amr_string: str) -> str:
        """
        Full preprocessing pipeline
        
        Pipeline:
        1. Extract variable→concept mapping
        2. Replace variable references with concepts (PRESERVE COREFERENCE)
        3. Remove variable declarations
        4. Normalize concepts
        5. Remove wiki tags
        6. Fix malformed structures
        7. Linearize
        8. Validate
        """
        try:
            self.stats['processed'] += 1
            
            # Step 1: Extract variable mapping
            var_to_concept = self.extract_variable_to_concept_map(amr_string)
            
            # Step 2: Replace variable references (KEY STEP!)
            if self.config.get('preserve_coreference', True):
                amr_string = self.replace_variable_references(amr_string, var_to_concept)
            
            # Step 3: Remove variable declarations
            if self.config.get("remove_variables", True):
                amr_string = self.remove_variables(amr_string)
            
            # Step 4: Normalize concepts
            amr_string = self.normalize_concepts(amr_string)
            
            # Step 5: Remove wiki tags
            amr_string = self.remove_wiki_tags(amr_string)
            
            # Step 6: Fix malformed structures
            if self.config.get('fix_malformed_amr', True):
                amr_string = self.fix_malformed_structures(amr_string)
            
            # Step 7: Linearize
            amr_string = self.linearize(amr_string)
            
            # Step 8: Validate
            is_valid, error = self.validate_preprocessed(amr_string)
            if not is_valid:
                logger.warning(f"Validation failed: {error}")
                # Return original linearized version if validation fails
                return self.linearize(amr_string)
            
            return amr_string
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Preprocessing error: {e}")
            # Fallback: just linearize
            return self.linearize(amr_string)
    
    def get_stats(self) -> Dict:
        """Get preprocessing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'processed': 0,
            'errors': 0,
            'variable_replacements': 0,
            'concept_normalizations': 0
        }


def test_preprocessor():
    """Test the preprocessor"""
    preprocessor = ImprovedAMRPreprocessor(config={
        'preserve_coreference': True,
        'normalize_concepts': True,
        'fix_malformed_amr': True
    })
    
    # Test example
    amr = """(n / nhớ
    :pivot(t / tôi)
    :theme(l / lời
        :poss(c / chủ tịch
            :mod(x / xã)
            :agent-of(n1 / nhắc
                :topic(q / quan trọng)))))"""
    
    result = preprocessor.preprocess(amr)
    print("Input:")
    print(amr)
    print("\nOutput:")
    print(result)
    print("\nStats:")
    print(preprocessor.get_stats())


if __name__ == "__main__":
    test_preprocessor()
