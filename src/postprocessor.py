"""
Improved AMR Postprocessor for Vietnamese
Converts linearized AMR back to graph format with proper variable assignment
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ImprovedAMRPostprocessor:
    """
    Enhanced postprocessing with smart variable assignment
    Key features:
    1. Tracks repeated concepts for co-reference
    2. Assigns same variable to same concept (v2, v2, v2...)
    3. Proper graph formatting with indentation
    4. Validation of output AMR
    """
    
    def __init__(self):
        self.concept_to_var = {}  # Track concept → assigned variable
        self.var_counter = {}     # Counter for each variable letter
        self.stats = {
            'processed': 0,
            'errors': 0,
            'variables_assigned': 0,
            'coreferences_preserved': 0
        }
    
    def reset_state(self):
        """Reset state for new AMR"""
        self.concept_to_var = {}
        self.var_counter = {}
    
    def get_next_variable(self, concept: str) -> str:
        """
        Get variable for a concept
        - If concept seen before, return same variable
        - If new concept, assign new variable
        """
        # Check if we've seen this concept
        if concept in self.concept_to_var:
            self.stats['coreferences_preserved'] += 1
            return self.concept_to_var[concept]
        
        # New concept - assign variable
        # Use first letter of concept (or 'x' as fallback)
        base_letter = concept[0].lower() if concept and concept[0].isalpha() else 'x'
        
        # Get counter for this letter
        if base_letter not in self.var_counter:
            self.var_counter[base_letter] = 1
        else:
            self.var_counter[base_letter] += 1
        
        # Create variable: e.g., 'n', 'n2', 'n3', etc.
        if self.var_counter[base_letter] == 1:
            var = base_letter
        else:
            var = f"{base_letter}{self.var_counter[base_letter]}"
        
        # Store mapping
        self.concept_to_var[concept] = var
        self.stats['variables_assigned'] += 1
        
        return var
    
    def extract_concepts(self, linear_amr: str) -> List[str]:
        """
        Extract all concepts from linearized AMR
        Concepts appear as (concept) or after /
        """
        concepts = []
        
        # Pattern 1: (concept) without /
        pattern1 = r'\(([^/\s)]+)\)'
        matches1 = re.findall(pattern1, linear_amr)
        concepts.extend(matches1)
        
        # Pattern 2: After / in (var / concept)
        pattern2 = r'/\s*([^\s:)]+)'
        matches2 = re.findall(pattern2, linear_amr)
        concepts.extend(matches2)
        
        return concepts
    
    def add_variables_to_concepts(self, linear_amr: str) -> str:
        """
        Convert (concept) → (var / concept)
        Assigns variables smartly based on concept repetition
        """
        result = linear_amr
        
        # Find all (concept) patterns that don't have variables
        pattern = r'\(([^/\s)]+)\)'
        
        def replace_with_variable(match):
            concept = match.group(1)
            # Skip if it's a relation or special concept
            if concept.startswith(':') or concept in ['-', '+']:
                return match.group(0)
            
            var = self.get_next_variable(concept)
            return f'({var} / {concept})'
        
        result = re.sub(pattern, replace_with_variable, result)
        return result
    
    def format_graph(self, amr_string: str, indent: int = 4) -> str:
        """
        Format linearized AMR into indented graph structure
        Makes it human-readable and matches VLSP format
        """
        lines = []
        current_indent = 0
        current_line = ""
        
        i = 0
        while i < len(amr_string):
            char = amr_string[i]
            
            if char == '(':
                # Start new nested structure
                if current_line.strip():
                    lines.append(' ' * (current_indent * indent) + current_line.strip())
                    current_line = ""
                current_line += char
                current_indent += 1
                
            elif char == ')':
                # Close nested structure
                if current_line.strip():
                    lines.append(' ' * ((current_indent - 1) * indent) + current_line.strip())
                    current_line = ""
                current_indent -= 1
                lines.append(' ' * (current_indent * indent) + char)
                
            elif char == ':':
                # New relation - new line
                if current_line.strip() and not current_line.strip().endswith('('):
                    lines.append(' ' * (current_indent * indent) + current_line.strip())
                    current_line = ""
                # Look ahead to get full relation
                j = i + 1
                while j < len(amr_string) and amr_string[j] not in ' \t\n()':
                    j += 1
                relation = amr_string[i:j]
                current_line = relation
                i = j - 1
                
            else:
                current_line += char
            
            i += 1
        
        # Add any remaining line
        if current_line.strip():
            lines.append(' ' * (current_indent * indent) + current_line.strip())
        
        # Clean up
        formatted = '\n'.join(line.rstrip() for line in lines if line.strip())
        return formatted
    
    def validate_amr(self, amr_string: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the generated AMR
        Returns: (is_valid, error_message)
        """
        # Check balanced parentheses
        open_count = amr_string.count('(')
        close_count = amr_string.count(')')
        if open_count != close_count:
            return False, f"Unbalanced parentheses: {open_count} open, {close_count} close"
        
        # Check for variable/concept pattern
        if not re.search(r'\([a-z][0-9]?\s*/\s*[^\s)]+', amr_string):
            return False, "No valid (variable / concept) pattern found"
        
        # Check for at least one relation
        if not re.search(r':[a-zA-Z0-9\-_]+', amr_string):
            return False, "No relations found"
        
        return True, None
    
    def clean_output(self, amr_string: str) -> str:
        """
        Clean up common issues in model output
        """
        # Remove excessive whitespace
        amr_string = re.sub(r'\s+', ' ', amr_string)
        
        # Fix spacing around special characters
        amr_string = re.sub(r'\s*\(\s*', '(', amr_string)
        amr_string = re.sub(r'\s*\)\s*', ')', amr_string)
        amr_string = re.sub(r'\s*:\s*', ':', amr_string)
        amr_string = re.sub(r'\s*/\s*', ' / ', amr_string)
        
        # Remove any remaining artifacts
        amr_string = re.sub(r'[\r\n]+', ' ', amr_string)
        
        return amr_string.strip()
    
    def postprocess(self, linear_amr: str) -> str:
        """
        Full postprocessing pipeline
        
        Pipeline:
        1. Reset state
        2. Clean output
        3. Add variables to concepts
        4. Format as graph
        5. Validate
        """
        try:
            self.stats['processed'] += 1
            self.reset_state()
            
            # Step 1: Clean
            linear_amr = self.clean_output(linear_amr)
            
            # Step 2: Add variables
            linear_amr = self.add_variables_to_concepts(linear_amr)
            
            # Step 3: Format as graph
            graph_amr = self.format_graph(linear_amr)
            
            # Step 4: Validate
            is_valid, error = self.validate_amr(graph_amr)
            if not is_valid:
                logger.warning(f"Validation failed: {error}")
                # Return linearized version if formatting fails
                return linear_amr
            
            return graph_amr
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Postprocessing error: {e}")
            # Fallback: return cleaned input
            return self.clean_output(linear_amr)
    
    def get_stats(self) -> Dict:
        """Get postprocessing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'processed': 0,
            'errors': 0,
            'variables_assigned': 0,
            'coreferences_preserved': 0
        }


def test_postprocessor():
    """Test the postprocessor"""
    postprocessor = ImprovedAMRPostprocessor()
    
    # Test example - linearized AMR without variables
    linear = "(nhớ :pivot(tôi) :theme(lời :poss(chủ tịch :mod(xã))))"
    
    result = postprocessor.postprocess(linear)
    print("Input:")
    print(linear)
    print("\nOutput:")
    print(result)
    print("\nStats:")
    print(postprocessor.get_stats())
    print("\nConcept to Variable mapping:")
    print(postprocessor.concept_to_var)


if __name__ == "__main__":
    test_postprocessor()
