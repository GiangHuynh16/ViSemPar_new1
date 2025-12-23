"""
Inference Module for Vietnamese AMR Parser
Handles prediction generation with postprocessing
"""

import torch
import logging
import time
from typing import List, Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AMRInference:
    """
    Handles AMR generation from Vietnamese sentences
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        postprocessor,
        config: Dict,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.postprocessor = postprocessor
        self.config = config
        self.device = device
        
        # Move model to device
        if hasattr(self.model, 'to'):
            self.model.to(device)
        
        # Set to eval mode
        self.model.eval()
        
        logger.info(f"Inference engine initialized on {device}")
    
    def generate_amr(
        self,
        sentence: str,
        prompt_template: str,
        max_new_tokens: Optional[int] = None,
        return_metadata: bool = True
    ) -> Dict:
        """
        Generate AMR for a single sentence
        
        Returns:
            Dict with 'sentence', 'linear_amr', 'graph_amr', and optionally metadata
        """
        # Format prompt
        prompt = prompt_template.format(sentence=sentence)
        
        # Tokenize
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # Get inference config
        inference_cfg = self.config['INFERENCE_CONFIG']
        max_tokens = max_new_tokens or inference_cfg['max_new_tokens']
        
        # Generate
        with torch.no_grad():
            start_time = time.time()
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=inference_cfg['temperature'],
                top_p=inference_cfg['top_p'],
                top_k=inference_cfg.get('top_k', 50),
                do_sample=inference_cfg['do_sample'],
                repetition_penalty=inference_cfg['repetition_penalty'],
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            generation_time = time.time() - start_time
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract AMR from response
        try:
            # Look for response after prompt
            if "### Response:" in generated_text:
                linear_amr = generated_text.split("### Response:")[-1].strip()
            elif "Response:" in generated_text:
                linear_amr = generated_text.split("Response:")[-1].strip()
            else:
                # Fallback: take everything after input
                linear_amr = generated_text.split(sentence)[-1].strip()
        except:
            linear_amr = generated_text
        
        # Postprocess to graph format
        graph_amr = self.postprocessor.postprocess(linear_amr)
        
        # Validate
        is_valid, error = self.postprocessor.validate_amr(graph_amr)
        
        result = {
            'sentence': sentence,
            'linear_amr': linear_amr,
            'graph_amr': graph_amr,
            'is_valid': is_valid,
        }
        
        if return_metadata:
            result.update({
                'error': error if not is_valid else None,
                'generation_time': generation_time,
                'concept_to_var': dict(self.postprocessor.concept_to_var),
                'num_concepts': len(self.postprocessor.concept_to_var),
            })
        
        return result
    
    def generate_batch(
        self,
        sentences: List[str],
        prompt_template: str,
        batch_size: int = 8,
        show_progress: bool = True,
        return_metadata: bool = False
    ) -> List[Dict]:
        """
        Generate AMRs for multiple sentences
        
        Args:
            sentences: List of input sentences
            prompt_template: Prompt template
            batch_size: Number of sentences to process at once
            show_progress: Show progress bar
            return_metadata: Include generation metadata
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        # Create progress bar
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating AMRs", unit="batch")
        
        for i in iterator:
            batch_sentences = sentences[i:i + batch_size]
            
            for sentence in batch_sentences:
                try:
                    result = self.generate_amr(
                        sentence,
                        prompt_template,
                        return_metadata=return_metadata
                    )
                    predictions.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing sentence: {sentence[:50]}... Error: {e}")
                    # Add error result
                    predictions.append({
                        'sentence': sentence,
                        'linear_amr': '',
                        'graph_amr': f'# Error: {str(e)}',
                        'is_valid': False,
                        'error': str(e)
                    })
            
            # Optional: Clear cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        # Log statistics
        valid_count = sum(1 for p in predictions if p['is_valid'])
        logger.info(f"Generated {len(predictions)} AMRs")
        logger.info(f"Valid: {valid_count}/{len(predictions)} ({100 * valid_count / len(predictions):.1f}%)")
        
        return predictions
    
    def generate_from_dataframe(
        self,
        df,
        sentence_column: str = 'sentence',
        prompt_template: str = None,
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Generate AMRs from a pandas DataFrame
        
        Args:
            df: DataFrame with sentences
            sentence_column: Name of column containing sentences
            prompt_template: Prompt template (uses default if None)
            batch_size: Batch size for generation
            show_progress: Show progress bar
        
        Returns:
            List of predictions
        """
        if prompt_template is None:
            prompt_template = self.config['PROMPT_TEMPLATE']
        
        sentences = df[sentence_column].tolist()
        
        logger.info(f"Generating AMRs for {len(sentences)} sentences from DataFrame")
        
        predictions = self.generate_batch(
            sentences=sentences,
            prompt_template=prompt_template,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        return predictions


class StreamingInference:
    """
    Streaming inference for very large datasets
    Processes and writes results on-the-fly to save memory
    """
    
    def __init__(self, inference_engine: AMRInference, output_file: str):
        self.inference = inference_engine
        self.output_file = output_file
        self.processed_count = 0
    
    def process_stream(
        self,
        sentences: List[str],
        prompt_template: str,
        batch_size: int = 8
    ):
        """
        Process sentences and write results immediately
        """
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(sentences), batch_size), desc="Streaming"):
                batch = sentences[i:i + batch_size]
                
                for sentence in batch:
                    result = self.inference.generate_amr(
                        sentence,
                        prompt_template,
                        return_metadata=False
                    )
                    
                    # Write in VLSP format
                    f.write(f"#::snt {result['sentence']}\n")
                    f.write(f"{result['graph_amr']}\n\n")
                    
                    self.processed_count += 1
                
                # Periodic flush
                if i % (batch_size * 10) == 0:
                    f.flush()
                    torch.cuda.empty_cache()
        
        logger.info(f"Streaming complete: {self.processed_count} AMRs written to {self.output_file}")


def test_inference():
    """Test inference module"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.config import MODEL_NAME, INFERENCE_CONFIG, PROMPT_TEMPLATE
    from postprocessor import ImprovedAMRPostprocessor
    
    # This is just a structure test - actual model loading would happen in main.py
    print("Inference module structure validated!")
    print(f"Model: {MODEL_NAME}")
    print(f"Config: {INFERENCE_CONFIG}")


if __name__ == "__main__":
    from pathlib import Path
    test_inference()
