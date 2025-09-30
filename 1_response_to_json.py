#!/usr/bin/env python3
"""
Enhanced JSON Parser for LLM Responses
Handles various complex and compatibility issues in JSON formats
"""

import os
import json
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedJSONParser:
    """Enhanced JSON parser capable of handling various formatting issues"""
    
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'error_types': {}
        }
    
    def clean_json_string(self, text: str) -> str:
        """Clean JSON string by removing common issues"""
        # Remove markdown code fences
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Remove inline comments (// style)
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        
        # Remove parenthetical notes after numbers, e.g., "3 (Neutral)" -> "3"
        text = re.sub(r'(\d+)\s*\([^)]*\)', r'\1', text)
        
        # Fix property names missing double quotes
        text = re.sub(r'(\w+):', r'"\1":', text)
        
        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        return text.strip()
    
    def extract_json_blocks(self, text: str) -> List[str]:
        """Extract all possible JSON blocks from text"""
        json_blocks = []
        
        # Method 1: find complete JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        json_blocks.extend(matches)
        
        # Method 2: find array-form JSON
        array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        array_matches = re.findall(array_pattern, text, re.DOTALL)
        json_blocks.extend(array_matches)
        
        # Method 3: find nested JSON structures
        nested_pattern = r'\{.*?\}'
        nested_matches = re.findall(nested_pattern, text, re.DOTALL)
        json_blocks.extend(nested_matches)
        
        return json_blocks
    
    def try_parse_json(self, json_str: str) -> Optional[Dict]:
        """Try to parse a JSON string"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    def try_parse_with_ast(self, json_str: str) -> Optional[Dict]:
        """Use AST to try parsing Python-dict-like formats"""
        try:
            # Replace single quotes with double quotes
            json_str = json_str.replace("'", '"')
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return None
    
    def extract_scores_from_text(self, text: str) -> Dict[str, List[int]]:
        """Extract score information from free text"""
        result = {
            'ad_attitude_scores': [],
            'purchase_intention_scores': []
        }
        
        # Find numeric patterns
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            nums = [int(n) for n in numbers if 1 <= int(n) <= 7]
            
            # Infer score types from context
            if 'attitude' in text.lower() or 'like' in text.lower():
                result['ad_attitude_scores'] = nums[:4] if len(nums) >= 4 else nums
            if 'intention' in text.lower() or 'purchase' in text.lower():
                result['purchase_intention_scores'] = nums[4:7] if len(nums) >= 7 else nums[4:]
        
        return result
    
    def parse_response(self, response_text: str) -> Optional[Dict]:
        """Parse LLM response text"""
        if not response_text.strip():
            return None
        
        # Clean text
        cleaned_text = self.clean_json_string(response_text)
        
        # Method 1: try direct parse of cleaned text
        parsed = self.try_parse_json(cleaned_text)
        if parsed:
            return parsed
        
        # Method 2: extract JSON blocks and try to parse
        json_blocks = self.extract_json_blocks(cleaned_text)
        for block in json_blocks:
            parsed = self.try_parse_json(block)
            if parsed:
                return parsed
        
        # Method 3: try AST parsing
        for block in json_blocks:
            parsed = self.try_parse_with_ast(block)
            if parsed:
                return parsed
        
        # Method 4: extract scores from text as fallback
        scores = self.extract_scores_from_text(response_text)
        if scores['ad_attitude_scores'] or scores['purchase_intention_scores']:
            result = {'ad_type': 'unknown'}
            result.update(scores)
            return result
        
        return None
    
    def process_file(self, response_file: str, metadata_file: str) -> Tuple[bool, str, Optional[Dict]]:
        """Process a single response file"""
        try:
            # Read response file
            with open(response_file, 'r', encoding='utf-8') as f:
                response_text = f.read()
            
            # Read metadata file
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Parse response
            parsed_json = self.parse_response(response_text)
            
            if parsed_json:
                # Merge metadata with parsed results
                result = {**metadata, **parsed_json}
                return True, "Success", result
            else:
                return False, "No valid JSON found", None
                
        except Exception as e:
            return False, str(e), None
    
    def process_experiment_directory(self, response_dir: str, json_output_dir: str):
        """Process an entire experiment directory"""
        logger.info(f"Processing experiment directory: {response_dir}")
        
        # Create output directory
        os.makedirs(json_output_dir, exist_ok=True)
        
        # Collect all response files
        response_files = [f for f in os.listdir(response_dir) 
                         if f.endswith('.txt') and not f.endswith('_metadata.json')]
        
        success_count = 0
        error_count = 0
        error_details = []
        
        for response_file in response_files:
            self.stats['total_processed'] += 1
            
            response_path = os.path.join(response_dir, response_file)
            metadata_file = response_file.replace('.txt', '_metadata.json')
            metadata_path = os.path.join(response_dir, metadata_file)
            
            success, error_msg, result = self.process_file(response_path, metadata_path)
            
            if success:
                # Save successful JSON
                json_filename = response_file.replace('.txt', '.json')
                json_path = os.path.join(json_output_dir, json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                self.stats['successful'] += 1
            else:
                error_count += 1
                self.stats['failed'] += 1
                
                # Track error types
                error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                self.stats['error_types'][error_type] = self.stats['error_types'].get(error_type, 0) + 1
                
                error_details.append((response_file, error_msg))
                
                # Save error details
                error_filename = response_file.replace('.txt', '_parse_error.txt')
                error_path = os.path.join(json_output_dir, error_filename)
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(f"Parse Error: {error_msg}\n\n")
                    f.write("Raw Output:\n")
                    # Re-read response file content
                    with open(response_path, 'r', encoding='utf-8') as rf:
                        f.write(rf.read())
        
        logger.info(f"Completed {response_dir}: {success_count} successful, {error_count} errors")
        
        # Show error details
        if error_details:
            logger.info("Error details:")
            for file, error in error_details[:5]:  # 只显示前5个错误
                logger.info(f"  {file}: {error}")
            if len(error_details) > 5:
                logger.info(f"  ... and {len(error_details) - 5} more errors")
    
    def process_all_experiments(self, response_base_dir: str, json_base_dir: str):
        """Process all experiments"""
        logger.info("Starting enhanced JSON parsing...")
        
        if not os.path.exists(response_base_dir):
            logger.error(f"Response directory not found: {response_base_dir}")
            return
        
        # Iterate through all experiment folders
        for experiment_dir in os.listdir(response_base_dir):
            experiment_path = os.path.join(response_base_dir, experiment_dir)
            if not os.path.isdir(experiment_path):
                continue
            
            # Create corresponding JSON output directory
            json_experiment_dir = experiment_dir.replace('experiment_responses_', 'experiment_outputs_')
            json_output_dir = os.path.join(json_base_dir, json_experiment_dir)
            
            self.process_experiment_directory(experiment_path, json_output_dir)
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print statistics"""
        logger.info("=== Enhanced JSON Parser Statistics ===")
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            logger.info(f"Success rate: {success_rate:.2f}%")
        
        logger.info("Error types:")
        for error_type, count in sorted(self.stats['error_types'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {error_type}: {count}")


def main():
    """Main entry point"""
    parser = EnhancedJSONParser()
    
    # Configure paths
    response_base_dir = "./response_output"
    json_base_dir = "./json_output"
    
    # Process all experiments
    parser.process_all_experiments(response_base_dir, json_base_dir)


if __name__ == "__main__":
    main()
