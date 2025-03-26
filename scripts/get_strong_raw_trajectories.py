#!/usr/bin/env python3
"""
Extract strong model responses from log files.

This script parses log files in the specified directory, extracts the content
from API responses, and stores them in a structured format for later use
when prompting weaker models.
"""

import os
import re
import json
import html
import codecs
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResponseExtractor:
    """Extract and process strong model responses from log files."""
    
    def __init__(self, logs_dir: str, output_dir: str):
        """
        Initialize the extractor.
        
        Args:
            logs_dir: Directory containing the log files
            output_dir: Directory to save the extracted responses
        """
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def find_log_files(self) -> List[Path]:
        """Find all log files in the specified directory."""
        log_files = list(self.logs_dir.glob("*.log"))
        logger.info(f"Found {len(log_files)} log files in {self.logs_dir}")
        return log_files
    
    def extract_responses(self, log_file: Path) -> List[Dict[str, Any]]:
        """
        Extract API responses from a log file.
        
        Args:
            log_file: Path to the log file
            
        Returns:
            List of dictionaries containing extracted responses
        """
        responses = []
        instance_id = log_file.stem  # Extract instance_id from filename
        
        logger.info(f"Processing log file: {log_file}")
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find all API response entries
            # Handle both single and double quoted content fields
            api_response_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - API response ChatCompletion\(.*?message=ChatCompletionMessage\(content=([\'"])(.*?)\2, refusal=.*?\)\)\)'
            matches = re.findall(api_response_pattern, content, re.DOTALL)
            
            # Process matches
            processed_matches = []
            for match in matches:
                timestamp_str = match[0]
                # The content is in the third group (index 2) since the second group (index 1) is the quote type
                response_content = match[2]
                processed_matches.append((timestamp_str, response_content))
            
            matches = processed_matches
            
            if not matches:
                logger.warning(f"No API responses found in {log_file}")
                return responses
            
            for timestamp_str, response_content in matches:
                # Parse timestamp
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                
                # Clean up the response content (handle escaped characters, etc.)
                # Convert all encoded characters to their proper representation
                import html
                
                # First handle common escape sequences
                clean_content = response_content.replace("\\n", "\n")
                clean_content = clean_content.replace("\\'", "'")
                clean_content = clean_content.replace('\\"', '"')
                
                # Handle Unicode escapes with double backslashes
                clean_content = clean_content.replace('\\\\u', '\\u')
                
                # Function to decode all Unicode escape sequences
                def decode_all_unicode(text):
                    try:
                        # Decode standard Unicode escapes
                        text = codecs.decode(text, 'unicode_escape', errors='replace')
                        
                        # Handle HTML entities
                        text = html.unescape(text)
                        
                        # Additional replacements for specific problematic characters
                        replacements = {
                            '\u00e2\u0080\u0090': '-',  # Replace fancy hyphens
                            '\u00e2\u0080\u0091': '-',  # Replace fancy hyphens
                            '\u00e2\u0080\u0092': '-',  # Replace fancy hyphens
                            '\u00e2\u0080\u0093': '-',  # Replace en dash
                            '\u00e2\u0080\u0094': '-',  # Replace em dash
                            '\u00e2\u0080\u0098': "'",  # Replace fancy quotes
                            '\u00e2\u0080\u0099': "'",  # Replace fancy quotes
                            '\u00e2\u0080\u009c': '"',  # Replace fancy quotes
                            '\u00e2\u0080\u009d': '"',  # Replace fancy quotes
                            '\u2003': ' ',              # Replace em space
                        }
                        
                        for old, new in replacements.items():
                            text = text.replace(old, new)
                            
                        return text
                    except Exception as e:
                        logger.warning(f"Error decoding Unicode: {e}")
                        return text
                
                clean_content = decode_all_unicode(clean_content)
                
                response_data = {
                    "instance_id": instance_id,
                    "timestamp": timestamp.isoformat(),
                    "content": clean_content
                }
                
                responses.append(response_data)
            
            logger.info(f"Extracted {len(responses)} responses from {log_file}")
            return responses
            
        except Exception as e:
            logger.error(f"Error processing {log_file}: {e}")
            return []
    
    def save_responses(self, all_responses: Dict[str, List[Dict[str, Any]]]):
        """
        Save extracted responses to files.
        
        Args:
            all_responses: Dictionary mapping instance IDs to lists of responses
        """
        # Save each instance's responses to a separate file
        for instance_id, responses in all_responses.items():
            if not responses:
                continue
                
            # Sort responses by timestamp
            sorted_responses = sorted(responses, key=lambda x: x["timestamp"])
            
            # Create output file path
            output_file = self.output_dir / f"{instance_id}.json"
            
            # Save to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_responses, f, indent=2)
            
            logger.info(f"Saved {len(sorted_responses)} responses for {instance_id} to {output_file}")
            
        # Save an index file with summary information
        index_data = {
            "total_instances": len(all_responses),
            "instances": {}
        }
        
        for instance_id, responses in all_responses.items():
            index_data["instances"][instance_id] = {
                "response_count": len(responses),
                "filename": f"{instance_id}.json"
            }
            
        index_file = self.output_dir / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
            
        logger.info(f"Saved index file to {index_file}")
    
    def process_all_logs(self):
        """Process all log files and extract responses."""
        log_files = self.find_log_files()
        if not log_files:
            logger.error(f"No log files found in {self.logs_dir}")
            return
            
        # Dictionary to store all responses, keyed by instance_id
        all_responses = {}
        
        for log_file in log_files:
            instance_id = log_file.stem
            responses = self.extract_responses(log_file)
            
            if responses:
                all_responses[instance_id] = responses
        
        logger.info(f"Processed {len(log_files)} log files, found responses for {len(all_responses)} instances")
        self.save_responses(all_responses)
        logger.info("Processing complete")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Extract strong model responses from log files')
    parser.add_argument('--logs-dir', type=str, 
                        default='results/base_o3-mini-2025-01-31/logs',
                        help='Directory containing log files')
    parser.add_argument('--output-dir', type=str, 
                        default='data/strong_model_trajectories',
                        help='Directory to save extracted responses')
    args = parser.parse_args()
    
    logger.info(f"Starting extraction from {args.logs_dir}")
    extractor = ResponseExtractor(args.logs_dir, args.output_dir)
    extractor.process_all_logs()

if __name__ == "__main__":
    main()