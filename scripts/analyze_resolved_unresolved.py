import json
import os
import argparse
import re
import glob
from typing import Dict, List, Set, Tuple

def load_jsonl(file_path: str) -> Dict[str, dict]:
    """Load a JSONL file and index entries by instance_id."""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            if 'instance_id' in entry:
                results[entry['instance_id']] = entry
    return results

def load_report(file_path: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Load a report JSON file and return sets of resolved, unresolved, and incomplete IDs."""
    with open(file_path, 'r') as f:
        report = json.load(f)
    
    resolved_ids = set(report.get('resolved_ids', []))
    unresolved_ids = set(report.get('unresolved_ids', []))
    incomplete_ids = set(report.get('incomplete_ids', []))
    
    return resolved_ids, unresolved_ids, incomplete_ids

def extract_model_response_from_log(log_file_path: str) -> str:
    """Extract the model's complete response content from the log file."""
    if not os.path.exists(log_file_path):
        return "Log file not found"
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Find the last ChatCompletion response in the log
        matches = list(re.finditer(r"ChatCompletion\(.*?message=ChatCompletionMessage\(content='(.*?)', refusal=None", 
                                   log_content, re.DOTALL))
        if not matches:
            return "No model response found in log"
        
        # Get the last match
        last_match = matches[-1]
        # Extract and clean the content
        response_content = last_match.group(1)
        # Unescape single quotes and other special characters
        response_content = response_content.replace("\\'", "'").replace("\\n", "\n").replace("\\\"", "\"")
        
        return response_content
    except Exception as e:
        return f"Error extracting model response: {str(e)}"

def compare_responses(
    report_path1: str, 
    report_path2: str, 
    results_dir1: str, 
    results_dir2: str, 
    output_dir: str = "response_comparisons"
) -> None:
    """
    Compare model responses between two runs where instances are resolved in one but not in the other.
    
    Args:
        report_path1: Path to the first report JSON
        report_path2: Path to the second report JSON
        results_dir1: Path to the first results directory
        results_dir2: Path to the second results directory
        output_dir: Directory to save comparison files
    """
    # Load reports
    resolved1, unresolved1, incomplete1 = load_report(report_path1)
    resolved2, unresolved2, incomplete2 = load_report(report_path2)
    
    # Find instances resolved in report1 but not in report2
    resolved_in_1_not_2 = resolved1 - (resolved2 | incomplete2)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare and save results
    comparison_results = []
    
    # Get all_preds.jsonl files to extract patches
    patches1 = load_jsonl(os.path.join(results_dir1, "all_preds.jsonl"))
    patches2 = load_jsonl(os.path.join(results_dir2, "all_preds.jsonl"))
    
    for instance_id in resolved_in_1_not_2:
        # Get log files
        log_file1 = os.path.join(results_dir1, "logs", f"{instance_id}.log")
        log_file2 = os.path.join(results_dir2, "logs", f"{instance_id}.log")
        
        # Extract model responses
        response1 = extract_model_response_from_log(log_file1)
        response2 = extract_model_response_from_log(log_file2)
        
        # Get patches for reference
        patch1 = patches1.get(instance_id, {}).get('model_patch', 'No patch available')
        patch2 = patches2.get(instance_id, {}).get('model_patch', 'No patch available')
        
        comparison = {
            'instance_id': instance_id,
            'status': 'Resolved in report1, Unresolved in report2',
            'successful_response': response1,
            'unsuccessful_response': response2,
            'successful_patch': patch1,
            'unsuccessful_patch': patch2
        }
        comparison_results.append(comparison)
        
        # Save individual comparison to a text file
        with open(os.path.join(output_dir, f"{instance_id}_comparison.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Instance ID: {instance_id}\n")
            f.write(f"Status: Resolved in report1, Unresolved in report2\n\n")
            
            f.write("=== SUCCESSFUL MODEL RESPONSE (REPORT 1) ===\n\n")
            f.write(response1)
            f.write("\n\n=== UNSUCCESSFUL MODEL RESPONSE (REPORT 2) ===\n\n")
            f.write(response2)
            
            f.write("\n\n=== SUCCESSFUL PATCH (REPORT 1) ===\n\n")
            f.write(patch1)
            f.write("\n\n=== UNSUCCESSFUL PATCH (REPORT 2) ===\n\n")
            f.write(patch2)
    
    # Save a summary file
    with open(os.path.join(output_dir, "comparison_summary.txt"), 'w') as f:
        f.write(f"Total instances resolved in report1 but not in report2: {len(resolved_in_1_not_2)}\n")
        f.write("Instance IDs:\n")
        for instance_id in resolved_in_1_not_2:
            f.write(f"- {instance_id}\n")
    
    # Save all comparison data as JSON
    with open(os.path.join(output_dir, "all_comparisons.json"), 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"Found {len(resolved_in_1_not_2)} instances resolved in report1 but not in report2")
    print(f"Comparisons saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Compare model responses between two SWE-bench runs')
    parser.add_argument('--report1', required=True, help='Path to the first report JSON')
    parser.add_argument('--report2', required=True, help='Path to the second report JSON')
    parser.add_argument('--results_dir1', required=True, help='Path to the first results directory')
    parser.add_argument('--results_dir2', required=True, help='Path to the second results directory')
    parser.add_argument('--output', default='response_comparisons', help='Directory to save comparison files')
    
    args = parser.parse_args()
    
    compare_responses(
        args.report1,
        args.report2,
        args.results_dir1,
        args.results_dir2,
        args.output
    )

if __name__ == "__main__":
    main()