#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def compute_first_iteration_performance(
    predictions_file="results/first_strong_gpt-4o-mini-2024-07-18/all_preds.jsonl",
    report_file="sb-cli-reports/swe-bench_lite__test__agentless_lite_first_strong_gpt-4o-mini-2024-07-18.json"
):
    # Load the predictions file
    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    # Load the report file
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    # Get the resolved IDs from the report
    resolved_ids = set(report.get('resolved_ids', []))
    
    # Find predictions with 'gpt-4o-mini' in model_name_or_path_actual that are in resolved_ids
    successful_ids = []
    for pred in predictions:
        if ('gpt-4o-mini' in pred.get('model_name_or_path_actual', '') and 
            pred.get('instance_id') in resolved_ids):
            successful_ids.append(pred.get('instance_id'))
    
    # Calculate statistics
    total_count = 300  # As mentioned in the prompt
    successful_count = len(successful_ids)
    success_percentage = (successful_count / total_count) * 100
    
    # Print results
    print(f"Total successfully resolved instances: {successful_count}/{total_count} ({success_percentage:.2f}%)")
    print("\nSuccessfully resolved instance IDs:")
    for instance_id in successful_ids:
        print(instance_id)
    
    return successful_ids, successful_count, success_percentage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute first iteration performance for Agentless Lite")
    parser.add_argument(
        "--predictions_file", 
        type=str, 
        default="results/first_strong_gpt-4o-mini-2024-07-18/all_preds.jsonl",
        help="Path to the predictions JSONL file"
    )
    parser.add_argument(
        "--report_file", 
        type=str, 
        default="sb-cli-reports/swe-bench_lite__test__agentless_lite_first_strong_gpt-4o-mini-2024-07-18.json",
        help="Path to the SWE-bench report JSON file"
    )
    
    args = parser.parse_args()
    compute_first_iteration_performance(args.predictions_file, args.report_file)