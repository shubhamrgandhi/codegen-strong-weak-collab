import json
import os
import glob

def pool_resolved_ids():
    # Find all JSON files with results in the sb-cli-reports directory
    result_files = glob.glob("sb-cli-reports/swe-bench_lite__test__agentless_lite_sc_direct_gpt-4o-mini-2024-07-18_*.json")
    
    if not result_files:
        print("No result files found. Make sure the files are in the sb-cli-reports directory.")
        return
    
    # Initialize set to store unique resolved IDs
    all_resolved_ids = set()
    
    # Process each file
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Add resolved IDs to the set
                if "resolved_ids" in data:
                    resolved_ids = set(data["resolved_ids"])
                    all_resolved_ids.update(resolved_ids)
                else:
                    print(f"Warning: 'resolved_ids' not found in {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Calculate the resolution rate
    total_instances = 300  # As mentioned in your description
    resolution_rate = len(all_resolved_ids) / total_instances * 100
    
    # Print results
    print(f"Total unique resolved IDs: {len(all_resolved_ids)}")
    print(f"Total instances: {total_instances}")
    print(f"Pooled resolution rate: {resolution_rate:.2f}%")
    
    return all_resolved_ids, resolution_rate

if __name__ == "__main__":
    pool_resolved_ids()