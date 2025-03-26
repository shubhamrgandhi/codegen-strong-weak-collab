import json

def compare_entries(original_file, voyage_file):
    # Load the files
    original_entries = {}
    voyage_entries = {}
    
    # Load original file
    with open(original_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            original_entries[entry['instance_id']] = entry
    
    # Load voyage file
    with open(voyage_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            voyage_entries[entry['instance_id']] = entry
    
    # Find common instance_ids
    common_ids = set(original_entries.keys()) & set(voyage_entries.keys())
    print(f"Found {len(common_ids)} common instance_ids")
    
    # Compare found_files and file_contents for common IDs
    results = {}
    for instance_id in common_ids:
        original = original_entries[instance_id]
        voyage = voyage_entries[instance_id]
        
        # Compare found_files
        found_files_match = set(original['found_files']) == set(voyage['found_files'])
        
        # Compare file_contents
        file_contents_match = set(original['file_contents']) == set(voyage['file_contents'])
        
        results[instance_id] = {
            'found_files_match': found_files_match,
            'file_contents_match': file_contents_match,
            'both_match': found_files_match and file_contents_match
        }
    
    # Summarize results
    matches_both = sum(1 for res in results.values() if res['both_match'])
    matches_found_files = sum(1 for res in results.values() if res['found_files_match'])
    matches_file_contents = sum(1 for res in results.values() if res['file_contents_match'])
    
    print(f"Summary for {len(common_ids)} common instance_ids:")
    print(f"- Entries with matching found_files: {matches_found_files} ({matches_found_files/len(common_ids)*100:.1f}%)")
    print(f"- Entries with matching file_contents: {matches_file_contents} ({matches_file_contents/len(common_ids)*100:.1f}%)")
    print(f"- Entries with both matching: {matches_both} ({matches_both/len(common_ids)*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    original_file = "results/original/retrieval.jsonl"
    voyage_file = "results/base/retrieval.jsonl"
    
    results = compare_entries(original_file, voyage_file)
    
    # Optional: Print detailed results for non-matching entries
    print("\nDetailed mismatches:")
    mismatches = 0
    for instance_id, result in results.items():
        if not result['both_match']:
            mismatches += 1
            print(f"Instance ID: {instance_id}")
            print(f"  found_files match: {result['found_files_match']}")
            print(f"  file_contents match: {result['file_contents_match']}")
            
            if mismatches >= 10:  # Limit to first 10 mismatches to avoid overwhelming output
                print("... (more mismatches exist)")
                break