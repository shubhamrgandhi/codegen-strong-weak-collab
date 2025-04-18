import json
import argparse

def calculate_average_attempts(jsonl_path, expected_count=300):
    """
    Calculate the average number of attempts from entries in a JSONL file.
    For each missing entry (if total entries < expected_count), add 10 to the total attempts.
    
    Args:
        jsonl_path (str): Path to the JSONL file
        expected_count (int): Expected number of entries (default: 300)
        
    Returns:
        float: Average number of attempts
    """
    total_attempts = 0
    strong_attempts = 0
    weak_attempts = 0
    entry_count = 0
    
    try:
        with open(jsonl_path, 'r') as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    if "model_name_or_path_actual" in entry:
                        if "o3-mini" not in entry["model_name_or_path_actual"]:    
                                if "attempt" in entry:
                                    weak_attempts += entry["attempt"]
                                    total_attempts += entry["attempt"]
                                else:
                                    # Count entries without "attempt" field as having 10 attempts
                                    weak_attempts += 5
                                    total_attempts += 10
                        elif 'fallback' in jsonl_path:
                            if "attempt" in entry:
                                total_attempts += entry["attempt"]
                                weak_attempts += 5
                                strong_attempts += entry["attempt"]
                            else:
                                # Count entries without "attempt" field as having 10 attempts
                                total_attempts += 10
                                weak_attempts += 5
                                strong_attempts += 1
                        else: 
                            if "attempt" in entry:
                                total_attempts += entry["attempt"]
                            else:
                                # Count entries without "attempt" field as having 10 attempts
                                total_attempts += 10                            
                    else:
                        if "attempt" in entry:
                            total_attempts += entry["attempt"]
                        else:
                            # Count entries without "attempt" field as having 10 attempts
                            total_attempts += 10
                    entry_count += 1
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")
                    continue
        
        # Add 10 for each missing entry
        missing_entries = expected_count - entry_count
        if missing_entries > 0:
            print(f"Warning: Found {entry_count} entries, expected {expected_count}. Adding 10 attempts for each of the {missing_entries} missing entries.")
            total_attempts += missing_entries * 10
            strong_attempts += missing_entries * 1
            weak_attempts += missing_entries * 5
        
        # Always divide by the expected count
        return strong_attempts / expected_count, weak_attempts / expected_count
    
    except FileNotFoundError:
        print(f"Error: File '{jsonl_path}' not found")
        return None

def main():
    parser = argparse.ArgumentParser(description='Calculate average attempts from a JSONL file')
    parser.add_argument('--jsonl_path', type=str, help='Path to the JSONL file')
    args = parser.parse_args()
    
    # avg_attempts = calculate_average_attempts(args.jsonl_path)
    
    # if avg_attempts is not None:
    #     print(f"Average attempts: {avg_attempts:.2f}")
    
    avg_strong_attempts, avg_weak_attempts = calculate_average_attempts(args.jsonl_path)
    
    if avg_strong_attempts is not None and avg_weak_attempts is not None:
        print(f"Average strong attempts: {avg_strong_attempts:.2f}")
        print(f"Average weak attempts: {avg_weak_attempts:.2f}")

if __name__ == "__main__":
    main()