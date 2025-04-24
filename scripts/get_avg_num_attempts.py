import json
import argparse
import os
import glob

def calculate_average_attempts(jsonl_path, expected_count=300, strong_weak=False, router=False, routing_decisions_path=None):
    """
    Calculate the average number of attempts from entries in a JSONL file.
    For each missing entry (if total entries < expected_count), add 10 to the total attempts.
    
    Args:
        jsonl_path (str): Path to the JSONL file
        expected_count (int): Expected number of entries (default: 300)
        strong_weak (bool): Whether to calculate separate averages for strong and weak models
        router (bool): Whether to use routing decisions for missing entries
        routing_decisions_path (str): Path to the routing decisions JSONL
        
    Returns:
        float or tuple: Average number of attempts, or tuple of (strong_avg, weak_avg)
    """
    total_attempts = 0
    strong_attempts = 0
    weak_attempts = 0
    entry_count = 0
    processed_ids = set()
    
    try:
        with open(jsonl_path, 'r') as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    
                    # Track processed instance IDs for router option
                    if router and "instance_id" in entry:
                        processed_ids.add(entry["instance_id"])
                    
                    if "model_name_or_path_actual" in entry:
                        if 'fallback' in jsonl_path:
                            if "attempt" in entry:
                                total_attempts += entry["attempt"]
                                weak_attempts += 5
                                strong_attempts += entry["attempt"]
                            else:
                                # Count entries without "attempt" field as having 10 attempts
                                total_attempts += 10
                                weak_attempts += 5
                                strong_attempts += 1
                        elif "o3-mini" not in entry["model_name_or_path_actual"]:    
                            if "attempt" in entry:
                                weak_attempts += entry["attempt"]
                                total_attempts += entry["attempt"]
                            else:
                                # Count entries without "attempt" field as having 10 attempts
                                weak_attempts += 5
                                total_attempts += 10
                        else: 
                            if "attempt" in entry:
                                strong_attempts += entry["attempt"]
                                total_attempts += entry["attempt"]
                            else:
                                # Count entries without "attempt" field as having 10 attempts
                                strong_attempts += 10
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
            print(f"Warning: Found {entry_count} entries, expected {expected_count}. Adding attempts for each of the {missing_entries} missing entries.")
            
            if router and routing_decisions_path:
                # Get missing instance IDs and their routing decisions
                missing_ids_routing = get_missing_ids_routing(routing_decisions_path, processed_ids)
                
                # Count strong and weak attempts based on routing decisions
                for instance_id, model in missing_ids_routing.items():
                    if "o3-mini" in model:
                        strong_attempts += 10
                        total_attempts += 10
                    else:
                        weak_attempts += 10
                        total_attempts += 10
                
                # If there are still missing entries not found in routing decisions
                remaining_missing = missing_entries - len(missing_ids_routing)
                if remaining_missing > 0:
                    print(f"Warning: {remaining_missing} missing entries not found in routing decisions. Using default allocation.")
                    total_attempts += remaining_missing * 10
                    strong_attempts += remaining_missing * 1
                    weak_attempts += remaining_missing * 5
            else:
                # Use the original allocation
                total_attempts += missing_entries * 10
                strong_attempts += missing_entries * 1
                weak_attempts += missing_entries * 5
        
        # Always divide by the expected count
        if strong_weak:
            return strong_attempts / expected_count, weak_attempts / expected_count
        else:
            return total_attempts / expected_count
    
    except FileNotFoundError:
        print(f"Error: File '{jsonl_path}' not found")
        return None

def get_missing_ids_routing(routing_decisions_path, processed_ids):
    """
    Get routing decisions for missing instance IDs.
    
    Args:
        routing_decisions_path (str): Path to the routing decisions JSONL
        processed_ids (set): Set of already processed instance IDs
        
    Returns:
        dict: Mapping of missing instance IDs to their model assignments
    """
    missing_ids_routing = {}
    
    try:
        with open(routing_decisions_path, 'r') as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    instance_id = entry.get("instance_id")
                    model = entry.get("model", "")
                    
                    if instance_id and instance_id not in processed_ids:
                        missing_ids_routing[instance_id] = model
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Routing decisions file '{routing_decisions_path}' not found")
    
    return missing_ids_routing

def calculate_best_of_n_attempts(directory, expected_count=300, router=False, routing_decisions_path=None):
    """
    Calculate the average number of attempts from all temp_{i}.jsonl files in a directory.
    For each instance missing in a temp file, add 10 attempts.
    
    Args:
        directory (str): Path to the directory containing temp_{i}.jsonl files
        expected_count (int): Expected number of entries (default: 300)
        router (bool): Whether to use routing decisions for missing entries
        routing_decisions_path (str): Path to the routing decisions JSONL
        
    Returns:
        float: Average number of attempts
    """
    # Find all temp_{i}.jsonl files in the directory
    jsonl_files = glob.glob(os.path.join(directory, "temp_*.jsonl"))
    
    if not jsonl_files:
        print(f"Error: No temp_*.jsonl files found in directory '{directory}'")
        return None
    
    print(f"Found {len(jsonl_files)} temp_*.jsonl files in directory '{directory}'")
    
    # Set of all expected instance IDs (will be populated from files or routing decisions)
    all_instance_ids = set()
    
    # Dictionary to track instances found in each file
    instances_in_file = {file: set() for file in jsonl_files}
    
    # Dictionary to track attempts for each instance
    instance_attempts = {}
    
    # Process each temp_{i}.jsonl file
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r') as file:
                for line in file:
                    try:
                        entry = json.loads(line)
                        
                        if "instance_id" not in entry:
                            continue
                        
                        instance_id = entry["instance_id"]
                        all_instance_ids.add(instance_id)
                        instances_in_file[jsonl_file].add(instance_id)
                        
                        # Initialize instance if not seen before
                        if instance_id not in instance_attempts:
                            instance_attempts[instance_id] = 0
                        
                        # Add attempts for this instance
                        if "attempt" in entry:
                            instance_attempts[instance_id] += entry["attempt"]
                        else:
                            # Count entries without "attempt" field as having 10 attempts
                            instance_attempts[instance_id] += 10
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in file '{jsonl_file}'")
                        continue
        except FileNotFoundError:
            print(f"Error: File '{jsonl_file}' not found")
            continue
    
    # If router is enabled and we have routing_decisions_path, collect all expected instance IDs
    expected_instance_ids = set()
    if router and routing_decisions_path:
        try:
            with open(routing_decisions_path, 'r') as file:
                for line in file:
                    try:
                        entry = json.loads(line)
                        if "instance_id" in entry:
                            expected_instance_ids.add(entry["instance_id"])
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Error: Routing decisions file '{routing_decisions_path}' not found")
    
    # If we have expected instance IDs from routing decisions, use those
    # Otherwise, use the union of all instance IDs found in files
    if expected_instance_ids:
        all_instance_ids.update(expected_instance_ids)
    
    # Now, for each instance, add 10 attempts for each file it's missing from
    total_attempts = 0
    for instance_id in all_instance_ids:
        # Start with attempts we've already counted
        attempts = instance_attempts.get(instance_id, 0)
        
        # For each file, if the instance is missing, add 10 attempts
        for jsonl_file in jsonl_files:
            if instance_id not in instances_in_file[jsonl_file]:
                attempts += 10
        
        total_attempts += attempts
    
    # Handle missing instances if total is less than expected
    missing_instances = expected_count - len(all_instance_ids)
    if missing_instances > 0:
        print(f"Warning: Found {len(all_instance_ids)} unique instances, expected {expected_count}. Adding attempts for missing instances.")
        # Add 10 attempts per file for each entirely missing instance
        total_attempts += missing_instances * 10 * len(jsonl_files)
    
    # Calculate average
    return total_attempts / expected_count

def get_default_routing_path(jsonl_path):
    """
    Calculate default routing decisions path based on the jsonl path
    """
    # Implementation depends on your directory structure
    # This is a placeholder - modify as needed
    base_dir = os.path.dirname(jsonl_path)
    return os.path.join(base_dir, "routing_decisions.jsonl")

def main():
    parser = argparse.ArgumentParser(description='Calculate average attempts from JSONL files')
    parser.add_argument('--jsonl_path', type=str, help='Path to a single JSONL file')
    parser.add_argument('--strong_weak', action="store_true", help='Get strong-weak attempts')
    parser.add_argument('--router', action="store_true", help='Use routing decisions for missing entries')
    parser.add_argument('--routing_decisions_path', type=str, help='Path to the routing decisions JSONL')
    parser.add_argument('--best_of_n', action="store_true", help='Use best-of-n approach with multiple temp_*.jsonl files')
    parser.add_argument('--dir', type=str, help='Directory containing temp_*.jsonl files for best-of-n approach')
    args = parser.parse_args()
    
    # Validate arguments
    if args.best_of_n and not args.dir:
        print("Error: --dir is required when using --best_of_n")
        return
    
    if not args.best_of_n and not args.jsonl_path:
        print("Error: --jsonl_path is required when not using --best_of_n")
        return
    
    # Determine routing decisions path if needed
    routing_decisions_path = None
    if args.router:
        if args.routing_decisions_path:
            routing_decisions_path = args.routing_decisions_path
        else:
            if args.best_of_n:
                routing_decisions_path = os.path.join(args.dir, "routing_decisions.jsonl")
            else:
                routing_decisions_path = get_default_routing_path(args.jsonl_path)
            print(f"Using default routing decisions path: {routing_decisions_path}")
    
    if args.best_of_n:
        # Process multiple temp_*.jsonl files in directory
        # Note: strong_weak is ignored in best_of_n mode
        if args.strong_weak:
            print("Note: --strong_weak is ignored in --best_of_n mode")
            
        avg_attempts = calculate_best_of_n_attempts(
            args.dir,
            router=args.router,
            routing_decisions_path=routing_decisions_path
        )
        if avg_attempts is not None:
            print(f"Average attempts (best-of-n): {avg_attempts:.2f}")
    else:
        # Process single JSONL file
        if args.strong_weak:
            avg_strong_attempts, avg_weak_attempts = calculate_average_attempts(
                args.jsonl_path, 
                strong_weak=True, 
                router=args.router, 
                routing_decisions_path=routing_decisions_path
            )    
            if avg_strong_attempts is not None and avg_weak_attempts is not None:
                print(f"Average strong attempts: {avg_strong_attempts:.2f}")
                print(f"Average weak attempts: {avg_weak_attempts:.2f}")
        else:
            avg_attempts = calculate_average_attempts(
                args.jsonl_path,
                router=args.router,
                routing_decisions_path=routing_decisions_path
            )
            if avg_attempts is not None:
                print(f"Average attempts: {avg_attempts:.2f}")

if __name__ == "__main__":
    main()