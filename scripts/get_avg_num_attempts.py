import json
import argparse
import os

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

def main():
    parser = argparse.ArgumentParser(description='Calculate average attempts from a JSONL file')
    parser.add_argument('--jsonl_path', type=str, required=True, help='Path to the JSONL file')
    parser.add_argument('--strong_weak', action="store_true", help='Get strong-weak attempts')
    parser.add_argument('--router', action="store_true", help='Use routing decisions for missing entries')
    parser.add_argument('--routing_decisions_path', type=str, help='Path to the routing decisions JSONL')
    args = parser.parse_args()
    
    # Determine routing decisions path if needed
    routing_decisions_path = None
    if args.router:
        if args.routing_decisions_path:
            routing_decisions_path = args.routing_decisions_path
        else:
            routing_decisions_path = get_default_routing_path(args.jsonl_path)
            print(f"Using default routing decisions path: {routing_decisions_path}")
    
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