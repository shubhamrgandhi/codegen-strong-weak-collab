import json
from datasets import load_dataset
import os

# Load the SWE-bench Lite dataset
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

# Path to save the output JSONL file
output_file = "swebench_lite_instances.jsonl"

# Process dataset and save to JSONL
with open(output_file, "w") as f:
    for item in dataset:
        # Extract the instance_id and patch
        instance_data = {
            "instance_id": item["instance_id"],
            "patch": item["patch"]
        }
        
        # Write to JSONL file (one JSON object per line)
        f.write(json.dumps(instance_data) + "\n")

print(f"Successfully saved {len(dataset)} instances to {output_file}")

# Optional: Preview a few entries
print("\nPreview of the first 3 entries:")
with open(output_file, "r") as f:
    for _ in range(3):
        line = f.readline().strip()
        if line:
            print(line)