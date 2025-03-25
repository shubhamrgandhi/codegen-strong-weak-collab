import json
import os
import re
import argparse
from pathlib import Path
import html

def load_jsonl(file_path):
    """Load a JSONL file into a dictionary keyed by instance_id."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data[entry['instance_id']] = entry
    return data

def parse_patch(patch_text):
    """Parse a git patch to extract core changes with minimal context."""
    if not patch_text or patch_text.strip() == "":
        return {"file_path": "unknown", "changes": []}
    
    lines = patch_text.strip().split('\n')
    
    # Extract file path
    file_path = "unknown"
    for line in lines:
        if line.startswith('diff --git'):
            match = re.search(r'a/(.*) b/(.*)', line)
            if match:
                file_path = match.group(2)
                break
    
    # Extract just the essential changes
    changes = []
    additions = []
    removals = []
    
    in_hunk = False
    for line in lines:
        if line.startswith('@@'):
            in_hunk = True
            continue
        
        if not in_hunk:
            continue
        
        if line.startswith('+') and not line.startswith('+++'):
            additions.append(line[1:])
        elif line.startswith('-') and not line.startswith('---'):
            removals.append(line[1:])
    
    return {
        "file_path": file_path,
        "removals": removals,
        "additions": additions
    }

def create_html_slide(instance_id, gold_file, model_file, output_dir="slides"):
    """Create an HTML slide with a three-column layout for easy inclusion in presentations."""
    # Load data
    gold_data = load_jsonl(gold_file)
    model_data = load_jsonl(model_file)
    
    # Check if instance_id exists in both files
    if instance_id not in gold_data:
        print(f"Error: Instance ID '{instance_id}' not found in gold patch file.")
        return None
    
    if instance_id not in model_data:
        print(f"Error: Instance ID '{instance_id}' not found in model patch file.")
        return None
    
    # Get patches
    gold_patch = gold_data[instance_id]['patch']
    model_patch = model_data[instance_id]['model_patch']
    
    # Parse patches
    gold_parsed = parse_patch(gold_patch)
    model_parsed = parse_patch(model_patch)
    
    # Extract the key components
    buggy_code = [html.escape(line) for line in gold_parsed["removals"]]
    gold_solution = [html.escape(line) for line in gold_parsed["additions"]]
    model_solution = [html.escape(line) for line in model_parsed["additions"]]
    
    # Create HTML for a slide-friendly layout
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Patch Comparison: {instance_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        h1 {{
            font-size: 24px;
            margin-bottom: 20px;
        }}
        .container {{
            display: flex;
            justify-content: space-between;
        }}
        .column {{
            width: 32%;
            padding: 10px;
            border-radius: 5px;
        }}
        .buggy {{
            background-color: #ffdddd;
        }}
        .gold {{
            background-color: #ddffdd;
        }}
        .model {{
            background-color: #ddddff;
        }}
        pre {{
            white-space: pre-wrap;
            margin: 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}
        .column-title {{
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #ccc;
        }}
        .file-path {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            margin-bottom: 10px;
        }}
        code {{
            background-color: rgba(0,0,0,0.05);
            padding: 2px 4px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <h1>Patch Comparison: {instance_id}</h1>
    
    <div class="container">
        <div class="column buggy">
            <div class="column-title">Original Code (Bug)</div>
            <div class="file-path">File: {gold_parsed["file_path"]}</div>
            <pre>"""
    
    for line in buggy_code:
        html_content += f"{line}\n"
    
    html_content += """</pre>
        </div>
        
        <div class="column gold">
            <div class="column-title">Gold Solution</div>
            <div class="file-path">File: """ + gold_parsed["file_path"] + """</div>
            <pre>"""
    
    for line in gold_solution:
        html_content += f"{line}\n"
    
    html_content += """</pre>
        </div>
        
        <div class="column model">
            <div class="column-title">LLM-Generated Solution</div>
            <div class="file-path">File: """ + model_parsed["file_path"] + """</div>
            <pre>"""
    
    # Check each line of model solution against gold solution
    for line in model_solution:
        if line in gold_solution:
            html_content += f"<span style='color: green;'>{line} ✓</span>\n"
        else:
            html_content += f"{line}\n"
    
    html_content += """</pre>
        </div>
    </div>
</body>
</html>"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create safe filename
    safe_id = instance_id.replace('/', '_').replace('\\', '_')
    output_file = os.path.join(output_dir, f"{safe_id}.html")
    
    # Write HTML to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Slide created: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Create slides from patch comparisons")
    parser.add_argument("instance_id", help="Instance ID to create slide for")
    parser.add_argument("gold_file", help="Path to the gold patches JSONL file")
    parser.add_argument("model_file", help="Path to the model predictions JSONL file")
    parser.add_argument("--output-dir", "-o", default="slides", help="Output directory for slides")
    
    args = parser.parse_args()
    
    create_html_slide(args.instance_id, args.gold_file, args.model_file, args.output_dir)

if __name__ == "__main__":
    main()