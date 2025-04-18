import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from agentless_lite.util.backends import get_generator
from agentless_lite.util.repair import num_tokens_from_messages
from agentless_lite.util.prompts import *

# Define base directory for storing repos and insights
BASE_DIR = Path("data/swebench_repos")
INSIGHTS_DIR = Path("data/repo_insights")
FAQ_DIR = Path("data/repo_faqs")



def get_repo_structure(repo_path):
    """
    Extract the complete repository structure at directory level only (no files).
    Returns a dictionary of directories and their subdirectories.
    """
    repo_path = Path(repo_path)
    if not repo_path.exists() or not repo_path.is_dir():
        print(f"Error: Repository path {repo_path} does not exist or is not a directory")
        return {}
    
    structure = {}
    
    # Walk through the directory tree
    for root, dirs, _ in os.walk(repo_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Create relative path from repository root
        rel_path = os.path.relpath(root, repo_path)
        if rel_path == '.':
            rel_path = ''
            
        # Add this path to structure
        if rel_path not in structure:
            structure[rel_path] = {'dirs': []}
            
        # Add directories only
        structure[rel_path]['dirs'] = [d for d in dirs]
    
    return structure

def format_repo_structure(structure):
    """
    Format the repository structure into a hierarchical tree-like representation.
    Directory structure only, no files.
    """
    if not structure:
        return "Repository structure could not be extracted"
    
    # Build a nested representation first
    tree = {}
    
    # Sort paths by depth and alphabetically
    paths = sorted(structure.keys(), key=lambda x: (len(x.split(os.sep)), x))
    
    for path in paths:
        if path == '':  # Root directory
            tree = {
                'dirs': {d: {} for d in structure[path]['dirs']}
            }
        else:
            # Navigate to the correct position in the tree
            parts = path.split(os.sep)
            current = tree
            
            # Navigate to the parent directory
            for part in parts[:-1]:
                if part in current['dirs']:
                    current = current['dirs'][part]
            
            # Add the current directory and its contents
            if len(parts) > 0 and parts[-1] in current['dirs']:
                current['dirs'][parts[-1]] = {
                    'dirs': {d: {} for d in structure[path]['dirs']}
                }
    
    # Now format the tree into a string
    lines = []
    
    def _format_tree(node, prefix='', is_last=True, is_root=True):
        if 'dirs' not in node.keys(): return ''
        elif is_root:
            lines.append(prefix)
            
            # Process directories in root
            dirs = sorted(node['dirs'].keys())
            for i, d in enumerate(dirs):
                is_last_dir = i == len(dirs) - 1
                child_prefix = prefix + ('└── ' if is_last_dir else '├── ')
                lines.append(f"{child_prefix}{d}")
                
                # Recursively process the child directory
                child_node = node['dirs'][d]
                child_prefix = prefix + ('    ' if is_last_dir else '│   ')
                _format_tree(child_node, child_prefix, is_last_dir, False)
        else:
            # Process subdirectories
            dirs = sorted(node['dirs'].keys())
            for i, d in enumerate(dirs):
                is_last_dir = i == len(dirs) - 1
                child_prefix = prefix + ('└── ' if is_last_dir else '├── ')
                lines.append(f"{child_prefix}{d}")
                
                # Recursively process the child directory
                child_node = node['dirs'][d]
                child_prefix = prefix + ('    ' if is_last_dir else '│   ')
                _format_tree(child_node, child_prefix, is_last_dir, False)
    
    _format_tree(tree)
    return '\n'.join(lines)


def get_repo_readme(repo_path):
    """Get the content of the README file."""
    readme_candidates = [
        os.path.join(repo_path, "README.md"),
        os.path.join(repo_path, "README.rst"),
        os.path.join(repo_path, "README.txt"),
        os.path.join(repo_path, "README")
    ]
    
    for readme_path in readme_candidates:
        if os.path.exists(readme_path):
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Trim if too long
                    if len(content) > 4000:
                        content = content[:4000] + "...[content truncated]"
                    return content
            except Exception as e:
                print(f"Error reading README at {readme_path}: {e}")
    
    return "README not found"

def process_repository(repo_name, repo_path, args, file_lock):
    """Process a repository to generate high-level insights or FAQs."""
    print(f"Processing repository: {repo_name}")
    
    # Extract repository structure using the new functions (directories only)
    repo_structure = get_repo_structure(repo_path)
    formatted_structure = format_repo_structure(repo_structure)
    
    # Debug info
    print(f"Repository structure extraction results for {repo_name}:")
    print(f"- Path exists: {os.path.exists(repo_path)}")
    print(f"- Path is directory: {os.path.isdir(repo_path)}")
    
    readme_content = get_repo_readme(repo_path)
    
    # Decide whether to generate insights or FAQs
    if args.info:
        # Create the prompt for insights
        prompt = REPO_INSIGHT_PROMPT.format(
            repo_name=repo_name,
            repo_structure=formatted_structure,
            readme_content=readme_content
        )
        
        output_dir = INSIGHTS_DIR
        instance_id_prefix = "repo_insights"
        problem_description = f"Generate high-level insights for repository: {repo_name}"
        
    elif args.faq:
        # Create the prompt for FAQs
        prompt = REPO_FAQ_PROMPT.format(
            repo_name=repo_name,
            repo_structure=formatted_structure,
            readme_content=readme_content
        )
        
        output_dir = FAQ_DIR
        instance_id_prefix = "repo_faq"
        problem_description = f"Generate repository-level FAQs for repository: {repo_name}"
    else:
        # Default to insights if neither is specified
        prompt = REPO_INSIGHT_PROMPT.format(
            repo_name=repo_name,
            repo_structure=formatted_structure,
            readme_content=readme_content
        )
        
        output_dir = INSIGHTS_DIR
        instance_id_prefix = "repo_insights"
        problem_description = f"Generate high-level insights for repository: {repo_name}"
    
    # Check if the prompt is too long
    if num_tokens_from_messages(prompt) > args.max_input_tokens:
        print(f"Warning: Prompt for {repo_name} exceeds max tokens. Truncating...")
        
        # For directory-only structure, we'll limit depth by removing deepest directories
        lines = formatted_structure.split('\n')
        # Keep lines with fewer indentation levels (less nested directories)
        # Count the number of leading spaces as a proxy for nesting level
        simplified_lines = []
        for line in lines:
            # Calculate indentation level (number of leading spaces divided by 4)
            indent_level = (len(line) - len(line.lstrip())) // 4
            if indent_level <= 3:  # Keep only directories up to level 3
                simplified_lines.append(line)
        
        simplified_structure = '\n'.join(simplified_lines)
        if not simplified_structure.strip():
            # If we removed everything, at least keep the first few levels
            simplified_structure = '\n'.join(lines[:20])
        
        # Truncate README if needed
        truncated_readme = readme_content[:2000] + "...[content truncated]" if len(readme_content) > 2000 else readme_content
        
        # Create a shortened prompt based on what we're generating
        if args.faq:
            prompt = REPO_FAQ_PROMPT.format(
                repo_name=repo_name,
                repo_structure=simplified_structure,
                readme_content=truncated_readme
            )
        else:
            prompt = REPO_INSIGHT_PROMPT.format(
                repo_name=repo_name,
                repo_structure=simplified_structure,
                readme_content=truncated_readme
            )
    
    # Get the appropriate generator
    generator = get_generator(args.backend)
    if not generator:
        raise ValueError(f"Unsupported backend: {args.backend}")
    
    # Initialize output files
    generator.initialize_output_files(args)
    
    # Create a mock instance to use existing generation code
    instance = {
        "instance_id": f"{instance_id_prefix}_{repo_name}",
        "problem_description": problem_description
    }
    
    # Generate content
    generated_content, output_entry = generator.generate_plan(
        instance,
        prompt,
        args,
        file_lock,
        args.output_file,
        defer_writing=True
    )
    
    # Save content to repository-specific file
    output_path = os.path.join(output_dir, f"{repo_name}_{instance_id_prefix}.json")
    field_name = 'repo_faq' if args.faq else 'insights'
    repo_output = {
        "repo_name": repo_name,
        "structure": formatted_structure,  # Store the full formatted structure
        "readme": readme_content,
        field_name: generated_content
    }
    
    with file_lock:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(repo_output, f, indent=2)
    
    print(f"Content for {repo_name} saved to {output_path}")
    return repo_output


def get_repositories_from_swebench():
    """Get list of repositories from SWE-bench Lite."""
    return [
        "astropy/astropy",
        "django/django",
        "matplotlib/matplotlib",
        "mwaskom/seaborn",
        "pallets/flask",
        "psf/requests",
        "pydata/xarray",
        "pylint-dev/pylint",
        "pytest-dev/pytest",
        "scikit-learn/scikit-learn",
        "sphinx-doc/sphinx",
        "sympy/sympy"
    ]

def clone_repository(repo_full_name):
    """Clone a repository if it doesn't exist."""
    repo_name = repo_full_name.split('/')[1]
    repo_path = BASE_DIR / repo_name
    
    if repo_path.exists():
        print(f"Repository {repo_name} already exists at {repo_path}")
        return repo_path
    
    print(f"Cloning repository: {repo_full_name}")
    try:
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo_full_name}.git", str(repo_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Successfully cloned {repo_full_name}")
        return repo_path
    except subprocess.CalledProcessError as e:
        print(f"Error cloning {repo_full_name}: {e}")
        return None

def generate_repo_content(args):
    """Generate insights or FAQs for all repositories."""
    file_lock = threading.Lock()
    
    # Create directories if they don't exist
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    FAQ_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine output directory based on what we're generating
    output_dir = FAQ_DIR if args.faq else INSIGHTS_DIR
    content_type = "faqs" if args.faq else "insights"
    
    # Get repositories
    repositories = get_repositories_from_swebench()
    if args.repo_name:
        repositories = [repo for repo in repositories if args.repo_name in repo]
    
    all_content = {}
    
    if args.num_threads == 1:
        for repo_full_name in repositories:
            repo_name = repo_full_name.split('/')[1]
            repo_path = clone_repository(repo_full_name)
            if repo_path:
                content = process_repository(repo_name, repo_path, args, file_lock)
                all_content[repo_name] = content
    else:
        # First clone all repositories
        repo_paths = {}
        for repo_full_name in repositories:
            repo_name = repo_full_name.split('/')[1]
            repo_path = clone_repository(repo_full_name)
            if repo_path:
                repo_paths[repo_name] = repo_path
        
        # Then process them in parallel
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [
                executor.submit(process_repository, repo_name, repo_path, args, file_lock)
                for repo_name, repo_path in repo_paths.items()
            ]
            for future in futures:
                result = future.result()
                if result:
                    all_content[result["repo_name"]] = result
    
    # Save combined content
    combined_path = output_dir / f"all_repo_{content_type}.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_content, f, indent=2)
    
    print(f"All repository {content_type} saved to {combined_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate high-level repository insights for SWE-bench repositories"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="repo_output.jsonl",
        help="Path to save the generated content",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="agentless_lite",
        help="Base directory path for the project",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=111000,
        help="Maximum number of tokens allowed in the input prompt",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=10000,
        help="Maximum number of tokens allowed in the completion response",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o3-mini",
        help="Model to use for generating insights or FAQs",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        help="Filter to specific repository (e.g., 'django')",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of concurrent threads for processing",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Maximum number of retries to generate a valid response",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "deepseek", "open_router"],
        help="The backend service to use for generation",
    )
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/repo_output",
        help="Folder to save the output files",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Generate high-level repository insights",
    )
    parser.add_argument(
        "--faq",
        action="store_true",
        help="Generate repository-level FAQs",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    args.testbed_dir = os.path.join(args.base_path, "testbed")
    
    # Create logs directory
    os.makedirs(os.path.join(args.output_folder, "logs"), exist_ok=True)
    args.output_file = os.path.join(
        args.output_folder, os.path.basename(args.output_file)
    )
    
    # Default to info if neither info nor faq is specified
    if not args.info and not args.faq:
        args.info = True
    
    # Set content type for log file name
    content_type = "faq" if args.faq else "insights"
    log_file = os.path.join(args.output_folder, "logs", f"repo_{content_type}_parameters.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        args_dict = {
            k: v
            for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, list, dict))
        }
        json.dump(args_dict, f, indent=2)
    
    generate_repo_content(args)
    print(f"Finished generating {content_type} for all repositories.")