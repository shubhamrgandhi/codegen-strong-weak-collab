import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from agentless_lite.util.backends import get_generator
from agentless_lite.util.repair import num_tokens_from_messages

# Define base directory for storing repos and insights
BASE_DIR = Path("../data/swebench_repos")
INSIGHTS_DIR = Path("../data/repo_insights")

REPO_INSIGHT_PROMPT = """I need you to provide high-level insights about the following repository: {repo_name}

Based on the repository structure and README below, generate a comprehensive overview of this repository that could help guide a language model in solving technical issues.

Repository Structure:
{repo_structure}

README Content:
{readme_content}

Please provide the following insights. For each point, provide concrete details and specific examples from the codebase - high-level doesn't mean vague, it means providing a clear architectural overview with specific names, patterns, and implementations:

1. Core Purpose and Functionality: 
    - What specific problem does this repository solve?
    - What are its primary features and capabilities?

2. Main Architectural Patterns:
    - Identify concrete architectural patterns used in this codebase
    - EXAMPLE: Plugin based architecture, layered architecture, etc

3. Module Organization:
    - Name the specific key modules and their exact responsibilities
    - EXAMPLE: I/O module, error-handling module, etc

4. Key Abstractions and Concepts:
    - List the actual fundamental abstractions used in the codebase
    - EXAMPLE: Quantity class for numerical values, Logger class for logging, etc

5. Design Patterns:
    - Identify specific recurring code patterns with examples
    - EXAMPLE: Factory methods, Decorators, etc

6. Error Handling Approaches:
    - Describe precise error handling mechanisms used in the codebase
    - EXAMPLE: Custom exception hierarchies, warnings, etc

Focus on providing actionable architectural insights that would be valuable for understanding the repository's design philosophy and core abstractions. Your response should contain specific implementation details that would help someone understand how to navigate, extend, and debug the codebase to solve issues.
"""

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
    """Process a repository to generate high-level insights."""
    print(f"Processing repository: {repo_name}")
    
    # Extract repository structure using the new functions (directories only)
    repo_structure = get_repo_structure(repo_path)
    formatted_structure = format_repo_structure(repo_structure)
    
    # Debug info
    print(f"Repository structure extraction results for {repo_name}:")
    print(f"- Path exists: {os.path.exists(repo_path)}")
    print(f"- Path is directory: {os.path.isdir(repo_path)}")
    
    readme_content = get_repo_readme(repo_path)
    
    # Create the prompt (using the original prompt)
    prompt = REPO_INSIGHT_PROMPT.format(
        repo_name=repo_name,
        repo_structure=formatted_structure,
        readme_content=readme_content
    )
    
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
        
        # Create a shortened prompt
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
        "instance_id": f"repo_insights_{repo_name}",
        "problem_description": f"Generate high-level insights for repository: {repo_name}"
    }
    
    # Generate insights
    insights, output_entry = generator.generate_plan(
        instance,
        prompt,
        args,
        file_lock,
        args.output_file,
        defer_writing=True
    )
    
    # Save insights to repository-specific file
    output_path = os.path.join(INSIGHTS_DIR, f"{repo_name}_insights.json")
    repo_output = {
        "repo_name": repo_name,
        "structure": formatted_structure,  # Store the full formatted structure
        "readme": readme_content,
        "insights": insights
    }
    
    with file_lock:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(repo_output, f, indent=2)
    
    print(f"Insights for {repo_name} saved to {output_path}")
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

def generate_repo_insights(args):
    """Generate insights for all repositories."""
    file_lock = threading.Lock()
    
    # Create directories if they don't exist
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get repositories
    repositories = get_repositories_from_swebench()
    if args.repo_name:
        repositories = [repo for repo in repositories if args.repo_name in repo]
    
    all_insights = {}
    
    if args.num_threads == 1:
        for repo_full_name in repositories:
            repo_name = repo_full_name.split('/')[1]
            repo_path = clone_repository(repo_full_name)
            if repo_path:
                insights = process_repository(repo_name, repo_path, args, file_lock)
                all_insights[repo_name] = insights
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
                    all_insights[result["repo_name"]] = result
    
    # Save combined insights
    combined_path = INSIGHTS_DIR / "all_repo_insights.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_insights, f, indent=2)
    
    print(f"All repository insights saved to {combined_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate high-level repository insights for SWE-bench repositories"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="repo_insights.jsonl",
        help="Path to save the generated insights",
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
        default=4000,
        help="Maximum number of tokens allowed in the completion response",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for generating insights",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        help="Filter to specific repository (e.g., 'django')",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
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
        default="../data/repo_insights",
        help="Folder to save the output files",
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
    
    log_file = os.path.join(args.output_folder, "logs", "repo_insights_parameters.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        args_dict = {
            k: v
            for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, list, dict))
        }
        json.dump(args_dict, f, indent=2)
    
    generate_repo_insights(args)
    print("Finished generating insights for all repositories.")