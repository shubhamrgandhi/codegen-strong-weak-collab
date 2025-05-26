#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def prep_dirs():
    # List of standard experiment names (using gpt-4o-mini-2024-07-18)
    standard_experiments = [
        "first_strong",
        "prompt_reduction",
        "fallback",
        "plan",
        "instance_faq",
        "router_weak",
        "router_strong",
        "fs_random_successful_1",
        "fs_random_successful_5",
        "fs_similarity_successful_1",
        "fs_similarity_successful_5",
        "repograph",
        "repo_faq",
        "info",
        "sc_direct",
        "sc_clustering",
        "sc_universal"
    ]
    # List of special experiment names (using o4-mini-2025-04-16)
    # special_experiments = ["generate_plan", "generate_instance_faq"]
    special_experiments = []
    
    # Base directory path
    base_dir = Path("results/base copy")
    
    # Check if base directory exists
    if not base_dir.exists():
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return
    
    # Process standard experiments
    for exp in standard_experiments:
        # Create target directory name with gpt-4o-mini-2024-07-18 suffix
        target_dir_name = f"{exp}_gpt-4o-mini-2024-07-18"
        target_dir = Path(f"results/{target_dir_name}")
        
        # Create copy directory name
        copy_dir_name = f"{exp}_gpt-4o-mini-2024-07-18 copy"
        copy_dir = Path(f"results/{copy_dir_name}")
        
        # Create directories
        create_experiment_dirs(exp, base_dir, target_dir, copy_dir)
    
    # Process special experiments
    for exp in special_experiments:
        # Create target directory name with o4-mini-2025-04-16 suffix
        target_dir_name = f"{exp}_o4-mini-2025-04-16"
        target_dir = Path(f"results/{target_dir_name}")
        
        # Create copy directory name
        copy_dir_name = f"{exp}_o4-mini-2025-04-16 copy"
        copy_dir = Path(f"results/{copy_dir_name}")
        
        # Create directories
        create_experiment_dirs(exp, base_dir, target_dir, copy_dir)

def create_experiment_dirs(exp, base_dir, target_dir, copy_dir):
    """Create the target directory and its copy for a given experiment."""
    # Create primary directory if it doesn't exist
    if target_dir.exists():
        print(f"Directory '{target_dir}' already exists. Skipping creation.")
    else:
        # Create parent directory if it doesn't exist
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the base directory to the new location
        print(f"Creating directory for experiment: {exp}")
        try:
            shutil.copytree(base_dir, target_dir)
            print(f"Successfully created: {target_dir}")
        except Exception as e:
            print(f"Error creating directory for {exp}: {e}")
            return  # Skip creating copy if primary fails
    
    # Create copy directory
    if copy_dir.exists():
        print(f"Directory '{copy_dir}' already exists. Skipping creation.")
    else:
        # Create copy of the target directory
        print(f"Creating copy directory for experiment: {exp}")
        try:
            shutil.copytree(target_dir, copy_dir)
            print(f"Successfully created copy: {copy_dir}")
        except Exception as e:
            print(f"Error creating copy directory for {exp}: {e}")

if __name__ == "__main__":
    prep_dirs()