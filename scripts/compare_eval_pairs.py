import json
from collections import Counter

def compare_reports(file1, file2):
    # Load the two JSON reports
    with open(file1, 'r') as f1:
        report1 = json.load(f1)
    
    with open(file2, 'r') as f2:
        report2 = json.load(f2)
    
    # Extract resolved instance IDs from both reports
    resolved1 = set(report1["resolved_ids"])
    resolved2 = set(report2["resolved_ids"])
    
    # Calculate different categories
    resolved_both = resolved1.intersection(resolved2)
    resolved_only_report1 = resolved1 - resolved2
    resolved_only_report2 = resolved2 - resolved1
    
    # Get repository statistics
    repos1 = Counter([id.split('__')[0] for id in resolved1])
    repos2 = Counter([id.split('__')[0] for id in resolved2])
    repos_both = Counter([id.split('__')[0] for id in resolved_both])
    repos_only1 = Counter([id.split('__')[0] for id in resolved_only_report1])
    repos_only2 = Counter([id.split('__')[0] for id in resolved_only_report2])
    
    # Print results
    print(f"Total instances with gpt-4o-mini without file contents: {report1['total_instances']}")
    print(f"Total instances with gpt-4o-mini: {report2['total_instances']}")
    print(f"Completed instances (was able to generate valid patch) with gpt-4o-mini without file contents: {report1['completed_instances']}")
    print(f"Completed instances (was able to generate valid patch) with gpt-4o-mini: {report2['completed_instances']}")
    print(f"Resolved instances with gpt-4o-mini without file contents: {len(resolved1)}")
    print(f"Resolved instances with gpt-4o-mini: {len(resolved2)}")
    print(f"Instances resolved in both reports: {len(resolved_both)}")
    print(f"Instances resolved only with gpt-4o-mini without file contents: {resolved_only_report1}")
    print(f"Instances resolved only with gpt-4o-mini: {len(resolved_only_report2)}")
    
    print("\nRepository breakdown for instances resolved in both reports:")
    for repo, count in sorted(repos_both.items(), key=lambda x: x[1], reverse=True):
        print(f"  {repo}: {count}")
    
    print("\nRepository breakdown for instances resolved only with gpt-4o-mini without file contents:")
    for repo, count in sorted(repos_only1.items(), key=lambda x: x[1], reverse=True):
        print(f"  {repo}: {count}")
    
    print("\nRepository breakdown for instances resolved only with gpt-4o-mini:")
    for repo, count in sorted(repos_only2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {repo}: {count}")

# Call the function with your file paths
compare_reports('sb-cli-reports/swe-bench_lite__test__agentless_lite_fs_similarity_successful_5_gpt-4o-mini-2024-07-18_1.json', 'sb-cli-reports/swe-bench_lite__test__agentless_lite_fs_with_window_similarity_successful_5_gpt-4o-mini-2024-07-18_1.json')