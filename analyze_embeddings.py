import json

# Path to the JSON file
file_path = 'voyage_lite/astropy__astropy-6938/default__vector_store.json'

# Read and parse the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Variable to store the sum
total_length = 0

# Check if data is a dictionary with an embedding_dict key
if isinstance(data, dict) and "embedding_dict" in data:
    embedding_dict = data["embedding_dict"]
    # Iterate through all values in the dictionary
    for key, value in embedding_dict.items():
        if isinstance(value, list):
            total_length += len(value)
    
    print(f"Sum of lengths of all lists in embedding_dict: {total_length}")
    print(f"Number of key-value pairs: {len(embedding_dict)}")
else:
    # Print information about the structure to debug
    print(f"Data type: {type(data)}")
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
            # Try to inspect the first item if it might contain the embedding_dict
            if isinstance(data[0], dict):
                print(f"Keys in first item: {list(data[0].keys())}")