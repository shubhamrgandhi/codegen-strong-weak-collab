import requests
import os
import numpy as np

def test_voyage_api(model="voyage-code-3"):
    """
    Test the Voyage API key stored in VOYAGE_API_KEY environment variable
    by generating embeddings for a sample text.
    
    Args:
        model (str): The model to use, defaults to voyage-code-3
    
    Returns:
        bool: True if the API key is valid, False otherwise
    """
    # Get the API key from environment variable
    api_key = os.environ.get("VOYAGE_API_KEY")
    
    if not api_key:
        print("❌ VOYAGE_API_KEY environment variable not found")
        return False
    
    # Define the API endpoint for embeddings
    url = "https://api.voyageai.com/v1/embeddings"
    
    # Sample text to embed
    sample_text = "def hello_world():\n    print('Hello, World!')"
    
    # Prepare the request payload
    payload = {
        "model": model,
        "input": sample_text
    }
    
    # Set up headers with the API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        # Make the API request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            embedding = data.get("data", [{}])[0].get("embedding", [])
            
            if embedding:
                embedding_array = np.array(embedding)
                print(f"✅ API key is valid for model: {model}")
                print(f"✅ Successfully generated embedding with dimension: {len(embedding)}")
                print(f"✅ Sample values: {embedding[:5]}...")
                return True
            else:
                print(f"❌ API response did not contain valid embeddings")
                print(f"Response: {data}")
                return False
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"Error message: {response.text}")
            return False
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    # Use command-line argument for model if provided, otherwise default to voyage-code-3
    model = sys.argv[1] if len(sys.argv) > 1 else "voyage-code-3"
    
    test_voyage_api(model)