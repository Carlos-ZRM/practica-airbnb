"""
Debug script to test vLLM endpoint connectivity
Run this to verify your endpoint and model configuration before using GrammarBot
"""

import requests
import json
from openai import OpenAI

# Configuration
ENDPOINT = "https://llama-32-3b-instruct-my-first-model.apps.ocp.gp6sl.sandbox2065.opentlc.com/v1"
MODEL_NAME = "llama-32-3b-instruct"
API_KEY = ""  # Replace with your actual API key

print("=" * 70)
print("vLLM Endpoint Connectivity Test")
print("=" * 70)

# Test 1: Check if endpoint is reachable
print("\n[Test 1] Checking endpoint connectivity...")
try:
    response = requests.get(
        f"{ENDPOINT}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=5
    )
    print(f"✓ Endpoint reachable (Status: {response.status_code})")
except requests.exceptions.RequestException as e:
    print(f"✗ Endpoint not reachable: {e}")
    exit(1)

# Test 2: Get list of available models
print("\n[Test 2] Fetching available models...")
try:
    response = requests.get(
        f"{ENDPOINT}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=5
    )
    if response.status_code == 200:
        models_data = response.json()
        if "data" in models_data:
            models = [m["id"] for m in models_data["data"]]
            print(f"✓ Found {len(models)} model(s):")
            for model in models:
                print(f"  - {model}")
            
            # Check if our model exists
            if MODEL_NAME in models:
                print(f"\n✓ Model '{MODEL_NAME}' is available!")
            else:
                print(f"\n✗ Model '{MODEL_NAME}' NOT found in available models")
                print(f"  → Use one of the models listed above instead")
        else:
            print(f"✗ Unexpected response format: {models_data}")
    else:
        print(f"✗ Failed to fetch models (Status: {response.status_code})")
        print(f"  Response: {response.text}")
except Exception as e:
    print(f"✗ Error fetching models: {e}")

# Test 3: Test actual API call
print("\n[Test 3] Testing actual API call...")
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=ENDPOINT
    )
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello'"}
        ],
        temperature=0.0,
        max_tokens=10
    )
    
    print(f"✓ API call successful!")
    print(f"  Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"✗ API call failed: {e}")
    print(f"  Error type: {type(e).__name__}")

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)