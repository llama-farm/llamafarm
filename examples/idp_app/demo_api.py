import requests
import json
import time

# Configuration
API_URL = "http://localhost:8000/v1/projects/finance-legal/idp-app-demo/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL = "idp-agent"

def run_scenario(name, document_text):
    print(f"\n--- Scenario: {name} ---")
    print(f"Document: {document_text[:50]}...")
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": f"Process this document: {document_text}"}
        ],
        "temperature": 0.1
    }
    
    try:
        start = time.time()
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        duration = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"Response ({duration:.2f}s):\n{content}")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    print("Running IDP API Demo...")
    
    # 1. Invoice
    run_scenario("Invoice Processing", """
    INVOICE #9923
    Vendor: Nvidia Corp
    Amount: $250,000.00
    Items: H100 GPU Cluster
    """)

    # 2. Anomaly
    run_scenario("Anomaly Detection", """
    UNKNOWN DATA STREAM
    x98989898
    Corrupted Header
    """)
