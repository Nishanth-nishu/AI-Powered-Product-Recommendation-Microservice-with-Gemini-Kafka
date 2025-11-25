"""
Example usage of the Recommendation Microservice API
Run this after starting the service to see it in action
"""
import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def print_response(title, response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def main():
    """Demonstrate the API functionality"""
    
    print("🎯 Product Recommendation Microservice - Demo")
    print("=" * 60)
    
    # 1. Check service health
    print("\n1️⃣ Checking service health...")
    response = requests.get(f"{BASE_URL}/health")
    print_response("Health Check", response)
    
    # 2. Track user interactions
    print("\n2️⃣ Tracking user interactions...")
    
    interactions = [
        {"user_id": "alice", "product_id": "laptop_123", "interaction_type": "view"},
        {"user_id": "alice", "product_id": "laptop_123", "interaction_type": "click"},
        {"user_id": "alice", "product_id": "mouse_456", "interaction_type": "view"},
        {"user_id": "bob", "product_id": "laptop_123", "interaction_type": "view"},
        {"user_id": "bob", "product_id": "keyboard_789", "interaction_type": "purchase"},
        {"user_id": "charlie", "product_id": "mouse_456", "interaction_type": "view"},
        {"user_id": "charlie", "product_id": "laptop_123", "interaction_type": "add_to_cart"},
    ]
    
    for interaction in interactions:
        response = requests.post(f"{BASE_URL}/api/v1/interactions", json=interaction)
        print(f"  ✓ Tracked: {interaction['user_id']} → {interaction['product_id']} ({interaction['interaction_type']})")
        time.sleep(0.2)  # Small delay to process events
    
    # Wait for processing
    print("\n⏳ Waiting for events to be processed...")
    time.sleep(2)
    
    # 3. Get service statistics
    print("\n3️⃣ Getting service statistics...")
    response = requests.get(f"{BASE_URL}/api/v1/stats")
    print_response("Service Statistics", response)
    
    # 4. Trigger model retraining
    print("\n4️⃣ Triggering model retraining...")
    response = requests.post(f"{BASE_URL}/api/v1/retrain")
    print_response("Model Retraining", response)
    
    # Wait for retraining
    time.sleep(1)
    
    # 5. Get recommendations for Alice
    print("\n5️⃣ Getting recommendations for Alice...")
    recommendation_request = {
        "user_id": "alice",
        "limit": 5,
        "exclude_products": []
    }
    response = requests.post(f"{BASE_URL}/api/v1/recommendations", json=recommendation_request)
    print_response("Recommendations for Alice", response)
    
    # 6. Get recommendations for Bob
    print("\n6️⃣ Getting recommendations for Bob...")
    recommendation_request = {
        "user_id": "bob",
        "limit": 3,
        "exclude_products": []
    }
    response = requests.post(f"{BASE_URL}/api/v1/recommendations", json=recommendation_request)
    print_response("Recommendations for Bob", response)
    
    # 7. Get recommendations for new user
    print("\n7️⃣ Getting recommendations for new user (popular products)...")
    recommendation_request = {
        "user_id": "new_user_dave",
        "limit": 5
    }
    response = requests.post(f"{BASE_URL}/api/v1/recommendations", json=recommendation_request)
    print_response("Recommendations for New User", response)
    
    # 8. Test with exclusions
    print("\n8️⃣ Getting recommendations with exclusions...")
    recommendation_request = {
        "user_id": "alice",
        "limit": 5,
        "exclude_products": ["laptop_123"]
    }
    response = requests.post(f"{BASE_URL}/api/v1/recommendations", json=recommendation_request)
    print_response("Recommendations (excluding laptop_123)", response)
    
    print("\n" + "="*60)
    print("✅ Demo completed!")
    print("="*60)
    print("\n💡 Next steps:")
    print("  • View API docs: http://localhost:8000/docs")
    print("  • Check service stats: http://localhost:8000/api/v1/stats")
    print("  • Monitor logs for event processing")
    print()


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the service.")
        print("Please make sure the service is running:")
        print("  docker-compose up --build")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
