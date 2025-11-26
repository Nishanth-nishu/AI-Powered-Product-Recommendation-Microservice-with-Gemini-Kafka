"""
Enhanced Demo with Real Product Data
Demonstrates the recommendation system with actual products
"""
import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_product(product, score=None, reason=None):
    """Pretty print product"""
    print(f"\n {product.get('title', 'Unknown')}")
    print(f"   ID: {product.get('product_id', 'N/A')}")
    print(f"   Category: {product.get('category', 'N/A')}")
    print(f"   Brand: {product.get('brand', 'N/A')}")
    print(f"   Price: ${product.get('price', 0)}")
    print(f"   Rating: {product.get('rating', 0)}")
    if score:
        print(f"   Match Score: {score:.2f}")
    if reason:
        print(f"   Why: {reason}")


def main():
    print(" Real Product Recommendation Demo")
    print("="*70)
    
    # 1. Check service health
    print_section("1️ Checking Service Health")
    response = requests.get(f"{BASE_URL}/health")
    health = response.json()
    print(f"Status: {health['status']}")
    print(f"Gemini Enabled: {health.get('gemini_enabled', False)}")
    
    # 2. Get available categories
    print_section("2️ Fetching Available Categories")
    response = requests.get(f"{BASE_URL}/api/v1/categories")
    categories = response.json()['categories']
    print(f"Found {len(categories)} categories:")
    for cat in categories[:10]:
        print(f"  • {cat}")
    
    # 3. Search for products
    print_section("3️ Searching for Products")
    
    searches = ["laptop", "phone", "watch"]
    all_products = {}
    
    for query in searches:
        response = requests.get(
            f"{BASE_URL}/api/v1/products/search",
            params={"query": query, "limit": 5}
        )
        products = response.json()['products']
        all_products[query] = products
        
        print(f"\n Search: '{query}' - Found {len(products)} products")
        for product in products[:3]:
            print(f"  • {product['title']} (${product['price']})")
    
    # 4. Simulate user (Alice) shopping journey
    print_section("4️ Simulating Alice's Shopping Journey")
    
    alice_id = "alice_123"
    
    # Alice browses laptops
    print("\n👤 Alice is browsing laptops...")
    laptop_products = all_products.get("laptop", [])
    
    if laptop_products:
        # View first laptop
        laptop = laptop_products[0]
        print(f"   Views: {laptop['title']}")
        requests.post(f"{BASE_URL}/api/v1/interactions", json={
            "user_id": alice_id,
            "product_id": str(laptop['id']),
            "interaction_type": "view"
        })
        time.sleep(0.5)
        
        # Click on it
        print(f"   Clicks on: {laptop['title']}")
        requests.post(f"{BASE_URL}/api/v1/interactions", json={
            "user_id": alice_id,
            "product_id": str(laptop['id']),
            "interaction_type": "click"
        })
        time.sleep(0.5)
        
        # Purchase it
        print(f" Purchases: {laptop['title']} for ${laptop['price']}")
        requests.post(f"{BASE_URL}/api/v1/interactions", json={
            "user_id": alice_id,
            "product_id": str(laptop['id']),
            "interaction_type": "purchase"
        })
        time.sleep(0.5)
        
        # View some phones
        print("\n   Browses phones...")
        phone_products = all_products.get("phone", [])
        if phone_products:
            for phone in phone_products[:2]:
                print(f"   Views: {phone['title']}")
                requests.post(f"{BASE_URL}/api/v1/interactions", json={
                    "user_id": alice_id,
                    "product_id": str(phone['id']),
                    "interaction_type": "view"
                })
                time.sleep(0.3)
    
    # Wait for processing
    print("\nProcessing interactions...")
    time.sleep(2)
    
    # 5. Get personalized recommendations for Alice
    print_section("5️ Personalized Recommendations for Alice")
    
    response = requests.post(f"{BASE_URL}/api/v1/recommendations", json={
        "user_id": alice_id,
        "limit": 5
    })
    
    recommendations = response.json()
    print(f"\n Based on Alice's shopping behavior:")
    print(f"   • Purchased: 1 laptop")
    print(f"   • Viewed: Multiple electronics")
    print(f"\n Top {len(recommendations['recommendations'])} Recommendations:\n")
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        print(f"{i}. ", end="")
        print_product(rec, rec['score'], rec['reason'])
    
    # 6. Simulate user (Bob) with different preferences
    print_section("6️ Simulating Bob's Shopping Journey")
    
    bob_id = "bob_456"
    print("\n👤 Bob is interested in watches and accessories...")
    
    watch_products = all_products.get("watch", [])
    if watch_products:
        for watch in watch_products[:3]:
            print(f"   Likes: {watch['title']}")
            requests.post(f"{BASE_URL}/api/v1/interactions", json={
                "user_id": bob_id,
                "product_id": str(watch['id']),
                "interaction_type": "like"
            })
            time.sleep(0.3)
    
    time.sleep(2)
    
    # 7. Get recommendations for Bob
    print_section("7️ Personalized Recommendations for Bob")
    
    response = requests.post(f"{BASE_URL}/api/v1/recommendations", json={
        "user_id": bob_id,
        "limit": 5
    })
    
    recommendations = response.json()
    print(f"\n🎯 Based on Bob's interests:\n")
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        print(f"{i}. ", end="")
        print_product(rec, rec['score'], rec['reason'])
    
    # 8. Cold start - new user
    print_section("8️ Cold Start - New User Recommendations")
    
    response = requests.post(f"{BASE_URL}/api/v1/recommendations", json={
        "user_id": "new_user_789",
        "limit": 5
    })
    
    recommendations = response.json()
    print(f"\n New user with no history")
    print(f"Popular Products:\n")
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        print(f"{i}. ", end="")
        print_product(rec, rec['score'], rec['reason'])
    
    # 9. Product details
    print_section("9️Detailed Product Information")
    
    if laptop_products:
        product_id = laptop_products[0]['id']
        response = requests.get(f"{BASE_URL}/api/v1/products/{product_id}")
        product = response.json()
        
        print(f"\nProduct Details:")
        print(f"   Title: {product['title']}")
        print(f"   Description: {product['description'][:100]}...")
        print(f"   Price: ${product['price']}")
        print(f"   Category: {product['category']}")
        print(f"   Brand: {product['brand']}")
        print(f"   Rating: {product['rating']}")
        print(f"   Stock: {product['stock']} units")
        print(f"   Discount: {product.get('discountPercentage', 0)}%")
    
    # 10. Service statistics
    print_section("Service Statistics")
    
    response = requests.get(f"{BASE_URL}/api/v1/stats")
    stats = response.json()
    
    print(f"\nSystem Performance:")
    print(f"   Total Interactions: {stats['total_interactions']}")
    print(f"   Unique Users: {stats['unique_users']}")
    print(f"   Model Version: {stats['model_version']}")
    print(f"   Gemini Enabled: {stats.get('gemini_enabled', False)}")
    
    # Summary
    print_section("Demo Complete!")
    print("""
What we demonstrated:
   ✓ Real product search and discovery
   ✓ User interaction tracking
   ✓ Personalized AI recommendations
   ✓ Different user preferences
   ✓ Cold start handling
   ✓ Rich product metadata

Next Steps:
   • View API docs: http://localhost:8000/docs
   • Try different searches: /api/v1/products/search?query=YOUR_QUERY
   • Explore categories: /api/v1/categories
   • Get product details: /api/v1/products/{id}
   • Build your own shopping journey!

Real-world applications:
   • E-commerce platforms
   • Product discovery systems
   • Personalized shopping experiences
   • Cross-sell and upsell features
    """)
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n Error: Could not connect to the service.")
        print("Please make sure the service is running:")
        print("  docker-compose up --build")
        print("  OR")
        print("  uvicorn main:app --reload")
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
