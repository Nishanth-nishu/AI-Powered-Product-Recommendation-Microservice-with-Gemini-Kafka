"""
Product Service - Fetches real product data from external APIs
Integrates with DummyJSON API for demo purposes (can be replaced with real e-commerce APIs)
"""
import logging
import httpx
from typing import List, Dict, Optional
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProductService:
    """
    Service to fetch and manage real product data
    Uses DummyJSON API as example (can integrate with Amazon, eBay, etc.)
    """
    
    def __init__(self):
        self.base_url = "https://dummyjson.com"
        self.cache: Dict[str, Dict] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(hours=1)
        self.timeout = httpx.Timeout(10.0)
    
    async def search_products(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for products by query
        
        Args:
            query: Search query (e.g., "laptop", "phone")
            limit: Maximum number of results
        
        Returns:
            List of product dictionaries
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/products/search",
                    params={"q": query, "limit": limit}
                )
                response.raise_for_status()
                data = response.json()
                
                products = data.get("products", [])
                logger.info(f"Found {len(products)} products for query: {query}")
                
                # Cache products
                for product in products:
                    self._cache_product(product)
                
                return products
                
        except Exception as e:
            logger.error(f"Error searching products: {str(e)}")
            return []
    
    async def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """
        Get detailed product information by ID
        
        Args:
            product_id: Product ID
        
        Returns:
            Product dictionary or None
        """
        # Check cache first
        cached = self._get_from_cache(product_id)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/products/{product_id}")
                response.raise_for_status()
                product = response.json()
                
                self._cache_product(product)
                return product
                
        except Exception as e:
            logger.error(f"Error fetching product {product_id}: {str(e)}")
            return None
    
    async def get_products_by_category(
        self,
        category: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get products by category
        
        Args:
            category: Category name (e.g., "smartphones", "laptops")
            limit: Maximum number of results
        
        Returns:
            List of product dictionaries
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/products/category/{category}",
                    params={"limit": limit}
                )
                response.raise_for_status()
                data = response.json()
                
                products = data.get("products", [])
                logger.info(f"Found {len(products)} products in category: {category}")
                
                for product in products:
                    self._cache_product(product)
                
                return products
                
        except Exception as e:
            logger.error(f"Error fetching category {category}: {str(e)}")
            return []
    
    async def get_all_categories(self) -> List[str]:
        """
        Get all available product categories
        
        Returns:
            List of category names
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/products/categories")
                response.raise_for_status()
                categories = response.json()
                
                logger.info(f"Found {len(categories)} categories")
                return categories
                
        except Exception as e:
            logger.error(f"Error fetching categories: {str(e)}")
            return []
    
    async def get_random_products(self, limit: int = 10) -> List[Dict]:
        """
        Get random products for cold start
        
        Args:
            limit: Maximum number of products
        
        Returns:
            List of product dictionaries
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/products",
                    params={"limit": limit}
                )
                response.raise_for_status()
                data = response.json()
                
                products = data.get("products", [])
                
                for product in products:
                    self._cache_product(product)
                
                return products
                
        except Exception as e:
            logger.error(f"Error fetching random products: {str(e)}")
            return []
    
    async def enrich_product_data(self, product_id: str) -> Dict:
        """
        Get enriched product data with all details
        
        Args:
            product_id: Product ID
        
        Returns:
            Enriched product dictionary
        """
        product = await self.get_product_by_id(product_id)
        
        if not product:
            return {
                "id": product_id,
                "title": f"Product {product_id}",
                "description": "No description available",
                "price": 0,
                "category": "unknown",
                "brand": "unknown",
                "rating": 0,
                "thumbnail": "",
                "images": []
            }
        
        return {
            "id": str(product.get("id", product_id)),
            "title": product.get("title", "Unknown Product"),
            "description": product.get("description", ""),
            "price": product.get("price", 0),
            "category": product.get("category", "unknown"),
            "brand": product.get("brand", "unknown"),
            "rating": product.get("rating", 0),
            "stock": product.get("stock", 0),
            "thumbnail": product.get("thumbnail", ""),
            "images": product.get("images", []),
            "discount": product.get("discountPercentage", 0)
        }
    
    def _cache_product(self, product: Dict):
        """Cache product data"""
        product_id = str(product.get("id"))
        self.cache[product_id] = product
        self.cache_expiry[product_id] = datetime.utcnow() + self.cache_duration
    
    def _get_from_cache(self, product_id: str) -> Optional[Dict]:
        """Get product from cache if not expired"""
        if product_id in self.cache:
            if datetime.utcnow() < self.cache_expiry.get(product_id, datetime.utcnow()):
                logger.debug(f"Cache hit for product {product_id}")
                return self.cache[product_id]
            else:
                # Expired, remove from cache
                del self.cache[product_id]
                del self.cache_expiry[product_id]
        
        return None
    
    async def search_and_recommend(
        self,
        user_history: List[Dict],
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for products similar to user's history
        
        Args:
            user_history: List of user's interacted products
            limit: Number of recommendations
        
        Returns:
            List of recommended products with metadata
        """
        if not user_history:
            # Cold start - return random products
            return await self.get_random_products(limit)
        
        # Extract categories and brands from user history
        categories = set()
        brands = set()
        
        for interaction in user_history[-5:]:  # Last 5 interactions
            product_id = interaction.get("product_id")
            product = await self.get_product_by_id(product_id)
            
            if product:
                categories.add(product.get("category", ""))
                brands.add(product.get("brand", ""))
        
        # Search for similar products
        recommendations = []
        seen_ids = set(i.get("product_id") for i in user_history)
        
        # Search by categories
        for category in categories:
            if category:
                products = await self.get_products_by_category(category, limit=5)
                for product in products:
                    product_id = str(product.get("id"))
                    if product_id not in seen_ids:
                        recommendations.append(product)
                        seen_ids.add(product_id)
                    
                    if len(recommendations) >= limit:
                        break
            
            if len(recommendations) >= limit:
                break
        
        # Fill with random if needed
        if len(recommendations) < limit:
            random_products = await self.get_random_products(limit * 2)
            for product in random_products:
                product_id = str(product.get("id"))
                if product_id not in seen_ids:
                    recommendations.append(product)
                    seen_ids.add(product_id)
                
                if len(recommendations) >= limit:
                    break
        
        return recommendations[:limit]


# Global instance
product_service = ProductService()
