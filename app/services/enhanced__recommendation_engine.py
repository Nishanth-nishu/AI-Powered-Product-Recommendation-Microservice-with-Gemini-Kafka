"""
Enhanced Recommendation Engine with Real Product Integration
Combines Gemini AI with real product data for practical recommendations
"""
import logging
import json
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
import asyncio

from google import genai
from google.genai import types

from app.models.schemas import ProductRecommendation, UserInteractionEvent
from app.services.product_service import product_service
from app.config import settings

logger = logging.getLogger(__name__)


class EnhancedRecommendationEngine:
    """
    Enhanced recommendation engine with real product integration
    """
    
    def __init__(self):
        self.model_version = settings.MODEL_VERSION
        self.user_interactions: Dict[str, List[Dict]] = defaultdict(list)
        self.lock = asyncio.Lock()
        
        # Initialize Gemini client
        self.client: Optional[genai.Client] = None
        self._initialize_client()
        
        # Interaction weights for scoring
        self.interaction_weights = {
            "view": 1.0,
            "click": 2.0,
            "add_to_cart": 3.0,
            "like": 2.5,
            "share": 3.0,
            "purchase": 5.0
        }
    
    def _initialize_client(self):
        """Initialize Gemini client with API key"""
        api_key = settings.gemini_key
        
        if not api_key:
            logger.warning(
                "Gemini API key not found. Recommendations will use fallback logic."
            )
            return
        
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            self.client = None
    
    async def process_interaction(self, interaction: UserInteractionEvent):
        """Process a user interaction event"""
        async with self.lock:
            try:
                # Fetch real product data
                product_data = await product_service.get_product_by_id(interaction.product_id)
                
                interaction_data = {
                    "product_id": interaction.product_id,
                    "interaction_type": interaction.interaction_type,
                    "timestamp": interaction.timestamp.isoformat(),
                    "weight": self.interaction_weights.get(interaction.interaction_type, 1.0),
                    "product_data": product_data,
                    "metadata": interaction.metadata
                }
                
                self.user_interactions[interaction.user_id].append(interaction_data)
                
                logger.info(
                    f"Processed interaction: user={interaction.user_id}, "
                    f"product={interaction.product_id} ({product_data.get('title', 'Unknown') if product_data else 'Unknown'})"
                )
                
            except Exception as e:
                logger.error(f"Error processing interaction: {str(e)}")
                raise
    
    async def get_recommendations(
        self,
        user_id: str,
        limit: int = 5,
        exclude_products: Optional[List[str]] = None
    ) -> List[ProductRecommendation]:
        """
        Generate enhanced recommendations with real product data
        """
        exclude_products = exclude_products or []
        
        try:
            user_history = self.user_interactions.get(user_id, [])
            
            # Get product-based recommendations
            similar_products = await product_service.search_and_recommend(
                user_history,
                limit=limit * 2  # Get more for filtering
            )
            
            # Filter out excluded and already interacted products
            interacted_ids = set(i["product_id"] for i in user_history)
            available_products = [
                p for p in similar_products
                if str(p.get("id")) not in interacted_ids
                and str(p.get("id")) not in exclude_products
            ]
            
            # If user has enough history and Gemini is available, use AI
            if len(user_history) >= settings.MIN_INTERACTIONS_FOR_RECOMMENDATIONS and self.client:
                try:
                    recommendations = await self._get_gemini_recommendations(
                        user_history, available_products, limit
                    )
                    if recommendations:
                        return recommendations
                except Exception as e:
                    logger.warning(f"Gemini failed, using product-based: {str(e)}")
            
            # Fallback: Use product-based recommendations
            return await self._create_recommendations_from_products(
                available_products[:limit],
                "Similar to your interests"
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            # Return random products as last resort
            random_products = await product_service.get_random_products(limit)
            return await self._create_recommendations_from_products(
                random_products,
                "Popular products"
            )
    
    async def _get_gemini_recommendations(
        self,
        user_history: List[Dict],
        available_products: List[Dict],
        limit: int
    ) -> List[ProductRecommendation]:
        """Use Gemini AI to rank and explain recommendations"""
        try:
            # Create user profile
            user_profile = self._create_user_profile(user_history)
            
            # Create prompt with real product data
            prompt = self._create_enhanced_prompt(user_profile, available_products, limit)
            
            # Call Gemini API
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )
            
            # Parse and enrich recommendations
            return await self._parse_and_enrich_response(response.text, available_products)
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise
    
    def _create_user_profile(self, user_history: List[Dict]) -> Dict:
        """Create user profile from history"""
        categories = defaultdict(int)
        brands = defaultdict(int)
        price_range = []
        
        for interaction in user_history:
            product_data = interaction.get("product_data")
            if product_data:
                categories[product_data.get("category", "unknown")] += 1
                brands[product_data.get("brand", "unknown")] += 1
                price_range.append(product_data.get("price", 0))
        
        top_products = [
            {
                "title": i.get("product_data", {}).get("title", "Unknown"),
                "category": i.get("product_data", {}).get("category", "unknown"),
                "type": i.get("interaction_type")
            }
            for i in user_history[-5:]
            if i.get("product_data")
        ]
        
        return {
            "top_categories": dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]),
            "favorite_brands": dict(sorted(brands.items(), key=lambda x: x[1], reverse=True)[:3]),
            "avg_price": sum(price_range) / len(price_range) if price_range else 0,
            "recent_products": top_products,
            "total_interactions": len(user_history)
        }
    
    def _create_enhanced_prompt(
        self,
        user_profile: Dict,
        available_products: List[Dict],
        limit: int
    ) -> str:
        """Create enhanced prompt with real product data"""
        
        # Format product list
        product_list = []
        for p in available_products[:20]:  # Limit to avoid token limits
            product_list.append({
                "id": str(p.get("id")),
                "title": p.get("title"),
                "category": p.get("category"),
                "brand": p.get("brand"),
                "price": p.get("price"),
                "rating": p.get("rating")
            })
        
        prompt = f"""You are an intelligent e-commerce recommendation system. Analyze the user's shopping behavior and recommend the best products.

User Shopping Profile:
- Favorite Categories: {user_profile['top_categories']}
- Preferred Brands: {user_profile['favorite_brands']}
- Average Spending: ${user_profile['avg_price']:.2f}
- Recent Products: {user_profile['recent_products']}
- Total Interactions: {user_profile['total_interactions']}

Available Products:
{json.dumps(product_list, indent=2)}

Task: Recommend the top {limit} products that best match this user's preferences and shopping behavior.

IMPORTANT: Respond with ONLY valid JSON in this exact format:
{{
  "recommendations": [
    {{
      "product_id": "1",
      "score": 0.95,
      "reason": "Matches your interest in smartphones and preferred price range"
    }}
  ]
}}

Guidelines:
1. Consider category preferences, brands, and price range
2. Score 0.0 to 1.0 based on relevance
3. Provide specific, personalized reasons
4. Return exactly {limit} recommendations
5. Prioritize products similar to user's history
"""
        return prompt
    
    async def _parse_and_enrich_response(
        self,
        response_text: str,
        available_products: List[Dict]
    ) -> List[ProductRecommendation]:
        """Parse Gemini response and enrich with product data"""
        try:
            # Clean response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            data = json.loads(response_text)
            
            # Create product lookup
            product_map = {str(p.get("id")): p for p in available_products}
            
            recommendations = []
            for item in data.get("recommendations", []):
                product_id = str(item["product_id"])
                product = product_map.get(product_id)
                
                if product:
                    recommendations.append(ProductRecommendation(
                        product_id=product_id,
                        title=product.get("title", "Unknown Product"),
                        description=product.get("description", ""),
                        price=product.get("price", 0),
                        category=product.get("category", "unknown"),
                        brand=product.get("brand"),
                        rating=product.get("rating"),
                        thumbnail=product.get("thumbnail"),
                        score=min(max(float(item["score"]), 0.0), 1.0),
                        reason=item.get("reason", "AI recommended")
                    ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {str(e)}")
            raise
    
    async def _create_recommendations_from_products(
        self,
        products: List[Dict],
        default_reason: str
    ) -> List[ProductRecommendation]:
        """Convert product list to recommendations"""
        recommendations = []
        
        for i, product in enumerate(products):
            score = 0.9 - (i * 0.1)  # Decreasing score
            
            recommendations.append(ProductRecommendation(
                product_id=str(product.get("id")),
                title=product.get("title", "Unknown Product"),
                description=product.get("description", ""),
                price=product.get("price", 0),
                category=product.get("category", "unknown"),
                brand=product.get("brand"),
                rating=product.get("rating"),
                thumbnail=product.get("thumbnail"),
                score=max(score, 0.5),
                reason=default_reason
            ))
        
        return recommendations
    
    async def retrain_model(self):
        """Trigger for model updates"""
        async with self.lock:
            logger.info(
                f"Model refresh. Users: {len(self.user_interactions)}, "
                f"Gemini: {self.client is not None}"
            )
    
    def get_stats(self) -> Dict:
        """Get service statistics"""
        total_interactions = sum(
            len(history) for history in self.user_interactions.values()
        )
        
        return {
            "total_interactions": total_interactions,
            "unique_users": len(self.user_interactions),
            "model_version": self.model_version,
            "gemini_model": settings.GEMINI_MODEL,
            "gemini_enabled": self.client is not None,
            "last_training": datetime.utcnow().isoformat()
        }
