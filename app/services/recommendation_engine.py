"""
AI-Powered Recommendation Engine using Google Gemini
Uses Gemini API for intelligent, context-aware recommendations
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
from app.config import settings

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    AI-powered recommendation engine using Google Gemini
    Leverages LLM for intelligent product recommendations
    """
    
    def __init__(self):
        self.model_version = settings.MODEL_VERSION
        self.user_interactions: Dict[str, List[Dict]] = defaultdict(list)
        self.product_metadata: Dict[str, Dict] = {}
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
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable. "
                "Recommendations will use fallback logic."
            )
            return
        
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            self.client = None
    
    async def process_interaction(self, interaction: UserInteractionEvent):
        """
        Process a user interaction event
        Stores interaction data for AI-powered recommendations
        """
        async with self.lock:
            try:
                interaction_data = {
                    "product_id": interaction.product_id,
                    "interaction_type": interaction.interaction_type,
                    "timestamp": interaction.timestamp.isoformat(),
                    "weight": self.interaction_weights.get(interaction.interaction_type, 1.0),
                    "metadata": interaction.metadata
                }
                
                self.user_interactions[interaction.user_id].append(interaction_data)
                
                # Update product metadata if provided
                if interaction.metadata:
                    if interaction.product_id not in self.product_metadata:
                        self.product_metadata[interaction.product_id] = {}
                    self.product_metadata[interaction.product_id].update(interaction.metadata)
                
                logger.info(
                    f"Processed interaction: user={interaction.user_id}, "
                    f"product={interaction.product_id}, type={interaction.interaction_type}"
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
        Generate AI-powered recommendations using Gemini
        Falls back to popularity-based if Gemini is unavailable
        """
        exclude_products = exclude_products or []
        
        try:
            # Check if user has interaction history
            user_history = self.user_interactions.get(user_id, [])
            
            if len(user_history) < settings.MIN_INTERACTIONS_FOR_RECOMMENDATIONS:
                logger.info(f"User {user_id} has insufficient history, using popular products")
                return await self._get_popular_recommendations(limit, exclude_products)
            
            # Try Gemini-powered recommendations
            if self.client:
                try:
                    recommendations = await self._get_gemini_recommendations(
                        user_id, user_history, limit, exclude_products
                    )
                    if recommendations:
                        return recommendations
                except Exception as e:
                    logger.warning(f"Gemini recommendation failed, using fallback: {str(e)}")
            
            # Fallback to rule-based recommendations
            return await self._get_fallback_recommendations(
                user_id, user_history, limit, exclude_products
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return await self._get_popular_recommendations(limit, exclude_products)
    
    async def _get_gemini_recommendations(
        self,
        user_id: str,
        user_history: List[Dict],
        limit: int,
        exclude_products: List[str]
    ) -> List[ProductRecommendation]:
        """
        Use Gemini AI to generate intelligent recommendations
        """
        try:
            # Prepare user profile for Gemini
            user_profile = self._create_user_profile(user_history)
            all_products = list(self.product_metadata.keys())
            
            # Create prompt for Gemini
            prompt = self._create_recommendation_prompt(
                user_profile, all_products, exclude_products, limit
            )
            
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
            
            # Parse Gemini response
            recommendations = self._parse_gemini_response(response.text, limit)
            
            logger.info(f"Generated {len(recommendations)} Gemini recommendations for {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise
    
    def _create_user_profile(self, user_history: List[Dict]) -> Dict:
        """Create a user profile summary from interaction history"""
        product_scores = defaultdict(float)
        interaction_counts = defaultdict(int)
        
        for interaction in user_history:
            product_id = interaction["product_id"]
            weight = interaction["weight"]
            interaction_type = interaction["interaction_type"]
            
            product_scores[product_id] += weight
            interaction_counts[interaction_type] += 1
        
        # Get top interacted products
        top_products = sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "top_products": [p[0] for p in top_products],
            "interaction_summary": dict(interaction_counts),
            "total_interactions": len(user_history)
        }
    
    def _create_recommendation_prompt(
        self,
        user_profile: Dict,
        all_products: List[str],
        exclude_products: List[str],
        limit: int
    ) -> str:
        """Create a prompt for Gemini to generate recommendations"""
        available_products = [p for p in all_products if p not in exclude_products]
        available_products = [p for p in available_products if p not in user_profile["top_products"]]
        
        prompt = f"""You are an AI recommendation system. Based on the user's interaction history, recommend {limit} products.

User Profile:
- Previously interacted products: {user_profile['top_products']}
- Interaction types: {user_profile['interaction_summary']}
- Total interactions: {user_profile['total_interactions']}

Available products to recommend from: {available_products[:50]}

IMPORTANT: Respond ONLY with valid JSON in this exact format, no markdown, no explanation:
{{
  "recommendations": [
    {{"product_id": "product_name", "score": 0.95, "reason": "Brief reason"}},
    {{"product_id": "product_name", "score": 0.87, "reason": "Brief reason"}}
  ]
}}

Rules:
1. Recommend products similar to what the user interacted with
2. Score between 0.0 and 1.0 (higher = more relevant)
3. Provide brief, compelling reasons
4. Return exactly {limit} recommendations
5. Do not recommend products the user already interacted with
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str, limit: int) -> List[ProductRecommendation]:
        """Parse Gemini's JSON response into ProductRecommendation objects"""
        try:
            # Clean response text
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            data = json.loads(response_text)
            recommendations = []
            
            for item in data.get("recommendations", [])[:limit]:
                recommendations.append(ProductRecommendation(
                    product_id=item["product_id"],
                    score=min(max(float(item["score"]), 0.0), 1.0),
                    reason=item.get("reason", "AI recommended")
                ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {str(e)}")
            raise
    
    async def _get_fallback_recommendations(
        self,
        user_id: str,
        user_history: List[Dict],
        limit: int,
        exclude_products: List[str]
    ) -> List[ProductRecommendation]:
        """
        Fallback rule-based recommendations
        Recommends products similar to user's most interacted items
        """
        # Calculate product scores from user history
        product_scores = defaultdict(float)
        for interaction in user_history:
            product_id = interaction["product_id"]
            product_scores[product_id] += interaction["weight"]
        
        # Get user's top products
        user_top_products = set([p[0] for p in sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]])
        
        # Get all products the user hasn't interacted with
        all_interacted = set(p["product_id"] for p in user_history)
        available_products = [
            p for p in self.product_metadata.keys()
            if p not in all_interacted and p not in exclude_products
        ]
        
        # Score available products (simple similarity)
        recommendations = []
        for product_id in available_products[:limit]:
            score = 0.6 + (0.4 * len(user_top_products.intersection({product_id})))
            recommendations.append(ProductRecommendation(
                product_id=product_id,
                score=score,
                reason="Based on your interests"
            ))
        
        # Fill with popular if needed
        if len(recommendations) < limit:
            popular = await self._get_popular_recommendations(
                limit - len(recommendations),
                exclude_products + [r.product_id for r in recommendations]
            )
            recommendations.extend(popular)
        
        return recommendations[:limit]
    
    async def _get_popular_recommendations(
        self,
        limit: int,
        exclude_products: List[str]
    ) -> List[ProductRecommendation]:
        """Get popular products as cold-start recommendations"""
        product_popularity = defaultdict(int)
        
        # Calculate popularity from all interactions
        for user_history in self.user_interactions.values():
            for interaction in user_history:
                product_id = interaction["product_id"]
                product_popularity[product_id] += 1
        
        # Sort by popularity
        popular_products = sorted(
            product_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = []
        max_popularity = max(product_popularity.values()) if product_popularity else 1
        
        for product_id, count in popular_products:
            if product_id in exclude_products:
                continue
            
            score = count / max_popularity
            recommendations.append(ProductRecommendation(
                product_id=product_id,
                score=score,
                reason="Popular choice"
            ))
            
            if len(recommendations) >= limit:
                break
        
        return recommendations
    
    async def retrain_model(self):
        """
        Trigger for model updates
        With Gemini, this is mostly for logging/metrics
        """
        async with self.lock:
            logger.info(
                f"Model refresh triggered. Using Gemini {settings.GEMINI_MODEL}. "
                f"Total users: {len(self.user_interactions)}, "
                f"Total products: {len(self.product_metadata)}"
            )
    
    def get_stats(self) -> Dict:
        """Get service statistics"""
        total_interactions = sum(
            len(history) for history in self.user_interactions.values()
        )
        
        return {
            "total_interactions": total_interactions,
            "unique_users": len(self.user_interactions),
            "unique_products": len(self.product_metadata),
            "model_version": self.model_version,
            "gemini_model": settings.GEMINI_MODEL,
            "gemini_enabled": self.client is not None,
            "last_training": datetime.utcnow().isoformat()
        }
