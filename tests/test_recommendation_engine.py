"""
Unit tests for Recommendation Engine
"""
import pytest
from datetime import datetime

from app.services.recommendation_engine import RecommendationEngine
from app.models.schemas import UserInteractionEvent, InteractionType


@pytest.mark.asyncio
async def test_process_interaction():
    """Test processing a single interaction"""
    engine = RecommendationEngine()
    
    interaction = UserInteractionEvent(
        user_id="user_1",
        product_id="product_a",
        interaction_type=InteractionType.VIEW,
        timestamp=datetime.utcnow()
    )
    
    await engine.process_interaction(interaction)
    
    assert "user_1" in engine.user_interactions
    assert "product_a" in engine.user_interactions["user_1"]
    assert engine.user_interactions["user_1"]["product_a"] == 1.0


@pytest.mark.asyncio
async def test_interaction_weights():
    """Test different interaction type weights"""
    engine = RecommendationEngine()
    
    # View interaction (weight: 1.0)
    view_interaction = UserInteractionEvent(
        user_id="user_1",
        product_id="product_a",
        interaction_type=InteractionType.VIEW
    )
    await engine.process_interaction(view_interaction)
    
    # Purchase interaction (weight: 5.0)
    purchase_interaction = UserInteractionEvent(
        user_id="user_2",
        product_id="product_b",
        interaction_type=InteractionType.PURCHASE
    )
    await engine.process_interaction(purchase_interaction)
    
    assert engine.user_interactions["user_1"]["product_a"] == 1.0
    assert engine.user_interactions["user_2"]["product_b"] == 5.0


@pytest.mark.asyncio
async def test_get_popular_recommendations():
    """Test getting popular product recommendations"""
    engine = RecommendationEngine()
    
    # Add some interactions
    for i in range(5):
        interaction = UserInteractionEvent(
            user_id=f"user_{i}",
            product_id="popular_product",
            interaction_type=InteractionType.VIEW
        )
        await engine.process_interaction(interaction)
    
    # Get recommendations for new user
    recommendations = await engine.get_recommendations("new_user", limit=3)
    
    assert len(recommendations) > 0
    assert recommendations[0].product_id == "popular_product"


@pytest.mark.asyncio
async def test_retrain_model():
    """Test model retraining"""
    engine = RecommendationEngine()
    
    # Add interactions
    for i in range(3):
        interaction = UserInteractionEvent(
            user_id=f"user_{i}",
            product_id=f"product_{i}",
            interaction_type=InteractionType.VIEW
        )
        await engine.process_interaction(interaction)
    
    # Retrain model
    await engine.retrain_model()
    
    assert engine.user_product_matrix is not None
    assert len(engine.user_index) == 3
    assert engine.last_training is not None


@pytest.mark.asyncio
async def test_recommendations_with_exclusions():
    """Test recommendations exclude specified products"""
    engine = RecommendationEngine()
    
    # Add interactions
    interaction = UserInteractionEvent(
        user_id="user_1",
        product_id="product_a",
        interaction_type=InteractionType.VIEW
    )
    await engine.process_interaction(interaction)
    
    # Get recommendations with exclusions
    recommendations = await engine.get_recommendations(
        user_id="user_2",
        limit=5,
        exclude_products=["product_a"]
    )
    
    recommended_ids = [r.product_id for r in recommendations]
    assert "product_a" not in recommended_ids


@pytest.mark.asyncio
async def test_get_stats():
    """Test getting engine statistics"""
    engine = RecommendationEngine()
    
    # Add some interactions
    interaction1 = UserInteractionEvent(
        user_id="user_1",
        product_id="product_a",
        interaction_type=InteractionType.VIEW
    )
    interaction2 = UserInteractionEvent(
        user_id="user_1",
        product_id="product_b",
        interaction_type=InteractionType.PURCHASE
    )
    
    await engine.process_interaction(interaction1)
    await engine.process_interaction(interaction2)
    
    stats = engine.get_stats()
    
    assert stats["total_interactions"] == 2
    assert stats["unique_users"] == 1
    assert stats["unique_products"] == 2
    assert stats["model_version"] == "v1.0.0"


@pytest.mark.asyncio
async def test_multiple_interactions_same_product():
    """Test multiple interactions on same product accumulate scores"""
    engine = RecommendationEngine()
    
    # Multiple views
    for _ in range(3):
        interaction = UserInteractionEvent(
            user_id="user_1",
            product_id="product_a",
            interaction_type=InteractionType.VIEW
        )
        await engine.process_interaction(interaction)
    
    assert engine.user_interactions["user_1"]["product_a"] == 3.0
