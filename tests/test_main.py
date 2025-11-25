"""
Unit tests for FastAPI endpoints
"""
import pytest
from httpx import AsyncClient, ASGITransport
from datetime import datetime

from main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Product Recommendation Microservice"
        assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "recommendation-service"
        assert "kafka_connected" in data


@pytest.mark.asyncio
async def test_track_interaction():
    """Test tracking user interaction"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        interaction_data = {
            "user_id": "test_user_123",
            "product_id": "test_product_456",
            "interaction_type": "view",
            "metadata": {"source": "test"}
        }
        
        response = await client.post("/api/v1/interactions", json=interaction_data)
        assert response.status_code == 202
        data = response.json()
        assert data["user_id"] == "test_user_123"
        assert data["product_id"] == "test_product_456"


@pytest.mark.asyncio
async def test_get_recommendations():
    """Test getting recommendations"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        request_data = {
            "user_id": "test_user_123",
            "limit": 5
        }
        
        response = await client.post("/api/v1/recommendations", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test_user_123"
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) <= 5


@pytest.mark.asyncio
async def test_get_stats():
    """Test service statistics endpoint"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_interactions" in data
        assert "unique_users" in data
        assert "unique_products" in data
        assert "model_version" in data


@pytest.mark.asyncio
async def test_trigger_retrain():
    """Test model retraining trigger"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/retrain")
        assert response.status_code == 202
        data = response.json()
        assert "message" in data
        assert data["status"] == "processing"


@pytest.mark.asyncio
async def test_invalid_interaction():
    """Test invalid interaction data"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        invalid_data = {
            "user_id": "",  # Empty user_id
            "product_id": "test_product",
            "interaction_type": "view"
        }
        
        response = await client.post("/api/v1/interactions", json=invalid_data)
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_recommendations_with_exclusions():
    """Test recommendations with product exclusions"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        request_data = {
            "user_id": "test_user_123",
            "limit": 3,
            "exclude_products": ["prod_1", "prod_2"]
        }
        
        response = await client.post("/api/v1/recommendations", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        # Verify excluded products are not in recommendations
        recommended_ids = [r["product_id"] for r in data["recommendations"]]
        assert "prod_1" not in recommended_ids
        assert "prod_2" not in recommended_ids
