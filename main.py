"""
FastAPI application for Real-time Product Recommendation System
Event-driven microservice with Kafka integration and Web UI
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
from typing import List
import os

from app.models.schemas import (
    UserInteractionEvent,
    RecommendationRequest,
    RecommendationResponse,
    HealthResponse
)
from app.services.enhanced_recommendation_engine import SOTARecommendationEngine
from app.services.product_service import product_service
from app.services.kafka_producer import KafkaProducerService
from app.services.kafka_consumer import KafkaConsumerService
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services with SOTA engine
recommendation_engine = SOTARecommendationEngine()
kafka_producer = KafkaProducerService()
kafka_consumer = KafkaConsumerService(recommendation_engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    # Startup
    logger.info("Starting recommendation microservice...")
    await kafka_producer.start()
    await kafka_consumer.start()
    logger.info("Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down recommendation microservice...")
    await kafka_producer.stop()
    await kafka_consumer.stop()
    logger.info("Service stopped successfully")


# Initialize FastAPI app
app = FastAPI(
    title="Product Recommendation Microservice",
    description="Event-driven AI microservice for real-time product recommendations with Web UI",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=FileResponse)
async def serve_ui():
    """Serve the web UI"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {
        "service": "Product Recommendation Microservice",
        "version": "2.0.0",
        "status": "running",
        "ui": "Web UI not available. Create static/index.html to enable.",
        "api_docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="recommendation-service",
        kafka_connected=kafka_producer.is_connected()
    )


@app.post("/api/v1/interactions", status_code=202)
async def track_user_interaction(
    interaction: UserInteractionEvent,
    background_tasks: BackgroundTasks
):
    """
    Track user interaction. 
    If Kafka is available, publish there. Otherwise, process directly.
    """
    try:
        # Check if Kafka is connected
        if kafka_producer.is_connected():
            # Preferred: Publish to Kafka
            background_tasks.add_task(
                kafka_producer.publish_interaction,
                interaction
            )
        else:
            # Fallback: Direct processing (Important for your setup!)
            # logger.warning("Kafka offline. Processing interaction directly.")
            background_tasks.add_task(
                recommendation_engine.process_interaction,
                interaction
            )
        
        logger.info(f"Interaction tracked: user={interaction.user_id}, product={interaction.product_id}")
        
        return {
            "message": "Interaction tracked successfully",
            "user_id": interaction.user_id,
            "product_id": interaction.product_id
        }
    except Exception as e:
        logger.error(f"Error tracking interaction: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to track interaction")

@app.post("/api/v1/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get real-time product recommendations for a user
    Uses trained ML model for personalized suggestions
    """
    try:
        recommendations = await recommendation_engine.get_recommendations(
            user_id=request.user_id,
            limit=request.limit,
            exclude_products=request.exclude_products
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {request.user_id}")
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_version=recommendation_engine.model_version
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")


@app.get("/api/v1/stats", response_model=dict)
async def get_service_stats():
    """Get service statistics"""
    try:
        stats = recommendation_engine.get_stats()
        return {
            "total_interactions": stats["total_interactions"],
            "unique_users": stats["unique_users"],
            "model_version": stats["model_version"],
            "gemini_model": stats.get("gemini_model", "N/A"),
            "gemini_enabled": stats.get("gemini_enabled", False),
            "last_training": stats.get("last_training")
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@app.post("/api/v1/retrain", status_code=202)
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with latest interaction data
    Training happens asynchronously
    """
    try:
        background_tasks.add_task(recommendation_engine.retrain_model)
        
        logger.info("Model retraining triggered")
        
        return {
            "message": "Model retraining initiated",
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error triggering retraining: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")


@app.get("/api/v1/products/search")
async def search_products(query: str, limit: int = 10):
    """
    Search for real products by query
    
    Args:
        query: Search term (e.g., "laptop", "phone")
        limit: Number of results (max 30)
    """
    try:
        if limit > 30:
            limit = 30
        
        products = await product_service.search_products(query, limit)
        
        return {
            "query": query,
            "total": len(products),
            "products": products
        }
    except Exception as e:
        logger.error(f"Error searching products: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search products")


@app.get("/api/v1/products/{product_id}")
async def get_product_details(product_id: str):
    """
    Get detailed product information
    
    Args:
        product_id: Product ID
    """
    try:
        product = await product_service.get_product_by_id(product_id)
        
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return product
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching product: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch product")


@app.get("/api/v1/products/category/{category}")
async def get_products_by_category(category: str, limit: int = 10):
    """
    Get products by category
    
    Args:
        category: Category name
        limit: Number of results
    """
    try:
        products = await product_service.get_products_by_category(category, limit)
        
        return {
            "category": category,
            "total": len(products),
            "products": products
        }
    except Exception as e:
        logger.error(f"Error fetching category: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch products")


@app.get("/api/v1/categories")
async def get_all_categories():
    """
    Get all available product categories
    """
    try:
        categories = await product_service.get_all_categories()
        
        return {
            "total": len(categories),
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Error fetching categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch categories")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )