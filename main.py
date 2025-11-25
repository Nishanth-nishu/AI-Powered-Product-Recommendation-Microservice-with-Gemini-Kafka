"""
FastAPI application for Real-time Product Recommendation System
Event-driven microservice with Kafka integration
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import List

from app.models.schemas import (
    UserInteractionEvent,
    RecommendationRequest,
    RecommendationResponse,
    HealthResponse
)
from app.services.recommendation_engine import RecommendationEngine
from app.services.kafka_producer import KafkaProducerService
from app.services.kafka_consumer import KafkaConsumerService
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
recommendation_engine = RecommendationEngine()
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
    description="Event-driven AI microservice for real-time product recommendations",
    version="1.0.0",
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


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Product Recommendation Microservice",
        "version": "1.0.0",
        "status": "running"
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
    Track user interaction and publish to Kafka
    Events are processed asynchronously
    """
    try:
        # Publish event to Kafka in background
        background_tasks.add_task(
            kafka_producer.publish_interaction,
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
    stats = recommendation_engine.get_stats()
    return {
        "total_interactions": stats["total_interactions"],
        "unique_users": stats["unique_users"],
        "unique_products": stats["unique_products"],
        "model_version": stats["model_version"],
        "last_training": stats["last_training"]
    }


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
