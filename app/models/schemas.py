"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum


class InteractionType(str, Enum):
    """Types of user interactions"""
    VIEW = "view"
    CLICK = "click"
    ADD_TO_CART = "add_to_cart"
    PURCHASE = "purchase"
    LIKE = "like"
    SHARE = "share"


class UserInteractionEvent(BaseModel):
    """User interaction event schema"""
    user_id: str = Field(..., description="Unique user identifier")
    product_id: str = Field(..., description="Unique product identifier")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('user_id', 'product_id')
    @classmethod
    def validate_ids(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("ID cannot be empty")
        return v.strip()


class ProductRecommendation(BaseModel):
    """Single product recommendation with full details"""
    product_id: str = Field(..., description="Recommended product ID")
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation confidence score")
    reason: Optional[str] = Field(None, description="Reason for recommendation")
    
    # Enhanced fields for UI display
    title: Optional[str] = Field("Unknown", description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: Optional[float] = Field(0.0, description="Product price")
    category: Optional[str] = Field(None, description="Product category")
    brand: Optional[str] = Field(None, description="Product brand")
    rating: Optional[float] = Field(None, description="Product rating")
    thumbnail: Optional[str] = Field(None, description="Product thumbnail URL")


class RecommendationRequest(BaseModel):
    """Request for product recommendations"""
    user_id: str = Field(..., description="User ID to get recommendations for")
    limit: int = Field(5, ge=1, le=50, description="Number of recommendations to return")
    exclude_products: Optional[List[str]] = Field(default_factory=list, description="Products to exclude")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("User ID cannot be empty")
        return v.strip()


class RecommendationResponse(BaseModel):
    """Response containing product recommendations"""
    user_id: str
    recommendations: List[ProductRecommendation]
    model_version: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    kafka_connected: bool = Field(..., description="Kafka connection status")
    gemini_enabled: bool = Field(default=False, description="Is Gemini AI enabled")
    timestamp: datetime = Field(default_factory=datetime.utcnow)