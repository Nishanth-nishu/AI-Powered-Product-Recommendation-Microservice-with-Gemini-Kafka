"""
Configuration management using Pydantic settings
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "recommendation-service"
    SERVICE_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Kafka configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_INTERACTION_TOPIC: str = "user-interactions"
    KAFKA_CONSUMER_GROUP: str = "recommendation-service-group"
    KAFKA_AUTO_OFFSET_RESET: str = "earliest"
    
    # Gemini API configuration
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None  # Takes precedence if both are set
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    
    # Database configuration
    DATABASE_URL: str = "sqlite:///./test.db"  # Default to SQLite for dev
    REDIS_URL: Optional[str] = None  # e.g., "redis://localhost:6379/0"
    
    # JWT Authentication
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # ML Model configuration
    MODEL_VERSION: str = "v2.0.0-gemini"
    MIN_INTERACTIONS_FOR_RECOMMENDATIONS: int = 3
    
    # API configuration
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    @property
    def gemini_key(self) -> Optional[str]:
        """Get Gemini API key with precedence: GOOGLE_API_KEY > GEMINI_API_KEY"""
        return self.GOOGLE_API_KEY or self.GEMINI_API_KEY
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()