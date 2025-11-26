"""
Database configuration and models
Supports PostgreSQL for persistence and Redis for caching
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import redis
import json
import logging
from typing import Optional
from app.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()


class UserInteraction(Base):
    """User interaction model"""
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    product_id = Column(String, index=True, nullable=False)
    interaction_type = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default={})
    weight = Column(Float, default=1.0)


class UserProfile(Base):
    """User profile model"""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    favorite_categories = Column(JSON, default={})
    favorite_brands = Column(JSON, default={})
    avg_price = Column(Float, default=0.0)
    total_interactions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.redis_client = None
        self._initialize_db()
        self._initialize_redis()
    
    def _initialize_db(self):
        """Initialize PostgreSQL database"""
        try:
            database_url = settings.DATABASE_URL
            if database_url and database_url != "sqlite:///./test.db":
                self.engine = create_engine(
                    database_url,
                    pool_pre_ping=True,
                    pool_size=10,
                    max_overflow=20
                )
                self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
                
                # Create tables
                Base.metadata.create_all(bind=self.engine)
                logger.info("PostgreSQL database initialized")
            else:
                logger.warning("DATABASE_URL not configured, using in-memory storage only")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            logger.warning("Continuing with in-memory storage")
    
    def _initialize_redis(self):
        """Initialize Redis cache"""
        try:
            if settings.REDIS_URL:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            else:
                logger.warning("REDIS_URL not configured, caching disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            logger.warning("Continuing without Redis cache")
            self.redis_client = None
    
    def get_db(self):
        """Get database session"""
        if self.SessionLocal:
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()
        else:
            yield None
    
    async def save_interaction(self, interaction_data: dict):
        """Save interaction to database"""
        if not self.SessionLocal:
            return
        
        try:
            db = next(self.get_db())
            db_interaction = UserInteraction(**interaction_data)
            db.add(db_interaction)
            db.commit()
            db.refresh(db_interaction)
            logger.debug(f"Saved interaction {db_interaction.id} to database")
        except Exception as e:
            logger.error(f"Error saving interaction: {str(e)}")
            if db:
                db.rollback()
    
    async def get_user_interactions(self, user_id: str, limit: int = 100):
        """Get user interactions from database"""
        if not self.SessionLocal:
            return []
        
        try:
            db = next(self.get_db())
            interactions = db.query(UserInteraction)\
                .filter(UserInteraction.user_id == user_id)\
                .order_by(UserInteraction.timestamp.desc())\
                .limit(limit)\
                .all()
            
            return [
                {
                    "user_id": i.user_id,
                    "product_id": i.product_id,
                    "interaction_type": i.interaction_type,
                    "timestamp": i.timestamp.isoformat(),
                    "weight": i.weight,
                    "metadata": i.metadata
                }
                for i in interactions
            ]
        except Exception as e:
            logger.error(f"Error fetching interactions: {str(e)}")
            return []
    
    async def update_user_profile(self, user_id: str, profile_data: dict):
        """Update user profile in database"""
        if not self.SessionLocal:
            return
        
        try:
            db = next(self.get_db())
            profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            
            if profile:
                for key, value in profile_data.items():
                    setattr(profile, key, value)
                profile.updated_at = datetime.utcnow()
            else:
                profile = UserProfile(user_id=user_id, **profile_data)
                db.add(profile)
            
            db.commit()
            logger.debug(f"Updated profile for user {user_id}")
        except Exception as e:
            logger.error(f"Error updating profile: {str(e)}")
            if db:
                db.rollback()
    
    async def get_user_profile(self, user_id: str) -> Optional[dict]:
        """Get user profile from database"""
        if not self.SessionLocal:
            return None
        
        try:
            db = next(self.get_db())
            profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            
            if profile:
                return {
                    "user_id": profile.user_id,
                    "favorite_categories": profile.favorite_categories,
                    "favorite_brands": profile.favorite_brands,
                    "avg_price": profile.avg_price,
                    "total_interactions": profile.total_interactions
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching profile: {str(e)}")
            return None
    
    # Redis cache methods
    async def cache_set(self, key: str, value: dict, ttl: int = 3600):
        """Set value in Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )
            return True
        except Exception as e:
            logger.error(f"Redis set error: {str(e)}")
            return False
    
    async def cache_get(self, key: str) -> Optional[dict]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {str(e)}")
            return None
    
    async def cache_delete(self, key: str):
        """Delete value from Redis cache"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {str(e)}")


# Global database manager
db_manager = DatabaseManager()
