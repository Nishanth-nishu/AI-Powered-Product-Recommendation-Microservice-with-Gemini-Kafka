"""
JWT Authentication Handler
Provides secure user authentication
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token
security = HTTPBearer()


class TokenData(BaseModel):
    """Token data model"""
    user_id: str
    exp: Optional[datetime] = None


class UserAuth(BaseModel):
    """User authentication model"""
    username: str
    password: str


def create_access_token(user_id: str) -> str:
    """
    Create JWT access token
    
    Args:
        user_id: User identifier
    
    Returns:
        JWT token string
    """
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """
    Verify and decode JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        TokenData object
    
    Raises:
        HTTPException: If token is invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        return TokenData(user_id=user_id, exp=payload.get("exp"))
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Get current authenticated user from JWT token
    
    Args:
        credentials: HTTP Authorization credentials
    
    Returns:
        User ID
    
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    token_data = verify_token(token)
    return token_data.user_id


def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against hash"""
    return pwd_context.verify(plain_password, hashed_password)
