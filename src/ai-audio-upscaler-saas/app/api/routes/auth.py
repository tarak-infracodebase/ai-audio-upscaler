"""
Authentication Routes
Handles user authentication, token management, and authorization
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, EmailStr, validator
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.auth import user_manager, get_current_user, require_admin
from app.models.user import User, UserRole
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()

# Request/Response Models
class AuthCallbackRequest(BaseModel):
    """Azure B2C authentication callback request"""
    authorization_code: str
    redirect_uri: str

class TokenRefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str

class AuthResponse(BaseModel):
    """Authentication response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes
    user: Dict[str, Any]

class TokenResponse(BaseModel):
    """Token refresh response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 1800

class UserProfileResponse(BaseModel):
    """User profile response"""
    id: int
    email: str
    name: Optional[str]
    role: str
    is_active: bool
    avatar_url: Optional[str]
    total_processing_minutes: int
    monthly_processing_minutes: int
    created_at: datetime
    last_login_at: Optional[datetime]

    class Config:
        from_attributes = True

class UserUpdateRequest(BaseModel):
    """User profile update request"""
    name: Optional[str] = None
    avatar_url: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if v and (len(v) < 2 or len(v) > 100):
            raise ValueError('Name must be between 2 and 100 characters')
        return v

    @validator('avatar_url')
    def validate_avatar_url(cls, v):
        if v and len(v) > 500:
            raise ValueError('Avatar URL too long')
        return v

class AdminUserUpdateRequest(BaseModel):
    """Admin user update request"""
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

@router.get("/login-url")
async def get_login_url(request: Request):
    """Get Azure B2C login URL"""
    settings = get_settings()

    # Build Azure B2C login URL
    tenant_name = settings.AZURE_B2C_TENANT_NAME
    policy_name = settings.AZURE_B2C_POLICY_NAME
    client_id = settings.AZURE_B2C_CLIENT_ID

    # Get base URL from request
    base_url = f"{request.url.scheme}://{request.url.netloc}"
    redirect_uri = f"{base_url}/auth/callback"

    login_url = (
        f"https://{tenant_name}.b2clogin.com/{tenant_name}.onmicrosoft.com/"
        f"{policy_name}/oauth2/v2.0/authorize?"
        f"client_id={client_id}&"
        f"response_type=code&"
        f"redirect_uri={redirect_uri}&"
        f"response_mode=query&"
        f"scope=openid%20profile%20email%20offline_access&"
        f"state={request.headers.get('x-request-id', 'default')}"
    )

    return {
        "login_url": login_url,
        "redirect_uri": redirect_uri
    }

@router.post("/callback", response_model=AuthResponse)
async def auth_callback(
    callback_request: AuthCallbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """Handle Azure B2C authentication callback"""
    try:
        # Exchange authorization code for B2C token
        b2c_manager = user_manager.b2c_manager
        token_response = await b2c_manager.exchange_authorization_code(
            callback_request.authorization_code,
            callback_request.redirect_uri
        )

        # Extract B2C access token
        b2c_access_token = token_response.get("access_token")
        if not b2c_access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No access token received from Azure B2C"
            )

        # Authenticate user and create API tokens
        user, access_token, refresh_token = await user_manager.authenticate_user(
            db, b2c_access_token
        )

        logger.info(
            "User authentication successful",
            user_id=user.id,
            email=user.email
        )

        return AuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user={
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role.value,
                "is_active": user.is_active,
                "avatar_url": user.avatar_url
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authentication callback failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_request: TokenRefreshRequest,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token"""
    try:
        access_token = await user_manager.refresh_user_token(
            db, refresh_request.refresh_token
        )

        return TokenResponse(access_token=access_token)

    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )

@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get current user profile"""
    return UserProfileResponse.from_orm(current_user)

@router.put("/me", response_model=UserProfileResponse)
async def update_current_user_profile(
    update_request: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user profile"""
    try:
        # Update user fields
        if update_request.name is not None:
            current_user.name = update_request.name

        if update_request.avatar_url is not None:
            current_user.avatar_url = update_request.avatar_url

        current_user.updated_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(current_user)

        logger.info(
            "User profile updated",
            user_id=current_user.id,
            email=current_user.email
        )

        return UserProfileResponse.from_orm(current_user)

    except Exception as e:
        logger.error("Profile update failed", error=str(e), user_id=current_user.id)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout user (client should delete tokens)"""
    logger.info("User logged out", user_id=current_user.id, email=current_user.email)

    return {
        "message": "Logged out successfully",
        "user_id": current_user.id
    }

@router.get("/usage")
async def get_usage_stats(current_user: User = Depends(get_current_user)):
    """Get user's usage statistics"""
    # Calculate limits based on role
    if current_user.role == UserRole.ADMIN:
        monthly_limit = None  # Unlimited
    elif current_user.role == UserRole.PREMIUM:
        monthly_limit = 300  # 5 hours
    else:
        monthly_limit = 60   # 1 hour

    return {
        "user_id": current_user.id,
        "role": current_user.role.value,
        "total_processing_minutes": current_user.total_processing_minutes,
        "monthly_processing_minutes": current_user.monthly_processing_minutes,
        "monthly_limit_minutes": monthly_limit,
        "remaining_minutes": (
            monthly_limit - current_user.monthly_processing_minutes
            if monthly_limit is not None else None
        ),
        "last_reset_date": current_user.last_reset_date
    }

# Admin Routes
@router.get("/admin/users")
async def list_users(
    skip: int = 0,
    limit: int = 100,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """List all users (admin only)"""
    try:
        from sqlalchemy import select, func

        # Get total count
        count_query = select(func.count(User.id))
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Get paginated users
        users_query = select(User).offset(skip).limit(limit)
        users_result = await db.execute(users_query)
        users = users_result.scalars().all()

        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "users": [
                {
                    "id": user.id,
                    "email": user.email,
                    "name": user.name,
                    "role": user.role.value,
                    "is_active": user.is_active,
                    "total_processing_minutes": user.total_processing_minutes,
                    "monthly_processing_minutes": user.monthly_processing_minutes,
                    "created_at": user.created_at,
                    "last_login_at": user.last_login_at
                }
                for user in users
            ]
        }

    except Exception as e:
        logger.error("Failed to list users", error=str(e), admin_user_id=admin_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

@router.get("/admin/users/{user_id}", response_model=UserProfileResponse)
async def get_user_by_id(
    user_id: int,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get user by ID (admin only)"""
    try:
        from sqlalchemy import select

        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return UserProfileResponse.from_orm(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user", error=str(e), user_id=user_id, admin_user_id=admin_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )

@router.put("/admin/users/{user_id}")
async def update_user_by_admin(
    user_id: int,
    update_request: AdminUserUpdateRequest,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update user by admin"""
    try:
        from sqlalchemy import select

        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Update fields
        if update_request.role is not None:
            user.role = update_request.role

        if update_request.is_active is not None:
            user.is_active = update_request.is_active

        user.updated_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(user)

        logger.info(
            "User updated by admin",
            user_id=user.id,
            admin_user_id=admin_user.id,
            changes=update_request.dict(exclude_unset=True)
        )

        return UserProfileResponse.from_orm(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user", error=str(e), user_id=user_id, admin_user_id=admin_user.id)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )