"""
Authentication and Authorization Module
Handles JWT tokens, Azure AD B2C integration, and user management
"""

import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
import httpx
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.core.database import get_db
from app.models.user import User, UserRole
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logger = structlog.get_logger(__name__)
security = HTTPBearer()

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class AuthorizationError(Exception):
    """Custom authorization error"""
    pass

class JWTManager:
    """
    JWT token management for API authentication
    Handles token creation, validation, and refresh
    """

    def __init__(self):
        self.settings = get_settings()
        self.algorithm = "RS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7

    def create_access_token(self, user_id: str, user_email: str, roles: List[str]) -> str:
        """Create JWT access token"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "sub": user_id,
            "email": user_email,
            "roles": roles,
            "iat": now,
            "exp": expire,
            "type": "access"
        }

        try:
            token = jwt.encode(
                payload,
                self.settings.JWT_SECRET_KEY,
                algorithm=self.algorithm
            )

            logger.info(
                "Access token created",
                user_id=user_id,
                expires_at=expire.isoformat()
            )

            return token

        except Exception as e:
            logger.error("Failed to create access token", error=str(e))
            raise AuthenticationError("Failed to create access token")

    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "sub": user_id,
            "iat": now,
            "exp": expire,
            "type": "refresh"
        }

        try:
            token = jwt.encode(
                payload,
                self.settings.JWT_SECRET_KEY,
                algorithm=self.algorithm
            )

            logger.info(
                "Refresh token created",
                user_id=user_id,
                expires_at=expire.isoformat()
            )

            return token

        except Exception as e:
            logger.error("Failed to create refresh token", error=str(e))
            raise AuthenticationError("Failed to create refresh token")

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.settings.JWT_SECRET_KEY,
                algorithms=[self.algorithm]
            )

            # Check token type
            if payload.get("type") not in ["access", "refresh"]:
                raise AuthenticationError("Invalid token type")

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, timezone.utc) < datetime.now(timezone.utc):
                raise AuthenticationError("Token has expired")

            logger.debug("Token verified successfully", user_id=payload.get("sub"))
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            raise AuthenticationError("Invalid token")
        except Exception as e:
            logger.error("Token verification failed", error=str(e))
            raise AuthenticationError("Token verification failed")

class AzureB2CManager:
    """
    Azure AD B2C integration manager
    Handles user authentication and profile management
    """

    def __init__(self):
        self.settings = get_settings()
        self.tenant_name = self.settings.AZURE_B2C_TENANT_NAME
        self.policy_name = self.settings.AZURE_B2C_POLICY_NAME
        self.client_id = self.settings.AZURE_B2C_CLIENT_ID
        self.client_secret = self.settings.AZURE_B2C_CLIENT_SECRET

        # Build B2C URLs
        self.authority = f"https://{self.tenant_name}.b2clogin.com/{self.tenant_name}.onmicrosoft.com/{self.policy_name}"
        self.token_endpoint = f"{self.authority}/oauth2/v2.0/token"
        self.jwks_uri = f"{self.authority}/discovery/v2.0/keys"

        self._jwks_cache = {}
        self._jwks_cache_time = None
        self._cache_duration = timedelta(hours=24)

    @asynccontextmanager
    async def http_client(self):
        """Create async HTTP client with proper configuration"""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            verify=True,
            follow_redirects=True
        ) as client:
            yield client

    async def get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set from Azure B2C"""
        now = datetime.now(timezone.utc)

        # Check cache
        if (self._jwks_cache and self._jwks_cache_time and
            now - self._jwks_cache_time < self._cache_duration):
            return self._jwks_cache

        try:
            async with self.http_client() as client:
                response = await client.get(self.jwks_uri)
                response.raise_for_status()

                jwks = response.json()
                self._jwks_cache = jwks
                self._jwks_cache_time = now

                logger.info("JWKS retrieved from Azure B2C")
                return jwks

        except Exception as e:
            logger.error("Failed to retrieve JWKS", error=str(e))
            raise AuthenticationError("Failed to retrieve JWKS from Azure B2C")

    async def validate_b2c_token(self, token: str) -> Dict[str, Any]:
        """Validate Azure B2C token"""
        try:
            # Get JWKS for token validation
            jwks = await self.get_jwks()

            # Decode token header to get key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            if not kid:
                raise AuthenticationError("Token missing key ID")

            # Find matching public key
            public_key = None
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
                    break

            if not public_key:
                raise AuthenticationError("Public key not found for token")

            # Verify token
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=self.client_id,
                issuer=f"https://{self.tenant_name}.b2clogin.com/{self.tenant_name}.onmicrosoft.com/v2.0/"
            )

            logger.info(
                "Azure B2C token validated",
                user_id=payload.get("sub"),
                email=payload.get("email")
            )

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Azure B2C token expired")
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid Azure B2C token", error=str(e))
            raise AuthenticationError("Invalid token")
        except Exception as e:
            logger.error("Azure B2C token validation failed", error=str(e))
            raise AuthenticationError("Token validation failed")

    async def exchange_authorization_code(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "scope": "openid profile email offline_access"
        }

        try:
            async with self.http_client() as client:
                response = await client.post(
                    self.token_endpoint,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                response.raise_for_status()

                tokens = response.json()

                logger.info("Authorization code exchanged successfully")
                return tokens

        except httpx.HTTPStatusError as e:
            logger.error("Token exchange failed", status_code=e.response.status_code, response=e.response.text)
            raise AuthenticationError("Failed to exchange authorization code")
        except Exception as e:
            logger.error("Token exchange error", error=str(e))
            raise AuthenticationError("Token exchange failed")

class UserManager:
    """
    User management service
    Handles user creation, updates, and role management
    """

    def __init__(self):
        self.jwt_manager = JWTManager()
        self.b2c_manager = AzureB2CManager()

    async def get_or_create_user(
        self,
        db: AsyncSession,
        b2c_user_id: str,
        email: str,
        name: Optional[str] = None
    ) -> User:
        """Get existing user or create new one from B2C profile"""
        try:
            # Try to find existing user
            result = await db.execute(
                select(User).where(User.azure_b2c_id == b2c_user_id)
            )
            user = result.scalar_one_or_none()

            if user:
                # Update user info if needed
                if user.email != email or (name and user.name != name):
                    user.email = email
                    if name:
                        user.name = name
                    user.updated_at = datetime.now(timezone.utc)

                    logger.info("User profile updated", user_id=user.id, email=email)

                return user

            # Create new user
            user = User(
                azure_b2c_id=b2c_user_id,
                email=email,
                name=name or email.split("@")[0],
                role=UserRole.USER,  # Default role
                is_active=True
            )

            db.add(user)
            await db.flush()  # Get the ID
            await db.refresh(user)

            logger.info("New user created", user_id=user.id, email=email)
            return user

        except Exception as e:
            logger.error("Failed to get or create user", error=str(e), email=email)
            await db.rollback()
            raise AuthenticationError("User management failed")

    async def authenticate_user(self, db: AsyncSession, b2c_token: str) -> tuple[User, str, str]:
        """Authenticate user with Azure B2C token and create API tokens"""
        try:
            # Validate B2C token
            b2c_payload = await self.b2c_manager.validate_b2c_token(b2c_token)

            b2c_user_id = b2c_payload.get("sub")
            email = b2c_payload.get("email") or b2c_payload.get("emails", [None])[0]
            name = b2c_payload.get("name")

            if not b2c_user_id or not email:
                raise AuthenticationError("Invalid B2C token payload")

            # Get or create user
            user = await self.get_or_create_user(db, b2c_user_id, email, name)

            if not user.is_active:
                raise AuthorizationError("User account is disabled")

            # Create API tokens
            access_token = self.jwt_manager.create_access_token(
                user_id=str(user.id),
                user_email=user.email,
                roles=[user.role.value]
            )

            refresh_token = self.jwt_manager.create_refresh_token(str(user.id))

            # Update last login
            user.last_login_at = datetime.now(timezone.utc)
            await db.commit()

            logger.info(
                "User authenticated successfully",
                user_id=user.id,
                email=user.email,
                role=user.role.value
            )

            return user, access_token, refresh_token

        except (AuthenticationError, AuthorizationError):
            raise
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            raise AuthenticationError("Authentication process failed")

    async def refresh_user_token(self, db: AsyncSession, refresh_token: str) -> str:
        """Refresh user access token"""
        try:
            # Verify refresh token
            payload = self.jwt_manager.verify_token(refresh_token)

            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid refresh token")

            user_id = payload.get("sub")
            if not user_id:
                raise AuthenticationError("Invalid token payload")

            # Get user from database
            result = await db.execute(
                select(User).where(User.id == int(user_id), User.is_active == True)
            )
            user = result.scalar_one_or_none()

            if not user:
                raise AuthenticationError("User not found or inactive")

            # Create new access token
            access_token = self.jwt_manager.create_access_token(
                user_id=str(user.id),
                user_email=user.email,
                roles=[user.role.value]
            )

            logger.info("Access token refreshed", user_id=user.id)
            return access_token

        except AuthenticationError:
            raise
        except Exception as e:
            logger.error("Token refresh failed", error=str(e))
            raise AuthenticationError("Token refresh failed")

# Dependency injection functions
jwt_manager = JWTManager()
user_manager = UserManager()

async def get_current_user_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get and verify current user's JWT token"""
    try:
        token = credentials.credentials
        payload = jwt_manager.verify_token(token)

        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )

        return payload

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

async def get_current_user(
    token_payload: Dict[str, Any] = Depends(get_current_user_token),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )

        result = await db.execute(
            select(User).where(User.id == int(user_id), User.is_active == True)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get current user", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

async def require_role(required_role: UserRole):
    """Create dependency that requires specific user role"""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {required_role.value}"
            )
        return current_user
    return role_checker

async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

async def require_premium(current_user: User = Depends(get_current_user)) -> User:
    """Require premium or admin role"""
    if current_user.role not in [UserRole.PREMIUM, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_user