"""
Token Revocation and Session Management
Implements secure token blacklisting and session management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Set
import redis.asyncio as redis
from redis.asyncio import Redis
import json

from .config import get_settings

logger = logging.getLogger(__name__)

class TokenRevocationService:
    """
    Manages token revocation using Redis for high-performance blacklisting
    """

    def __init__(self, redis_client: Optional[Redis] = None):
        self.settings = get_settings()
        self.redis_client = redis_client
        self._revoked_tokens: Set[str] = set()  # In-memory cache for critical performance

    async def connect(self):
        """Initialize Redis connection if not provided"""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                self.settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True
            )

        # Test connection
        try:
            await self.redis_client.ping()
            logger.info("Token revocation service connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def revoke_token(self, token_jti: str, expires_at: datetime, user_id: str) -> bool:
        """
        Revoke a token by adding it to the blacklist

        Args:
            token_jti: JWT ID (unique identifier for the token)
            expires_at: When the token expires naturally
            user_id: User ID for audit logging

        Returns:
            bool: True if successfully revoked
        """
        try:
            # Calculate TTL (time until token would expire naturally)
            ttl_seconds = max(0, int((expires_at - datetime.utcnow()).total_seconds()))

            if ttl_seconds <= 0:
                # Token already expired, no need to revoke
                return True

            # Store in Redis with expiration
            revocation_key = f"revoked_token:{token_jti}"
            revocation_data = {
                "user_id": user_id,
                "revoked_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat()
            }

            # Use Redis transaction for atomicity
            async with self.redis_client.pipeline() as pipe:
                await pipe.setex(
                    revocation_key,
                    ttl_seconds,
                    json.dumps(revocation_data)
                )

                # Add to user's revoked tokens list (for bulk revocation)
                user_tokens_key = f"user_tokens:{user_id}"
                await pipe.sadd(user_tokens_key, token_jti)
                await pipe.expire(user_tokens_key, ttl_seconds)

                # Execute transaction
                await pipe.execute()

            # Update in-memory cache
            self._revoked_tokens.add(token_jti)

            logger.info(
                f"Token revoked successfully",
                extra={
                    "token_jti": token_jti,
                    "user_id": user_id,
                    "ttl_seconds": ttl_seconds
                }
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to revoke token: {e}",
                extra={
                    "token_jti": token_jti,
                    "user_id": user_id,
                    "error": str(e)
                }
            )
            return False

    async def is_token_revoked(self, token_jti: str) -> bool:
        """
        Check if a token is revoked

        Uses multi-layer caching for performance:
        1. In-memory cache (fastest)
        2. Redis lookup (fast)
        3. Return False if Redis is unavailable (fail open with logging)
        """

        # Check in-memory cache first (fastest)
        if token_jti in self._revoked_tokens:
            return True

        try:
            # Check Redis
            revocation_key = f"revoked_token:{token_jti}"
            is_revoked = await self.redis_client.exists(revocation_key)

            # Update in-memory cache if revoked
            if is_revoked:
                self._revoked_tokens.add(token_jti)
                return True

            return False

        except Exception as e:
            # Log error but don't block authentication
            # This is a "fail open" approach - prefer availability over security
            # In production, you might want to "fail closed" depending on requirements
            logger.error(
                f"Failed to check token revocation status: {e}",
                extra={
                    "token_jti": token_jti,
                    "error": str(e)
                }
            )
            return False

    async def revoke_all_user_tokens(self, user_id: str) -> int:
        """
        Revoke all tokens for a specific user (e.g., on password change, account compromise)

        Returns:
            int: Number of tokens revoked
        """
        try:
            user_tokens_key = f"user_tokens:{user_id}"
            token_jtis = await self.redis_client.smembers(user_tokens_key)

            if not token_jtis:
                return 0

            revoked_count = 0
            async with self.redis_client.pipeline() as pipe:
                for token_jti in token_jtis:
                    # Check if token still exists in revocation store
                    revocation_key = f"revoked_token:{token_jti}"
                    if await self.redis_client.exists(revocation_key):
                        # Extend expiration for bulk revocation tracking
                        await pipe.expire(revocation_key, 86400)  # 24 hours
                        revoked_count += 1

                    # Update in-memory cache
                    self._revoked_tokens.add(token_jti)

                # Clear user's token list
                await pipe.delete(user_tokens_key)
                await pipe.execute()

            logger.warning(
                f"Revoked all tokens for user",
                extra={
                    "user_id": user_id,
                    "revoked_count": revoked_count
                }
            )

            return revoked_count

        except Exception as e:
            logger.error(
                f"Failed to revoke all user tokens: {e}",
                extra={
                    "user_id": user_id,
                    "error": str(e)
                }
            )
            return 0

    async def cleanup_expired_tokens(self) -> int:
        """
        Cleanup expired tokens from in-memory cache
        This is called periodically to prevent memory leaks
        """
        # Redis handles expiration automatically, but we need to clean in-memory cache
        cleaned_count = 0
        tokens_to_remove = set()

        try:
            # Check each token in memory cache
            for token_jti in list(self._revoked_tokens):
                revocation_key = f"revoked_token:{token_jti}"
                if not await self.redis_client.exists(revocation_key):
                    tokens_to_remove.add(token_jti)
                    cleaned_count += 1

            # Remove expired tokens from memory
            self._revoked_tokens -= tokens_to_remove

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired tokens from memory cache")

            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
            return 0

    async def get_revocation_stats(self) -> dict:
        """Get statistics about token revocations"""
        try:
            # Count revoked tokens in Redis
            cursor = 0
            revoked_count = 0

            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match="revoked_token:*",
                    count=1000
                )
                revoked_count += len(keys)

                if cursor == 0:
                    break

            return {
                "total_revoked_tokens": revoked_count,
                "memory_cache_size": len(self._revoked_tokens),
                "redis_connected": await self.redis_client.ping() == True
            }

        except Exception as e:
            logger.error(f"Failed to get revocation stats: {e}")
            return {
                "error": str(e),
                "memory_cache_size": len(self._revoked_tokens),
                "redis_connected": False
            }

# Global instance
_token_revocation_service: Optional[TokenRevocationService] = None

async def get_token_revocation_service() -> TokenRevocationService:
    """Get or create the global token revocation service"""
    global _token_revocation_service

    if _token_revocation_service is None:
        _token_revocation_service = TokenRevocationService()
        await _token_revocation_service.connect()

    return _token_revocation_service

# Cleanup task
async def start_cleanup_task():
    """Start background task for token cleanup"""
    async def cleanup_loop():
        while True:
            try:
                service = await get_token_revocation_service()
                await service.cleanup_expired_tokens()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Token cleanup task error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error

    # Start as background task
    asyncio.create_task(cleanup_loop())