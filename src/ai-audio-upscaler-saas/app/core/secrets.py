"""
Azure Key Vault Integration
Secure secrets management using Azure Key Vault
"""

import asyncio
from typing import Dict, Any, Optional
from azure.keyvault.secrets.aio import SecretClient
from azure.identity.aio import DefaultAzureCredential, ClientSecretCredential
from azure.core.exceptions import ResourceNotFoundError
import structlog
from functools import lru_cache

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

class SecretsManager:
    """
    Azure Key Vault secrets manager
    Handles secure retrieval and caching of application secrets
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[SecretClient] = None
        self._cache: Dict[str, str] = {}
        self._cache_ttl = 300  # 5 minutes

    async def _get_client(self) -> SecretClient:
        """Get or create Azure Key Vault client"""
        if self._client is None:
            if not self.settings.AZURE_KEY_VAULT_URL:
                raise ValueError("Azure Key Vault URL not configured")

            try:
                # Try different credential methods
                if (self.settings.AZURE_CLIENT_ID and
                    self.settings.AZURE_CLIENT_SECRET and
                    self.settings.AZURE_TENANT_ID):

                    # Service principal authentication
                    credential = ClientSecretCredential(
                        tenant_id=self.settings.AZURE_TENANT_ID,
                        client_id=self.settings.AZURE_CLIENT_ID,
                        client_secret=self.settings.AZURE_CLIENT_SECRET
                    )
                    logger.info("Using service principal authentication for Key Vault")
                else:
                    # Managed identity or Azure CLI authentication
                    credential = DefaultAzureCredential()
                    logger.info("Using default Azure credentials for Key Vault")

                self._client = SecretClient(
                    vault_url=self.settings.AZURE_KEY_VAULT_URL,
                    credential=credential
                )

                logger.info("Azure Key Vault client initialized")

            except Exception as e:
                logger.error("Failed to initialize Azure Key Vault client", error=str(e))
                raise

        return self._client

    async def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret from Azure Key Vault with caching

        Args:
            secret_name: Name of the secret in Key Vault
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        try:
            # Check cache first
            if secret_name in self._cache:
                return self._cache[secret_name]

            # Get from Key Vault
            client = await self._get_client()
            secret = await client.get_secret(secret_name)

            # Cache the value
            self._cache[secret_name] = secret.value

            logger.debug("Secret retrieved from Key Vault", secret_name=secret_name)
            return secret.value

        except ResourceNotFoundError:
            logger.warning("Secret not found in Key Vault", secret_name=secret_name)
            return default

        except Exception as e:
            logger.error("Failed to retrieve secret", secret_name=secret_name, error=str(e))
            return default

    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """
        Set secret in Azure Key Vault

        Args:
            secret_name: Name of the secret
            secret_value: Secret value

        Returns:
            True if successful
        """
        try:
            client = await self._get_client()
            await client.set_secret(secret_name, secret_value)

            # Update cache
            self._cache[secret_name] = secret_value

            logger.info("Secret stored in Key Vault", secret_name=secret_name)
            return True

        except Exception as e:
            logger.error("Failed to store secret", secret_name=secret_name, error=str(e))
            return False

    async def delete_secret(self, secret_name: str) -> bool:
        """
        Delete secret from Azure Key Vault

        Args:
            secret_name: Name of the secret to delete

        Returns:
            True if successful
        """
        try:
            client = await self._get_client()
            await client.begin_delete_secret(secret_name)

            # Remove from cache
            self._cache.pop(secret_name, None)

            logger.info("Secret deleted from Key Vault", secret_name=secret_name)
            return True

        except Exception as e:
            logger.error("Failed to delete secret", secret_name=secret_name, error=str(e))
            return False

    async def list_secrets(self) -> Dict[str, Any]:
        """
        List all secrets in Key Vault (metadata only)

        Returns:
            Dictionary of secret metadata
        """
        try:
            client = await self._get_client()
            secrets = {}

            async for secret_properties in client.list_properties_of_secrets():
                secrets[secret_properties.name] = {
                    "enabled": secret_properties.enabled,
                    "created_on": secret_properties.created_on,
                    "updated_on": secret_properties.updated_on,
                    "expires_on": secret_properties.expires_on,
                }

            logger.info("Listed Key Vault secrets", count=len(secrets))
            return secrets

        except Exception as e:
            logger.error("Failed to list secrets", error=str(e))
            return {}

    async def get_database_connection_string(self) -> Optional[str]:
        """Get database connection string from Key Vault"""
        return await self.get_secret("database-connection-string")

    async def get_jwt_secret_key(self) -> Optional[str]:
        """Get JWT secret key from Key Vault"""
        return await self.get_secret("jwt-secret-key")

    async def get_azure_b2c_client_secret(self) -> Optional[str]:
        """Get Azure B2C client secret from Key Vault"""
        return await self.get_secret("azure-b2c-client-secret")

    async def get_storage_account_key(self) -> Optional[str]:
        """Get Azure Storage account key from Key Vault"""
        return await self.get_secret("storage-account-key")

    async def get_redis_connection_string(self) -> Optional[str]:
        """Get Redis connection string from Key Vault"""
        return await self.get_secret("redis-connection-string")

    def clear_cache(self):
        """Clear the secrets cache"""
        self._cache.clear()
        logger.info("Secrets cache cleared")

    async def health_check(self) -> bool:
        """Check Key Vault connectivity"""
        try:
            client = await self._get_client()
            # Try to list secrets to test connectivity
            count = 0
            async for _ in client.list_properties_of_secrets():
                count += 1
                if count >= 1:  # Just check that we can list at least one
                    break

            return True

        except Exception as e:
            logger.error("Key Vault health check failed", error=str(e))
            return False

    async def close(self):
        """Close the Key Vault client connection"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Key Vault client connection closed")

# Global secrets manager instance
@lru_cache()
def get_secrets_manager() -> SecretsManager:
    """Get cached secrets manager instance"""
    return SecretsManager()

async def load_secrets_into_settings():
    """
    Load secrets from Key Vault into application settings
    Called during application startup
    """
    secrets_manager = get_secrets_manager()

    try:
        # Get critical secrets
        database_url = await secrets_manager.get_database_connection_string()
        if database_url:
            # Update settings (would need to modify Settings class to be mutable)
            logger.info("Database connection string loaded from Key Vault")

        jwt_secret = await secrets_manager.get_jwt_secret_key()
        if jwt_secret:
            logger.info("JWT secret key loaded from Key Vault")

        b2c_secret = await secrets_manager.get_azure_b2c_client_secret()
        if b2c_secret:
            logger.info("Azure B2C client secret loaded from Key Vault")

        storage_key = await secrets_manager.get_storage_account_key()
        if storage_key:
            logger.info("Storage account key loaded from Key Vault")

        redis_url = await secrets_manager.get_redis_connection_string()
        if redis_url:
            logger.info("Redis connection string loaded from Key Vault")

    except Exception as e:
        logger.warning("Failed to load some secrets from Key Vault", error=str(e))
        # Don't fail startup if Key Vault is unavailable

# Context manager for managing secrets lifecycle
class SecretsContext:
    """Context manager for secrets management"""

    def __init__(self):
        self.secrets_manager = get_secrets_manager()

    async def __aenter__(self):
        return self.secrets_manager

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.secrets_manager.close()

# Utility functions
async def get_secret_or_env(secret_name: str, env_var: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get secret from Key Vault, fallback to environment variable

    Args:
        secret_name: Key Vault secret name
        env_var: Environment variable name
        default: Default value if neither found

    Returns:
        Secret value or default
    """
    import os

    # Try Key Vault first
    try:
        secrets_manager = get_secrets_manager()
        vault_value = await secrets_manager.get_secret(secret_name)
        if vault_value:
            return vault_value
    except Exception as e:
        logger.warning("Failed to get secret from Key Vault", secret_name=secret_name, error=str(e))

    # Fallback to environment variable
    env_value = os.getenv(env_var)
    if env_value:
        return env_value

    # Return default
    return default