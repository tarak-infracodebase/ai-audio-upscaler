"""
Security Tests
Comprehensive security testing for authentication, authorization, and security features
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
import jwt
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import JWTManager, AzureB2CManager, UserManager, AuthenticationError, AuthorizationError
from app.core.security import (
    validate_filename, validate_file_content, validate_processing_parameters,
    RateLimiter, generate_secure_token, constant_time_compare, hash_password, verify_password
)
from app.models.user import User, UserRole

class TestJWTManager:
    """Test JWT token management"""

    def setup_method(self):
        """Setup test method"""
        with patch('app.core.auth.get_settings') as mock_settings:
            mock_settings.return_value.JWT_SECRET_KEY = "test-secret-key"
            self.jwt_manager = JWTManager()

    def test_create_access_token(self):
        """Test access token creation"""
        token = self.jwt_manager.create_access_token(
            user_id="123",
            user_email="test@example.com",
            roles=["user"]
        )

        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify token
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == "123"
        assert payload["email"] == "test@example.com"
        assert payload["roles"] == ["user"]
        assert payload["type"] == "access"

    def test_create_refresh_token(self):
        """Test refresh token creation"""
        token = self.jwt_manager.create_refresh_token("123")

        assert isinstance(token, str)
        assert len(token) > 0

        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == "123"
        assert payload["type"] == "refresh"

    def test_verify_valid_token(self):
        """Test valid token verification"""
        token = self.jwt_manager.create_access_token("123", "test@example.com", ["user"])
        payload = self.jwt_manager.verify_token(token)

        assert payload["sub"] == "123"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "access"

    def test_verify_expired_token(self):
        """Test expired token verification"""
        # Create token that expires immediately
        past_time = datetime.now(timezone.utc) - timedelta(seconds=1)
        payload = {
            "sub": "123",
            "email": "test@example.com",
            "roles": ["user"],
            "iat": past_time,
            "exp": past_time,
            "type": "access"
        }

        expired_token = jwt.encode(payload, "test-secret-key", algorithm="HS256")

        with pytest.raises(AuthenticationError, match="Token has expired"):
            self.jwt_manager.verify_token(expired_token)

    def test_verify_invalid_token(self):
        """Test invalid token verification"""
        with pytest.raises(AuthenticationError, match="Invalid token"):
            self.jwt_manager.verify_token("invalid-token")

    def test_verify_wrong_secret(self):
        """Test token with wrong secret"""
        wrong_token = jwt.encode({"sub": "123"}, "wrong-secret", algorithm="HS256")

        with pytest.raises(AuthenticationError):
            self.jwt_manager.verify_token(wrong_token)

class TestAzureB2CManager:
    """Test Azure B2C integration"""

    def setup_method(self):
        """Setup test method"""
        with patch('app.core.auth.get_settings') as mock_settings:
            mock_settings.return_value.AZURE_B2C_TENANT_NAME = "test-tenant"
            mock_settings.return_value.AZURE_B2C_POLICY_NAME = "B2C_1_signupsignin"
            mock_settings.return_value.AZURE_B2C_CLIENT_ID = "test-client-id"
            mock_settings.return_value.AZURE_B2C_CLIENT_SECRET = "test-client-secret"
            self.b2c_manager = AzureB2CManager()

    @pytest.mark.asyncio
    async def test_get_jwks_success(self):
        """Test successful JWKS retrieval"""
        mock_jwks = {"keys": [{"kid": "test-key", "n": "test-n", "e": "AQAB"}]}

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = mock_jwks
            mock_response.raise_for_status = Mock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            jwks = await self.b2c_manager.get_jwks()

            assert jwks == mock_jwks
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_jwks_http_error(self):
        """Test JWKS retrieval HTTP error"""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("HTTP error")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(AuthenticationError):
                await self.b2c_manager.get_jwks()

    @pytest.mark.asyncio
    async def test_validate_b2c_token_missing_kid(self):
        """Test B2C token validation with missing key ID"""
        test_token = jwt.encode({"sub": "123"}, "secret", algorithm="HS256")

        # Mock jwt.get_unverified_header to return empty dict
        with patch('jwt.get_unverified_header', return_value={}):
            with pytest.raises(AuthenticationError, match="Token missing key ID"):
                await self.b2c_manager.validate_b2c_token(test_token)

class TestSecurityValidation:
    """Test security validation functions"""

    def test_validate_filename_valid(self):
        """Test valid filename validation"""
        valid_names = [
            "test.wav",
            "audio_file.mp3",
            "My-Song.flac",
            "recording123.ogg"
        ]

        for filename in valid_names:
            result = validate_filename(filename)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_validate_filename_invalid_extension(self):
        """Test filename with invalid extension"""
        with pytest.raises(Exception):  # Should raise ValidationError
            validate_filename("test.exe")

    def test_validate_filename_path_traversal(self):
        """Test filename with path traversal attempt"""
        dangerous_names = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\calc.exe",
            "/etc/shadow.wav",
            "C:\\Windows\\System32\\cmd.exe.mp3"
        ]

        for filename in dangerous_names:
            result = validate_filename(filename)
            # Should strip path components
            assert "/" not in result
            assert "\\" not in result
            assert not result.startswith(".")

    def test_validate_filename_empty(self):
        """Test empty filename validation"""
        with pytest.raises(Exception):
            validate_filename("")

    def test_validate_filename_too_long(self):
        """Test filename too long"""
        long_name = "a" * 300 + ".wav"
        with pytest.raises(Exception):
            validate_filename(long_name)

    def test_validate_file_content_valid(self):
        """Test valid file content validation"""
        # Mock WAV file header
        wav_header = b'RIFF\x24\x08\x00\x00WAVE'
        result = validate_file_content(wav_header, "test.wav")
        assert result is True

    def test_validate_file_content_empty(self):
        """Test empty file content"""
        with pytest.raises(Exception):
            validate_file_content(b'', "test.wav")

    def test_validate_file_content_too_large(self):
        """Test file content too large"""
        # Create content larger than MAX_FILE_SIZE
        large_content = b'x' * (600 * 1024 * 1024)  # 600MB
        with pytest.raises(Exception):
            validate_file_content(large_content, "test.wav")

    def test_validate_processing_parameters_valid(self):
        """Test valid processing parameters"""
        params = {
            'target_sample_rate': 48000,
            'mode': 'ai',
            'use_ai': True,
            'transient_strength': 0.5
        }

        result = validate_processing_parameters(params)
        assert result['target_sample_rate'] == 48000
        assert result['mode'] == 'ai'
        assert result['use_ai'] is True
        assert result['transient_strength'] == 0.5

    def test_validate_processing_parameters_invalid_sample_rate(self):
        """Test invalid sample rate"""
        params = {'target_sample_rate': 1000000}  # Too high
        with pytest.raises(Exception):
            validate_processing_parameters(params)

    def test_validate_processing_parameters_invalid_mode(self):
        """Test invalid mode"""
        params = {'mode': 'invalid_mode'}
        with pytest.raises(Exception):
            validate_processing_parameters(params)

class TestRateLimiter:
    """Test rate limiting functionality"""

    def test_rate_limiter_creation(self):
        """Test rate limiter creation"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.max_requests == 5
        assert limiter.window_seconds == 60

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        # Should allow first 3 requests
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True

        # Should deny 4th request
        assert limiter.is_allowed("user1") is False

    def test_rate_limiter_different_keys(self):
        """Test rate limiter with different keys"""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Different keys should have separate limits
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user2") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user2") is True

        # Both should be at limit now
        assert limiter.is_allowed("user1") is False
        assert limiter.is_allowed("user2") is False

    def test_rate_limiter_remaining_count(self):
        """Test remaining request count"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        assert limiter.get_remaining("user1") == 5

        limiter.is_allowed("user1")
        assert limiter.get_remaining("user1") == 4

        limiter.is_allowed("user1")
        assert limiter.get_remaining("user1") == 3

class TestSecurityUtilities:
    """Test security utility functions"""

    def test_generate_secure_token(self):
        """Test secure token generation"""
        token1 = generate_secure_token()
        token2 = generate_secure_token()

        assert len(token1) > 0
        assert len(token2) > 0
        assert token1 != token2  # Should be random

        # Test custom length
        long_token = generate_secure_token(64)
        assert len(long_token) > len(token1)

    def test_constant_time_compare_equal(self):
        """Test constant time string comparison with equal strings"""
        assert constant_time_compare("password", "password") is True
        assert constant_time_compare("", "") is True

    def test_constant_time_compare_not_equal(self):
        """Test constant time string comparison with different strings"""
        assert constant_time_compare("password", "wrong") is False
        assert constant_time_compare("password", "Password") is False
        assert constant_time_compare("password", "") is False

    def test_hash_password(self):
        """Test password hashing"""
        password = "test_password_123"
        hashed, salt = hash_password(password)

        assert isinstance(hashed, str)
        assert isinstance(salt, bytes)
        assert len(hashed) > 0
        assert len(salt) > 0

        # Same password should produce different hashes with different salts
        hashed2, salt2 = hash_password(password)
        assert hashed != hashed2
        assert salt != salt2

    def test_verify_password_correct(self):
        """Test password verification with correct password"""
        password = "test_password_123"
        hashed, salt = hash_password(password)

        assert verify_password(password, hashed, salt) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password"""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed, salt = hash_password(password)

        assert verify_password(wrong_password, hashed, salt) is False

    def test_verify_password_invalid_hash(self):
        """Test password verification with invalid hash"""
        password = "test_password_123"
        _, salt = hash_password(password)

        # Invalid hash should return False, not raise exception
        assert verify_password(password, "invalid_hash", salt) is False

@pytest.mark.asyncio
class TestUserManager:
    """Test user management functionality"""

    def setup_method(self):
        """Setup test method"""
        with patch('app.core.auth.get_settings') as mock_settings:
            mock_settings.return_value.JWT_SECRET_KEY = "test-secret"
            self.user_manager = UserManager()

    async def test_get_or_create_user_new(self):
        """Test creating new user"""
        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None  # No existing user
        mock_db.execute.return_value = mock_result

        # Mock user creation
        mock_user = Mock()
        mock_user.id = 1
        mock_user.email = "test@example.com"
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()

        with patch('app.models.user.User', return_value=mock_user):
            user = await self.user_manager.get_or_create_user(
                db=mock_db,
                b2c_user_id="b2c-123",
                email="test@example.com",
                name="Test User"
            )

            mock_db.add.assert_called_once()
            mock_db.flush.assert_called_once()

    async def test_authenticate_user_invalid_token(self):
        """Test user authentication with invalid B2C token"""
        mock_db = AsyncMock()

        # Mock B2C manager to raise authentication error
        with patch.object(self.user_manager.b2c_manager, 'validate_b2c_token') as mock_validate:
            mock_validate.side_effect = AuthenticationError("Invalid token")

            with pytest.raises(AuthenticationError):
                await self.user_manager.authenticate_user(mock_db, "invalid_token")

@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security features"""

    def test_security_headers_middleware(self):
        """Test security headers are applied"""
        from app.api.middleware import SecurityHeadersMiddleware
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_rate_limit_middleware(self):
        """Test rate limiting middleware"""
        from app.api.middleware import RateLimitMiddleware
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        # First request should succeed
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers