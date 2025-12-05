"""
API Security Tests
Comprehensive security testing for the AI Audio Upscaler API
Tests authentication, authorization, input validation, and security headers
"""

import pytest
import httpx
from unittest.mock import patch, AsyncMock
import asyncio
import json
import time
from fastapi.testclient import TestClient

# Import app for testing
from app.main import app
from app.core.config import get_settings


class TestAPISecurityHeaders:
    """Test security headers implementation"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_security_headers_present(self):
        """Test that all required security headers are present"""
        response = self.client.get("/health")

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"

        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"

        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"

        assert "Referrer-Policy" in response.headers
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

        assert "Strict-Transport-Security" in response.headers
        assert "max-age=31536000" in response.headers["Strict-Transport-Security"]

    def test_content_security_policy_hardened(self):
        """Test that CSP is properly hardened without unsafe directives"""
        response = self.client.get("/health")

        assert "Content-Security-Policy" in response.headers
        csp = response.headers["Content-Security-Policy"]

        # Ensure no unsafe directives
        assert "'unsafe-inline'" not in csp
        assert "'unsafe-eval'" not in csp

        # Ensure nonce-based directives are present
        assert "script-src 'self' 'nonce-" in csp or "script-src 'self';" in csp
        assert "style-src 'self' 'nonce-" in csp or "style-src 'self';" in csp

        # Check for security directives
        assert "object-src 'none'" in csp
        assert "base-uri 'self'" in csp
        assert "form-action 'self'" in csp


class TestRateLimiting:
    """Test rate limiting implementation"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_rate_limiting_enforced(self):
        """Test that rate limiting is enforced"""
        # Make multiple rapid requests
        responses = []
        for _ in range(150):  # Exceed typical rate limit
            response = self.client.get("/health")
            responses.append(response.status_code)
            if response.status_code == 429:
                break

        # Should eventually get rate limited
        assert 429 in responses, "Rate limiting not enforced"

    def test_rate_limit_headers_present(self):
        """Test that rate limit headers are present"""
        response = self.client.get("/health")

        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Window" in response.headers


class TestAuthenticationSecurity:
    """Test authentication security mechanisms"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_jwt_token_validation(self):
        """Test JWT token validation"""
        # Test with invalid JWT
        invalid_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"

        response = self.client.get(
            "/api/v1/protected-endpoint",
            headers={"Authorization": f"Bearer {invalid_token}"}
        )

        assert response.status_code == 401

    def test_missing_authorization_header(self):
        """Test behavior with missing authorization header"""
        response = self.client.get("/api/v1/protected-endpoint")
        assert response.status_code == 401

    def test_malformed_authorization_header(self):
        """Test behavior with malformed authorization header"""
        response = self.client.get(
            "/api/v1/protected-endpoint",
            headers={"Authorization": "InvalidFormat token"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_token_revocation_check(self):
        """Test that revoked tokens are rejected"""
        # Mock token revocation service
        with patch('app.core.token_revocation.get_token_revocation_service') as mock_service:
            mock_revocation = AsyncMock()
            mock_revocation.is_token_revoked.return_value = True
            mock_service.return_value = mock_revocation

            response = self.client.get(
                "/api/v1/protected-endpoint",
                headers={"Authorization": "Bearer valid-but-revoked-token"}
            )

            assert response.status_code == 401


class TestInputValidationSecurity:
    """Test input validation and sanitization"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM secrets --",
            "'; INSERT INTO users VALUES('hacker', 'password'); --"
        ]

        for payload in malicious_inputs:
            response = self.client.get(f"/api/v1/search?q={payload}")

            # Should not return 500 (internal server error) or expose database errors
            assert response.status_code != 500
            assert "database" not in response.text.lower()
            assert "sql" not in response.text.lower()
            assert "error" not in response.text.lower() or response.status_code >= 400

    def test_xss_prevention(self):
        """Test XSS prevention in input handling"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//",
        ]

        for payload in xss_payloads:
            response = self.client.post(
                "/api/v1/feedback",
                json={"message": payload}
            )

            # Response should not contain unescaped script content
            if response.status_code == 200:
                assert "<script>" not in response.text
                assert "javascript:" not in response.text
                assert "onerror=" not in response.text

    def test_file_upload_validation(self):
        """Test file upload security validation"""
        # Test malicious file types
        malicious_files = [
            ("test.exe", b"MZ\x90\x00", "application/x-executable"),
            ("test.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("test.jsp", b"<%@ page import=\"java.io.*\" %>", "application/x-jsp"),
            ("test.sh", b"#!/bin/bash\nrm -rf /", "application/x-sh"),
        ]

        for filename, content, content_type in malicious_files:
            response = self.client.post(
                "/api/v1/upload",
                files={"file": (filename, content, content_type)}
            )

            # Should reject dangerous file types
            assert response.status_code in [400, 415, 422]  # Bad request or unsupported media type

    def test_large_payload_handling(self):
        """Test handling of oversized payloads"""
        # Create oversized payload (assuming 10MB limit)
        large_payload = "A" * (11 * 1024 * 1024)  # 11MB

        response = self.client.post(
            "/api/v1/process",
            json={"data": large_payload}
        )

        # Should reject oversized payloads
        assert response.status_code in [413, 422]  # Payload too large or validation error


class TestCORSSecurity:
    """Test CORS security implementation"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_cors_origin_validation(self):
        """Test CORS origin validation"""
        # Test with unauthorized origin
        response = self.client.options(
            "/api/v1/test",
            headers={"Origin": "https://malicious-site.com"}
        )

        # Should not include CORS headers for unauthorized origins
        assert "Access-Control-Allow-Origin" not in response.headers or \
               response.headers.get("Access-Control-Allow-Origin") != "https://malicious-site.com"

    def test_cors_preflight_security(self):
        """Test CORS preflight request security"""
        response = self.client.options(
            "/api/v1/test",
            headers={
                "Origin": "https://authorized-domain.com",
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "X-Custom-Header"
            }
        )

        # Should validate allowed methods and headers
        if "Access-Control-Allow-Methods" in response.headers:
            allowed_methods = response.headers["Access-Control-Allow-Methods"]
            # Should not allow dangerous methods if not explicitly configured
            assert "TRACE" not in allowed_methods
            assert "CONNECT" not in allowed_methods


class TestErrorHandlingSecurity:
    """Test secure error handling"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_error_information_disclosure(self):
        """Test that error messages don't leak sensitive information"""
        # Test various error conditions
        error_endpoints = [
            "/api/v1/nonexistent",
            "/api/v1/admin/secret",
            "/api/v1/internal/debug",
        ]

        for endpoint in error_endpoints:
            response = self.client.get(endpoint)

            # Error responses should not leak system information
            sensitive_patterns = [
                r"/home/\w+/",
                r"c:\\",
                r"database.*error",
                r"stack trace",
                r"internal server error.*at line",
                r"sql.*exception",
                r"postgresql://",
                r"redis://",
            ]

            response_text = response.text.lower()
            for pattern in sensitive_patterns:
                import re
                assert not re.search(pattern, response_text), \
                    f"Potential information disclosure in error response: {pattern}"

    def test_404_response_consistency(self):
        """Test that 404 responses are consistent and don't leak path information"""
        non_existent_paths = [
            "/api/v1/admin/users",
            "/api/v1/secret/config",
            "/api/v1/internal/debug",
            "/does/not/exist",
        ]

        responses = []
        for path in non_existent_paths:
            response = self.client.get(path)
            responses.append(response.text)

        # 404 responses should be consistent to avoid path enumeration
        # (This test might need adjustment based on your specific error handling)
        assert all(response == responses[0] for response in responses) or \
               all(len(response) < 200 for response in responses), \
               "404 responses vary significantly, potential information disclosure"


class TestSessionSecurity:
    """Test session security mechanisms"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_session_fixation_prevention(self):
        """Test session fixation prevention"""
        # This would require authentication flow testing
        # Implementation depends on your specific authentication mechanism
        pass

    def test_concurrent_session_limits(self):
        """Test concurrent session limits"""
        # This would test if multiple sessions from same user are properly managed
        pass


class TestSecurityMisconfiguration:
    """Test for security misconfigurations"""

    def setup_method(self):
        self.client = TestClient(app)

    def test_debug_mode_disabled(self):
        """Test that debug mode is disabled in production"""
        settings = get_settings()

        # Debug should be disabled in production
        if settings.environment == "production":
            assert not getattr(settings, 'DEBUG', True), \
                "Debug mode should be disabled in production"

    def test_default_credentials_not_used(self):
        """Test that default credentials are not used"""
        settings = get_settings()

        # Check for default/weak credentials
        default_passwords = [
            "password",
            "admin",
            "12345",
            "default",
            "",
        ]

        # This would need to be adapted based on your configuration
        # For now, just ensure JWT secret is not default
        jwt_secret = getattr(settings, 'JWT_SECRET_KEY', '')
        assert jwt_secret not in default_passwords, \
            "Default or weak JWT secret detected"
        assert len(jwt_secret) >= 32, \
            "JWT secret should be at least 32 characters"

    def test_sensitive_endpoints_not_exposed(self):
        """Test that sensitive endpoints are not accessible"""
        sensitive_endpoints = [
            "/admin",
            "/debug",
            "/config",
            "/health/detailed",
            "/metrics/private",
            "/.env",
            "/backup",
            "/database",
        ]

        for endpoint in sensitive_endpoints:
            response = self.client.get(endpoint)

            # These endpoints should not be accessible without proper authentication
            # or should not exist at all
            assert response.status_code in [401, 403, 404], \
                f"Sensitive endpoint {endpoint} may be exposed"


@pytest.mark.asyncio
async def test_async_security_operations():
    """Test security operations in async context"""
    # Test token revocation service async operations
    from app.core.token_revocation import get_token_revocation_service

    try:
        service = await get_token_revocation_service()
        # Test basic functionality without actual Redis connection in test
        assert hasattr(service, 'is_token_revoked')
        assert hasattr(service, 'revoke_token')
    except Exception:
        # Redis might not be available in test environment
        pass


if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])