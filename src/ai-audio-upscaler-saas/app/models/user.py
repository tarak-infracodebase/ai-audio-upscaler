"""
User Models
Database models for user management and authentication
"""

from datetime import datetime, timezone
from typing import Optional, List
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from app.core.database import Base

class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "admin"
    PREMIUM = "premium"
    USER = "user"

class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"

class User(Base):
    """User model for authentication and profile management"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    # Authentication fields
    azure_b2c_id = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)

    # Profile fields
    name = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)

    # Role and permissions
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    status = Column(SQLEnum(UserStatus), default=UserStatus.ACTIVE, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Usage tracking
    total_processing_minutes = Column(Integer, default=0, nullable=False)
    monthly_processing_minutes = Column(Integer, default=0, nullable=False)
    last_reset_date = Column(DateTime(timezone=True), nullable=True)

    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    # Preferences
    preferences = Column(Text, nullable=True)  # JSON string for user preferences

    # Relationships
    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role"""
        return self.role == UserRole.ADMIN

    @property
    def is_premium(self) -> bool:
        """Check if user has premium access"""
        return self.role in [UserRole.PREMIUM, UserRole.ADMIN]

    @property
    def display_name(self) -> str:
        """Get display name for user"""
        return self.name or self.email.split("@")[0]

    def can_process_audio(self, estimated_minutes: int = 0) -> bool:
        """Check if user can process audio based on role and usage"""
        if self.role == UserRole.ADMIN:
            return True

        if self.role == UserRole.PREMIUM:
            # Premium users have higher limits
            monthly_limit = 300  # 5 hours per month
        else:
            # Free users have basic limits
            monthly_limit = 60   # 1 hour per month

        return (self.monthly_processing_minutes + estimated_minutes) <= monthly_limit

    def add_processing_time(self, minutes: int) -> None:
        """Add processing time to user's usage tracking"""
        self.total_processing_minutes += minutes
        self.monthly_processing_minutes += minutes
        self.updated_at = datetime.now(timezone.utc)

    def reset_monthly_usage(self) -> None:
        """Reset monthly usage counter (typically called monthly)"""
        self.monthly_processing_minutes = 0
        self.last_reset_date = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)