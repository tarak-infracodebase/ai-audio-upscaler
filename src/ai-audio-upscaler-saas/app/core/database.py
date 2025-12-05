"""
Database Configuration and Session Management
PostgreSQL async database setup with SQLAlchemy
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
import structlog
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# Create declarative base
Base = declarative_base()

# Global engine and session maker
engine = None
async_session_maker = None

def create_database_engine():
    """Create database engine with proper configuration"""
    settings = get_settings()

    return create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_POOL_OVERFLOW,
        pool_timeout=settings.DATABASE_POOL_TIMEOUT,
        pool_pre_ping=True,  # Validate connections before use
        poolclass=NullPool if settings.ENVIRONMENT == "test" else None,
    )

async def init_database():
    """Initialize database connection"""
    global engine, async_session_maker

    if engine is not None:
        return

    try:
        engine = create_database_engine()
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise

async def close_database():
    """Close database connections"""
    global engine, async_session_maker

    if engine:
        await engine.dispose()
        engine = None
        async_session_maker = None
        logger.info("Database connections closed")

async def create_tables():
    """Create all database tables"""
    if not engine:
        await init_database()

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")

    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise

async def drop_tables():
    """Drop all database tables (for testing)"""
    if not engine:
        await init_database()

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")

    except Exception as e:
        logger.error("Failed to drop database tables", error=str(e))
        raise

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session

    Usage:
        async def endpoint(db: AsyncSession = Depends(get_db)):
            # Use db session
            pass
    """
    if not async_session_maker:
        await init_database()

    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Health check function
async def check_database_health() -> bool:
    """Check database connectivity"""
    try:
        if not engine:
            await init_database()

        async with engine.begin() as conn:
            await conn.execute("SELECT 1")

        return True

    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False