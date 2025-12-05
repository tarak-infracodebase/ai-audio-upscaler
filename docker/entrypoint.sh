#!/bin/bash
set -e

# AI Audio Upscaler Pro - Production Entrypoint Script
# Handles different service modes and graceful startup

# Default environment variables
export PYTHONPATH=${PYTHONPATH:-/app}
export ENVIRONMENT=${ENVIRONMENT:-production}
export LOG_LEVEL=${LOG_LEVEL:-info}

# Function to wait for dependencies
wait_for_redis() {
    echo "Waiting for Redis..."
    until python -c "import redis; r=redis.Redis(host='${REDIS_HOST:-redis}', port=${REDIS_PORT:-6379}); r.ping()"; do
        echo "Redis is unavailable - sleeping"
        sleep 2
    done
    echo "Redis is ready!"
}

wait_for_postgres() {
    echo "Waiting for PostgreSQL..."
    until python -c "import asyncpg; import asyncio; asyncio.run(asyncpg.connect('${DATABASE_URL}'))"; do
        echo "PostgreSQL is unavailable - sleeping"
        sleep 2
    done
    echo "PostgreSQL is ready!"
}

# Function to run database migrations
run_migrations() {
    echo "Running database migrations..."
    cd /app
    alembic upgrade head
    echo "Migrations completed"
}

# Function to check GPU availability
check_gpu() {
    echo "Checking GPU availability..."
    python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB')
else:
    print('Running in CPU-only mode')
"
}

# Function to validate environment
validate_environment() {
    echo "Validating environment..."
    required_vars=("DATABASE_URL" "REDIS_URL" "AZURE_STORAGE_CONNECTION_STRING")

    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "ERROR: Required environment variable $var is not set"
            exit 1
        fi
    done
    echo "Environment validation passed"
}

# Main service startup
case "$1" in
    api)
        echo "Starting AI Audio Upscaler API Service..."
        validate_environment
        wait_for_redis
        wait_for_postgres
        run_migrations
        check_gpu

        # Start the FastAPI application with Gunicorn
        exec gunicorn app.api.main:app \
            --bind 0.0.0.0:8000 \
            --workers ${API_WORKERS:-4} \
            --worker-class uvicorn.workers.UvicornWorker \
            --worker-connections ${API_WORKER_CONNECTIONS:-1000} \
            --max-requests ${API_MAX_REQUESTS:-1000} \
            --max-requests-jitter ${API_MAX_REQUESTS_JITTER:-100} \
            --timeout ${API_TIMEOUT:-120} \
            --keep-alive ${API_KEEP_ALIVE:-2} \
            --log-level ${LOG_LEVEL} \
            --access-logfile - \
            --error-logfile - \
            --log-config /app/docker/logging.conf
        ;;

    worker)
        echo "Starting AI Audio Upscaler Celery Worker..."
        validate_environment
        wait_for_redis
        wait_for_postgres
        check_gpu

        # Start Celery worker for audio processing
        exec celery -A app.worker.celery worker \
            --loglevel=${LOG_LEVEL} \
            --concurrency=${WORKER_CONCURRENCY:-2} \
            --max-tasks-per-child=${WORKER_MAX_TASKS:-10} \
            --time-limit=${WORKER_TIME_LIMIT:-3600} \
            --soft-time-limit=${WORKER_SOFT_TIME_LIMIT:-3300} \
            --prefetch-multiplier=${WORKER_PREFETCH:-1} \
            --queues=audio_processing,high_priority,low_priority
        ;;

    scheduler)
        echo "Starting Celery Beat Scheduler..."
        validate_environment
        wait_for_redis

        # Start Celery beat for scheduled tasks
        exec celery -A app.worker.celery beat \
            --loglevel=${LOG_LEVEL} \
            --schedule-filename=/tmp/celerybeat-schedule \
            --pidfile=/tmp/celerybeat.pid
        ;;

    flower)
        echo "Starting Flower (Celery Monitoring)..."
        wait_for_redis

        # Start Flower for monitoring Celery
        exec celery -A app.worker.celery flower \
            --port=5555 \
            --basic_auth=${FLOWER_USER:-admin}:${FLOWER_PASSWORD:-admin} \
            --url_prefix=${FLOWER_URL_PREFIX:-/flower}
        ;;

    migrate)
        echo "Running database migrations only..."
        wait_for_postgres
        run_migrations
        echo "Migration complete, exiting"
        exit 0
        ;;

    shell)
        echo "Starting interactive shell..."
        exec python
        ;;

    test)
        echo "Running tests..."
        exec python -m pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing
        ;;

    *)
        echo "Usage: $0 {api|worker|scheduler|flower|migrate|shell|test}"
        echo "Available commands:"
        echo "  api       - Start FastAPI application server"
        echo "  worker    - Start Celery worker for audio processing"
        echo "  scheduler - Start Celery beat scheduler"
        echo "  flower    - Start Flower monitoring interface"
        echo "  migrate   - Run database migrations"
        echo "  shell     - Start Python interactive shell"
        echo "  test      - Run test suite"
        exit 1
        ;;
esac