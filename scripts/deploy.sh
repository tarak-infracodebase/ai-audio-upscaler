#!/bin/bash
set -euo pipefail

# AI Audio Upscaler Pro - Deployment Script
# Production-ready deployment automation

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-production}"
AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-ai-upscaler-rg}"
AKS_CLUSTER="${AKS_CLUSTER:-ai-upscaler-aks}"
ACR_NAME="${ACR_NAME:-aiupscaleracr}"
IMAGE_NAME="${IMAGE_NAME:-ai-audio-upscaler}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
trap 'log_error "Deployment failed on line $LINENO. Exit code: $?"' ERR

# Usage function
usage() {
    cat << EOF
AI Audio Upscaler Pro Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    build           Build and push Docker images
    deploy          Deploy to Kubernetes
    migrate         Run database migrations
    rollback        Rollback to previous version
    status          Check deployment status
    logs            View application logs
    scale           Scale deployments
    cleanup         Clean up resources

Options:
    -e, --environment   Environment (production, staging) [default: production]
    -t, --tag           Docker image tag [default: latest]
    -r, --resource-group Azure resource group [default: ai-upscaler-rg]
    -c, --cluster       AKS cluster name [default: ai-upscaler-aks]
    -n, --namespace     Kubernetes namespace [default: ai-upscaler]
    -d, --dry-run       Show what would be done without executing
    -v, --verbose       Verbose output
    -h, --help          Show this help message

Examples:
    $0 build --tag v1.2.3
    $0 deploy --environment staging
    $0 scale --replicas 5
    $0 rollback --version v1.2.2
EOF
}

# Parse command line arguments
COMMAND=""
TAG="latest"
NAMESPACE="ai-upscaler"
DRY_RUN=false
VERBOSE=false
REPLICAS=""
VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -r|--resource-group)
            AZURE_RESOURCE_GROUP="$2"
            shift 2
            ;;
        -c|--cluster)
            AKS_CLUSTER="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        build|deploy|migrate|rollback|status|logs|scale|cleanup)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate command
if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    usage
    exit 1
fi

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Helper functions
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check required tools
    local tools=("az" "kubectl" "docker" "helm")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done

    # Check Azure login
    if ! az account show &> /dev/null; then
        log_error "Not logged into Azure. Run 'az login' first."
        exit 1
    fi

    # Check Kubernetes context
    if ! kubectl cluster-info &> /dev/null; then
        log_warning "No active Kubernetes context. Will attempt to get AKS credentials."
        get_aks_credentials
    fi

    log_success "Prerequisites check passed"
}

get_aks_credentials() {
    log_info "Getting AKS credentials..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would get credentials for cluster: $AKS_CLUSTER"
        return
    fi

    az aks get-credentials \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --name "$AKS_CLUSTER" \
        --overwrite-existing

    log_success "AKS credentials configured"
}

build_and_push_image() {
    log_info "Building and pushing Docker image..."

    local image_tag="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${TAG}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build and push image: $image_tag"
        return
    fi

    # Login to ACR
    az acr login --name "$ACR_NAME"

    # Build image
    log_info "Building Docker image: $image_tag"
    docker build -t "$image_tag" "$PROJECT_ROOT"

    # Push image
    log_info "Pushing image to ACR: $image_tag"
    docker push "$image_tag"

    # Tag as latest if this is production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        local latest_tag="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:latest"
        docker tag "$image_tag" "$latest_tag"
        docker push "$latest_tag"
    fi

    log_success "Image built and pushed: $image_tag"
}

deploy_application() {
    log_info "Deploying application to $ENVIRONMENT environment..."

    local k8s_dir="$PROJECT_ROOT/k8s/$ENVIRONMENT"

    if [[ ! -d "$k8s_dir" ]]; then
        log_error "Kubernetes manifests directory not found: $k8s_dir"
        exit 1
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy to namespace: $NAMESPACE"
        log_info "[DRY RUN] Would use image tag: $TAG"
        return
    fi

    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Update image tag in deployment manifests
    local temp_dir=$(mktemp -d)
    cp -r "$k8s_dir"/* "$temp_dir"/

    # Replace IMAGE_TAG placeholder with actual tag
    find "$temp_dir" -name "*.yaml" -exec sed -i "s|IMAGE_TAG|$TAG|g" {} \;

    # Apply manifests
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f "$temp_dir" --namespace="$NAMESPACE"

    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    kubectl rollout status deployment/ai-audio-upscaler-api -n "$NAMESPACE" --timeout=600s
    kubectl rollout status deployment/ai-audio-upscaler-worker-cpu -n "$NAMESPACE" --timeout=600s

    # Check if GPU workers exist and wait for them
    if kubectl get deployment ai-audio-upscaler-worker-gpu -n "$NAMESPACE" &> /dev/null; then
        kubectl rollout status deployment/ai-audio-upscaler-worker-gpu -n "$NAMESPACE" --timeout=600s
    fi

    # Cleanup temp directory
    rm -rf "$temp_dir"

    log_success "Application deployed successfully"

    # Show deployment status
    show_status
}

run_database_migration() {
    log_info "Running database migrations..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run database migrations"
        return
    fi

    # Create migration job
    local migration_job="
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: migrate
        image: ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${TAG}
        command: [\"alembic\", \"upgrade\", \"head\"]
        envFrom:
        - configMapRef:
            name: ai-upscaler-config
        - secretRef:
            name: ai-upscaler-secrets
  backoffLimit: 2
"

    echo "$migration_job" | kubectl apply -f -

    # Wait for job to complete
    local job_name=$(echo "$migration_job" | grep "name:" | head -1 | awk '{print $2}')
    kubectl wait --for=condition=complete job/"$job_name" -n "$NAMESPACE" --timeout=300s

    # Show job logs
    kubectl logs job/"$job_name" -n "$NAMESPACE"

    log_success "Database migration completed"
}

rollback_deployment() {
    log_info "Rolling back deployment..."

    if [[ -z "$VERSION" ]]; then
        log_info "No version specified, rolling back to previous version"

        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would rollback deployments"
            return
        fi

        kubectl rollout undo deployment/ai-audio-upscaler-api -n "$NAMESPACE"
        kubectl rollout undo deployment/ai-audio-upscaler-worker-cpu -n "$NAMESPACE"

        if kubectl get deployment ai-audio-upscaler-worker-gpu -n "$NAMESPACE" &> /dev/null; then
            kubectl rollout undo deployment/ai-audio-upscaler-worker-gpu -n "$NAMESPACE"
        fi
    else
        log_info "Rolling back to version: $VERSION"

        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would rollback to version: $VERSION"
            return
        fi

        # Update deployments with specific version
        kubectl set image deployment/ai-audio-upscaler-api api="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${VERSION}" -n "$NAMESPACE"
        kubectl set image deployment/ai-audio-upscaler-worker-cpu worker="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${VERSION}" -n "$NAMESPACE"

        if kubectl get deployment ai-audio-upscaler-worker-gpu -n "$NAMESPACE" &> /dev/null; then
            kubectl set image deployment/ai-audio-upscaler-worker-gpu worker="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${VERSION}" -n "$NAMESPACE"
        fi
    fi

    # Wait for rollout
    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/ai-audio-upscaler-api -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/ai-audio-upscaler-worker-cpu -n "$NAMESPACE" --timeout=300s

    if kubectl get deployment ai-audio-upscaler-worker-gpu -n "$NAMESPACE" &> /dev/null; then
        kubectl rollout status deployment/ai-audio-upscaler-worker-gpu -n "$NAMESPACE" --timeout=300s
    fi

    log_success "Rollback completed"
}

show_status() {
    log_info "Deployment status for namespace: $NAMESPACE"

    echo
    echo "=== Deployments ==="
    kubectl get deployments -n "$NAMESPACE"

    echo
    echo "=== Pods ==="
    kubectl get pods -n "$NAMESPACE"

    echo
    echo "=== Services ==="
    kubectl get services -n "$NAMESPACE"

    echo
    echo "=== HPA Status ==="
    kubectl get hpa -n "$NAMESPACE"

    # Check if load balancer has external IP
    echo
    echo "=== Load Balancer Status ==="
    local lb_ip=$(kubectl get service ai-audio-upscaler-api-lb -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

    if [[ -n "$lb_ip" ]]; then
        log_success "Load balancer is ready: http://$lb_ip"

        # Test health endpoint
        if curl -s "http://$lb_ip/health" > /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed"
        fi
    else
        log_warning "Load balancer IP not yet assigned"
    fi
}

show_logs() {
    log_info "Application logs for namespace: $NAMESPACE"

    echo
    echo "=== API Logs ==="
    kubectl logs -l app=ai-audio-upscaler-api -n "$NAMESPACE" --tail=50

    echo
    echo "=== Worker Logs ==="
    kubectl logs -l component=worker-cpu -n "$NAMESPACE" --tail=50

    if kubectl get pods -l component=worker-gpu -n "$NAMESPACE" &> /dev/null; then
        echo
        echo "=== GPU Worker Logs ==="
        kubectl logs -l component=worker-gpu -n "$NAMESPACE" --tail=50
    fi
}

scale_deployment() {
    if [[ -z "$REPLICAS" ]]; then
        log_error "No replica count specified. Use --replicas option."
        exit 1
    fi

    log_info "Scaling API deployment to $REPLICAS replicas..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would scale to $REPLICAS replicas"
        return
    fi

    kubectl scale deployment ai-audio-upscaler-api --replicas="$REPLICAS" -n "$NAMESPACE"
    kubectl rollout status deployment/ai-audio-upscaler-api -n "$NAMESPACE"

    log_success "Scaling completed"
    show_status
}

cleanup_resources() {
    log_warning "This will delete all resources in namespace: $NAMESPACE"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would delete namespace: $NAMESPACE"
        return
    fi

    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi

    log_info "Deleting namespace and all resources..."
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true

    log_success "Cleanup completed"
}

# Main execution
main() {
    log_info "Starting deployment script for environment: $ENVIRONMENT"
    log_info "Target namespace: $NAMESPACE"
    log_info "Image tag: $TAG"

    check_prerequisites

    case $COMMAND in
        build)
            build_and_push_image
            ;;
        deploy)
            build_and_push_image
            deploy_application
            ;;
        migrate)
            run_database_migration
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        scale)
            scale_deployment
            ;;
        cleanup)
            cleanup_resources
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac

    log_success "Operation completed successfully!"
}

# Run main function
main "$@"