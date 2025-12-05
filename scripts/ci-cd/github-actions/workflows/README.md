# CI/CD Pipeline Documentation

## Overview

The AI Audio Upscaler Pro uses a comprehensive CI/CD pipeline built with GitHub Actions that ensures code quality, security, and reliable deployments.

## Pipeline Architecture

### Stages

1. **Security & Code Quality**
   - Trivy vulnerability scanning
   - Bandit security linting
   - Black code formatting
   - isort import sorting
   - Flake8 linting
   - MyPy type checking

2. **Testing**
   - Unit tests with pytest
   - Integration tests
   - Code coverage reporting
   - Test result artifacts

3. **Build & Containerization**
   - Multi-architecture Docker builds (amd64/arm64)
   - Azure Container Registry push
   - Container vulnerability scanning
   - Image signing and verification

4. **Infrastructure Validation**
   - Terraform format checking
   - Terraform validation
   - Infrastructure planning (PR only)
   - Plan commenting on PRs

5. **Deployment**
   - Staging deployment (develop branch)
   - Production deployment (main branch + tags)
   - Blue-green deployment strategy
   - Automated rollback on failure

6. **Post-Deployment**
   - Database migrations
   - Performance testing
   - Security scanning
   - Cleanup tasks

## Required Secrets

Configure these secrets in GitHub repository settings:

### Azure Authentication
- `AZURE_CREDENTIALS` - Service principal JSON for Azure login
- `AZURE_CLIENT_ID` - Service principal client ID
- `AZURE_CLIENT_SECRET` - Service principal client secret
- `AZURE_SUBSCRIPTION_ID` - Azure subscription ID
- `AZURE_TENANT_ID` - Azure tenant ID

### Container Registry
- `ACR_USERNAME` - Azure Container Registry username
- `ACR_PASSWORD` - Azure Container Registry password

### Database
- `PRODUCTION_DATABASE_URL` - Production PostgreSQL connection string

### Notifications
- `SLACK_WEBHOOK` - Slack webhook URL for deployment notifications

## Deployment Environments

### Staging
- **Trigger**: Push to `develop` branch
- **URL**: https://staging.aiupscaler.com
- **Resources**: Scaled-down replicas for cost efficiency
- **Purpose**: Integration testing and preview

### Production
- **Trigger**: Push to `main` branch with version tag (v*)
- **URL**: https://aiupscaler.com
- **Resources**: Full production scale with auto-scaling
- **Purpose**: Live customer traffic

## Deployment Process

### Automatic Deployments

1. **Code Push** → Triggers pipeline
2. **Quality Gates** → All checks must pass
3. **Build & Test** → Create container images
4. **Deploy** → Blue-green deployment
5. **Health Check** → Verify deployment success
6. **Notification** → Slack notification

### Manual Deployments

Use the deployment script for manual control:

```bash
# Deploy to staging
./scripts/deploy.sh deploy --environment staging --tag v1.2.3

# Deploy to production
./scripts/deploy.sh deploy --environment production --tag v1.2.3

# Rollback if needed
./scripts/deploy.sh rollback --version v1.2.2
```

## Quality Gates

All deployments must pass these gates:

### Security
- ✅ No critical vulnerabilities (Trivy)
- ✅ No security issues (Bandit)
- ✅ Container scanning passed

### Code Quality
- ✅ Code formatted (Black)
- ✅ Imports sorted (isort)
- ✅ Linting passed (Flake8)
- ✅ Type checking passed (MyPy)

### Testing
- ✅ Unit tests passed (pytest)
- ✅ Integration tests passed
- ✅ Code coverage > 80%

### Infrastructure
- ✅ Terraform validation passed
- ✅ No infrastructure drift

## Monitoring Integration

The pipeline integrates with monitoring systems:

- **Deployment Events** → Grafana annotations
- **Performance Tests** → Metrics collection
- **Error Rates** → Automatic alerting
- **Health Checks** → Uptime monitoring

## Rollback Strategy

### Automatic Rollback
- Health checks fail → Automatic rollback
- High error rate detected → Automatic rollback
- Performance degradation → Alert for manual review

### Manual Rollback
```bash
# Rollback to previous version
kubectl rollout undo deployment/ai-audio-upscaler-api

# Rollback to specific version
./scripts/deploy.sh rollback --version v1.2.2
```

## Branch Strategy

### Main Branch (`main`)
- **Purpose**: Production releases
- **Protection**: Requires PR approval, status checks
- **Deployments**: Production only
- **Tagging**: Semantic versioning (v1.2.3)

### Development Branch (`develop`)
- **Purpose**: Integration and staging
- **Protection**: Requires status checks
- **Deployments**: Staging environment
- **Features**: Feature branches merge here first

### Feature Branches (`feature/*`)
- **Purpose**: Feature development
- **Protection**: None (for development flexibility)
- **Deployments**: None (CI checks only)
- **Naming**: `feature/description-of-feature`

## Performance Testing

Automated performance tests run after production deployments:

### Load Testing (k6)
- **Concurrent Users**: 10-20 users
- **Duration**: 12 minutes
- **Thresholds**: 95th percentile < 2s, Error rate < 10%

### Stress Testing (k6)
- **Peak Load**: 400 concurrent users
- **Purpose**: Find breaking points
- **Monitoring**: System resource limits

## Security Scanning

### Static Analysis
- **Bandit**: Python security linting
- **Trivy**: Filesystem vulnerability scanning
- **Container Scanning**: Pre-deployment image scanning

### Dynamic Analysis
- **OWASP ZAP**: Post-deployment security scanning
- **Penetration Testing**: Scheduled external scans

## Troubleshooting

### Common Issues

**Pipeline Fails at Security Scan**
- Check Trivy/Bandit reports in artifacts
- Update dependencies with security patches
- Add security exceptions if false positives

**Tests Failing**
- Check test logs in Actions output
- Verify database connectivity for integration tests
- Update test data if schema changed

**Deployment Timeout**
- Check Kubernetes events: `kubectl get events`
- Verify resource limits and requests
- Check node capacity and scheduling

**Health Checks Failing**
- Verify service dependencies (DB, Redis)
- Check application logs for errors
- Validate configuration and secrets

### Debug Commands

```bash
# Check pipeline status
gh run list

# View specific run logs
gh run view <run-id>

# Check deployment status
./scripts/deploy.sh status

# View application logs
./scripts/deploy.sh logs

# Manual health check
curl -f https://aiupscaler.com/health
```

## Best Practices

### Code Changes
1. Create feature branch from `develop`
2. Make atomic commits with clear messages
3. Add tests for new functionality
4. Update documentation as needed
5. Create PR to `develop` branch

### Releases
1. Merge `develop` → `main` via PR
2. Create semantic version tag (v1.2.3)
3. Monitor deployment progress
4. Verify production health checks
5. Announce release to team

### Hotfixes
1. Create hotfix branch from `main`
2. Implement minimal fix
3. Test thoroughly in staging
4. Deploy directly to production
5. Merge back to `develop` and `main`

## Metrics and KPIs

Track these deployment metrics:

- **Deployment Frequency**: Daily/Weekly releases
- **Lead Time**: Time from commit to production
- **MTTR**: Mean time to recovery from failures
- **Change Failure Rate**: % of deployments causing issues
- **Deployment Success Rate**: % of successful deployments

## Compliance

The pipeline ensures:

- **Audit Trail**: All changes tracked in Git
- **Approval Process**: Required PR reviews
- **Security Scanning**: Automated vulnerability checks
- **Quality Gates**: No degraded deployments
- **Rollback Capability**: Quick recovery from issues