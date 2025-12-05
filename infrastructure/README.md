# AI Audio Upscaler Infrastructure

This directory contains the Terraform infrastructure code for the AI Audio Upscaler Pro platform.

## Directory Structure

```
infrastructure/
├── terraform/              # Current enhanced infrastructure (recommended)
│   ├── main.tf             # Main infrastructure resources
│   ├── variables.tf        # Input variables with validation
│   ├── outputs.tf          # Output values
│   ├── terraform.tf        # Provider configuration
│   ├── .gitignore          # Terraform-specific gitignore
│   ├── *.tfvars.example    # Environment-specific examples
│   └── modules/            # Reusable Terraform modules
│       ├── network/        # Network infrastructure
│       ├── aks/           # Azure Kubernetes Service
│       └── storage/       # Secure storage setup
└── terraform-legacy/      # Original infrastructure code (backup)
```

## Enhanced Infrastructure Features

The current Terraform configuration (`./terraform/`) includes:

### **Modern Architecture**
- **Modular Design**: Separate modules for network, AKS, and storage
- **Latest Providers**: AzureRM 4.55+, AzureAD 3.1+, AzAPI 2.1+
- **Best Practices**: Following Terraform and Azure security guidelines

### **Security-First Approach**
- Private networking with Network Security Groups
- Private DNS zones for all services
- Private endpoints for data plane access
- RBAC authorization throughout
- Zero-trust network architecture

### **Production-Ready AKS**
- Uses proven `claranet/aks-light` module
- Azure CNI Overlay with Cilium network policy
- Workload Identity and OIDC issuer enabled
- GPU node pools for AI workloads
- Auto-scaling and maintenance windows
- Container insights and monitoring

### **Advanced Storage**
- Blob versioning and soft delete
- Point-in-time restore capabilities
- Lifecycle management policies
- Private endpoint connectivity
- Customer-managed encryption keys

## Getting Started

### Prerequisites
- Azure CLI authenticated
- Terraform 1.8+ installed
- Appropriate Azure permissions

### Environment Setup

1. **Copy environment template:**
   ```bash
   cp terraform.dev.tfvars.example terraform.dev.tfvars
   # Edit with your values
   ```

2. **Set required secrets:**
   ```bash
   export TF_VAR_postgres_admin_password="your-secure-password"
   ```

3. **Initialize and plan:**
   ```bash
   terraform init
   terraform plan -var-file="terraform.dev.tfvars"
   ```

4. **Apply infrastructure:**
   ```bash
   terraform apply -var-file="terraform.dev.tfvars"
   ```

### Environment Files

- `terraform.dev.tfvars` - Development environment (cost-optimized)
- `terraform.staging.tfvars` - Staging environment
- `terraform.production.tfvars` - Production environment (high-availability)

## Architecture Overview

The infrastructure deploys:

1. **Resource Group** - Container for all resources
2. **Virtual Network** - Private network with multiple subnets
3. **AKS Cluster** - Kubernetes with system, CPU, and GPU node pools
4. **Container Registry** - Private Docker image registry
5. **Storage Account** - Audio file storage with lifecycle policies
6. **Redis Cache** - Session and data caching
7. **PostgreSQL** - Application database with high availability
8. **Key Vault** - Secrets and certificate management
9. **Log Analytics** - Centralized logging and monitoring

## Cost Optimization

### Development Environment
- Basic/Standard SKUs
- Minimal node counts
- Shorter retention periods
- Local redundancy

### Production Environment
- Premium SKUs for performance
- Auto-scaling enabled
- Extended backups and retention
- Zone/Geo redundancy

## Security Considerations

- All data plane access via private endpoints
- Network security groups with minimal access
- RBAC for all Azure services
- Secrets stored in Key Vault
- TLS 1.2+ encryption everywhere
- Regular security patching enabled

## Monitoring & Observability

- Container insights for AKS
- Storage analytics and metrics
- Database performance monitoring
- Diagnostic settings on all resources
- Log Analytics workspace integration

## Backup & Disaster Recovery

- Automated PostgreSQL backups (7-35 days)
- Storage account soft delete and versioning
- Cross-region replication for production
- Point-in-time restore capabilities

## Migration from Legacy

If you need to migrate from the legacy infrastructure:

1. Review current resources: `terraform state list`
2. Plan migration strategy
3. Test in development environment first
4. Use blue-green deployment for production

## Support

For infrastructure issues:
1. Check Terraform plan output
2. Review Azure Activity Log
3. Consult module documentation
4. Check provider compatibility

---

**Note**: The `terraform-legacy/` directory contains the original infrastructure code for reference and emergency rollback purposes. The enhanced version in `terraform/` is the recommended approach.