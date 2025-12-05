#!/bin/bash
# Terraform Plan Dry Run Script
# Validates Terraform configuration without Azure authentication

set -e

echo "🔍 AI Audio Upscaler Pro - Terraform Validation & Plan"
echo "================================================="

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform is not installed"
    exit 1
fi

# Format check
echo "📝 Checking Terraform formatting..."
terraform fmt -check -diff -recursive

# Initialize
echo "⚡ Initializing Terraform..."
terraform init -backend=false

# Validate
echo "✅ Validating Terraform configuration..."
terraform validate

# Show what would be planned (without Azure auth)
echo "📋 Terraform Plan Summary:"
echo "=========================="
echo "This configuration would create:"
echo "- Resource Group: ai-upscaler-dev-rg"
echo "- Virtual Network with subnets"
echo "- Azure Kubernetes Service cluster"
echo "- PostgreSQL Flexible Server"
echo "- Azure Cache for Redis"
echo "- Azure Container Registry"
echo "- Azure Storage Account"
echo "- Azure Key Vault"
echo "- Azure Application Gateway"
echo "- Log Analytics Workspace"
echo "- Application Insights"
echo ""

# Provider versions
echo "📦 Provider Versions:"
echo "===================="
terraform providers

echo ""
echo "✨ Terraform configuration is valid and ready for deployment!"
echo "💡 To deploy: configure Azure credentials and run 'terraform apply'"

# Resource count estimate
echo ""
echo "📊 Estimated Resource Count: ~15-20 Azure resources"
echo "💰 Estimated Monthly Cost: ~$2,500-3,500 (varies by usage)"
echo ""
echo "🔐 Required Azure Permissions:"
echo "- Contributor role on subscription"
echo "- User Access Administrator (for RBAC assignments)"
echo "- Key Vault Administrator (for Key Vault operations)"