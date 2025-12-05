# Enhanced Outputs for AI Audio Upscaler Pro Infrastructure

# Resource Group
output "resource_group_name" {
  description = "Name of the main resource group"
  value       = azurerm_resource_group.main.name
}

output "resource_group_id" {
  description = "ID of the main resource group"
  value       = azurerm_resource_group.main.id
}

output "location" {
  description = "Azure region where resources are deployed"
  value       = azurerm_resource_group.main.location
}

# Network Outputs
output "network" {
  description = "Network configuration details"
  value = {
    vnet_id    = module.network.vnet_id
    vnet_name  = module.network.vnet_name
    subnet_ids = module.network.subnet_ids
  }
}

# AKS Cluster Outputs
output "aks_cluster" {
  description = "AKS cluster details"
  value = {
    id                       = module.aks.cluster_id
    name                     = module.aks.cluster_name
    fqdn                     = module.aks.cluster_fqdn
    private_fqdn            = module.aks.private_fqdn
    node_resource_group_name = module.aks.node_resource_group_name
    oidc_issuer_url         = module.aks.oidc_issuer_url
  }
}

output "aks_identities" {
  description = "AKS managed identities"
  value = {
    principal_id    = module.aks.identity_principal_id
    client_id       = module.aks.identity_client_id
    kubelet_identity = module.aks.kubelet_identity
  }
}

output "kube_config_raw" {
  description = "Raw kubeconfig for the AKS cluster"
  value       = module.aks.kube_config_raw
  sensitive   = true
}

# Container Registry Outputs
output "container_registry" {
  description = "Azure Container Registry details"
  value = {
    id           = azurerm_container_registry.main.id
    name         = azurerm_container_registry.main.name
    login_server = azurerm_container_registry.main.login_server
  }
}

# Storage Account Outputs
output "storage_account" {
  description = "Storage account details"
  value = {
    name                   = module.storage.storage_account_name
    primary_blob_endpoint  = module.storage.primary_blob_endpoint
    containers            = module.storage.container_names
  }
}

output "storage_connection_string" {
  description = "Primary connection string for storage account"
  value       = module.storage.primary_connection_string
  sensitive   = true
}

# Redis Cache Outputs
output "redis_cache" {
  description = "Redis cache details"
  value = {
    name     = azurerm_redis_cache.main.name
    hostname = azurerm_redis_cache.main.hostname
    port     = azurerm_redis_cache.main.port
    ssl_port = azurerm_redis_cache.main.ssl_port
  }
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "rediss://:${azurerm_redis_cache.main.primary_access_key}@${azurerm_redis_cache.main.hostname}:${azurerm_redis_cache.main.ssl_port}/0"
  sensitive   = true
}

# PostgreSQL Database Outputs
output "postgresql_server" {
  description = "PostgreSQL server details"
  value = {
    name          = azurerm_postgresql_flexible_server.main.name
    fqdn          = azurerm_postgresql_flexible_server.main.fqdn
    database_name = azurerm_postgresql_flexible_server_database.main.name
  }
}

output "postgresql_connection_string" {
  description = "PostgreSQL connection string"
  value       = "postgresql+asyncpg://${var.postgres_admin_login}:${var.postgres_admin_password}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/${azurerm_postgresql_flexible_server_database.main.name}"
  sensitive   = true
}

# Key Vault Outputs
output "key_vault" {
  description = "Key Vault details"
  value = {
    name = azurerm_key_vault.main.name
    uri  = azurerm_key_vault.main.vault_uri
  }
}

# Log Analytics Workspace
output "log_analytics_workspace" {
  description = "Log Analytics workspace details"
  value = {
    id   = module.aks.log_analytics_workspace_id
    name = split("/", module.aks.log_analytics_workspace_id)[8]
  }
}

# Application Configuration (for deployment)
output "application_config" {
  description = "Configuration values for application deployment"
  value = {
    # Database configuration
    database_url = "postgresql+asyncpg://${var.postgres_admin_login}:${var.postgres_admin_password}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/${azurerm_postgresql_flexible_server_database.main.name}"

    # Cache configuration
    redis_url = "rediss://:${azurerm_redis_cache.main.primary_access_key}@${azurerm_redis_cache.main.hostname}:${azurerm_redis_cache.main.ssl_port}/0"

    # Storage configuration
    storage_connection_string = module.storage.primary_connection_string
    storage_containers = {
      inputs     = "audio-inputs"
      outputs    = "audio-outputs"
      models     = "ml-models"
      temp       = "temp-processing"
    }

    # Key Vault configuration
    key_vault_uri = azurerm_key_vault.main.vault_uri

    # Container Registry
    acr_login_server = azurerm_container_registry.main.login_server
  }
  sensitive = true
}

# Kubernetes Configuration Commands
output "kubectl_commands" {
  description = "Commands for Kubernetes configuration"
  value = {
    get_credentials = "az aks get-credentials --resource-group ${azurerm_resource_group.main.name} --name ${module.aks.cluster_name}"
    acr_login      = "az acr login --name ${azurerm_container_registry.main.name}"
  }
}

# Terraform State Information
output "terraform_workspace" {
  description = "Current Terraform workspace"
  value       = terraform.workspace
}

output "deployment_timestamp" {
  description = "Timestamp of this deployment"
  value       = timestamp()
}

# Environment Information
output "environment_summary" {
  description = "Summary of the deployed environment"
  value = {
    environment    = var.environment
    location      = var.location
    naming_suffix = random_string.suffix.result

    # Resource counts
    aks_enabled          = true
    gpu_nodes_enabled    = var.enable_gpu_nodes
    private_networking   = var.disable_public_access
    app_gateway_enabled  = var.enable_app_gateway

    # SKU information
    aks_sku          = var.aks_sku_tier
    acr_sku          = var.acr_sku
    redis_sku        = var.redis_sku_name
    postgres_sku     = var.postgres_sku_name
    storage_tier     = var.storage_account_tier

    # Security features
    private_cluster     = var.private_cluster_enabled
    rbac_enabled       = var.rbac_enabled
    key_vault_rbac     = var.key_vault_rbac_enabled
    private_endpoints  = var.create_private_endpoints
  }
}

# Cost estimation context
output "cost_estimation_context" {
  description = "Information for cost estimation"
  value = {
    # AKS node information
    system_nodes = {
      vm_size    = var.system_node_vm_size
      min_count  = var.system_node_min_count
      max_count  = var.system_node_max_count
    }
    cpu_workers = {
      vm_size    = var.cpu_worker_vm_size
      min_count  = var.cpu_worker_min_count
      max_count  = var.cpu_worker_max_count
    }
    gpu_workers = var.enable_gpu_nodes ? {
      vm_size    = var.gpu_vm_size
      min_count  = var.gpu_worker_min_count
      max_count  = var.gpu_worker_max_count
    } : null

    # Storage information
    storage_replication = var.storage_replication_type
    postgres_storage_gb = var.postgres_storage_mb / 1024
    redis_capacity      = var.redis_capacity

    # Backup settings
    backup_retention_days = var.backup_retention_days
    log_retention_days   = var.log_retention_days
  }
}