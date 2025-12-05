# Enhanced AI Audio Upscaler Infrastructure
# Uses modern Terraform modules with latest security practices

# Data sources
data "azuread_client_config" "current" {}
data "azurerm_client_config" "current" {}

# Generate random suffix for unique naming
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

# Local variables with enhanced naming and configuration
locals {
  project_name = "ai-audio-upscaler"
  environment  = var.environment
  location     = var.location
  suffix       = random_string.suffix.result

  # Naming convention following Azure best practices
  naming_prefix = "${local.project_name}-${local.environment}"

  # Location mapping for short names
  location_short_map = {
    "East US 2"      = "eus2"
    "West Europe"    = "weu"
    "Southeast Asia" = "sea"
    "Australia East" = "aue"
  }
  location_short = local.location_short_map[local.location]

  # Common tags following best practices
  common_tags = {
    Project             = "AI Audio Upscaler Pro"
    Environment         = local.environment
    Owner               = "Platform Team"
    BusinessUnit        = "Engineering"
    CostCenter          = "R&D"
    ManagedBy          = "Terraform"
    CreatedDate        = formatdate("YYYY-MM-DD", timestamp())
    Repository         = "ai-audio-upscaler"
    TerraformWorkspace = terraform.workspace
  }

  # Network configuration with proper CIDR allocation
  vnet_address_space = ["10.0.0.0/16"]
  subnet_cidrs = {
    aks               = "10.0.1.0/24"
    postgres          = "10.0.2.0/24"
    private_endpoints = "10.0.3.0/24"
    app_gateway       = "10.0.4.0/24"
  }

  # Service CIDR for Kubernetes services
  service_cidr = "10.1.0.0/16"
}

# Main Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${local.naming_prefix}-rg-${local.suffix}"
  location = local.location
  tags     = local.common_tags
}

# Network Module
module "network" {
  source = "./modules/network"

  vnet_name           = "${local.naming_prefix}-vnet-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  environment         = local.environment

  vnet_address_space    = local.vnet_address_space
  subnet_cidrs          = local.subnet_cidrs
  enable_app_gateway    = var.enable_app_gateway

  tags = local.common_tags
}

# Container Registry with enhanced security
resource "azurerm_container_registry" "main" {
  name                = "${replace(local.naming_prefix, "-", "")}acr${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.acr_sku
  admin_enabled       = false

  # Public access prevention
  public_network_access_enabled = !var.disable_public_access

  # Network access rules
  network_rule_set {
    default_action = var.disable_public_access ? "Deny" : "Allow"

    dynamic "virtual_network" {
      for_each = var.disable_public_access ? [1] : []
      content {
        action    = "Allow"
        subnet_id = module.network.subnet_ids.aks
      }
    }
  }

  # Retention policy for production
  dynamic "retention_policy" {
    for_each = var.acr_sku == "Premium" ? [1] : []
    content {
      days    = var.acr_retention_days
      enabled = true
    }
  }

  # Trust policy for production
  dynamic "trust_policy" {
    for_each = var.acr_sku == "Premium" && local.environment == "production" ? [1] : []
    content {
      enabled = true
    }
  }

  # Quarantine policy for security scanning
  dynamic "quarantine_policy" {
    for_each = var.acr_sku == "Premium" ? [1] : []
    content {
      enabled = true
    }
  }

  # Identity for customer-managed keys
  identity {
    type = "SystemAssigned"
  }

  tags = local.common_tags
}

# Enhanced AKS Module
module "aks" {
  source = "./modules/aks"

  # Required parameters
  client_name         = var.client_name
  environment         = local.environment
  stack               = var.stack
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  location_short      = local.location_short

  # Network configuration
  vnet_id         = module.network.vnet_id
  vnet_name       = module.network.vnet_name
  aks_subnet_name = "aks-subnet"
  service_cidr    = local.service_cidr

  # AKS configuration
  aks_cluster_name   = "${local.naming_prefix}-aks-${local.suffix}"
  kubernetes_version = var.kubernetes_version

  # System nodes configuration
  system_node_vm_size   = var.system_node_vm_size
  system_node_count     = var.system_node_count
  system_node_min_count = var.system_node_min_count
  system_node_max_count = var.system_node_max_count

  # Additional node pools for different workloads
  node_pools = [
    # CPU workers for general processing
    {
      name                 = "cpuworkers"
      vm_size              = var.cpu_worker_vm_size
      node_count           = var.cpu_worker_count
      auto_scaling_enabled = true
      min_count            = var.cpu_worker_min_count
      max_count            = var.cpu_worker_max_count
      node_labels = {
        "workload-type" = "cpu-intensive"
        "nodepool-type" = "worker"
      }
    },
    # GPU workers for AI inference (conditional)
    var.enable_gpu_nodes ? {
      name                 = "gpuworkers"
      vm_size              = var.gpu_vm_size
      node_count           = var.gpu_worker_count
      auto_scaling_enabled = true
      min_count            = var.gpu_worker_min_count
      max_count            = var.gpu_worker_max_count
      node_labels = {
        "workload-type" = "gpu-intensive"
        "nodepool-type" = "gpu-worker"
        "accelerator"   = "nvidia-gpu"
      }
      node_taints = [
        "nvidia.com/gpu=true:NoSchedule",
        "sku=${var.gpu_vm_size}:NoSchedule"
      ]
    } : null
  ]

  # Security configuration
  private_cluster_enabled   = var.private_cluster_enabled
  rbac_enabled             = var.rbac_enabled
  admin_group_object_ids   = var.admin_group_object_ids

  # Monitoring
  log_analytics_name = "${local.naming_prefix}-logs-${local.suffix}"
  log_retention_days = var.log_retention_days

  # Identity
  identity_name         = "${local.naming_prefix}-aks-identity-${local.suffix}"
  container_registry_id = azurerm_container_registry.main.id

  # Other configurations
  key_vault_secrets_provider_enabled = var.key_vault_secrets_provider_enabled
  automatic_upgrade_channel          = var.automatic_upgrade_channel
  sku_tier                          = var.aks_sku_tier

  tags = local.common_tags

  depends_on = [module.network]
}

# Enhanced Storage Module
module "storage" {
  source = "./modules/storage"

  storage_account_name = "${replace(local.naming_prefix, "-", "")}sa${local.suffix}"
  resource_group_name  = azurerm_resource_group.main.name
  location             = azurerm_resource_group.main.location
  environment          = local.environment

  # Storage configuration
  account_tier      = var.storage_account_tier
  replication_type  = var.storage_replication_type

  # Security settings
  disable_public_access      = var.disable_public_access
  shared_access_key_enabled  = var.storage_shared_access_key_enabled
  allowed_subnet_ids         = [module.network.subnet_ids.aks]

  # Data protection
  versioning_enabled              = var.storage_versioning_enabled
  blob_soft_delete_days          = var.blob_soft_delete_days
  container_soft_delete_days     = var.container_soft_delete_days
  point_in_time_restore_days     = var.point_in_time_restore_days

  # Private endpoints
  create_private_endpoints    = var.create_private_endpoints
  private_endpoint_subnet_id  = module.network.subnet_ids.private_endpoints
  blob_private_dns_zone_id    = module.network.private_dns_zone_ids.storage

  # Lifecycle management
  enable_lifecycle_policy = var.enable_storage_lifecycle_policy

  # Monitoring
  enable_diagnostics             = var.enable_diagnostics
  log_analytics_workspace_id     = module.aks.log_analytics_workspace_id
  diagnostic_retention_days      = var.diagnostic_retention_days

  tags = local.common_tags

  depends_on = [module.network, module.aks]
}

# Redis Cache with enhanced security
resource "azurerm_redis_cache" "main" {
  name                = "${local.naming_prefix}-redis-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku_name

  # Security settings
  minimum_tls_version       = "1.2"
  public_network_access_enabled = !var.disable_public_access

  # Network integration for Premium SKU
  subnet_id = var.redis_sku_name == "Premium" ? module.network.subnet_ids.aks : null

  # Redis configuration with security hardening
  redis_configuration {
    authentication_enabled = true
    maxmemory_policy      = "allkeys-lru"

    # Disable potentially dangerous commands
    notify_keyspace_events = ""
  }

  # Patch schedule for maintenance
  dynamic "patch_schedule" {
    for_each = local.environment == "production" ? [1] : []
    content {
      day_of_week    = "Sunday"
      start_hour_utc = 2
    }
  }

  tags = local.common_tags
}

# PostgreSQL with enhanced security
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${local.naming_prefix}-postgres-${local.suffix}"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = var.postgres_version
  delegated_subnet_id    = module.network.subnet_ids.postgres
  private_dns_zone_id    = module.network.private_dns_zone_ids.postgres
  administrator_login    = var.postgres_admin_login
  administrator_password = var.postgres_admin_password

  # Server configuration
  storage_mb = var.postgres_storage_mb
  sku_name   = var.postgres_sku_name

  # Security settings
  public_network_access_enabled = false
  ssl_enforcement_enabled        = true
  ssl_minimal_tls_version_enforced = "TLS1_2"

  # Backup and availability settings
  backup_retention_days        = local.environment == "production" ? 35 : var.backup_retention_days
  geo_redundant_backup_enabled = local.environment == "production" ? true : var.geo_redundant_backup_enabled

  # High availability for production
  dynamic "high_availability" {
    for_each = local.environment == "production" ? [1] : []
    content {
      mode                      = "ZoneRedundant"
      standby_availability_zone = "2"
    }
  }

  # Maintenance window
  maintenance_window {
    day_of_week  = 0  # Sunday
    start_hour   = 2
    start_minute = 0
  }

  tags = local.common_tags
}

# PostgreSQL database
resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = var.postgres_database_name
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Key Vault for secrets management
resource "azurerm_key_vault" "main" {
  name                = "${local.naming_prefix}-kv-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = var.key_vault_sku_name

  # Enhanced security settings
  enabled_for_disk_encryption     = true
  enabled_for_template_deployment = false
  enabled_for_deployment         = false
  enable_rbac_authorization      = var.key_vault_rbac_enabled
  public_network_access_enabled  = !var.disable_public_access

  # Network ACLs
  network_acls {
    bypass                     = "AzureServices"
    default_action             = var.disable_public_access ? "Deny" : "Allow"
    virtual_network_subnet_ids = var.disable_public_access ? [module.network.subnet_ids.aks] : []
  }

  # Soft delete and purge protection
  soft_delete_retention_days = local.environment == "production" ? 90 : var.key_vault_soft_delete_days
  purge_protection_enabled   = local.environment == "production" ? true : var.key_vault_purge_protection

  tags = local.common_tags
}

# Key Vault access policies
resource "azurerm_key_vault_access_policy" "current_user" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id

  secret_permissions = [
    "Get", "List", "Set", "Delete", "Purge", "Recover"
  ]

  key_permissions = [
    "Get", "List", "Create", "Delete", "Purge", "Recover",
    "Encrypt", "Decrypt", "WrapKey", "UnwrapKey"
  ]

  certificate_permissions = [
    "Get", "List", "Create", "Delete", "Purge", "Recover"
  ]
}

# Key Vault access policy for AKS
resource "azurerm_key_vault_access_policy" "aks" {
  count = var.key_vault_secrets_provider_enabled ? 1 : 0

  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = module.aks.key_vault_secrets_provider_identity[0].object_id

  secret_permissions = [
    "Get", "List"
  ]
}