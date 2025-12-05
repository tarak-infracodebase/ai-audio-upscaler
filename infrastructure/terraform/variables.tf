# Enhanced Variables for AI Audio Upscaler Pro Infrastructure
# Following Terraform best practices with comprehensive validation

# General Configuration
variable "client_name" {
  description = "Client name/account used in naming"
  type        = string
  default     = "ai-upscaler"

  validation {
    condition     = length(var.client_name) > 0 && length(var.client_name) <= 20
    error_message = "Client name must be between 1 and 20 characters."
  }
}

variable "stack" {
  description = "Project stack name"
  type        = string
  default     = "audio-processing"

  validation {
    condition     = length(var.stack) > 0 && length(var.stack) <= 20
    error_message = "Stack name must be between 1 and 20 characters."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "East US 2"

  validation {
    condition = contains([
      "East US 2", "West Europe", "Southeast Asia", "Australia East"
    ], var.location)
    error_message = "Location must be one of the supported regions."
  }
}

# AKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.29"

  validation {
    condition     = can(regex("^1\\.(2[6-9]|[3-9][0-9])$", var.kubernetes_version))
    error_message = "Kubernetes version must be 1.26 or higher."
  }
}

variable "private_cluster_enabled" {
  description = "Enable private AKS cluster"
  type        = bool
  default     = true
}

variable "rbac_enabled" {
  description = "Enable Azure AD RBAC for AKS"
  type        = bool
  default     = true
}

variable "admin_group_object_ids" {
  description = "Object IDs of Azure AD groups with admin access"
  type        = list(string)
  default     = []
}

# System Node Pool Configuration
variable "system_node_vm_size" {
  description = "VM size for system nodes"
  type        = string
  default     = "Standard_D4s_v5"

  validation {
    condition = contains([
      "Standard_D2s_v5", "Standard_D4s_v5", "Standard_D8s_v5"
    ], var.system_node_vm_size)
    error_message = "System node VM size must be a supported SKU."
  }
}

variable "system_node_count" {
  description = "Number of system nodes"
  type        = number
  default     = 2

  validation {
    condition     = var.system_node_count >= 1 && var.system_node_count <= 10
    error_message = "System node count must be between 1 and 10."
  }
}

variable "system_node_min_count" {
  description = "Minimum number of system nodes"
  type        = number
  default     = 1

  validation {
    condition     = var.system_node_min_count >= 1 && var.system_node_min_count <= 5
    error_message = "System node minimum count must be between 1 and 5."
  }
}

variable "system_node_max_count" {
  description = "Maximum number of system nodes"
  type        = number
  default     = 5

  validation {
    condition     = var.system_node_max_count >= 2 && var.system_node_max_count <= 20
    error_message = "System node maximum count must be between 2 and 20."
  }
}

# CPU Worker Node Configuration
variable "cpu_worker_vm_size" {
  description = "VM size for CPU worker nodes"
  type        = string
  default     = "Standard_D8s_v5"

  validation {
    condition = contains([
      "Standard_D4s_v5", "Standard_D8s_v5", "Standard_D16s_v5", "Standard_D32s_v5"
    ], var.cpu_worker_vm_size)
    error_message = "CPU worker VM size must be a supported compute-optimized SKU."
  }
}

variable "cpu_worker_count" {
  description = "Initial number of CPU worker nodes"
  type        = number
  default     = 2

  validation {
    condition     = var.cpu_worker_count >= 0 && var.cpu_worker_count <= 20
    error_message = "CPU worker count must be between 0 and 20."
  }
}

variable "cpu_worker_min_count" {
  description = "Minimum number of CPU worker nodes"
  type        = number
  default     = 0

  validation {
    condition     = var.cpu_worker_min_count >= 0 && var.cpu_worker_min_count <= 10
    error_message = "CPU worker minimum count must be between 0 and 10."
  }
}

variable "cpu_worker_max_count" {
  description = "Maximum number of CPU worker nodes"
  type        = number
  default     = 10

  validation {
    condition     = var.cpu_worker_max_count >= 1 && var.cpu_worker_max_count <= 50
    error_message = "CPU worker maximum count must be between 1 and 50."
  }
}

# GPU Configuration
variable "enable_gpu_nodes" {
  description = "Enable GPU node pool for AI inference"
  type        = bool
  default     = true
}

variable "gpu_vm_size" {
  description = "VM size for GPU nodes"
  type        = string
  default     = "Standard_NC4as_T4_v3"

  validation {
    condition = contains([
      "Standard_NC4as_T4_v3",  # 1x NVIDIA T4
      "Standard_NC8as_T4_v3",  # 1x NVIDIA T4
      "Standard_NC16as_T4_v3", # 1x NVIDIA T4
      "Standard_NC6s_v3",      # 1x NVIDIA V100
      "Standard_NC12s_v3",     # 2x NVIDIA V100
      "Standard_NC24s_v3",     # 4x NVIDIA V100
    ], var.gpu_vm_size)
    error_message = "GPU VM size must be a valid NVIDIA GPU-enabled SKU."
  }
}

variable "gpu_worker_count" {
  description = "Initial number of GPU worker nodes"
  type        = number
  default     = 0

  validation {
    condition     = var.gpu_worker_count >= 0 && var.gpu_worker_count <= 10
    error_message = "GPU worker count must be between 0 and 10."
  }
}

variable "gpu_worker_min_count" {
  description = "Minimum number of GPU worker nodes"
  type        = number
  default     = 0

  validation {
    condition     = var.gpu_worker_min_count >= 0 && var.gpu_worker_min_count <= 5
    error_message = "GPU worker minimum count must be between 0 and 5."
  }
}

variable "gpu_worker_max_count" {
  description = "Maximum number of GPU worker nodes"
  type        = number
  default     = 5

  validation {
    condition     = var.gpu_worker_max_count >= 1 && var.gpu_worker_max_count <= 20
    error_message = "GPU worker maximum count must be between 1 and 20."
  }
}

# Container Registry Configuration
variable "acr_sku" {
  description = "SKU for Azure Container Registry"
  type        = string
  default     = "Standard"

  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "ACR SKU must be Basic, Standard, or Premium."
  }
}

variable "acr_retention_days" {
  description = "Retention days for ACR (Premium only)"
  type        = number
  default     = 30

  validation {
    condition     = var.acr_retention_days >= 1 && var.acr_retention_days <= 365
    error_message = "ACR retention days must be between 1 and 365."
  }
}

# Storage Configuration
variable "storage_account_tier" {
  description = "Performance tier for storage account"
  type        = string
  default     = "Standard"

  validation {
    condition     = contains(["Standard", "Premium"], var.storage_account_tier)
    error_message = "Storage account tier must be Standard or Premium."
  }
}

variable "storage_replication_type" {
  description = "Replication type for storage account"
  type        = string
  default     = "LRS"

  validation {
    condition     = contains(["LRS", "GRS", "ZRS", "GZRS"], var.storage_replication_type)
    error_message = "Storage replication type must be LRS, GRS, ZRS, or GZRS."
  }
}

variable "storage_shared_access_key_enabled" {
  description = "Enable shared access keys for storage account"
  type        = bool
  default     = false
}

variable "storage_versioning_enabled" {
  description = "Enable blob versioning"
  type        = bool
  default     = true
}

variable "blob_soft_delete_days" {
  description = "Soft delete retention days for blobs"
  type        = number
  default     = 30

  validation {
    condition     = var.blob_soft_delete_days >= 1 && var.blob_soft_delete_days <= 365
    error_message = "Blob soft delete days must be between 1 and 365."
  }
}

variable "container_soft_delete_days" {
  description = "Soft delete retention days for containers"
  type        = number
  default     = 7

  validation {
    condition     = var.container_soft_delete_days >= 1 && var.container_soft_delete_days <= 365
    error_message = "Container soft delete days must be between 1 and 365."
  }
}

variable "point_in_time_restore_days" {
  description = "Point-in-time restore days (0 to disable)"
  type        = number
  default     = 7

  validation {
    condition     = var.point_in_time_restore_days >= 0 && var.point_in_time_restore_days <= 365
    error_message = "Point-in-time restore days must be between 0 and 365."
  }
}

variable "enable_storage_lifecycle_policy" {
  description = "Enable storage lifecycle management policy"
  type        = bool
  default     = true
}

# Redis Configuration
variable "redis_sku_name" {
  description = "SKU name for Redis Cache"
  type        = string
  default     = "Standard"

  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.redis_sku_name)
    error_message = "Redis SKU must be Basic, Standard, or Premium."
  }
}

variable "redis_family" {
  description = "Redis family (C for Basic/Standard, P for Premium)"
  type        = string
  default     = "C"

  validation {
    condition     = contains(["C", "P"], var.redis_family)
    error_message = "Redis family must be C or P."
  }
}

variable "redis_capacity" {
  description = "Redis capacity (0-6 for C family, 1-5 for P family)"
  type        = number
  default     = 2

  validation {
    condition     = var.redis_capacity >= 0 && var.redis_capacity <= 6
    error_message = "Redis capacity must be between 0 and 6."
  }
}

# PostgreSQL Configuration
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15"

  validation {
    condition     = contains(["13", "14", "15", "16"], var.postgres_version)
    error_message = "PostgreSQL version must be 13, 14, 15, or 16."
  }
}

variable "postgres_admin_login" {
  description = "Administrator login for PostgreSQL"
  type        = string
  default     = "aiupscaleradmin"
  sensitive   = true

  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9_]*$", var.postgres_admin_login))
    error_message = "PostgreSQL admin login must start with a letter and contain only letters, numbers, and underscores."
  }
}

variable "postgres_admin_password" {
  description = "Administrator password for PostgreSQL"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.postgres_admin_password) >= 12
    error_message = "PostgreSQL password must be at least 12 characters long."
  }
}

variable "postgres_sku_name" {
  description = "SKU name for PostgreSQL Flexible Server"
  type        = string
  default     = "GP_Standard_D2s_v3"

  validation {
    condition     = can(regex("^(B_Standard_B[124]ms|GP_Standard_D[248]s_v3|MO_Standard_E[248]s_v3)$", var.postgres_sku_name))
    error_message = "Invalid PostgreSQL SKU name."
  }
}

variable "postgres_storage_mb" {
  description = "Storage size in MB for PostgreSQL"
  type        = number
  default     = 65536

  validation {
    condition     = var.postgres_storage_mb >= 32768 && var.postgres_storage_mb <= 16777216
    error_message = "PostgreSQL storage must be between 32768 MB (32 GB) and 16777216 MB (16 TB)."
  }
}

variable "postgres_database_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "ai_audio_upscaler"

  validation {
    condition     = can(regex("^[a-z][a-z0-9_]*$", var.postgres_database_name))
    error_message = "Database name must start with lowercase letter and contain only lowercase letters, numbers, and underscores."
  }
}

variable "backup_retention_days" {
  description = "Backup retention days for PostgreSQL"
  type        = number
  default     = 7

  validation {
    condition     = var.backup_retention_days >= 7 && var.backup_retention_days <= 35
    error_message = "Backup retention days must be between 7 and 35."
  }
}

variable "geo_redundant_backup_enabled" {
  description = "Enable geo-redundant backups"
  type        = bool
  default     = false
}

# Key Vault Configuration
variable "key_vault_sku_name" {
  description = "SKU name for Key Vault"
  type        = string
  default     = "standard"

  validation {
    condition     = contains(["standard", "premium"], var.key_vault_sku_name)
    error_message = "Key Vault SKU must be standard or premium."
  }
}

variable "key_vault_rbac_enabled" {
  description = "Enable RBAC authorization for Key Vault"
  type        = bool
  default     = true
}

variable "key_vault_soft_delete_days" {
  description = "Soft delete retention days for Key Vault"
  type        = number
  default     = 7

  validation {
    condition     = var.key_vault_soft_delete_days >= 7 && var.key_vault_soft_delete_days <= 90
    error_message = "Key Vault soft delete days must be between 7 and 90."
  }
}

variable "key_vault_purge_protection" {
  description = "Enable purge protection for Key Vault"
  type        = bool
  default     = false
}

variable "key_vault_secrets_provider_enabled" {
  description = "Enable Key Vault secrets provider for AKS"
  type        = bool
  default     = true
}

# Network Security Configuration
variable "disable_public_access" {
  description = "Disable public access to Azure services"
  type        = bool
  default     = false
}

variable "create_private_endpoints" {
  description = "Create private endpoints for Azure services"
  type        = bool
  default     = false
}

variable "enable_app_gateway" {
  description = "Enable Application Gateway for ingress"
  type        = bool
  default     = false
}

# AKS Additional Configuration
variable "automatic_upgrade_channel" {
  description = "Automatic upgrade channel for AKS"
  type        = string
  default     = "stable"

  validation {
    condition     = contains(["patch", "rapid", "node-image", "stable", "none"], var.automatic_upgrade_channel)
    error_message = "Automatic upgrade channel must be patch, rapid, node-image, stable, or none."
  }
}

variable "aks_sku_tier" {
  description = "SKU tier for AKS"
  type        = string
  default     = "Standard"

  validation {
    condition     = contains(["Free", "Standard", "Premium"], var.aks_sku_tier)
    error_message = "AKS SKU tier must be Free, Standard, or Premium."
  }
}

# Monitoring Configuration
variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30

  validation {
    condition     = var.log_retention_days >= 30 && var.log_retention_days <= 730
    error_message = "Log retention days must be between 30 and 730."
  }
}

variable "enable_diagnostics" {
  description = "Enable diagnostic settings"
  type        = bool
  default     = true
}

variable "diagnostic_retention_days" {
  description = "Diagnostic data retention period in days"
  type        = number
  default     = 90

  validation {
    condition     = var.diagnostic_retention_days >= 1 && var.diagnostic_retention_days <= 365
    error_message = "Diagnostic retention days must be between 1 and 365."
  }
}