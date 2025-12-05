# Variables for AI Audio Upscaler Pro Infrastructure

# General
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
}

# AKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.28"
}

variable "aks_system_node_count" {
  description = "Number of nodes in the system node pool"
  type        = number
  default     = 2

  validation {
    condition     = var.aks_system_node_count >= 1 && var.aks_system_node_count <= 10
    error_message = "System node count must be between 1 and 10."
  }
}

variable "aks_cpu_node_count" {
  description = "Initial number of CPU worker nodes"
  type        = number
  default     = 2

  validation {
    condition     = var.aks_cpu_node_count >= 0 && var.aks_cpu_node_count <= 20
    error_message = "CPU node count must be between 0 and 20."
  }
}

variable "enable_gpu_nodes" {
  description = "Enable GPU node pool for AI inference"
  type        = bool
  default     = true
}

variable "aks_gpu_node_count" {
  description = "Initial number of GPU worker nodes"
  type        = number
  default     = 0

  validation {
    condition     = var.aks_gpu_node_count >= 0 && var.aks_gpu_node_count <= 10
    error_message = "GPU node count must be between 0 and 10."
  }
}

variable "gpu_vm_size" {
  description = "VM size for GPU nodes"
  type        = string
  default     = "Standard_NC6s_v3"

  validation {
    condition = contains([
      "Standard_NC6s_v3",
      "Standard_NC12s_v3",
      "Standard_NC24s_v3",
      "Standard_ND40rs_v2",
      "Standard_NC4as_T4_v3",
      "Standard_NC8as_T4_v3",
      "Standard_NC16as_T4_v3"
    ], var.gpu_vm_size)
    error_message = "GPU VM size must be a valid NVIDIA GPU-enabled SKU."
  }
}

# Container Registry
variable "acr_sku" {
  description = "SKU for Azure Container Registry"
  type        = string
  default     = "Standard"

  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "ACR SKU must be Basic, Standard, or Premium."
  }
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
  description = "Redis capacity"
  type        = number
  default     = 2

  validation {
    condition     = var.redis_capacity >= 0 && var.redis_capacity <= 6
    error_message = "Redis capacity must be between 0 and 6."
  }
}

# PostgreSQL Configuration
variable "postgres_admin_login" {
  description = "Administrator login for PostgreSQL"
  type        = string
  default     = "aiupscaleradmin"
  sensitive   = true
}

variable "postgres_admin_password" {
  description = "Administrator password for PostgreSQL"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.postgres_admin_password) >= 8
    error_message = "PostgreSQL password must be at least 8 characters long."
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
  default     = 32768

  validation {
    condition     = var.postgres_storage_mb >= 32768 && var.postgres_storage_mb <= 16777216
    error_message = "PostgreSQL storage must be between 32768 and 16777216 MB."
  }
}

# Application Gateway
variable "enable_app_gateway" {
  description = "Enable Application Gateway for ingress"
  type        = bool
  default     = false
}

# Monitoring
variable "enable_monitoring" {
  description = "Enable advanced monitoring with Application Insights"
  type        = bool
  default     = true
}

# Auto-scaling
variable "enable_keda" {
  description = "Enable KEDA for advanced auto-scaling"
  type        = bool
  default     = true
}

# Security
variable "enable_private_cluster" {
  description = "Enable private AKS cluster"
  type        = bool
  default     = false
}

variable "authorized_ip_ranges" {
  description = "IP ranges authorized to access the Kubernetes API server"
  type        = list(string)
  default     = []
}

# Backup and DR
variable "backup_retention_days" {
  description = "Backup retention days"
  type        = number
  default     = 7

  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 35
    error_message = "Backup retention days must be between 1 and 35."
  }
}

# Cost management
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "max_spot_percentage" {
  description = "Maximum percentage of spot instances in node pools"
  type        = number
  default     = 50

  validation {
    condition     = var.max_spot_percentage >= 0 && var.max_spot_percentage <= 100
    error_message = "Spot percentage must be between 0 and 100."
  }
}