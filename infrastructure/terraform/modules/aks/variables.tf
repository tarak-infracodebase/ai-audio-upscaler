# AKS Module Variables

# Required parameters
variable "client_name" {
  description = "Client name/account used in naming"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "stack" {
  description = "Project stack name"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "location" {
  description = "Azure region for resources"
  type        = string
}

variable "location_short" {
  description = "Short string for Azure location"
  type        = string
}

# Network configuration
variable "vnet_id" {
  description = "ID of the virtual network"
  type        = string
}

variable "vnet_name" {
  description = "Name of the virtual network"
  type        = string
}

variable "aks_subnet_name" {
  description = "Name of the AKS subnet"
  type        = string
}

variable "service_cidr" {
  description = "CIDR used by Kubernetes services"
  type        = string
  default     = "10.1.0.0/16"
}

# AKS configuration
variable "aks_cluster_name" {
  description = "Name of the AKS cluster"
  type        = string
}

variable "kubernetes_version" {
  description = "Version of Kubernetes to deploy"
  type        = string
  default     = "1.29"
}

variable "system_node_vm_size" {
  description = "VM size for system nodes"
  type        = string
  default     = "Standard_D4s_v5"
}

variable "system_node_count" {
  description = "Number of system nodes"
  type        = number
  default     = 2
}

variable "system_node_min_count" {
  description = "Minimum number of system nodes"
  type        = number
  default     = 1
}

variable "system_node_max_count" {
  description = "Maximum number of system nodes"
  type        = number
  default     = 5
}

variable "availability_zones" {
  description = "Availability zones for nodes"
  type        = list(number)
  default     = [1, 2, 3]
}

# Node pools configuration
variable "node_pools" {
  description = "Additional node pools configuration"
  type = list(object({
    name                 = string
    vm_size              = optional(string, "Standard_D4s_v5")
    node_count           = optional(number, 2)
    auto_scaling_enabled = optional(bool, true)
    min_count            = optional(number, 0)
    max_count            = optional(number, 10)
    zones                = optional(list(number), [1, 2, 3])
    node_labels          = optional(map(string), {})
    node_taints          = optional(list(string), [])
    os_sku               = optional(string, "Ubuntu")
    os_disk_type         = optional(string, "Ephemeral")
    priority             = optional(string, "Regular")
  }))
  default = []
}

# Security configuration
variable "private_cluster_enabled" {
  description = "Enable private AKS cluster"
  type        = bool
  default     = true
}

variable "rbac_enabled" {
  description = "Enable Azure AD RBAC"
  type        = bool
  default     = true
}

variable "admin_group_object_ids" {
  description = "Object IDs of Azure AD groups with admin access"
  type        = list(string)
  default     = []
}

# Monitoring configuration
variable "log_analytics_name" {
  description = "Name of the Log Analytics workspace"
  type        = string
}

variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30
}

# Identity configuration
variable "identity_name" {
  description = "Name of the user assigned identity"
  type        = string
}

variable "container_registry_id" {
  description = "ID of the Azure Container Registry"
  type        = string
  default     = null
}

# Key Vault integration
variable "key_vault_secrets_provider_enabled" {
  description = "Enable Key Vault secrets provider"
  type        = bool
  default     = true
}

# Maintenance configuration
variable "maintenance_window" {
  description = "Maintenance window configuration"
  type = object({
    allowed = optional(list(object({
      day   = string
      hours = list(number)
    })), [])
    not_allowed = optional(list(object({
      start = string
      end   = string
    })), [])
  })
  default = {
    allowed = [
      {
        day   = "Sunday"
        hours = [2, 3, 4, 5]
      }
    ]
  }
}

# Auto-scaling configuration
variable "auto_scaler_profile" {
  description = "Auto-scaler profile configuration"
  type = object({
    balance_similar_node_groups      = optional(bool, false)
    expander                         = optional(string, "random")
    max_graceful_termination_sec     = optional(number, 600)
    max_node_provisioning_time       = optional(string, "15m")
    max_unready_nodes                = optional(number, 3)
    max_unready_percentage           = optional(number, 45)
    new_pod_scale_up_delay           = optional(string, "10s")
    scale_down_delay_after_add       = optional(string, "10m")
    scale_down_delay_after_delete    = optional(string, "10s")
    scale_down_delay_after_failure   = optional(string, "3m")
    scan_interval                    = optional(string, "10s")
    scale_down_unneeded              = optional(string, "10m")
    scale_down_unready               = optional(string, "20m")
    scale_down_utilization_threshold = optional(number, 0.5)
    empty_bulk_delete_max            = optional(number, 10)
    skip_nodes_with_local_storage    = optional(bool, true)
    skip_nodes_with_system_pods      = optional(bool, true)
  })
  default = {}
}

# Upgrade settings
variable "automatic_upgrade_channel" {
  description = "Automatic upgrade channel for AKS"
  type        = string
  default     = "stable"

  validation {
    condition     = contains(["patch", "rapid", "node-image", "stable", "none"], var.automatic_upgrade_channel)
    error_message = "Automatic upgrade channel must be one of: patch, rapid, node-image, stable, none."
  }
}

# Cost optimization
variable "sku_tier" {
  description = "SKU tier for AKS"
  type        = string
  default     = "Standard"

  validation {
    condition     = contains(["Free", "Standard", "Premium"], var.sku_tier)
    error_message = "SKU tier must be Free, Standard, or Premium."
  }
}

# Private endpoint configuration
variable "create_private_endpoint" {
  description = "Create private endpoint for AKS API server"
  type        = bool
  default     = false
}

variable "private_endpoint_subnet_id" {
  description = "Subnet ID for private endpoints"
  type        = string
  default     = null
}

variable "aks_private_dns_zone_id" {
  description = "Private DNS zone ID for AKS"
  type        = string
  default     = null
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}