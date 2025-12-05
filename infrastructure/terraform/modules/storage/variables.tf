# Storage Module Variables

variable "storage_account_name" {
  description = "Name of the storage account"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.storage_account_name))
    error_message = "Storage account name must be 3-24 characters, lowercase letters and numbers only."
  }
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "location" {
  description = "Azure region for resources"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "account_tier" {
  description = "Performance tier for storage account"
  type        = string
  default     = "Standard"

  validation {
    condition     = contains(["Standard", "Premium"], var.account_tier)
    error_message = "Account tier must be Standard or Premium."
  }
}

variable "replication_type" {
  description = "Replication type for storage account"
  type        = string
  default     = "LRS"

  validation {
    condition     = contains(["LRS", "GRS", "ZRS", "GZRS"], var.replication_type)
    error_message = "Replication type must be LRS, GRS, ZRS, or GZRS."
  }
}

variable "disable_public_access" {
  description = "Disable public access to storage account"
  type        = bool
  default     = false
}

variable "shared_access_key_enabled" {
  description = "Enable shared access keys for storage account"
  type        = bool
  default     = true
}

variable "allowed_ip_ranges" {
  description = "List of allowed IP ranges for storage account access"
  type        = list(string)
  default     = []
}

variable "allowed_subnet_ids" {
  description = "List of allowed subnet IDs for storage account access"
  type        = list(string)
  default     = []
}

variable "versioning_enabled" {
  description = "Enable blob versioning"
  type        = bool
  default     = true
}

variable "change_feed_enabled" {
  description = "Enable change feed for blobs"
  type        = bool
  default     = false
}

variable "change_feed_retention_days" {
  description = "Change feed retention days"
  type        = number
  default     = 7
}

variable "blob_soft_delete_days" {
  description = "Soft delete retention days for blobs"
  type        = number
  default     = 7

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
  default     = 0

  validation {
    condition     = var.point_in_time_restore_days >= 0 && var.point_in_time_restore_days <= 365
    error_message = "Point-in-time restore days must be between 0 and 365."
  }
}

variable "last_access_time_enabled" {
  description = "Enable last access time tracking"
  type        = bool
  default     = false
}

variable "queue_log_retention_days" {
  description = "Queue log retention days"
  type        = number
  default     = 30
}

variable "create_private_endpoints" {
  description = "Create private endpoints for storage account"
  type        = bool
  default     = false
}

variable "private_endpoint_subnet_id" {
  description = "Subnet ID for private endpoints"
  type        = string
  default     = null
}

variable "blob_private_dns_zone_id" {
  description = "Private DNS zone ID for blob private endpoint"
  type        = string
  default     = null
}

variable "customer_managed_key" {
  description = "Customer managed key configuration"
  type = object({
    key_vault_id              = string
    key_name                  = string
    key_version               = optional(string)
    user_assigned_identity_id = string
  })
  default = null
}

variable "enable_lifecycle_policy" {
  description = "Enable storage lifecycle management policy"
  type        = bool
  default     = true
}

variable "enable_diagnostics" {
  description = "Enable diagnostic settings"
  type        = bool
  default     = true
}

variable "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID for diagnostics"
  type        = string
  default     = null
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

variable "sftp_enabled" {
  description = "Enable SFTP for secure file transfers"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}