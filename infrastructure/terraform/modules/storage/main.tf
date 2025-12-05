# Secure Storage Module for AI Audio Upscaler
# Implements latest security best practices for Azure Storage

# Storage Account with advanced security features
resource "azurerm_storage_account" "main" {
  name                = var.storage_account_name
  resource_group_name = var.resource_group_name
  location            = var.location

  account_tier             = var.account_tier
  account_replication_type = var.replication_type
  account_kind             = "StorageV2"
  access_tier              = "Hot"

  # Security settings
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = var.shared_access_key_enabled
  https_traffic_only_enabled      = true

  # Enable SFTP if needed for secure file transfers
  sftp_enabled              = var.sftp_enabled
  is_hns_enabled           = var.sftp_enabled # Required for SFTP
  nfsv3_enabled            = false

  # Cross-tenant replication prevention
  cross_tenant_replication_enabled = false

  # Public access prevention
  public_network_access_enabled = !var.disable_public_access

  # Network rules
  network_rules {
    default_action             = var.disable_public_access ? "Deny" : "Allow"
    bypass                     = ["AzureServices", "Logging", "Metrics"]
    ip_rules                   = var.allowed_ip_ranges
    virtual_network_subnet_ids = var.allowed_subnet_ids
  }

  # Blob properties with advanced security
  blob_properties {
    # Soft delete for blobs
    delete_retention_policy {
      days = var.blob_soft_delete_days
    }

    # Soft delete for containers
    container_delete_retention_policy {
      days = var.container_soft_delete_days
    }

    # Versioning for data protection
    versioning_enabled       = var.versioning_enabled
    change_feed_enabled      = var.change_feed_enabled
    change_feed_retention_in_days = var.change_feed_enabled ? var.change_feed_retention_days : null

    # Restore policy for point-in-time recovery
    dynamic "restore_policy" {
      for_each = var.point_in_time_restore_days > 0 ? [1] : []
      content {
        days = var.point_in_time_restore_days
      }
    }

    # Last access time tracking
    last_access_time_enabled = var.last_access_time_enabled
  }

  # Queue properties
  queue_properties {
    # Queue logging
    logging {
      delete  = true
      read    = true
      write   = true
      version = "1.0"
      retention_policy_days = var.queue_log_retention_days
    }
  }

  # Advanced threat protection
  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Storage containers for different types of audio files
resource "azurerm_storage_container" "audio_inputs" {
  name                  = "audio-inputs"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"

  metadata = {
    description = "Input audio files for processing"
    environment = var.environment
  }
}

resource "azurerm_storage_container" "audio_outputs" {
  name                  = "audio-outputs"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"

  metadata = {
    description = "Processed output audio files"
    environment = var.environment
  }
}

resource "azurerm_storage_container" "ml_models" {
  name                  = "ml-models"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"

  metadata = {
    description = "Machine learning models and weights"
    environment = var.environment
  }
}

resource "azurerm_storage_container" "temp_processing" {
  name                  = "temp-processing"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"

  metadata = {
    description = "Temporary files during processing"
    environment = var.environment
  }
}

# Private endpoint for secure connectivity
resource "azurerm_private_endpoint" "storage_blob" {
  count = var.create_private_endpoints ? 1 : 0

  name                = "${var.storage_account_name}-blob-pe"
  location            = var.location
  resource_group_name = var.resource_group_name
  subnet_id           = var.private_endpoint_subnet_id

  private_service_connection {
    name                           = "${var.storage_account_name}-blob-psc"
    private_connection_resource_id = azurerm_storage_account.main.id
    subresource_names              = ["blob"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "blob-dns-zone-group"
    private_dns_zone_ids = [var.blob_private_dns_zone_id]
  }

  tags = var.tags
}

# Storage Account Customer-Managed Keys (if enabled)
resource "azurerm_storage_account_customer_managed_key" "main" {
  count = var.customer_managed_key != null ? 1 : 0

  storage_account_id = azurerm_storage_account.main.id
  key_vault_id       = var.customer_managed_key.key_vault_id
  key_name           = var.customer_managed_key.key_name
  key_version        = var.customer_managed_key.key_version
  user_assigned_identity_id = var.customer_managed_key.user_assigned_identity_id
}

# Lifecycle management policy
resource "azurerm_storage_management_policy" "main" {
  count = var.enable_lifecycle_policy ? 1 : 0

  storage_account_id = azurerm_storage_account.main.id

  # Rule for temporary processing files
  rule {
    name    = "temp-processing-cleanup"
    enabled = true
    filters {
      prefix_match = ["temp-processing/"]
      blob_types   = ["blockBlob"]
    }
    actions {
      base_blob {
        delete_after_days_since_modification_greater_than = 7
      }
      snapshot {
        delete_after_days_since_creation_greater_than = 7
      }
      version {
        delete_after_days_since_creation_greater_than = 7
      }
    }
  }

  # Rule for input files archiving
  rule {
    name    = "audio-inputs-archiving"
    enabled = true
    filters {
      prefix_match = ["audio-inputs/"]
      blob_types   = ["blockBlob"]
    }
    actions {
      base_blob {
        tier_to_cool_after_days_since_modification_greater_than    = 30
        tier_to_archive_after_days_since_modification_greater_than = 90
      }
    }
  }

  # Rule for output files lifecycle
  rule {
    name    = "audio-outputs-lifecycle"
    enabled = true
    filters {
      prefix_match = ["audio-outputs/"]
      blob_types   = ["blockBlob"]
    }
    actions {
      base_blob {
        tier_to_cool_after_days_since_modification_greater_than = 60
      }
    }
  }
}

# Diagnostic settings for monitoring
resource "azurerm_monitor_diagnostic_setting" "storage" {
  count = var.enable_diagnostics ? 1 : 0

  name               = "${var.storage_account_name}-diagnostics"
  target_resource_id = "${azurerm_storage_account.main.id}/blobServices/default"
  log_analytics_workspace_id = var.log_analytics_workspace_id

  # Enabled log categories
  enabled_log {
    category = "StorageRead"
  }
  enabled_log {
    category = "StorageWrite"
  }
  enabled_log {
    category = "StorageDelete"
  }

  # Metrics
  metric {
    category = "Transaction"
    enabled  = true

    retention_policy {
      enabled = true
      days    = var.diagnostic_retention_days
    }
  }
}