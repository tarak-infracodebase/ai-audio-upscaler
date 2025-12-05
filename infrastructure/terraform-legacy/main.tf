# AI Audio Upscaler Pro - Azure Infrastructure
# Production-ready Terraform configuration for Azure deployment

terraform {
  required_version = ">= 1.6"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.117"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.53"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
  }

  # Configure remote state (uncomment for production)
  # backend "azurerm" {
  #   resource_group_name  = "ai-upscaler-tfstate-rg"
  #   storage_account_name = "aiupscalertfstate"
  #   container_name       = "tfstate"
  #   key                  = "prod.terraform.tfstate"
  #   use_azuread_auth     = true
  # }
}

# Configure providers
provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    virtual_machine {
      delete_os_disk_on_deletion     = true
      graceful_shutdown              = false
      skip_shutdown_and_force_delete = false
    }
    log_analytics_workspace {
      permanently_delete_on_destroy = true
    }
  }
}

provider "azuread" {}

# Data sources
data "azuread_client_config" "current" {}
data "azurerm_client_config" "current" {}

# Generate random suffix for unique naming
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

# Local variables
locals {
  project_name = "ai-upscaler"
  environment  = var.environment
  location     = var.location
  suffix       = random_string.suffix.result

  # Naming convention
  naming_prefix = "${local.project_name}-${local.environment}"

  # Common tags
  common_tags = {
    Project     = "AI Audio Upscaler Pro"
    Environment = local.environment
    ManagedBy   = "Terraform"
    CreatedDate = formatdate("YYYY-MM-DD", timestamp())
    CostCenter  = "Engineering"
  }

  # Network configuration
  vnet_address_space = ["10.0.0.0/16"]
  subnets = {
    aks = {
      name             = "aks-subnet"
      address_prefixes = ["10.0.1.0/24"]
    }
    aci = {
      name             = "aci-subnet"
      address_prefixes = ["10.0.2.0/24"]
    }
    postgres = {
      name             = "postgres-subnet"
      address_prefixes = ["10.0.3.0/24"]
    }
    private_endpoints = {
      name             = "private-endpoints-subnet"
      address_prefixes = ["10.0.4.0/24"]
    }
  }
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${local.naming_prefix}-rg-${local.suffix}"
  location = local.location
  tags     = local.common_tags
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${local.naming_prefix}-vnet-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  address_space       = local.vnet_address_space
  tags                = local.common_tags
}

# Subnets
resource "azurerm_subnet" "subnets" {
  for_each = local.subnets

  name                 = each.value.name
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = each.value.address_prefixes

  # Delegate postgres subnet to PostgreSQL
  dynamic "delegation" {
    for_each = each.key == "postgres" ? [1] : []
    content {
      name = "postgresql_delegation"
      service_delegation {
        name = "Microsoft.DBforPostgreSQL/flexibleServers"
        actions = [
          "Microsoft.Network/virtualNetworks/subnets/join/action"
        ]
      }
    }
  }
}

# Network Security Groups
resource "azurerm_network_security_group" "aks" {
  name                = "${local.naming_prefix}-aks-nsg-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTP"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Associate NSG with AKS subnet
resource "azurerm_subnet_network_security_group_association" "aks" {
  subnet_id                 = azurerm_subnet.subnets["aks"].id
  network_security_group_id = azurerm_network_security_group.aks.id
}

# Key Vault for secrets
resource "azurerm_key_vault" "main" {
  name                = "${local.naming_prefix}-kv-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  # Network access
  network_acls {
    bypass                     = "AzureServices"
    default_action             = "Deny"
    virtual_network_subnet_ids = [azurerm_subnet.subnets["aks"].id]
  }

  # Enable soft delete and purge protection for production
  soft_delete_retention_days = var.environment == "production" ? 90 : 7
  purge_protection_enabled   = var.environment == "production"

  tags = local.common_tags
}

# Key Vault access policy for current user
resource "azurerm_key_vault_access_policy" "current_user" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id

  secret_permissions = [
    "Get", "List", "Set", "Delete", "Purge", "Recover"
  ]
}

# Storage Account
resource "azurerm_storage_account" "main" {
  name                = "${replace(local.naming_prefix, "-", "")}sa${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  account_tier             = "Standard"
  account_replication_type = var.environment == "production" ? "ZRS" : "LRS"
  account_kind             = "StorageV2"
  access_tier              = "Hot"

  # Security settings
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = true
  https_traffic_only_enabled      = true

  # Network access
  network_rules {
    default_action             = "Deny"
    bypass                     = ["AzureServices"]
    virtual_network_subnet_ids = [azurerm_subnet.subnets["aks"].id]
  }

  # Blob properties
  blob_properties {
    delete_retention_policy {
      days = var.environment == "production" ? 30 : 7
    }
    container_delete_retention_policy {
      days = var.environment == "production" ? 30 : 7
    }
    versioning_enabled = var.environment == "production"
  }

  tags = local.common_tags
}

# Storage containers
resource "azurerm_storage_container" "audio_inputs" {
  name                  = "audio-inputs"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "audio_outputs" {
  name                  = "audio-outputs"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# Redis Cache
resource "azurerm_redis_cache" "main" {
  name                = "${local.naming_prefix}-redis-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku_name

  # Security
  minimum_tls_version = "1.2"

  # Network access
  subnet_id = var.redis_sku_name == "Premium" ? azurerm_subnet.subnets["aks"].id : null

  # Redis configuration
  redis_configuration {
    authentication_enabled = true
    maxmemory_policy       = "allkeys-lru"
  }

  tags = local.common_tags
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${local.naming_prefix}-postgres-${local.suffix}"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "15"
  delegated_subnet_id    = azurerm_subnet.subnets["postgres"].id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  administrator_login    = var.postgres_admin_login
  administrator_password = var.postgres_admin_password

  storage_mb = var.postgres_storage_mb
  sku_name   = var.postgres_sku_name

  # Backup settings
  backup_retention_days        = var.environment == "production" ? 35 : 7
  geo_redundant_backup_enabled = var.environment == "production"

  # High availability for production
  dynamic "high_availability" {
    for_each = var.environment == "production" ? [1] : []
    content {
      mode = "ZoneRedundant"
    }
  }

  tags       = local.common_tags
  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]
}

# PostgreSQL database
resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = "ai_audio_upscaler"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Private DNS zone for PostgreSQL
resource "azurerm_private_dns_zone" "postgres" {
  name                = "${local.naming_prefix}-postgres-${local.suffix}.private.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "postgres-vnet-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
  tags                  = local.common_tags
}