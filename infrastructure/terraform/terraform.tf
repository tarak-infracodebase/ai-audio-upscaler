# Terraform configuration with latest provider versions
terraform {
  required_version = ">= 1.8"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.55"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 3.1"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
    azapi = {
      source  = "azure/azapi"
      version = "~> 2.1"
    }
  }

  # Backend configuration for Terraform Cloud/Enterprise
  # Uncomment and configure for production
  # backend "azurerm" {
  #   resource_group_name  = "ai-upscaler-tfstate-rg"
  #   storage_account_name = "aiupscalertfstate"
  #   container_name       = "tfstate"
  #   key                  = "ai-upscaler.tfstate"
  #   use_azuread_auth     = true
  # }
}

# Configure AzureRM Provider features
provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy          = true
      purge_soft_deleted_keys_on_destroy    = true
      purge_soft_deleted_secrets_on_destroy = true
      purge_soft_deleted_certificates_on_destroy = true
      recover_soft_deleted_key_vaults       = true
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
    cognitive_account {
      purge_soft_delete_on_destroy = true
    }
  }
}

provider "azuread" {}
provider "azapi" {}