# Modern AKS Module with Latest Security Best Practices

# Log Analytics Workspace for container insights
resource "azurerm_log_analytics_workspace" "aks" {
  name                = var.log_analytics_name
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days

  tags = var.tags
}

# User Assigned Identity for AKS
resource "azurerm_user_assigned_identity" "aks" {
  name                = var.identity_name
  location            = var.location
  resource_group_name = var.resource_group_name

  tags = var.tags
}

# Role assignments for AKS identity
resource "azurerm_role_assignment" "aks_network_contributor" {
  scope                = var.vnet_id
  role_definition_name = "Network Contributor"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

resource "azurerm_role_assignment" "aks_acr_pull" {
  count = var.container_registry_id != null ? 1 : 0

  scope                = var.container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

# Use the modern AKS module from the registry
module "aks" {
  source  = "claranet/aks-light/azurerm"
  version = "8.11.0"

  # Required parameters
  client_name         = var.client_name
  environment         = var.environment
  stack               = var.stack
  resource_group_name = var.resource_group_name
  location            = var.location
  location_short      = var.location_short

  # Network configuration
  nodes_subnet = {
    name                 = var.aks_subnet_name
    virtual_network_name = var.vnet_name
    resource_group_name  = var.resource_group_name
  }

  service_cidr = var.service_cidr

  # Network plugin configuration with Overlay CNI
  network_plugin = {
    name     = "azure"
    cni_mode = "overlay"
  }
  network_policy = "cilium"

  # Kubernetes version
  kubernetes_version = var.kubernetes_version

  # Identity configuration
  user_assigned_identity = {
    id        = azurerm_user_assigned_identity.aks.id
    client_id = azurerm_user_assigned_identity.aks.client_id
    object_id = azurerm_user_assigned_identity.aks.principal_id
  }

  # Security settings
  private_cluster_enabled = var.private_cluster_enabled
  azure_policy_enabled    = true

  # RBAC configuration
  azure_active_directory_rbac = var.rbac_enabled ? {
    azure_rbac_enabled     = true
    admin_group_object_ids = var.admin_group_object_ids
  } : null

  # Default node pool configuration
  default_node_pool = {
    name                 = "system"
    vm_size              = var.system_node_vm_size
    node_count           = var.system_node_count
    auto_scaling_enabled = true
    min_count            = var.system_node_min_count
    max_count            = var.system_node_max_count
    os_sku               = "Ubuntu"
    os_disk_type         = "Ephemeral"
    zones                = var.availability_zones

    node_labels = {
      "nodepool-type"    = "system"
      "environment"      = var.environment
      "workload-type"    = "system"
    }

    node_taints = [
      "CriticalAddonsOnly=true:NoSchedule"
    ]

    upgrade_settings = {
      max_surge = "33%"
    }
  }

  # Additional node pools
  node_pools = var.node_pools

  # Monitoring configuration
  oms_agent = {
    log_analytics_workspace_id      = azurerm_log_analytics_workspace.aks.id
    msi_auth_for_monitoring_enabled = true
  }

  # Key Vault secrets provider
  key_vault_secrets_provider = var.key_vault_secrets_provider_enabled ? {
    secret_rotation_enabled  = true
    secret_rotation_interval = "2m"
  } : {}

  # Workload Identity
  workload_identity_enabled = true
  oidc_issuer_enabled      = true

  # Maintenance windows
  maintenance_window = var.maintenance_window

  # Auto-scaling configuration
  auto_scaler_profile = var.auto_scaler_profile

  # Upgrade settings
  automatic_upgrade_channel = var.automatic_upgrade_channel
  node_os_upgrade_channel   = "SecurityPatch"

  # Storage configuration
  storage_profile = {
    blob_driver_enabled         = true
    disk_driver_enabled         = true
    file_driver_enabled         = true
    snapshot_controller_enabled = true
  }

  # Image cleaner
  image_cleaner_configuration = {
    enabled        = true
    interval_hours = 24
  }

  # Monitoring destinations
  logs_destinations_ids = [azurerm_log_analytics_workspace.aks.id]

  # Cost optimization
  sku_tier = var.sku_tier

  # Tags
  extra_tags = var.tags
}

# Private endpoints for secure connectivity (if private cluster is enabled)
resource "azurerm_private_endpoint" "aks_api" {
  count = var.private_cluster_enabled && var.create_private_endpoint ? 1 : 0

  name                = "${var.aks_cluster_name}-api-pe"
  location            = var.location
  resource_group_name = var.resource_group_name
  subnet_id           = var.private_endpoint_subnet_id

  private_service_connection {
    name                           = "${var.aks_cluster_name}-api-psc"
    private_connection_resource_id = module.aks.id
    subresource_names              = ["management"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "aks-dns-zone-group"
    private_dns_zone_ids = [var.aks_private_dns_zone_id]
  }

  tags = var.tags
}