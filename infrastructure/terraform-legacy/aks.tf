# Azure Kubernetes Service Configuration
# Production-ready AKS cluster with GPU node pools

# Log Analytics Workspace for AKS monitoring
resource "azurerm_log_analytics_workspace" "aks" {
  name                = "${local.naming_prefix}-aks-logs-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = var.environment == "production" ? 30 : 7
  tags                = local.common_tags
}

# Managed Identity for AKS
resource "azurerm_user_assigned_identity" "aks" {
  name                = "${local.naming_prefix}-aks-identity-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.common_tags
}

# Role assignments for AKS managed identity
resource "azurerm_role_assignment" "aks_network_contributor" {
  scope                = azurerm_virtual_network.main.id
  role_definition_name = "Network Contributor"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${local.naming_prefix}-aks-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${local.naming_prefix}-aks-${local.suffix}"
  kubernetes_version  = var.kubernetes_version

  # Automatic upgrade channel
  automatic_channel_upgrade = var.environment == "production" ? "stable" : "patch"

  # Default node pool (system nodes)
  default_node_pool {
    name                = "system"
    node_count          = var.aks_system_node_count
    vm_size             = "Standard_D4s_v3"
    os_disk_size_gb     = 128
    os_disk_type        = "Managed"
    vnet_subnet_id      = azurerm_subnet.subnets["aks"].id
    type                = "VirtualMachineScaleSets"
    enable_auto_scaling = true
    min_count           = 1
    max_count           = 5

    # Only system workloads
    only_critical_addons_enabled = true

    # Node labels
    node_labels = {
      "nodepool-type" = "system"
      "environment"   = local.environment
      "nodepoolos"    = "linux"
      "app"           = "system"
    }

    # Node taints for system nodes
    node_taints = [
      "CriticalAddonsOnly=true:NoSchedule"
    ]

    tags = local.common_tags
  }

  # Identity
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.aks.id]
  }

  # Network configuration
  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    dns_service_ip    = "10.0.0.10"
    service_cidr      = "10.0.0.0/24"
    load_balancer_sku = "standard"
  }

  # Add-ons
  oms_agent {
    log_analytics_workspace_id      = azurerm_log_analytics_workspace.aks.id
    msi_auth_for_monitoring_enabled = true
  }

  azure_policy_enabled             = true
  http_application_routing_enabled = false

  # Key Vault integration
  key_vault_secrets_provider {
    secret_rotation_enabled  = true
    secret_rotation_interval = "2m"
  }

  # Auto-scaler settings
  auto_scaler_profile {
    balance_similar_node_groups      = false
    expander                         = "random"
    max_graceful_termination_sec     = 600
    max_node_provisioning_time       = "15m"
    max_unready_nodes                = 3
    max_unready_percentage           = 45
    new_pod_scale_up_delay           = "10s"
    scale_down_delay_after_add       = "10m"
    scale_down_delay_after_delete    = "10s"
    scale_down_delay_after_failure   = "3m"
    scan_interval                    = "10s"
    scale_down_unneeded              = "10m"
    scale_down_unready               = "20m"
    scale_down_utilization_threshold = 0.5
    empty_bulk_delete_max            = 10
    skip_nodes_with_local_storage    = true
    skip_nodes_with_system_pods      = true
  }

  tags = local.common_tags
}

# CPU Worker Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "cpu_workers" {
  name                  = "cpuworkers"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = "Standard_D8s_v3"
  node_count            = var.aks_cpu_node_count
  os_disk_size_gb       = 128
  os_disk_type          = "Managed"
  vnet_subnet_id        = azurerm_subnet.subnets["aks"].id

  # Auto-scaling
  enable_auto_scaling = true
  min_count           = 0
  max_count           = 10

  # Node configuration
  node_labels = {
    "nodepool-type" = "cpu-worker"
    "workload-type" = "cpu-intensive"
    "environment"   = local.environment
  }

  tags = local.common_tags
}

# GPU Worker Node Pool (for AI inference)
resource "azurerm_kubernetes_cluster_node_pool" "gpu_workers" {
  count = var.enable_gpu_nodes ? 1 : 0

  name                  = "gpuworkers"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.gpu_vm_size
  node_count            = var.aks_gpu_node_count
  os_disk_size_gb       = 256
  os_disk_type          = "Managed"
  vnet_subnet_id        = azurerm_subnet.subnets["aks"].id

  # Auto-scaling
  enable_auto_scaling = true
  min_count           = 0
  max_count           = 5

  # Node configuration
  node_labels = {
    "nodepool-type" = "gpu-worker"
    "workload-type" = "gpu-intensive"
    "accelerator"   = "nvidia-gpu"
    "environment"   = local.environment
    "sku"           = var.gpu_vm_size
  }

  # Taints to ensure only GPU workloads are scheduled
  node_taints = [
    "nvidia.com/gpu=true:NoSchedule",
    "sku=${var.gpu_vm_size}:NoSchedule"
  ]

  tags = local.common_tags
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = "${replace(local.naming_prefix, "-", "")}acr${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.acr_sku
  admin_enabled       = false

  # Network access
  network_rule_set {
    default_action = "Deny"
    virtual_network {
      action    = "Allow"
      subnet_id = azurerm_subnet.subnets["aks"].id
    }
  }

  # Retention policy for production
  dynamic "retention_policy" {
    for_each = var.acr_sku == "Premium" ? [1] : []
    content {
      days    = 30
      enabled = true
    }
  }

  # Trust policy for production
  dynamic "trust_policy" {
    for_each = var.acr_sku == "Premium" && var.environment == "production" ? [1] : []
    content {
      enabled = true
    }
  }

  tags = local.common_tags
}

# Application Gateway for ingress (optional)
resource "azurerm_public_ip" "app_gateway" {
  count = var.enable_app_gateway ? 1 : 0

  name                = "${local.naming_prefix}-appgw-pip-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  allocation_method   = "Static"
  sku                 = "Standard"
  zones               = var.environment == "production" ? ["1", "2", "3"] : null

  tags = local.common_tags
}

resource "azurerm_application_gateway" "main" {
  count = var.enable_app_gateway ? 1 : 0

  name                = "${local.naming_prefix}-appgw-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  sku {
    name     = "Standard_v2"
    tier     = "Standard_v2"
    capacity = 2
  }

  gateway_ip_configuration {
    name      = "appGatewayIpConfig"
    subnet_id = azurerm_subnet.subnets["aks"].id
  }

  frontend_port {
    name = "port_80"
    port = 80
  }

  frontend_port {
    name = "port_443"
    port = 443
  }

  frontend_ip_configuration {
    name                 = "appGwPublicFrontendIp"
    public_ip_address_id = azurerm_public_ip.app_gateway[0].id
  }

  backend_address_pool {
    name = "defaultaddresspool"
  }

  backend_http_settings {
    name                  = "defaulthttpsetting"
    cookie_based_affinity = "Disabled"
    port                  = 80
    protocol              = "Http"
    request_timeout       = 60
  }

  http_listener {
    name                           = "defaulthttplistener"
    frontend_ip_configuration_name = "appGwPublicFrontendIp"
    frontend_port_name             = "port_80"
    protocol                       = "Http"
  }

  request_routing_rule {
    name                       = "defaultroutingrule"
    rule_type                  = "Basic"
    http_listener_name         = "defaulthttplistener"
    backend_address_pool_name  = "defaultaddresspool"
    backend_http_settings_name = "defaulthttpsetting"
    priority                   = 100
  }

  tags = local.common_tags
}