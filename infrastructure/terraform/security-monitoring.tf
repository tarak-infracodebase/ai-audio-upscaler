# Azure Security Center and Monitoring Configuration
# Comprehensive security monitoring and alerting for 10/10 security score

# Azure Security Center (Microsoft Defender for Cloud) subscription-level configuration
resource "azurerm_security_center_subscription_pricing" "storage" {
  tier          = var.environment == "production" ? "Standard" : "Free"
  resource_type = "StorageAccounts"
}

resource "azurerm_security_center_subscription_pricing" "containers" {
  tier          = var.environment == "production" ? "Standard" : "Free"
  resource_type = "Containers"
}

resource "azurerm_security_center_subscription_pricing" "kubernetes" {
  tier          = var.environment == "production" ? "Standard" : "Free"
  resource_type = "KubernetesService"
}

resource "azurerm_security_center_subscription_pricing" "key_vault" {
  tier          = var.environment == "production" ? "Standard" : "Free"
  resource_type = "KeyVaults"
}

resource "azurerm_security_center_subscription_pricing" "databases" {
  tier          = var.environment == "production" ? "Standard" : "Free"
  resource_type = "SqlServers"
}

# Security Center contact for alerts
resource "azurerm_security_center_contact" "security_contact" {
  email = var.security_contact_email
  phone = var.security_contact_phone

  alert_notifications = true
  alerts_to_admins    = true
}

# Microsoft Defender for Container Registries
resource "azurerm_security_center_auto_provisioning" "auto_provisioning" {
  auto_provision = "On"
}

# Log Analytics Workspace for security monitoring (if not already created)
locals {
  create_security_workspace = var.create_separate_security_workspace
}

resource "azurerm_log_analytics_workspace" "security" {
  count = local.create_security_workspace ? 1 : 0

  name                = "${var.project_name}-security-logs-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  sku               = "PerGB2018"
  retention_in_days = var.environment == "production" ? 90 : 30

  tags = local.common_tags
}

# Security solutions and monitoring
resource "azurerm_log_analytics_solution" "security_center" {
  count = local.create_security_workspace ? 1 : 0

  solution_name         = "Security"
  location              = azurerm_resource_group.main.location
  resource_group_name   = azurerm_resource_group.main.name
  workspace_resource_id = azurerm_log_analytics_workspace.security[0].id
  workspace_name        = azurerm_log_analytics_workspace.security[0].name

  plan {
    publisher = "Microsoft"
    product   = "OMSGallery/Security"
  }

  tags = local.common_tags
}

resource "azurerm_log_analytics_solution" "security_insights" {
  count = local.create_security_workspace ? 1 : 0

  solution_name         = "SecurityInsights"
  location              = azurerm_resource_group.main.location
  resource_group_name   = azurerm_resource_group.main.name
  workspace_resource_id = azurerm_log_analytics_workspace.security[0].id
  workspace_name        = azurerm_log_analytics_workspace.security[0].name

  plan {
    publisher = "Microsoft"
    product   = "OMSGallery/SecurityInsights"
  }

  tags = local.common_tags
}

# Azure Sentinel (now part of Microsoft Sentinel)
resource "azurerm_sentinel_log_analytics_workspace_onboarding" "sentinel" {
  count                        = local.create_security_workspace ? 1 : 0
  workspace_id                 = azurerm_log_analytics_workspace.security[0].id
  customer_managed_key_enabled = var.environment == "production"
}

# Action Groups for security alerts
resource "azurerm_monitor_action_group" "security_critical" {
  name                = "${var.project_name}-security-critical-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "sec-crit"

  email_receiver {
    name          = "security-team"
    email_address = var.security_contact_email
  }

  dynamic "sms_receiver" {
    for_each = var.security_contact_phone != null ? [var.security_contact_phone] : []
    content {
      name         = "security-sms"
      country_code = "1"
      phone_number = sms_receiver.value
    }
  }

  webhook_receiver {
    name        = "security-webhook"
    service_uri = var.security_webhook_url
  }

  tags = local.common_tags
}

resource "azurerm_monitor_action_group" "security_warning" {
  name                = "${var.project_name}-security-warning-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "sec-warn"

  email_receiver {
    name          = "ops-team"
    email_address = var.ops_contact_email
  }

  tags = local.common_tags
}

# Security Alert Rules
resource "azurerm_monitor_scheduled_query_rules_alert_v2" "failed_logins" {
  name                = "${var.project_name}-failed-logins-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  evaluation_frequency = "PT5M"
  window_duration      = "PT10M"
  scopes              = [azurerm_log_analytics_workspace.main.id]
  severity            = 2

  criteria {
    query                   = <<-QUERY
      AuditLogs
      | where TimeGenerated > ago(10m)
      | where OperationName contains "Sign-in activity"
      | where Result == "failure"
      | summarize FailedAttempts = count() by UserPrincipalName, IPAddress
      | where FailedAttempts >= 5
      QUERY
    time_aggregation_method = "Count"
    threshold               = 1
    operator                = "GreaterThan"

    failing_periods {
      minimum_failing_periods_to_trigger_alert = 1
      number_of_evaluation_periods             = 1
    }
  }

  action {
    action_groups = [azurerm_monitor_action_group.security_critical.id]
  }

  tags = local.common_tags
}

resource "azurerm_monitor_scheduled_query_rules_alert_v2" "unusual_api_activity" {
  name                = "${var.project_name}-unusual-api-activity-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  evaluation_frequency = "PT15M"
  window_duration      = "PT30M"
  scopes              = [azurerm_log_analytics_workspace.main.id]
  severity            = 3

  criteria {
    query                   = <<-QUERY
      ContainerLog
      | where TimeGenerated > ago(30m)
      | where LogEntry contains "ERROR" or LogEntry contains "CRITICAL"
      | where LogEntry contains "authentication" or LogEntry contains "authorization"
      | summarize ErrorCount = count() by Computer
      | where ErrorCount >= 10
      QUERY
    time_aggregation_method = "Count"
    threshold               = 1
    operator                = "GreaterThan"

    failing_periods {
      minimum_failing_periods_to_trigger_alert = 1
      number_of_evaluation_periods             = 1
    }
  }

  action {
    action_groups = [azurerm_monitor_action_group.security_warning.id]
  }

  tags = local.common_tags
}

resource "azurerm_monitor_scheduled_query_rules_alert_v2" "privilege_escalation" {
  name                = "${var.project_name}-privilege-escalation-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  evaluation_frequency = "PT5M"
  window_duration      = "PT10M"
  scopes              = [azurerm_log_analytics_workspace.main.id]
  severity            = 1

  criteria {
    query                   = <<-QUERY
      AuditLogs
      | where TimeGenerated > ago(10m)
      | where OperationName contains "Add member to role"
      | where TargetResources has "Global Administrator" or TargetResources has "Privileged Role Administrator"
      | project TimeGenerated, OperationName, InitiatedBy, TargetResources
      QUERY
    time_aggregation_method = "Count"
    threshold               = 1
    operator                = "GreaterThan"

    failing_periods {
      minimum_failing_periods_to_trigger_alert = 1
      number_of_evaluation_periods             = 1
    }
  }

  action {
    action_groups = [azurerm_monitor_action_group.security_critical.id]
  }

  tags = local.common_tags
}

# Storage account monitoring
resource "azurerm_monitor_metric_alert" "storage_delete_operations" {
  name                = "${var.project_name}-storage-delete-alert-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_storage_account.main.id]

  criteria {
    metric_namespace = "Microsoft.Storage/storageAccounts"
    metric_name      = "Transactions"
    aggregation      = "Total"
    operator         = "GreaterThan"
    threshold        = 10

    dimension {
      name     = "ApiName"
      operator = "Include"
      values   = ["DeleteBlob", "DeleteContainer"]
    }
  }

  action {
    action_group_id = azurerm_monitor_action_group.security_warning.id
  }

  frequency   = "PT5M"
  window_size = "PT15M"

  tags = local.common_tags
}

# Key Vault access monitoring
resource "azurerm_monitor_metric_alert" "key_vault_failed_requests" {
  name                = "${var.project_name}-keyvault-failed-requests-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_key_vault.main.id]

  criteria {
    metric_namespace = "Microsoft.KeyVault/vaults"
    metric_name      = "ServiceApiResult"
    aggregation      = "Count"
    operator         = "GreaterThan"
    threshold        = 5

    dimension {
      name     = "StatusCode"
      operator = "Include"
      values   = ["401", "403", "404"]
    }
  }

  action {
    action_group_id = azurerm_monitor_action_group.security_warning.id
  }

  frequency   = "PT5M"
  window_size = "PT15M"

  tags = local.common_tags
}

# AKS security monitoring
resource "azurerm_monitor_metric_alert" "aks_pod_failed" {
  name                = "${var.project_name}-aks-pod-failures-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [module.aks.aks_id]

  criteria {
    metric_namespace = "Microsoft.ContainerService/managedClusters"
    metric_name      = "PodReadyPercentage"
    aggregation      = "Average"
    operator         = "LessThan"
    threshold        = 90
  }

  action {
    action_group_id = azurerm_monitor_action_group.security_warning.id
  }

  frequency   = "PT5M"
  window_size = "PT15M"

  tags = local.common_tags
}

# Database security monitoring
resource "azurerm_monitor_metric_alert" "database_failed_connections" {
  name                = "${var.project_name}-db-failed-connections-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_postgresql_flexible_server.main.id]

  criteria {
    metric_namespace = "Microsoft.DBforPostgreSQL/flexibleServers"
    metric_name      = "connections_failed"
    aggregation      = "Total"
    operator         = "GreaterThan"
    threshold        = 10
  }

  action {
    action_group_id = azurerm_monitor_action_group.security_warning.id
  }

  frequency   = "PT5M"
  window_size = "PT15M"

  tags = local.common_tags
}

# Diagnostic settings for comprehensive logging
resource "azurerm_monitor_diagnostic_setting" "key_vault" {
  name               = "${var.project_name}-keyvault-diagnostics"
  target_resource_id = azurerm_key_vault.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  enabled_log {
    category = "AuditEvent"
  }

  metric {
    category = "AllMetrics"
  }
}

resource "azurerm_monitor_diagnostic_setting" "storage" {
  name               = "${var.project_name}-storage-diagnostics"
  target_resource_id = azurerm_storage_account.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  enabled_log {
    category = "StorageRead"
  }

  enabled_log {
    category = "StorageWrite"
  }

  enabled_log {
    category = "StorageDelete"
  }

  metric {
    category = "AllMetrics"
  }
}

# Network Watcher for network security monitoring
resource "azurerm_network_watcher" "main" {
  count = var.enable_network_watcher ? 1 : 0

  name                = "${var.project_name}-network-watcher-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = local.common_tags
}

# Network Security Group Flow Logs
resource "azurerm_network_watcher_flow_log" "nsg_flow_logs" {
  count = var.enable_network_watcher && var.enable_nsg_flow_logs ? 1 : 0

  network_watcher_name = azurerm_network_watcher.main[0].name
  resource_group_name  = azurerm_resource_group.main.name

  network_security_group_id = module.network.nsg_id
  storage_account_id        = azurerm_storage_account.main.id
  enabled                   = true

  retention_policy {
    enabled = true
    days    = var.environment == "production" ? 90 : 30
  }

  traffic_analytics {
    enabled               = true
    workspace_id          = azurerm_log_analytics_workspace.main.workspace_id
    workspace_region      = azurerm_log_analytics_workspace.main.location
    workspace_resource_id = azurerm_log_analytics_workspace.main.id
    interval_in_minutes   = 10
  }

  tags = local.common_tags
}