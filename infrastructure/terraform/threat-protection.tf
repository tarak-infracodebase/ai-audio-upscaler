# Advanced Threat Protection for Azure Services
# Microsoft Defender for Cloud comprehensive coverage

# Microsoft Defender for Storage
resource "azurerm_advanced_threat_protection" "storage" {
  target_resource_id = azurerm_storage_account.main.id
  enabled            = var.environment == "production" ? true : var.enable_threat_protection_dev
}

# Microsoft Defender for Key Vault (handled via Security Center pricing in security-monitoring.tf)
# Additional configuration for advanced threat detection

# Advanced Threat Protection for PostgreSQL
resource "azurerm_mssql_server_security_alert_policy" "postgres_threat_detection" {
  count = var.enable_database_threat_protection ? 1 : 0

  # Note: This is for SQL Server. For PostgreSQL, we use Log Analytics and custom queries
  # PostgreSQL flexible server has built-in security features we'll configure
}

# PostgreSQL Security Configuration with Enhanced Monitoring
resource "azurerm_postgresql_flexible_server_configuration" "log_checkpoints" {
  name      = "log_checkpoints"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_connections" {
  name      = "log_connections"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_disconnections" {
  name      = "log_disconnections"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_lock_waits" {
  name      = "log_lock_waits"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_min_duration_statement" {
  name      = "log_min_duration_statement"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "1000"  # Log queries taking longer than 1 second
}

resource "azurerm_postgresql_flexible_server_configuration" "connection_throttling" {
  name      = "connection_throttling.enable"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

# Container Registry Advanced Security
resource "azurerm_container_registry_task" "security_scan" {
  count = var.enable_container_security_scanning ? 1 : 0

  name                  = "security-scan-task"
  container_registry_id = azurerm_container_registry.main.id

  platform {
    os = "Linux"
  }

  docker_step {
    dockerfile_path      = "Dockerfile"
    context_path        = "https://github.com/your-repo/ai-audio-upscaler.git"
    context_access_token = var.github_token
    image_names         = ["{{.Run.Registry}}/ai-audio-upscaler:{{.Run.ID}}"]
  }

  agent_setting {
    cpu = 2
  }

  base_image_trigger {
    name = "base-image-trigger"
    type = "Runtime"
    status = "Enabled"
  }

  source_trigger {
    name = "source-trigger"
    events = ["commit"]

    source_control {
      repo_url    = "https://github.com/your-repo/ai-audio-upscaler.git"
      branch      = "main"
      source_control_type = "Github"
      token       = var.github_token
    }
  }

  tags = local.common_tags
}

# Network security with DDoS protection
resource "azurerm_network_ddos_protection_plan" "main" {
  count = var.enable_ddos_protection ? 1 : 0

  name                = "${var.project_name}-ddos-protection-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = local.common_tags
}

# Web Application Firewall (WAF) for Application Gateway
resource "azurerm_web_application_firewall_policy" "main" {
  count = var.enable_app_gateway ? 1 : 0

  name                = "${var.project_name}-waf-policy-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  policy_settings {
    enabled                     = true
    mode                       = var.environment == "production" ? "Prevention" : "Detection"
    request_body_check         = true
    file_upload_limit_in_mb    = 100
    max_request_body_size_in_kb = 128
  }

  managed_rules {
    managed_rule_set {
      type    = "OWASP"
      version = "3.2"

      rule_group_override {
        rule_group_name = "REQUEST-920-PROTOCOL-ENFORCEMENT"

        rule {
          id      = "920100"
          enabled = true
          action  = "Block"
        }
      }
    }

    managed_rule_set {
      type    = "Microsoft_BotManagerRuleSet"
      version = "0.1"
    }

    exclusion {
      match_variable          = "RequestHeaderNames"
      selector_match_operator = "Equals"
      selector                = "User-Agent"

      excluded_rule_set {
        type    = "OWASP"
        version = "3.2"

        rule_group {
          rule_group_name = "REQUEST-920-PROTOCOL-ENFORCEMENT"

          excluded_rule {
            rule_id = "920100"
          }
        }
      }
    }
  }

  custom_rules {
    name      = "RateLimitRule"
    priority  = 1
    rule_type = "RateLimitRule"
    action    = "Block"

    match_conditions {
      match_variables {
        variable_name = "RemoteAddr"
      }

      operator           = "IPMatch"
      negation_condition = false
      match_values       = ["0.0.0.0/0"]
    }

    rate_limit_duration    = "OneMin"
    rate_limit_threshold   = 100
  }

  custom_rules {
    name      = "GeoBlockRule"
    priority  = 2
    rule_type = "MatchRule"
    action    = "Block"

    match_conditions {
      match_variables {
        variable_name = "RemoteAddr"
      }

      operator           = "GeoMatch"
      negation_condition = false
      match_values       = var.blocked_countries
    }
  }

  tags = local.common_tags
}

# Azure Sentinel Analytics Rules for Advanced Threat Detection
resource "azurerm_sentinel_alert_rule_scheduled" "suspicious_login_activity" {
  count = var.create_separate_security_workspace ? 1 : 0

  name                       = "SuspiciousLoginActivity"
  log_analytics_workspace_id = azurerm_log_analytics_workspace.security[0].id
  display_name              = "Suspicious Login Activity Detected"
  severity                  = "High"
  enabled                   = true

  query_frequency   = "PT1H"
  query_period      = "PT2H"
  trigger_threshold = 1
  trigger_operator  = "GreaterThan"

  tactics = ["InitialAccess", "CredentialAccess"]

  query = <<QUERY
AuditLogs
| where TimeGenerated > ago(2h)
| where OperationName contains "Sign-in"
| where Result == "failure"
| summarize FailCount = count(), FirstFailTime = min(TimeGenerated), LastFailTime = max(TimeGenerated) by UserPrincipalName, IPAddress, LocationDetails.countryOrRegion
| where FailCount >= 10
| extend GeoInfo = LocationDetails_countryOrRegion
QUERY

  event_grouping {
    aggregation_method = "SingleAlert"
  }

  incident_configuration {
    create_incident = true

    grouping {
      enabled = true
      reopen_closed_incident = false
      lookback_duration = "P7D"
      entity_matching_method = "Selected"
      group_by_entities = ["Account", "IP"]
    }
  }

  alert_details_override {
    description_format   = "Suspicious login activity detected for user {{UserPrincipalName}} from IP {{IPAddress}}"
    display_name_format  = "Suspicious Login - {{UserPrincipalName}}"
    severity_column_name = "FailCount"
    tactics_column_name  = "tactics"
  }
}

resource "azurerm_sentinel_alert_rule_scheduled" "data_exfiltration_detection" {
  count = var.create_separate_security_workspace ? 1 : 0

  name                       = "DataExfiltrationDetection"
  log_analytics_workspace_id = azurerm_log_analytics_workspace.security[0].id
  display_name              = "Potential Data Exfiltration Detected"
  severity                  = "High"
  enabled                   = true

  query_frequency   = "PT30M"
  query_period      = "PT1H"
  trigger_threshold = 1
  trigger_operator  = "GreaterThan"

  tactics = ["Exfiltration"]

  query = <<QUERY
StorageBlobLogs
| where TimeGenerated > ago(1h)
| where OperationName == "GetBlob"
| summarize DownloadCount = count(), DataTransferred = sum(ResponseBodySize) by CallerIpAddress, AccountName
| where DownloadCount > 100 or DataTransferred > 1000000000  // 1GB threshold
| project TimeGenerated, CallerIpAddress, AccountName, DownloadCount, DataTransferred
QUERY

  event_grouping {
    aggregation_method = "SingleAlert"
  }

  incident_configuration {
    create_incident = true

    grouping {
      enabled = true
      reopen_closed_incident = false
      lookback_duration = "P1D"
      entity_matching_method = "Selected"
      group_by_entities = ["IP", "Account"]
    }
  }
}

# Threat Intelligence Integration
resource "azurerm_sentinel_data_connector_threat_intelligence" "threat_intel" {
  count                      = var.create_separate_security_workspace && var.enable_threat_intelligence ? 1 : 0
  name                       = "ThreatIntelligence"
  log_analytics_workspace_id = azurerm_log_analytics_workspace.security[0].id
}

# Microsoft Defender for Containers (AKS)
resource "azurerm_kubernetes_cluster_extension" "defender" {
  count = var.enable_defender_for_containers ? 1 : 0

  name           = "microsoft.azuredefender.kubernetes"
  cluster_id     = module.aks.aks_id
  extension_type = "microsoft.azuredefender.kubernetes"

  configuration_settings = {
    "logAnalyticsWorkspaceResourceID" = azurerm_log_analytics_workspace.main.id
  }

  depends_on = [
    module.aks
  ]
}

# Just-In-Time VM access configuration for any VM components
# Note: This is for future VM-based components if needed
resource "azurerm_security_center_jit_network_access_policy" "jit_policy" {
  count = var.enable_jit_access && length(var.jit_vm_ids) > 0 ? 1 : 0

  kind     = "Basic"
  location = azurerm_resource_group.main.location
  name     = "${var.project_name}-jit-policy-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name

  dynamic "virtual_machine" {
    for_each = var.jit_vm_ids
    content {
      id = virtual_machine.value

      port {
        number                     = 22
        protocol                   = "TCP"
        allowed_source_address_prefix = "*"
        max_request_access_duration = "PT3H"
      }
    }
  }
}

# Secure Score and Security Recommendations
resource "azurerm_security_center_setting" "mcas" {
  setting_name = "MCAS"
  enabled      = var.environment == "production" ? true : false
}

resource "azurerm_security_center_setting" "wdatp" {
  setting_name = "WDATP"
  enabled      = var.environment == "production" ? true : false
}

# Adaptive Network Hardening
resource "azurerm_security_center_workspace" "main" {
  count        = var.create_separate_security_workspace ? 1 : 0
  scope        = "/subscriptions/${data.azurerm_client_config.current.subscription_id}"
  workspace_id = azurerm_log_analytics_workspace.security[0].id
}