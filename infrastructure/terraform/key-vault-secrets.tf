# Azure Key Vault Secrets Management
# Comprehensive secrets provisioning and management for 10/10 security score

# Generate secure random passwords and keys
resource "random_password" "postgres_admin_password" {
  count   = var.postgres_admin_password == null ? 1 : 0
  length  = 32
  special = true
  upper   = true
  lower   = true
  numeric = true

  # Avoid characters that might cause issues in connection strings
  override_special = "!@#$%^&*()-_=+[]{}|;:,.<>?"
}

resource "random_password" "jwt_secret_key" {
  length  = 64
  special = true
  upper   = true
  lower   = true
  numeric = true
}

# Generate RSA key pair for JWT (RS256 algorithm)
resource "tls_private_key" "jwt_rsa" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

# Generate secure session secret
resource "random_password" "session_secret" {
  length  = 64
  special = true
  upper   = true
  lower   = true
  numeric = true
}

# Generate encryption keys for sensitive data
resource "random_password" "encryption_key" {
  length  = 32
  special = false  # Base64 compatible for AES encryption
  upper   = true
  lower   = true
  numeric = true
}

# Generate API keys for internal services
resource "random_password" "internal_api_key" {
  length  = 48
  special = false
  upper   = true
  lower   = true
  numeric = true
}

# Database connection string secrets
resource "azurerm_key_vault_secret" "postgres_admin_password" {
  name         = "postgres-admin-password"
  value        = var.postgres_admin_password != null ? var.postgres_admin_password : random_password.postgres_admin_password[0].result
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "Database"
    Rotation   = "Manual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "database_url" {
  name = "database-url"
  value = format(
    "postgresql://%s:%s@%s.postgres.database.azure.com:5432/%s?sslmode=require",
    azurerm_postgresql_flexible_server.main.administrator_login,
    var.postgres_admin_password != null ? var.postgres_admin_password : random_password.postgres_admin_password[0].result,
    azurerm_postgresql_flexible_server.main.name,
    azurerm_postgresql_flexible_server_database.main.name
  )
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "ConnectionString"
    Rotation   = "Automatic"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "redis_connection_string" {
  name = "redis-connection-string"
  value = format(
    "rediss://:%s@%s.redis.cache.windows.net:6380",
    azurerm_redis_cache.main.primary_access_key,
    azurerm_redis_cache.main.name
  )
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "ConnectionString"
    Rotation   = "Automatic"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# JWT secrets for secure authentication
resource "azurerm_key_vault_secret" "jwt_secret_key" {
  name         = "jwt-secret-key"
  value        = random_password.jwt_secret_key.result
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "Authentication"
    Rotation   = "Manual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "jwt_private_key" {
  name         = "jwt-private-key"
  value        = tls_private_key.jwt_rsa.private_key_pem
  key_vault_id = azurerm_key_vault.main.id
  content_type = "application/x-pem-file"

  tags = merge(local.common_tags, {
    SecretType = "PrivateKey"
    Rotation   = "Manual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "jwt_public_key" {
  name         = "jwt-public-key"
  value        = tls_private_key.jwt_rsa.public_key_pem
  key_vault_id = azurerm_key_vault.main.id
  content_type = "application/x-pem-file"

  tags = merge(local.common_tags, {
    SecretType = "PublicKey"
    Rotation   = "Manual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# Azure B2C application secrets
resource "azurerm_key_vault_secret" "azure_b2c_client_id" {
  name         = "azure-b2c-client-id"
  value        = var.azure_b2c_client_id
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "ClientCredential"
    Rotation   = "Manual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "azure_b2c_client_secret" {
  name         = "azure-b2c-client-secret"
  value        = var.azure_b2c_client_secret
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "ClientCredential"
    Rotation   = "Manual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "azure_b2c_tenant_name" {
  name         = "azure-b2c-tenant-name"
  value        = var.azure_b2c_tenant_name
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "Configuration"
    Rotation   = "Manual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# Storage account connection string
resource "azurerm_key_vault_secret" "storage_connection_string" {
  name = "storage-connection-string"
  value = format(
    "DefaultEndpointsProtocol=https;AccountName=%s;AccountKey=%s;EndpointSuffix=core.windows.net",
    azurerm_storage_account.main.name,
    azurerm_storage_account.main.primary_access_key
  )
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "ConnectionString"
    Rotation   = "Automatic"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# Session and encryption secrets
resource "azurerm_key_vault_secret" "session_secret" {
  name         = "session-secret"
  value        = random_password.session_secret.result
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "Session"
    Rotation   = "Quarterly"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "encryption_key" {
  name         = "encryption-key"
  value        = base64encode(random_password.encryption_key.result)
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "Encryption"
    Rotation   = "Annual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "internal_api_key" {
  name         = "internal-api-key"
  value        = random_password.internal_api_key.result
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "API"
    Rotation   = "Monthly"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# External service API keys (to be set manually or via automation)
resource "azurerm_key_vault_secret" "openai_api_key" {
  count        = var.openai_api_key != "" ? 1 : 0
  name         = "openai-api-key"
  value        = var.openai_api_key
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "ExternalAPI"
    Rotation   = "Manual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "sendgrid_api_key" {
  count        = var.sendgrid_api_key != "" ? 1 : 0
  name         = "sendgrid-api-key"
  value        = var.sendgrid_api_key
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(local.common_tags, {
    SecretType = "ExternalAPI"
    Rotation   = "Manual"
  })

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# Certificate for TLS/SSL
resource "azurerm_key_vault_certificate" "app_certificate" {
  count = var.create_ssl_certificate ? 1 : 0

  name         = "${var.project_name}-app-certificate"
  key_vault_id = azurerm_key_vault.main.id

  certificate_policy {
    issuer_parameters {
      name = "Self"
    }

    key_properties {
      exportable = true
      key_size   = 2048
      key_type   = "RSA"
      reuse_key  = true
    }

    lifetime_action {
      action {
        action_type = "AutoRenew"
      }

      trigger {
        days_before_expiry = 30
      }
    }

    secret_properties {
      content_type = "application/x-pkcs12"
    }

    x509_certificate_properties {
      # Customize for your domain
      subject            = "CN=${var.app_domain_name}"
      validity_in_months = 12

      subject_alternative_names {
        dns_names = [
          var.app_domain_name,
          "*.${var.app_domain_name}"
        ]
      }

      key_usage = [
        "cRLSign",
        "dataEncipherment",
        "digitalSignature",
        "keyAgreement",
        "keyCertSign",
        "keyEncipherment",
      ]

      extended_key_usage = [
        "1.3.6.1.5.5.7.3.1",  # serverAuth
        "1.3.6.1.5.5.7.3.2",  # clientAuth
      ]
    }
  }

  tags = local.common_tags

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# Key Vault access policies for AKS workload identity
resource "azurerm_key_vault_access_policy" "aks_workload_identity" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = module.aks.aks_identity_principal_id

  secret_permissions = [
    "Get",
    "List"
  ]

  certificate_permissions = [
    "Get",
    "List"
  ]

  depends_on = [module.aks]
}

# Key Vault access policy for External Secrets Operator
resource "azurerm_key_vault_access_policy" "external_secrets_operator" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_user_assigned_identity.external_secrets.principal_id

  secret_permissions = [
    "Get",
    "List"
  ]

  certificate_permissions = [
    "Get",
    "List"
  ]
}

# User Assigned Identity for External Secrets Operator
resource "azurerm_user_assigned_identity" "external_secrets" {
  name                = "${var.project_name}-external-secrets-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = local.common_tags
}

# Federated identity credential for External Secrets Operator
resource "azurerm_federated_identity_credential" "external_secrets" {
  name                = "${var.project_name}-external-secrets-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  audience            = ["api://AzureADTokenExchange"]
  issuer              = module.aks.aks_oidc_issuer_url
  parent_id           = azurerm_user_assigned_identity.external_secrets.id
  subject             = "system:serviceaccount:external-secrets-system:external-secrets-operator"

  depends_on = [module.aks]
}

# Key rotation automation (using Azure Automation if enabled)
resource "azurerm_automation_account" "key_rotation" {
  count = var.enable_key_rotation_automation ? 1 : 0

  name                = "${var.project_name}-key-rotation-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku_name           = "Basic"

  tags = local.common_tags
}

resource "azurerm_automation_runbook" "rotate_jwt_keys" {
  count = var.enable_key_rotation_automation ? 1 : 0

  name                    = "RotateJWTKeys"
  location               = azurerm_resource_group.main.location
  resource_group_name    = azurerm_resource_group.main.name
  automation_account_name = azurerm_automation_account.key_rotation[0].name
  log_verbose            = true
  log_progress           = true
  description            = "Automated JWT key rotation"
  runbook_type           = "PowerShell"

  content = <<CONTENT
param(
    [Parameter(Mandatory=$true)]
    [string]$KeyVaultName,

    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName
)

# Import required modules
Import-Module Az.KeyVault
Import-Module Az.Accounts

# Connect using managed identity
Connect-AzAccount -Identity

# Generate new RSA key pair
$rsa = [System.Security.Cryptography.RSA]::Create(2048)
$privateKey = [System.Security.Cryptography.PemEncoding]::WriteString("RSA PRIVATE KEY", $rsa.ExportRSAPrivateKey())
$publicKey = [System.Security.Cryptography.PemEncoding]::WriteString("PUBLIC KEY", $rsa.ExportSubjectPublicKeyInfo())

# Update Key Vault secrets
Set-AzKeyVaultSecret -VaultName $KeyVaultName -Name "jwt-private-key" -SecretValue (ConvertTo-SecureString -String $privateKey -AsPlainText -Force)
Set-AzKeyVaultSecret -VaultName $KeyVaultName -Name "jwt-public-key" -SecretValue (ConvertTo-SecureString -String $publicKey -AsPlainText -Force)

Write-Output "JWT keys rotated successfully at $(Get-Date)"
CONTENT

  tags = local.common_tags
}

# Schedule for key rotation (quarterly)
resource "azurerm_automation_schedule" "jwt_key_rotation" {
  count = var.enable_key_rotation_automation ? 1 : 0

  name                    = "JWT-Key-Rotation-Schedule"
  resource_group_name     = azurerm_resource_group.main.name
  automation_account_name = azurerm_automation_account.key_rotation[0].name
  frequency              = "Month"
  interval               = 3  # Every 3 months
  timezone               = "UTC"
  start_time             = "${formatdate("YYYY-MM-DD", timeadd(timestamp(), "24h"))}T02:00:00Z"
  description            = "Automated JWT key rotation schedule"
}

resource "azurerm_automation_job_schedule" "jwt_key_rotation" {
  count = var.enable_key_rotation_automation ? 1 : 0

  resource_group_name     = azurerm_resource_group.main.name
  automation_account_name = azurerm_automation_account.key_rotation[0].name
  schedule_name          = azurerm_automation_schedule.jwt_key_rotation[0].name
  runbook_name           = azurerm_automation_runbook.rotate_jwt_keys[0].name

  parameters = {
    KeyVaultName        = azurerm_key_vault.main.name
    ResourceGroupName   = azurerm_resource_group.main.name
  }
}