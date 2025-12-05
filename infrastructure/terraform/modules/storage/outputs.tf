# Storage Module Outputs

output "storage_account_id" {
  description = "ID of the storage account"
  value       = azurerm_storage_account.main.id
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.main.name
}

output "primary_blob_endpoint" {
  description = "Primary blob endpoint of the storage account"
  value       = azurerm_storage_account.main.primary_blob_endpoint
}

output "primary_connection_string" {
  description = "Primary connection string for the storage account"
  value       = azurerm_storage_account.main.primary_connection_string
  sensitive   = true
}

output "primary_access_key" {
  description = "Primary access key for the storage account"
  value       = azurerm_storage_account.main.primary_access_key
  sensitive   = true
}

output "container_names" {
  description = "Names of created storage containers"
  value = {
    inputs     = azurerm_storage_container.audio_inputs.name
    outputs    = azurerm_storage_container.audio_outputs.name
    models     = azurerm_storage_container.ml_models.name
    temp       = azurerm_storage_container.temp_processing.name
  }
}

output "private_endpoint_id" {
  description = "ID of the storage private endpoint"
  value       = var.create_private_endpoints ? azurerm_private_endpoint.storage_blob[0].id : null
}

output "identity_principal_id" {
  description = "Principal ID of the storage account system assigned identity"
  value       = azurerm_storage_account.main.identity[0].principal_id
}