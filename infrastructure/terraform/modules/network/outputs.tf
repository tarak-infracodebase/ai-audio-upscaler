# Network Module Outputs

output "vnet_id" {
  description = "ID of the virtual network"
  value       = azurerm_virtual_network.main.id
}

output "vnet_name" {
  description = "Name of the virtual network"
  value       = azurerm_virtual_network.main.name
}

output "subnet_ids" {
  description = "Map of subnet IDs"
  value = {
    aks               = azurerm_subnet.aks.id
    postgres          = azurerm_subnet.postgres.id
    private_endpoints = azurerm_subnet.private_endpoints.id
    app_gateway       = var.enable_app_gateway ? azurerm_subnet.app_gateway[0].id : null
  }
}

output "private_dns_zone_ids" {
  description = "Map of private DNS zone IDs"
  value = {
    postgres   = azurerm_private_dns_zone.postgres.id
    key_vault  = azurerm_private_dns_zone.key_vault.id
    storage    = azurerm_private_dns_zone.storage.id
    acr        = azurerm_private_dns_zone.acr.id
  }
}

output "nsg_ids" {
  description = "Map of Network Security Group IDs"
  value = {
    aks      = azurerm_network_security_group.aks.id
    postgres = azurerm_network_security_group.postgres.id
  }
}