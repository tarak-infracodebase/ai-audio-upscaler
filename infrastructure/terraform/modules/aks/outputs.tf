# AKS Module Outputs

output "cluster_id" {
  description = "ID of the AKS cluster"
  value       = module.aks.id
}

output "cluster_name" {
  description = "Name of the AKS cluster"
  value       = module.aks.name
}

output "cluster_fqdn" {
  description = "FQDN of the AKS cluster"
  value       = module.aks.public_fqdn
}

output "private_fqdn" {
  description = "Private FQDN of the AKS cluster"
  value       = module.aks.private_fqdn
}

output "kube_config_raw" {
  description = "Raw kubeconfig for the AKS cluster"
  value       = module.aks.kube_config_raw
  sensitive   = true
}

output "identity_principal_id" {
  description = "Principal ID of the AKS managed identity"
  value       = azurerm_user_assigned_identity.aks.principal_id
}

output "identity_client_id" {
  description = "Client ID of the AKS managed identity"
  value       = azurerm_user_assigned_identity.aks.client_id
}

output "node_resource_group_name" {
  description = "Resource group containing AKS nodes"
  value       = module.aks.nodes_resource_group_name
}

output "oidc_issuer_url" {
  description = "OIDC issuer URL for workload identity"
  value       = module.aks.oidc_issuer_url
}

output "kubelet_identity" {
  description = "Kubelet managed identity"
  value       = module.aks.kubelet_user_managed_identity
}

output "log_analytics_workspace_id" {
  description = "ID of the Log Analytics workspace"
  value       = azurerm_log_analytics_workspace.aks.id
}

output "key_vault_secrets_provider_identity" {
  description = "Key Vault secrets provider identity"
  value       = module.aks.key_vault_secrets_provider_identity
}