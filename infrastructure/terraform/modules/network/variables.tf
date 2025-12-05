# Network Module Variables

variable "vnet_name" {
  description = "Name of the virtual network"
  type        = string
}

variable "location" {
  description = "Azure region for resources"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vnet_address_space" {
  description = "Address space for the virtual network"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "subnet_cidrs" {
  description = "CIDR blocks for subnets"
  type = object({
    aks               = string
    postgres          = string
    private_endpoints = string
    app_gateway       = optional(string)
  })
}

variable "enable_app_gateway" {
  description = "Enable Application Gateway subnet"
  type        = bool
  default     = false
}

variable "custom_routes" {
  description = "Custom routes for the route table"
  type = list(object({
    name                   = string
    address_prefix         = string
    next_hop_type          = string
    next_hop_in_ip_address = optional(string)
  }))
  default = null
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}