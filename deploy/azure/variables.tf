# HF Path Simulator - Azure Terraform Variables

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "hfpathsim-rg"
}

variable "vm_size" {
  description = "Azure VM size with GPU"
  type        = string
  default     = "Standard_NC4as_T4_v3"
  validation {
    condition = contains([
      "Standard_NC4as_T4_v3",
      "Standard_NC8as_T4_v3",
      "Standard_NC16as_T4_v3",
      "Standard_NC24ads_A100_v4"
    ], var.vm_size)
    error_message = "VM size must be a supported GPU instance type."
  }
}

variable "admin_username" {
  description = "Admin username for the VM"
  type        = string
  default     = "azureuser"
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key file"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "os_disk_size_gb" {
  description = "OS disk size in GB"
  type        = number
  default     = 100
}

variable "docker_image" {
  description = "Docker image URI for HF Path Simulator"
  type        = string
  default     = "hfpathsim:latest"
}

variable "vnet_address_space" {
  description = "Virtual network address space"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "subnet_address_prefixes" {
  description = "Subnet address prefixes"
  type        = list(string)
  default     = ["10.0.1.0/24"]
}

variable "allowed_ssh_ips" {
  description = "IP addresses allowed for SSH access"
  type        = list(string)
  default     = ["*"]
}

variable "allowed_api_ips" {
  description = "IP addresses allowed for API access"
  type        = list(string)
  default     = ["*"]
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
