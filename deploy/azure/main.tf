# HF Path Simulator - Azure Terraform Configuration
# Deploys a GPU-enabled VM with Docker and HF Path Simulator

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "hfpathsim" {
  name     = "${var.environment}-${var.resource_group_name}"
  location = var.location

  tags = merge(var.tags, {
    environment = var.environment
    application = "hfpathsim"
  })
}

# Virtual Network
resource "azurerm_virtual_network" "hfpathsim" {
  name                = "${var.environment}-hfpathsim-vnet"
  address_space       = var.vnet_address_space
  location            = azurerm_resource_group.hfpathsim.location
  resource_group_name = azurerm_resource_group.hfpathsim.name

  tags = azurerm_resource_group.hfpathsim.tags
}

# Subnet
resource "azurerm_subnet" "hfpathsim" {
  name                 = "${var.environment}-hfpathsim-subnet"
  resource_group_name  = azurerm_resource_group.hfpathsim.name
  virtual_network_name = azurerm_virtual_network.hfpathsim.name
  address_prefixes     = var.subnet_address_prefixes
}

# Network Security Group
resource "azurerm_network_security_group" "hfpathsim" {
  name                = "${var.environment}-hfpathsim-nsg"
  location            = azurerm_resource_group.hfpathsim.location
  resource_group_name = azurerm_resource_group.hfpathsim.name

  # SSH
  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefixes    = var.allowed_ssh_ips
    destination_address_prefix = "*"
  }

  # HTTP
  security_rule {
    name                       = "HTTP"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  # HTTPS
  security_rule {
    name                       = "HTTPS"
    priority                   = 1003
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  # API
  security_rule {
    name                       = "API"
    priority                   = 1004
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefixes    = var.allowed_api_ips
    destination_address_prefix = "*"
  }

  # ZMQ Input
  security_rule {
    name                       = "ZMQ-Input"
    priority                   = 1005
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "5555"
    source_address_prefixes    = var.allowed_api_ips
    destination_address_prefix = "*"
  }

  # ZMQ Output
  security_rule {
    name                       = "ZMQ-Output"
    priority                   = 1006
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "5556"
    source_address_prefixes    = var.allowed_api_ips
    destination_address_prefix = "*"
  }

  tags = azurerm_resource_group.hfpathsim.tags
}

# Associate NSG with subnet
resource "azurerm_subnet_network_security_group_association" "hfpathsim" {
  subnet_id                 = azurerm_subnet.hfpathsim.id
  network_security_group_id = azurerm_network_security_group.hfpathsim.id
}

# Public IP
resource "azurerm_public_ip" "hfpathsim" {
  name                = "${var.environment}-hfpathsim-pip"
  location            = azurerm_resource_group.hfpathsim.location
  resource_group_name = azurerm_resource_group.hfpathsim.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = azurerm_resource_group.hfpathsim.tags
}

# Network Interface
resource "azurerm_network_interface" "hfpathsim" {
  name                = "${var.environment}-hfpathsim-nic"
  location            = azurerm_resource_group.hfpathsim.location
  resource_group_name = azurerm_resource_group.hfpathsim.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.hfpathsim.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.hfpathsim.id
  }

  tags = azurerm_resource_group.hfpathsim.tags
}

# Cloud-init configuration
locals {
  cloud_init = templatefile("${path.module}/cloud-init.yaml", {
    docker_image = var.docker_image
  })
}

# GPU Virtual Machine
resource "azurerm_linux_virtual_machine" "hfpathsim" {
  name                = "${var.environment}-hfpathsim-vm"
  resource_group_name = azurerm_resource_group.hfpathsim.name
  location            = azurerm_resource_group.hfpathsim.location
  size                = var.vm_size
  admin_username      = var.admin_username

  network_interface_ids = [
    azurerm_network_interface.hfpathsim.id,
  ]

  admin_ssh_key {
    username   = var.admin_username
    public_key = file(var.ssh_public_key_path)
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = var.os_disk_size_gb
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  custom_data = base64encode(local.cloud_init)

  tags = azurerm_resource_group.hfpathsim.tags
}
