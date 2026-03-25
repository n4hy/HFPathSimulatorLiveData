# HF Path Simulator - Azure Terraform Outputs

output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.hfpathsim.name
}

output "vm_name" {
  description = "Name of the virtual machine"
  value       = azurerm_linux_virtual_machine.hfpathsim.name
}

output "public_ip" {
  description = "Public IP address of the VM"
  value       = azurerm_public_ip.hfpathsim.ip_address
}

output "api_endpoint" {
  description = "HF Path Simulator API endpoint"
  value       = "http://${azurerm_public_ip.hfpathsim.ip_address}:8000"
}

output "api_docs" {
  description = "API documentation URL"
  value       = "http://${azurerm_public_ip.hfpathsim.ip_address}:8000/docs"
}

output "zmq_input" {
  description = "ZMQ input endpoint for signal streaming"
  value       = "tcp://${azurerm_public_ip.hfpathsim.ip_address}:5555"
}

output "zmq_output" {
  description = "ZMQ output endpoint for processed signals"
  value       = "tcp://${azurerm_public_ip.hfpathsim.ip_address}:5556"
}

output "ssh_command" {
  description = "SSH command to connect to the VM"
  value       = "ssh ${var.admin_username}@${azurerm_public_ip.hfpathsim.ip_address}"
}

output "vm_size" {
  description = "VM size (GPU configuration)"
  value       = azurerm_linux_virtual_machine.hfpathsim.size
}

output "location" {
  description = "Azure region where resources are deployed"
  value       = azurerm_resource_group.hfpathsim.location
}
