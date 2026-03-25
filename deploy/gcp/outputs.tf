# HF Path Simulator - GCP Terraform Outputs

output "instance_name" {
  description = "Name of the GCE instance"
  value       = google_compute_instance.hfpathsim.name
}

output "instance_zone" {
  description = "Zone where the instance is deployed"
  value       = google_compute_instance.hfpathsim.zone
}

output "public_ip" {
  description = "Public IP address of the instance"
  value       = google_compute_address.hfpathsim.address
}

output "api_endpoint" {
  description = "HF Path Simulator API endpoint"
  value       = "http://${google_compute_address.hfpathsim.address}:8000"
}

output "api_docs" {
  description = "API documentation URL"
  value       = "http://${google_compute_address.hfpathsim.address}:8000/docs"
}

output "zmq_input" {
  description = "ZMQ input endpoint for signal streaming"
  value       = "tcp://${google_compute_address.hfpathsim.address}:5555"
}

output "zmq_output" {
  description = "ZMQ output endpoint for processed signals"
  value       = "tcp://${google_compute_address.hfpathsim.address}:5556"
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "gcloud compute ssh ${google_compute_instance.hfpathsim.name} --zone=${google_compute_instance.hfpathsim.zone} --project=${var.project_id}"
}

output "network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.hfpathsim.name
}

output "gpu_info" {
  description = "GPU configuration"
  value       = "${var.gpu_count}x ${var.gpu_type}"
}
