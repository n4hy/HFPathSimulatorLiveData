# HF Path Simulator - GCP Terraform Variables

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
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

variable "machine_type" {
  description = "GCE machine type"
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "nvidia-tesla-t4"
  validation {
    condition     = contains(["nvidia-tesla-t4", "nvidia-l4", "nvidia-tesla-a100"], var.gpu_type)
    error_message = "GPU type must be nvidia-tesla-t4, nvidia-l4, or nvidia-tesla-a100."
  }
}

variable "gpu_count" {
  description = "Number of GPUs to attach"
  type        = number
  default     = 1
}

variable "boot_disk_size" {
  description = "Boot disk size in GB"
  type        = number
  default     = 100
}

variable "docker_image" {
  description = "Docker image URI for HF Path Simulator"
  type        = string
  default     = "hfpathsim:latest"
}

variable "network_name" {
  description = "VPC network name"
  type        = string
  default     = "hfpathsim-network"
}

variable "ssh_source_ranges" {
  description = "CIDR ranges allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "api_source_ranges" {
  description = "CIDR ranges allowed for API access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
