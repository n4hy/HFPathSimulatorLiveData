# HF Path Simulator - GCP Terraform Configuration
# Deploys a GPU-enabled VM with Docker and HF Path Simulator

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# VPC Network
resource "google_compute_network" "hfpathsim" {
  name                    = "${var.environment}-${var.network_name}"
  auto_create_subnetworks = false
  description             = "VPC network for HF Path Simulator"
}

# Subnet
resource "google_compute_subnetwork" "hfpathsim" {
  name          = "${var.environment}-hfpathsim-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.hfpathsim.id

  private_ip_google_access = true
}

# Firewall - SSH
resource "google_compute_firewall" "ssh" {
  name    = "${var.environment}-hfpathsim-ssh"
  network = google_compute_network.hfpathsim.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.ssh_source_ranges
  target_tags   = ["hfpathsim"]
}

# Firewall - HTTP/HTTPS
resource "google_compute_firewall" "web" {
  name    = "${var.environment}-hfpathsim-web"
  network = google_compute_network.hfpathsim.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["hfpathsim"]
}

# Firewall - API and ZMQ
resource "google_compute_firewall" "api" {
  name    = "${var.environment}-hfpathsim-api"
  network = google_compute_network.hfpathsim.name

  allow {
    protocol = "tcp"
    ports    = ["8000", "5555", "5556"]
  }

  source_ranges = var.api_source_ranges
  target_tags   = ["hfpathsim"]
}

# Static External IP
resource "google_compute_address" "hfpathsim" {
  name         = "${var.environment}-hfpathsim-ip"
  region       = var.region
  address_type = "EXTERNAL"
}

# Startup script for Docker and NVIDIA setup
locals {
  startup_script = <<-EOF
    #!/bin/bash
    set -ex

    # Log startup
    exec > >(tee /var/log/startup-script.log) 2>&1
    echo "Starting HF Path Simulator setup..."

    # Install Docker
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker

    # Install NVIDIA drivers (GCP provides a helper script)
    curl -fsSL https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py -o /tmp/install_gpu_driver.py
    python3 /tmp/install_gpu_driver.py

    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker

    # Pull and run HF Path Simulator
    docker pull ${var.docker_image}
    docker run -d \
      --name hfpathsim \
      --gpus all \
      -p 8000:8000 \
      -p 5555:5555 \
      -p 5556:5556 \
      --restart unless-stopped \
      ${var.docker_image}

    echo "HF Path Simulator setup complete!"
  EOF
}

# GPU Instance
resource "google_compute_instance" "hfpathsim" {
  name         = "${var.environment}-hfpathsim"
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["hfpathsim"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.boot_disk_size
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.hfpathsim.id

    access_config {
      nat_ip = google_compute_address.hfpathsim.address
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  metadata = {
    startup-script = local.startup_script
  }

  service_account {
    scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }

  labels = {
    environment = var.environment
    application = "hfpathsim"
  }

  lifecycle {
    ignore_changes = [
      metadata["ssh-keys"],
    ]
  }
}
