# Google Cloud Platform Deployment Guide

This guide covers deploying HF Path Simulator on GCP with GPU support.

## Prerequisites

- gcloud CLI installed and configured
- Project with Compute Engine API enabled
- GPU quota approved for your project

## Option 1: Compute Engine (Single Instance)

### Create GPU Instance

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Create instance with NVIDIA T4
gcloud compute instances create hfpathsim \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata=startup-script='#!/bin/bash
    # Install Docker
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker

    # Install NVIDIA drivers
    curl -fsSL https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py | python3

    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    systemctl restart docker

    # Run HF Path Simulator
    docker run -d --name hfpathsim --gpus all -p 8000:8000 -p 5555:5555 -p 5556:5556 --restart unless-stopped hfpathsim:latest
  '
```

### Firewall Rules

```bash
# Allow HTTP/HTTPS
gcloud compute firewall-rules create hfpathsim-web \
  --allow=tcp:80,tcp:443 \
  --target-tags=hfpathsim

# Allow API
gcloud compute firewall-rules create hfpathsim-api \
  --allow=tcp:8000 \
  --target-tags=hfpathsim

# Add tags to instance
gcloud compute instances add-tags hfpathsim \
  --zone=us-central1-a \
  --tags=hfpathsim
```

### Get External IP

```bash
gcloud compute instances describe hfpathsim \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

## Option 2: Google Kubernetes Engine (GKE)

### Create GPU-enabled Cluster

```bash
# Create cluster with GPU node pool
gcloud container clusters create hfpathsim-cluster \
  --zone=us-central1-a \
  --num-nodes=1 \
  --machine-type=n1-standard-2

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster=hfpathsim-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=1

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Deploy Application

```bash
# Get credentials
gcloud container clusters get-credentials hfpathsim-cluster --zone=us-central1-a

# Deploy
kubectl apply -f deploy/kubernetes/deployment.yaml
```

## GPU Types and Pricing (us-central1)

| GPU Type    | Memory | vCPU | Use Case    | Price/hr |
|-------------|--------|------|-------------|----------|
| NVIDIA T4   | 16GB   | 4    | Development | ~$0.35   |
| NVIDIA L4   | 24GB   | 4    | Balanced    | ~$0.70   |
| NVIDIA A100 | 40GB   | 12   | High-perf   | ~$2.93   |

## Container Registry

### Push to GCR

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/YOUR_PROJECT/hfpathsim:latest .
docker push gcr.io/YOUR_PROJECT/hfpathsim:latest
```

### Use Artifact Registry (Recommended)

```bash
# Create repository
gcloud artifacts repositories create hfpathsim \
  --repository-format=docker \
  --location=us-central1

# Configure and push
gcloud auth configure-docker us-central1-docker.pkg.dev
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT/hfpathsim/api:latest .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/hfpathsim/api:latest
```

## Load Balancing with HTTPS

### Create Managed Certificate

```bash
gcloud compute ssl-certificates create hfpathsim-cert \
  --domains=hfpathsim.example.com
```

### Create Load Balancer

```bash
# Create health check
gcloud compute health-checks create http hfpathsim-health \
  --port=8000 \
  --request-path=/api/v1/health

# Create backend service
gcloud compute backend-services create hfpathsim-backend \
  --protocol=HTTP \
  --port-name=http \
  --health-checks=hfpathsim-health \
  --global

# Create URL map and target proxy
gcloud compute url-maps create hfpathsim-map \
  --default-service=hfpathsim-backend

gcloud compute target-https-proxies create hfpathsim-proxy \
  --url-map=hfpathsim-map \
  --ssl-certificates=hfpathsim-cert

# Create forwarding rule
gcloud compute forwarding-rules create hfpathsim-https \
  --global \
  --target-https-proxy=hfpathsim-proxy \
  --ports=443
```

## Monitoring

### Cloud Monitoring

Enable GPU metrics:

```bash
# Install ops agent on instance
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
bash add-google-cloud-ops-agent-repo.sh --also-install
```

### Create Dashboard

```bash
gcloud monitoring dashboards create \
  --config-from-file=monitoring/gcp-dashboard.json
```

## Cleanup

```bash
# Delete instance
gcloud compute instances delete hfpathsim --zone=us-central1-a

# Delete GKE cluster
gcloud container clusters delete hfpathsim-cluster --zone=us-central1-a

# Delete firewall rules
gcloud compute firewall-rules delete hfpathsim-web hfpathsim-api
```
