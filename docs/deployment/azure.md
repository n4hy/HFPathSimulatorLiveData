# Azure Deployment Guide

This guide covers deploying HF Path Simulator on Azure with GPU support.

## Prerequisites

- Azure CLI installed and configured
- Subscription with GPU quota

## Option 1: Azure Virtual Machine with GPU

### Create Resource Group

```bash
az group create \
  --name hfpathsim-rg \
  --location eastus
```

### Create GPU VM

```bash
# Create VM with NVIDIA T4
az vm create \
  --resource-group hfpathsim-rg \
  --name hfpathsim-vm \
  --size Standard_NC4as_T4_v3 \
  --image Ubuntu2204 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --custom-data cloud-init.yaml
```

### cloud-init.yaml

```yaml
#cloud-config
package_update: true
packages:
  - docker.io
  - nvidia-driver-535
  - nvidia-container-toolkit

runcmd:
  - systemctl enable docker
  - systemctl start docker
  - nvidia-ctk runtime configure --runtime=docker
  - systemctl restart docker
  - docker run -d --name hfpathsim --gpus all -p 8000:8000 -p 5555:5555 -p 5556:5556 --restart unless-stopped hfpathsim:latest
```

### Open Ports

```bash
# Open API port
az vm open-port \
  --resource-group hfpathsim-rg \
  --name hfpathsim-vm \
  --port 8000 \
  --priority 1000

# Open web port
az vm open-port \
  --resource-group hfpathsim-rg \
  --name hfpathsim-vm \
  --port 80 \
  --priority 1001
```

### Get Public IP

```bash
az vm show \
  --resource-group hfpathsim-rg \
  --name hfpathsim-vm \
  --show-details \
  --query publicIps \
  --output tsv
```

## Option 2: Azure Kubernetes Service (AKS)

### Create AKS Cluster

```bash
# Create cluster
az aks create \
  --resource-group hfpathsim-rg \
  --name hfpathsim-aks \
  --node-count 1 \
  --node-vm-size Standard_DS2_v2 \
  --generate-ssh-keys

# Add GPU node pool
az aks nodepool add \
  --resource-group hfpathsim-rg \
  --cluster-name hfpathsim-aks \
  --name gpupool \
  --node-count 1 \
  --node-vm-size Standard_NC4as_T4_v3 \
  --node-taints sku=gpu:NoSchedule
```

### Install NVIDIA Device Plugin

```bash
# Get credentials
az aks get-credentials \
  --resource-group hfpathsim-rg \
  --name hfpathsim-aks

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

### Deploy Application

```bash
kubectl apply -f deploy/kubernetes/deployment.yaml
```

## GPU VM Sizes

| Size                    | GPU           | vCPU | RAM   | Use Case    |
|------------------------|---------------|------|-------|-------------|
| Standard_NC4as_T4_v3   | 1x T4 (16GB)  | 4    | 28GB  | Development |
| Standard_NC8as_T4_v3   | 1x T4 (16GB)  | 8    | 56GB  | Production  |
| Standard_NC24ads_A100_v4| 1x A100 (80GB)| 24   | 220GB | High-perf   |

## Container Registry

### Create ACR

```bash
az acr create \
  --resource-group hfpathsim-rg \
  --name hfpathsimacr \
  --sku Basic
```

### Push Images

```bash
# Login to ACR
az acr login --name hfpathsimacr

# Build and push
az acr build \
  --registry hfpathsimacr \
  --image hfpathsim:latest \
  .
```

### Attach ACR to AKS

```bash
az aks update \
  --resource-group hfpathsim-rg \
  --name hfpathsim-aks \
  --attach-acr hfpathsimacr
```

## Load Balancing with HTTPS

### Create Application Gateway

```bash
# Create public IP
az network public-ip create \
  --resource-group hfpathsim-rg \
  --name hfpathsim-pip \
  --sku Standard

# Create Application Gateway
az network application-gateway create \
  --resource-group hfpathsim-rg \
  --name hfpathsim-appgw \
  --sku Standard_v2 \
  --public-ip-address hfpathsim-pip \
  --capacity 1
```

### Enable HTTPS

```bash
# Add SSL certificate
az network application-gateway ssl-cert create \
  --resource-group hfpathsim-rg \
  --gateway-name hfpathsim-appgw \
  --name hfpathsim-cert \
  --cert-file cert.pfx \
  --cert-password $PASSWORD

# Add HTTPS listener
az network application-gateway http-listener create \
  --resource-group hfpathsim-rg \
  --gateway-name hfpathsim-appgw \
  --name https-listener \
  --frontend-port 443 \
  --ssl-cert hfpathsim-cert
```

## Monitoring

### Enable Azure Monitor

```bash
# Enable monitoring for AKS
az aks enable-addons \
  --resource-group hfpathsim-rg \
  --name hfpathsim-aks \
  --addons monitoring
```

### Create Alerts

```bash
az monitor metrics alert create \
  --name hfpathsim-cpu-alert \
  --resource-group hfpathsim-rg \
  --scopes /subscriptions/{sub}/resourceGroups/hfpathsim-rg/providers/Microsoft.Compute/virtualMachines/hfpathsim-vm \
  --condition "avg Percentage CPU > 80" \
  --window-size 5m \
  --evaluation-frequency 1m
```

## Cleanup

```bash
# Delete resource group (deletes everything)
az group delete --name hfpathsim-rg --yes --no-wait
```
