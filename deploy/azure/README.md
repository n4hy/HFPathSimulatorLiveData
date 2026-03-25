# HF Path Simulator - Azure Terraform Deployment

Deploy HF Path Simulator on Microsoft Azure with GPU support using Terraform.

## Prerequisites

1. **Azure CLI** - Install and authenticate:
   ```bash
   az login
   ```

2. **Terraform** - Version 1.0.0 or later:
   ```bash
   terraform --version
   ```

3. **GPU Quota** - Ensure your subscription has GPU VM quota:
   ```bash
   az vm list-usage --location eastus --query "[?contains(name.value, 'NC')]"
   ```

4. **SSH Key** - Generate if you don't have one:
   ```bash
   ssh-keygen -t rsa -b 4096
   ```

## Quick Start

1. **Initialize Terraform**:
   ```bash
   cd deploy/azure
   terraform init
   ```

2. **Create a variables file** (`terraform.tfvars`):
   ```hcl
   location            = "eastus"
   environment         = "dev"
   resource_group_name = "hfpathsim-rg"
   vm_size             = "Standard_NC4as_T4_v3"
   docker_image        = "your-registry/hfpathsim:latest"
   ssh_public_key_path = "~/.ssh/id_rsa.pub"
   ```

3. **Review the plan**:
   ```bash
   terraform plan
   ```

4. **Deploy**:
   ```bash
   terraform apply
   ```

5. **Get connection info**:
   ```bash
   terraform output
   ```

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `location` | Azure region | `eastus` |
| `environment` | Environment name | `dev` |
| `resource_group_name` | Resource group name | `hfpathsim-rg` |
| `vm_size` | VM size with GPU | `Standard_NC4as_T4_v3` |
| `admin_username` | VM admin username | `azureuser` |
| `ssh_public_key_path` | Path to SSH public key | `~/.ssh/id_rsa.pub` |
| `os_disk_size_gb` | OS disk size (GB) | `100` |
| `docker_image` | Docker image URI | `hfpathsim:latest` |

## GPU VM Sizes

| Size | GPU | vCPU | RAM | Use Case | Cost/hr* |
|------|-----|------|-----|----------|----------|
| `Standard_NC4as_T4_v3` | 1x T4 (16GB) | 4 | 28GB | Development | ~$0.53 |
| `Standard_NC8as_T4_v3` | 1x T4 (16GB) | 8 | 56GB | Production | ~$0.75 |
| `Standard_NC16as_T4_v3` | 1x T4 (16GB) | 16 | 112GB | High workload | ~$1.20 |
| `Standard_NC24ads_A100_v4` | 1x A100 (80GB) | 24 | 220GB | High-perf | ~$3.67 |

*Prices vary by region and are subject to change.

## Security Considerations

By default, the deployment allows access from any IP. For production:

```hcl
# Restrict SSH access
allowed_ssh_ips = ["YOUR_OFFICE_IP/32"]

# Restrict API access
allowed_api_ips = ["YOUR_CLIENT_CIDR/24"]
```

## Using Azure Container Registry

Push your image to ACR:

```bash
# Create ACR
az acr create \
  --resource-group hfpathsim-rg \
  --name hfpathsimacr \
  --sku Basic

# Login to ACR
az acr login --name hfpathsimacr

# Build and push
az acr build \
  --registry hfpathsimacr \
  --image hfpathsim:latest \
  .

# Use in terraform.tfvars
docker_image = "hfpathsimacr.azurecr.io/hfpathsim:latest"
```

## Accessing the VM

After deployment:

```bash
# SSH into the VM
$(terraform output -raw ssh_command)

# Check setup progress
tail -f /var/log/hfpathsim-setup.log

# Check Docker logs
sudo docker logs hfpathsim

# Check GPU status
nvidia-smi
```

## API Usage

```bash
# Get API endpoint
API_URL=$(terraform output -raw api_endpoint)

# Health check
curl $API_URL/api/v1/health

# View API docs
echo "Open in browser: $(terraform output -raw api_docs)"
```

## Cleanup

Remove all resources:

```bash
terraform destroy
```

Or delete the entire resource group via Azure CLI:

```bash
az group delete --name $(terraform output -raw resource_group_name) --yes --no-wait
```

## Troubleshooting

### GPU drivers not installed
Check the cloud-init log:
```bash
ssh azureuser@<IP> "cat /var/log/hfpathsim-setup.log"
```

### Docker container not running
```bash
ssh azureuser@<IP> "sudo docker ps -a"
ssh azureuser@<IP> "sudo docker logs hfpathsim"
```

### Quota exceeded
Request additional GPU quota through the Azure portal or CLI:
```bash
az vm list-usage --location eastus --query "[?contains(name.value, 'NC')]"
```

### Cloud-init still running
Check cloud-init status:
```bash
ssh azureuser@<IP> "cloud-init status"
```
