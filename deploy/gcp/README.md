# HF Path Simulator - GCP Terraform Deployment

Deploy HF Path Simulator on Google Cloud Platform with GPU support using Terraform.

## Prerequisites

1. **Google Cloud SDK** - Install and authenticate:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

2. **Terraform** - Version 1.0.0 or later:
   ```bash
   terraform --version
   ```

3. **GPU Quota** - Ensure your project has GPU quota in your target zone:
   ```bash
   gcloud compute regions describe us-central1 --format="json(quotas)"
   ```

4. **APIs Enabled**:
   ```bash
   gcloud services enable compute.googleapis.com
   ```

## Quick Start

1. **Initialize Terraform**:
   ```bash
   cd deploy/gcp
   terraform init
   ```

2. **Create a variables file** (`terraform.tfvars`):
   ```hcl
   project_id   = "your-gcp-project-id"
   region       = "us-central1"
   zone         = "us-central1-a"
   environment  = "dev"
   docker_image = "your-registry/hfpathsim:latest"
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
| `project_id` | GCP project ID | (required) |
| `region` | GCP region | `us-central1` |
| `zone` | GCP zone | `us-central1-a` |
| `environment` | Environment name | `dev` |
| `machine_type` | VM machine type | `n1-standard-4` |
| `gpu_type` | GPU type | `nvidia-tesla-t4` |
| `gpu_count` | Number of GPUs | `1` |
| `boot_disk_size` | Boot disk size (GB) | `100` |
| `docker_image` | Docker image URI | `hfpathsim:latest` |

## GPU Options

| GPU Type | Memory | Use Case | Hourly Cost* |
|----------|--------|----------|--------------|
| `nvidia-tesla-t4` | 16GB | Development | ~$0.35 |
| `nvidia-l4` | 24GB | Balanced | ~$0.70 |
| `nvidia-tesla-a100` | 40GB | Production | ~$2.93 |

*Prices vary by region and are subject to change.

## Security Considerations

By default, the deployment allows access from any IP (`0.0.0.0/0`). For production:

```hcl
# Restrict SSH access
ssh_source_ranges = ["YOUR_OFFICE_IP/32"]

# Restrict API access
api_source_ranges = ["YOUR_CLIENT_CIDR/24"]
```

## Using Container Registry

Push your image to Google Container Registry:

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/YOUR_PROJECT/hfpathsim:latest .
docker push gcr.io/YOUR_PROJECT/hfpathsim:latest

# Use in terraform.tfvars
docker_image = "gcr.io/YOUR_PROJECT/hfpathsim:latest"
```

Or use Artifact Registry (recommended):

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

## Accessing the Instance

After deployment:

```bash
# SSH into the instance
$(terraform output -raw ssh_command)

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

## Troubleshooting

### GPU drivers not installed
Check the startup script log:
```bash
gcloud compute ssh INSTANCE_NAME --command "cat /var/log/startup-script.log"
```

### Docker container not running
```bash
gcloud compute ssh INSTANCE_NAME --command "sudo docker ps -a"
gcloud compute ssh INSTANCE_NAME --command "sudo docker logs hfpathsim"
```

### Quota exceeded
Request additional GPU quota:
```bash
gcloud compute regions describe REGION --format="json(quotas)" | grep -A5 GPU
```
