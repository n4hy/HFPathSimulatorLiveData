# HF Path Simulator Helm Chart

Deploy HF Path Simulator on Kubernetes with GPU support using Helm.

## Prerequisites

1. **Kubernetes cluster** with GPU nodes
2. **NVIDIA GPU Operator** or device plugin installed:
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
   ```
3. **Helm** v3.0+:
   ```bash
   helm version
   ```

## Quick Start

1. **Install the chart**:
   ```bash
   helm install hfpathsim ./deploy/helm/hfpathsim \
     --namespace hfpathsim \
     --create-namespace
   ```

2. **Check deployment status**:
   ```bash
   kubectl get pods -n hfpathsim
   ```

3. **Access the API** (port-forward):
   ```bash
   kubectl port-forward svc/hfpathsim-api 8000:8000 -n hfpathsim
   curl http://localhost:8000/api/v1/health
   ```

## Configuration

Create a `values-custom.yaml` file:

```yaml
api:
  image:
    repository: your-registry/hfpathsim
    tag: v1.0.0
  resources:
    limits:
      nvidia.com/gpu: "1"

web:
  enabled: true
  image:
    repository: your-registry/hfpathsim-web
    tag: v1.0.0

ingress:
  enabled: true
  host: hfpathsim.yourdomain.com
  tls:
    enabled: true
    secretName: hfpathsim-tls
```

Install with custom values:

```bash
helm install hfpathsim ./deploy/helm/hfpathsim \
  -f values-custom.yaml \
  --namespace hfpathsim \
  --create-namespace
```

## Configuration Options

### API Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api.image.repository` | API image repository | `hfpathsim` |
| `api.image.tag` | API image tag | `latest` |
| `api.replicas` | Number of API replicas | `1` |
| `api.resources.limits.nvidia.com/gpu` | GPU limit | `1` |
| `api.service.type` | Service type | `ClusterIP` |
| `api.service.port` | API port | `8000` |

### Web Dashboard Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `web.enabled` | Enable web dashboard | `true` |
| `web.image.repository` | Web image repository | `hfpathsim-web` |
| `web.image.tag` | Web image tag | `latest` |
| `web.replicas` | Number of web replicas | `1` |
| `web.autoscaling.enabled` | Enable HPA | `false` |

### Ingress Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `false` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.host` | Ingress hostname | `hfpathsim.example.com` |
| `ingress.tls.enabled` | Enable TLS | `false` |

### GPU Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gpu.nodeSelector` | GPU node selector | `nvidia.com/gpu.present: "true"` |
| `gpu.tolerations` | GPU tolerations | `nvidia.com/gpu` |

## Upgrading

```bash
helm upgrade hfpathsim ./deploy/helm/hfpathsim \
  -f values-custom.yaml \
  --namespace hfpathsim
```

## Uninstalling

```bash
helm uninstall hfpathsim --namespace hfpathsim
kubectl delete namespace hfpathsim
```

## Examples

### Production with Ingress and TLS

```yaml
api:
  replicas: 1
  resources:
    requests:
      memory: "8Gi"
      cpu: "4"
    limits:
      memory: "32Gi"
      cpu: "16"
      nvidia.com/gpu: "1"

web:
  enabled: true
  replicas: 2
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10

ingress:
  enabled: true
  className: nginx
  host: hfpathsim.production.com
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  tls:
    enabled: true
    secretName: hfpathsim-tls
```

### Development (No Ingress)

```yaml
api:
  image:
    tag: dev
  resources:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "8Gi"
      cpu: "4"
      nvidia.com/gpu: "1"

web:
  enabled: true
  image:
    tag: dev

ingress:
  enabled: false
```

## Troubleshooting

### Pods pending due to GPU
Check if GPU nodes are available:
```bash
kubectl get nodes -l nvidia.com/gpu.present=true
kubectl describe node <gpu-node>
```

### API not responding
Check logs:
```bash
kubectl logs -l app.kubernetes.io/component=api -n hfpathsim
```

### GPU not detected in container
Verify NVIDIA device plugin:
```bash
kubectl get pods -n kube-system | grep nvidia
kubectl logs -n kube-system -l name=nvidia-device-plugin-ds
```
