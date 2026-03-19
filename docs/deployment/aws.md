# AWS Deployment Guide

This guide covers deploying HF Path Simulator on AWS with GPU support.

## Prerequisites

- AWS CLI configured with appropriate credentials
- EC2 Key Pair created
- Docker images pushed to ECR (or use Docker Hub)

## Option 1: CloudFormation (Recommended for Single Instance)

### Deploy the Stack

```bash
# Create the stack
aws cloudformation create-stack \
  --stack-name hfpathsim-dev \
  --template-body file://deploy/aws/cloudformation.yaml \
  --parameters \
    ParameterKey=EnvironmentName,ParameterValue=dev \
    ParameterKey=InstanceType,ParameterValue=g4dn.xlarge \
    ParameterKey=KeyPairName,ParameterValue=your-key-pair \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion
aws cloudformation wait stack-create-complete --stack-name hfpathsim-dev

# Get outputs
aws cloudformation describe-stacks --stack-name hfpathsim-dev \
  --query 'Stacks[0].Outputs'
```

### Instance Types

| Instance    | GPU       | vCPU | RAM  | Use Case        |
|-------------|-----------|------|------|-----------------|
| g4dn.xlarge | T4 (16GB) | 4    | 16GB | Development     |
| g4dn.2xlarge| T4 (16GB) | 8    | 32GB | Heavy workloads |
| g5.xlarge   | A10G (24GB)| 4   | 16GB | Production      |
| g5.2xlarge  | A10G (24GB)| 8   | 32GB | High performance|

### Cost Estimates (us-east-1)

- g4dn.xlarge: ~$0.526/hour (~$380/month)
- g5.xlarge: ~$1.006/hour (~$725/month)

## Option 2: ECS with GPU

### Create ECR Repository

```bash
aws ecr create-repository --repository-name hfpathsim
aws ecr create-repository --repository-name hfpathsim-web
```

### Push Images

```bash
# Get login token
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t hfpathsim .
docker tag hfpathsim:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hfpathsim:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hfpathsim:latest
```

### Create ECS Cluster with GPU

```bash
# Create GPU-enabled cluster
aws ecs create-cluster \
  --cluster-name hfpathsim \
  --capacity-providers FARGATE FARGATE_SPOT \
  --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1
```

Note: For GPU support, you need EC2 launch type with GPU instances.
See [AWS ECS GPU Guide](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html).

### Register Task Definition

```bash
# Update ACCOUNT_ID and REGION in the file first
aws ecs register-task-definition \
  --cli-input-json file://deploy/aws/ecs-task-definition.json
```

### Create Service

```bash
aws ecs create-service \
  --cluster hfpathsim \
  --service-name hfpathsim-service \
  --task-definition hfpathsim:1 \
  --desired-count 1 \
  --launch-type EC2 \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}"
```

## Networking

### Security Group Rules

| Port | Protocol | Source    | Description      |
|------|----------|-----------|------------------|
| 22   | TCP      | Your IP   | SSH              |
| 80   | TCP      | 0.0.0.0/0 | HTTP             |
| 443  | TCP      | 0.0.0.0/0 | HTTPS            |
| 8000 | TCP      | 0.0.0.0/0 | API              |
| 5555 | TCP      | Trusted   | ZMQ Input        |
| 5556 | TCP      | Trusted   | ZMQ Output       |

### Enable HTTPS

1. Get SSL certificate from ACM
2. Create Application Load Balancer
3. Configure HTTPS listener

```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name hfpathsim-alb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxx

# Create target group
aws elbv2 create-target-group \
  --name hfpathsim-api \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxx \
  --health-check-path /api/v1/health

# Add HTTPS listener with ACM certificate
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:... \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:... \
  --default-actions Type=forward,TargetGroupArn=arn:aws:...
```

## Monitoring

### CloudWatch Logs

Logs are automatically sent to CloudWatch Logs group `/ecs/hfpathsim`.

```bash
# View recent logs
aws logs tail /ecs/hfpathsim --follow
```

### CloudWatch Metrics

Create dashboard for:
- CPU/Memory utilization
- GPU utilization (requires CloudWatch agent)
- API response times
- WebSocket connections

### Alarms

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name hfpathsim-high-cpu \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:xxx:alerts
```

## Cleanup

```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name hfpathsim-dev

# Delete ECS resources
aws ecs delete-service --cluster hfpathsim --service hfpathsim-service --force
aws ecs delete-cluster --cluster hfpathsim
```
