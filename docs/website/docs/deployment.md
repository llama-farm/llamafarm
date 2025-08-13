---
sidebar_position: 5
title: Deployment
---

# Deployment Guide

Deploy LlamaFarm anywhere - from your laptop to global cloud infrastructure. This guide covers all deployment options and best practices.

## Deployment Philosophy

LlamaFarm follows a **"Write Once, Deploy Anywhere"** philosophy:

- 🏠 **Local First** - Start on your machine
- ☁️ **Cloud Ready** - Scale to any cloud provider
- 🌍 **Edge Capable** - Run on IoT and edge devices
- 🚀 **Production Grade** - Built for reliability and scale

## Quick Start Deployments

### Local Development

```bash
# Simple local deployment
llamafarm up

# With specific configuration
llamafarm up -f llamafarm.dev.yaml

# With hot reload
llamafarm up --watch
```

### Docker Deployment

```bash
# Build Docker image
llamafarm build --docker

# Run container
docker run -p 8080:8080 llamafarm:latest

# With GPU support
docker run --gpus all -p 8080:8080 llamafarm:latest
```

### Cloud Quick Deploy

```bash
# Deploy to AWS
llamafarm deploy aws --region us-east-1

# Deploy to Azure
llamafarm deploy azure --resource-group my-rg

# Deploy to GCP
llamafarm deploy gcp --project my-project
```

## Local Deployment

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 32GB+ |
| Storage | 20GB | 100GB+ SSD |
| GPU | Optional | NVIDIA 8GB+ VRAM |

### Configuration

```yaml title="deploy.local.yaml"
deploy:
  local:
    enable: true
    host: 0.0.0.0
    port: 8080
    workers: 4

    # GPU settings
    gpu:
      enable: true
      device: cuda:0
      memory_fraction: 0.8

    # CPU settings
    cpu:
      threads: 8
      enable_mkl: true

    # Memory settings
    memory:
      model_cache: 4GB
      max_batch_size: 32

    # Development features
    development:
      debug: true
      hot_reload: true
      swagger_ui: true
```

### Running Locally

```bash
# Start with defaults
llamafarm up

# Custom port
llamafarm up --port 3000

# Multiple models
llamafarm up --models llama2,mistral

# CPU only
llamafarm up --device cpu
```

## Cloud Deployments

### AWS Deployment

#### EC2 Deployment

```yaml title="deploy.aws.yaml"
deploy:
  aws:
    region: us-east-1

    # Instance configuration
    ec2:
      instance_type: g4dn.xlarge
      ami: ami-0abcdef1234567890  # LlamaFarm AMI
      key_pair: my-keypair

      # Auto-scaling
      auto_scaling:
        min: 1
        max: 10
        target_cpu: 70

      # Storage
      storage:
        type: gp3
        size: 100
        iops: 3000

    # Networking
    vpc:
      create_new: true
      cidr: 10.0.0.0/16

    security_group:
      ingress:
        - port: 8080
          protocol: tcp
          source: 0.0.0.0/0

    # Load balancer
    load_balancer:
      type: application
      scheme: internet-facing
      health_check:
        path: /health
        interval: 30
```

Deploy command:
```bash
llamafarm deploy aws --config deploy.aws.yaml
```

#### ECS Fargate Deployment

```yaml
deploy:
  aws:
    ecs:
      cluster: llamafarm-cluster
      service: llamafarm-service

      task_definition:
        cpu: 4096
        memory: 30720

      fargate:
        platform_version: LATEST
        assign_public_ip: true
```

#### Lambda Deployment

```yaml
deploy:
  aws:
    lambda:
      function_name: llamafarm-inference
      runtime: python3.9
      memory: 10240
      timeout: 900

      # Use container image for large models
      package_type: Image
      image_uri: ${ECR_URI}/llamafarm:latest
```

### Azure Deployment

#### Azure Container Instances

```yaml title="deploy.azure.yaml"
deploy:
  azure:
    resource_group: llamafarm-rg
    location: eastus

    container_instance:
      name: llamafarm-aci
      cpu: 4
      memory: 16
      gpu:
        count: 1
        sku: K80

    # Azure ML Integration
    ml_workspace:
      name: llamafarm-ml
      compute_target: gpu-cluster
```

Deploy command:
```bash
llamafarm deploy azure --config deploy.azure.yaml
```

### Google Cloud Platform

#### GKE Deployment

```yaml title="deploy.gcp.yaml"
deploy:
  gcp:
    project: my-project
    region: us-central1

    gke:
      cluster_name: llamafarm-cluster
      node_pools:
        - name: cpu-pool
          machine_type: n2-standard-8
          min_nodes: 1
          max_nodes: 10

        - name: gpu-pool
          machine_type: n1-standard-8
          accelerator:
            type: nvidia-tesla-t4
            count: 1
          min_nodes: 0
          max_nodes: 5
```

## Kubernetes Deployment

### Kubernetes Configuration

```yaml title="k8s-deployment.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamafarm
  namespace: ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llamafarm
  template:
    metadata:
      labels:
        app: llamafarm
    spec:
      containers:
      - name: llamafarm
        image: llamafarm/llamafarm:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "/models"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### Helm Chart

```bash
# Install with Helm
helm install llamafarm ./charts/llamafarm \
  --set model.type=llama2-13b \
  --set replicas=3 \
  --set gpu.enabled=true
```

## Edge Deployment

### Raspberry Pi

```yaml title="deploy.edge.yaml"
deploy:
  edge:
    device_type: raspberry-pi

    optimization:
      quantization: int4
      cpu_threads: 4

    models:
      - name: tiny-llama
        type: llama2-1b
        max_memory: 2GB
```

### NVIDIA Jetson

```yaml
deploy:
  edge:
    device_type: jetson-nano

    optimization:
      use_tensorrt: true
      precision: fp16

    models:
      - name: efficient-llama
        type: llama2-7b
        quantization: int8
```

### Deploy to Edge

```bash
# Build for ARM
llamafarm build --arch arm64

# Deploy to device
llamafarm deploy edge --host 192.168.1.100
```

## Production Best Practices

### High Availability

```yaml
deploy:
  high_availability:
    # Multiple regions
    regions:
      primary: us-east-1
      secondary: eu-west-1

    # Database replication
    database:
      type: postgres
      replicas: 3
      backup:
        enabled: true
        schedule: "0 2 * * *"

    # Model caching
    cache:
      type: redis
      cluster: true
      nodes: 6
```

### Monitoring & Observability

```yaml
monitoring:
  # Metrics
  prometheus:
    enabled: true
    scrape_interval: 15s

  # Logging
  logging:
    driver: fluentd
    options:
      fluentd-address: localhost:24224

  # Tracing
  tracing:
    enabled: true
    sampling_rate: 0.1
    exporter: jaeger

  # Alerts
  alerts:
    - name: high-latency
      condition: response_time > 2s
      action: scale_up

    - name: error-rate
      condition: error_rate > 0.05
      action: notify_oncall
```

### Security

```yaml
security:
  # Network security
  network:
    vpc_isolation: true
    private_subnets: true
    nat_gateway: true

  # Encryption
  encryption:
    at_rest: true
    in_transit: true
    kms_key: alias/llamafarm

  # Access control
  iam:
    role: llamafarm-role
    policies:
      - AmazonS3ReadOnlyAccess
      - CloudWatchLogsFullAccess

  # Secrets management
  secrets:
    provider: aws-secrets-manager
    rotation: true
```

### Scaling Strategies

```yaml
scaling:
  # Horizontal scaling
  horizontal:
    metrics:
      - type: cpu
        target: 70
      - type: memory
        target: 80
      - type: requests_per_second
        target: 1000

  # Vertical scaling
  vertical:
    enabled: true
    max_cpu: 16
    max_memory: 64Gi

  # Predictive scaling
  predictive:
    enabled: true
    lookahead: 1h
    algorithm: lstm
```

## Cost Optimization

### Multi-Cloud Cost Management

```yaml
cost_optimization:
  # Spot instances
  spot_instances:
    enabled: true
    max_price: 0.50
    fallback: on-demand

  # Reserved capacity
  reserved:
    term: 1_year
    payment: partial_upfront

  # Auto-shutdown
  schedules:
    - name: development
      start: "8:00"
      stop: "18:00"
      timezone: "America/New_York"
      days: ["Mon", "Tue", "Wed", "Thu", "Fri"]
```

### Model Optimization

```yaml
optimization:
  # Model quantization
  quantization:
    enabled: true
    bits: 4
    group_size: 128

  # Batch processing
  batching:
    enabled: true
    max_batch_size: 32
    timeout: 100ms

  # Caching
  cache:
    enabled: true
    ttl: 3600
    max_size: 10GB
```

## Deployment Automation

### CI/CD Pipeline

```yaml title=".github/workflows/deploy.yml"
name: Deploy LlamaFarm

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build
        run: llamafarm build

      - name: Test
        run: llamafarm test

      - name: Deploy to Staging
        run: llamafarm deploy staging

      - name: Run E2E Tests
        run: llamafarm test:e2e

      - name: Deploy to Production
        if: success()
        run: llamafarm deploy production
```

### Infrastructure as Code

```hcl title="terraform/main.tf"
module "llamafarm" {
  source = "./modules/llamafarm"

  environment = "production"
  region      = "us-east-1"

  instance_type = "g4dn.xlarge"
  min_instances = 2
  max_instances = 10

  model_config = {
    type = "llama2-13b"
    quantization = "int8"
  }
}
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce batch size, enable quantization |
| Slow inference | Enable GPU, use smaller model |
| Connection refused | Check firewall, security groups |
| Model not loading | Verify model path, check permissions |

### Health Checks

```bash
# Check deployment status
llamafarm status

# View logs
llamafarm logs --tail 100

# Run diagnostics
llamafarm diagnose

# Test endpoint
curl http://localhost:8080/health
```

## Next Steps

- Set up [Monitoring](#) for your deployment
- Configure [Security](#) best practices
- Optimize [Performance](#) for your use case