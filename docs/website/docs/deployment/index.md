---
title: Deployment
sidebar_label: Deployment
slug: /deployment
---

Deploy LlamaFarm locally, in containers, or on clusters.

## Philosophy

- Write once, deploy anywhere
- Local first, cloud ready, edge capable, production grade

## Quick starts

```bash
# Local
llamafarm up
llamafarm up -f llamafarm.dev.yaml
llamafarm up --watch

# Docker
llamafarm build --docker
docker run -p 8080:8080 llamafarm:latest

docker run --gpus all -p 8080:8080 llamafarm:latest

# Cloud (shortcuts)
llamafarm deploy aws --region us-east-1
llamafarm deploy azure --resource-group my-rg
llamafarm deploy gcp --project my-project
```

Continue with:

- Docker Compose → /deployment/docker-compose
- Kubernetes → /deployment/kubernetes

See also production best practices (HA, monitoring, security, scaling) in the sections below.
