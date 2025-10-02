---
title: Kubernetes
slug: /deployment/kubernetes
---

# Kubernetes

Package the server and RAG worker as deployments/stateful sets and manage them with your usual cluster tooling.

## Minimal Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamafarm-server
  namespace: ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llamafarm-server
  template:
    metadata:
      labels:
        app: llamafarm-server
    spec:
      containers:
        - name: server
          image: ghcr.io/your-org/llamafarm-server:latest
          ports:
            - containerPort: 8000
          env:
            - name: LLAMAFARM_CONFIG
              value: /app/llamafarm.yaml
            - name: RUNTIME_API_KEY
              valueFrom:
                secretKeyRef:
                  name: runtime-secrets
                  key: api-key
          volumeMounts:
            - name: config
              mountPath: /app/llamafarm.yaml
              subPath: llamafarm.yaml
      volumes:
        - name: config
          configMap:
            name: llamafarm-config
---
apiVersion: v1
kind: Service
metadata:
  name: llamafarm-server
spec:
  selector:
    app: llamafarm-server
  ports:
    - port: 8000
      targetPort: 8000
```

Add a separate deployment for the Celery worker (use the `rag` image) and scale it according to ingestion needs.

## Key Considerations

- **Config distribution**: ship `llamafarm.yaml` via ConfigMaps or shared storage.
- **Secrets**: load API keys through Kubernetes Secrets or external secret stores.
- **Runtime providers**: expose Ollama/vLLM services inside the cluster or configure access to external APIs.
- **Vector stores**: for production, use managed services (Qdrant Cloud, Pinecone) or stateful sets with persistent volumes.
- **Autoscaling**: use HPA/VPA to scale workers when ingestion spikes.

Helm charts and Terraform modules are on the roadmap; contributions welcome.

See [Deployment](./index.md) for general tips and the Docker Compose example if you need a quick local stack.
