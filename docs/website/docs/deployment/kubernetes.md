---
title: Kubernetes
sidebar_label: Kubernetes
slug: /deployment/kubernetes
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Deploy to Kubernetes clusters with manifests or Helm.

## Example manifest

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
              memory: '8Gi'
              cpu: '4'
              nvidia.com/gpu: 1
            limits:
              memory: '16Gi'
              cpu: '8'
              nvidia.com/gpu: 1
          env:
            - name: MODEL_PATH
              value: '/models'
          volumeMounts:
            - name: models
              mountPath: /models
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: models-pvc
```

## Helm

```bash
helm install llamafarm ./charts/llamafarm \
  --set model.type=llama2-13b \
  --set replicas=3 \
  --set gpu.enabled=true
```

## Tips

- Use node pools for CPU/GPU separation
- Configure HPA/VPA for scaling
- Mount secrets via Secret and CSI providers
