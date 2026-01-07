---
title: Deployment
sidebar_position: 9
---

# Deployment

LlamaFarm is designed for native deployment without containers. The application runs directly on your system using native binaries and the UV package manager for Python dependencies.

## Production Deployment

### Native Process Management

For production, use process supervisors to start services

**systemd (Linux)**:

```ini
[Unit]
Description=LlamaFarm Server
After=network.target

[Service]
Type=simple
User=llamafarm
WorkingDirectory=/opt/llamafarm
ExecStart=/opt/llamafarm/lf services start
Restart=always
Environment=LF_DATA_DIR=/var/lib/llamafarm

[Install]
WantedBy=multi-user.target
```

**PM2 (Node.js)**:

```bash
pm2 start "lf services start" --name llamafarm
pm2 save
```

**launchd (macOS)**:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.llamafarm.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/lf</string>
        <string>services</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

### Production Checklist

- **Environment variables**: Store API keys (OpenAI, Together, etc.) in `.env` files or secret managers. Update `runtime.api_key` to reference them.
- **Data directory**: Set `LF_DATA_DIR` to a persistent location with adequate storage for models and vector databases.
- **Firewall**: Restrict access to ports 8000 (API) and 11540 (Universal Runtime) as needed.
- **TLS termination**: Use a reverse proxy (Traefik, nginx, Caddy) for HTTPS in production.
- **Monitoring**: Enable logging and set up health checks against `/health` endpoint.

### Reverse Proxy Example (nginx)

```nginx
upstream llamafarm {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl;
    server_name api.yourcompany.com;

    ssl_certificate /etc/ssl/certs/api.yourcompany.com.crt;
    ssl_certificate_key /etc/ssl/private/api.yourcompany.com.key;

    location / {
        proxy_pass http://llamafarm;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # For streaming responses
        proxy_buffering off;
        proxy_cache off;
    }
}
```

## Scaling Considerations

- **Multiple instances**: Run multiple LlamaFarm instances behind a load balancer for horizontal scaling.
- **External model providers**: Use vLLM, Together, or OpenAI for model inference to offload compute.
- **Managed vector stores**: Swap `ChromaStore` for Qdrant Cloud, Pinecone, or another managed backend for larger deployments.
- **Shared storage**: Use NFS or object storage for `LF_DATA_DIR` when running multiple instances.

## Health Checks

LlamaFarm provides health endpoints for monitoring:

```bash
# Full health check
curl http://localhost:8000/health

# Liveness probe (for orchestrators)
curl http://localhost:8000/health/liveness
```

## Resources

- [Quickstart](../quickstart/index.md) – Local installation steps
- [Configuration Guide](../configuration/index.md) – Runtime/provider settings
- [Desktop App](../desktop-app/index.md) – Bundled application
- [Extending LlamaFarm](../extending/index.md) – Adapt to your infrastructure
