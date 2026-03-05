# Dashboard

The Dashboard is your project's home base — an at-a-glance view of your active project's configuration, data, and status.


## What You'll See

- **Project configuration summary** — runtime provider, model, enabled features
- **Dataset statistics** — number of datasets, total files, processing status
- **Quick actions** — common tasks like creating datasets, editing prompts, or opening chat
- **Getting Started Checklist** — personalized setup guide (see [Getting Started](./getting-started.md))

## Service Status

The header includes a **Service Status** indicator that shows the health of all connected services:

| Status | Meaning |
|---|---|
| 🟢 **Healthy** | All services operational |
| 🟡 **Degraded** | Some services need attention |
| 🔴 **Unhealthy** | Service issues detected |

Click the status indicator to see detailed health for each service (server, RAG worker, Universal Runtime, etc.) with specific status messages and links to documentation.

## Upgrade Banners

When a new version of LlamaFarm is available, a banner appears at the top of the page with the version number and a link to view details and upgrade. Banners can be dismissed per-context (home page vs. project page) and auto-dismiss after a successful upgrade.

## Route

```
/chat/dashboard
```
