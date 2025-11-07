<!-- 
  Shared snippet for Designer access instructions
  
  This file reduces duplication across documentation pages.
  
  Usage: Instead of repeating setup instructions, link to the Designer docs:
  See the [Designer documentation](../designer/index.md) for access instructions.
  
  The canonical setup instructions are in docs/website/docs/designer/index.md
  This file serves as a reference template if needed in the future.
-->

The Designer runs on different ports depending on how you start it. Both methods are fully functional—choose based on your workflow.

### Via the CLI (Recommended)

The easiest way to start the Designer is through the `lf start` command:

```bash
lf start
```

This automatically launches:
- The FastAPI server (port 8000)
- The RAG worker
- The Designer web UI (port **7724**)

Once started, open your browser to:

```
http://localhost:7724
```

**Why port 7724?** The CLI uses a dedicated port to avoid conflicts with other services and to allow running multiple LlamaFarm instances for different projects.

### Via Docker Compose

If you're using Docker Compose directly:

```bash
cd deployment/docker_compose
docker-compose up -d
```

The Designer will be available at:

```
http://localhost:3123
```

**Why port 3123?** Docker Compose uses a different port to distinguish between CLI-managed and Docker Compose-managed deployments, allowing both to run simultaneously if needed.

