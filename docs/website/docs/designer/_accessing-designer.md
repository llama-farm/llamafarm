<!--
  Shared snippet for Designer access instructions

  This file reduces duplication across documentation pages.

  Usage: Instead of repeating setup instructions, link to the Designer docs:
  See the [Designer documentation](../designer/index.md) for access instructions.

  The canonical setup instructions are in docs/website/docs/designer/index.md
  This file serves as a reference template if needed in the future.
-->

The easiest way to start the Designer is through the `lf start` command:

```bash
lf start
```

This automatically launches:

- The FastAPI server (port 8000)
- The RAG worker
- The Designer web UI (port **8000**)

Once started, open your browser to:

```
http://localhost:8000
```

The Designer is served by the same FastAPI server, so it shares port 8000 with the API.
