# Proposal: Bundle UI in Designer

## Summary

Add a "Bundle" experience to the Designer that lets Larry create distributable LlamaFarm archives for deploying to remote/air-gapped machines — without touching the terminal.

## Motivation

PR #772 added `lf bundle` as a CLI command, but Larry (our target user) shouldn't have to memorize flags like `--accelerator cuda --arch x86_64`. The Designer should guide him through building a bundle with smart defaults, plain-language explanations, and real-time progress — the same way it guides him through RAG and model setup.

## End-User Flow

### Larry's Journey

**1. Discovery — "I need to get this onto another machine"**

Larry has built an AI project in the Designer. Now he needs to deploy it to a GPU server in his office, or hand it to a teammate who's offline. He looks at the Dashboard and sees a **"Bundle for deployment"** action card (or a "Deploy" section in the nav).

**2. Configure — "What am I building for?"**

Larry clicks "Create Bundle" and gets a guided form:

```
┌─────────────────────────────────────────────────┐
│  Create Bundle                                   │
│                                                  │
│  Where is this going?                            │
│  ┌─────────────────────────────────────────────┐ │
│  │ 🐧 Linux Server    │ 🍎 Mac    │ 🪟 Windows │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  What chip?                                      │
│  ┌──────────────────────────────────┐            │
│  │ Intel/AMD (x86_64)  │ ARM (arm64)│            │
│  └──────────────────────────────────┘            │
│                                                  │
│  GPU acceleration                                │
│  ┌────────────────────────────────────────────┐  │
│  │ NVIDIA (CUDA) │ AMD (ROCm) │ CPU only │ ...│  │
│  └────────────────────────────────────────────┘  │
│  ℹ️ Not sure? "CPU only" works everywhere.       │
│     Pick NVIDIA if the target has a GeForce/     │
│     Tesla GPU.                                   │
│                                                  │
│  Addons (optional)                               │
│  ☐ Speech-to-Text (STT)                         │
│  ☐ Text-to-Speech (TTS)                         │
│                                                  │
│  Version: v0.8.0 (current)                       │
│                                                  │
│  Estimated size: ~1.8 GB                         │
│                                                  │
│           [Cancel]  [Create Bundle]              │
└─────────────────────────────────────────────────┘
```

Key UX decisions:
- **Visual selectors** (cards/pills), not dropdowns — Larry can scan options at a glance
- **Plain language** — "Where is this going?" not "Target platform". "What chip?" not "Architecture"
- **Smart defaults** — detect current platform, pre-select it, grey out invalid combos (e.g., Metal only on Mac)
- **Help text** — brief guidance under each option for Larry who doesn't know CUDA from ROCm
- **Size estimate** — updates dynamically based on selections (addons add ~200MB each, torch adds ~1GB for CUDA)

**3. Build — "It's working"**

Larry hits "Create Bundle" and sees real-time progress:

```
┌─────────────────────────────────────────────────┐
│  Building bundle...                              │
│                                                  │
│  ████████████░░░░░░░░░░░░░  48%    ~2m left     │
│                                                  │
│  ✓ CLI binary (12 MB)                            │
│  ✓ Server binary (45 MB)                         │
│  ⟳ RAG binary — downloading...                   │
│  ○ Runtime binary                                │
│  ○ CUDA torch wheels                             │
│  ○ Packaging archive                             │
│                                                  │
│           [Cancel]  [Background]                 │
└─────────────────────────────────────────────────┘
```

- **Real progress** from the server (SSE), not fake — each component download is a discrete step
- **Backgroundable** — minimize to a floating pill, keep working on the project
- **Cancellable** — stop mid-download, clean up temp files

**4. Success — "Here it is"**

```
┌─────────────────────────────────────────────────┐
│  🎉 Bundle ready!                                │
│                                                  │
│  llamafarm-v0.8.0-linux-x86_64-cuda.tar.gz      │
│  1.82 GB · Linux · x86_64 · CUDA                │
│                                                  │
│  Next steps:                                     │
│  1. Transfer this file to your target machine    │
│  2. Run: ./install.sh <filename>                 │
│                                                  │
│       [Download]  [Copy install command]  [Done] │
└─────────────────────────────────────────────────┘
```

- **Download button** — triggers browser download of the archive from the server
- **Copy install command** — one-click copy of the install.sh command
- **"Next steps"** — Larry doesn't have to read docs to know what to do next

**5. History — "What have I built before?"**

The Bundle page (replacing old Versions page) shows past bundles:

| Bundle | Target | Size | Date | Actions |
|--------|--------|------|------|---------|
| v0.8.0-linux-x86_64-cuda | Linux · CUDA | 1.82 GB | Feb 19, 2025 | ⬇️ 🗑️ |
| v0.7.5-darwin-arm64-metal | Mac · Metal | 1.45 GB | Feb 15, 2025 | ⬇️ 🗑️ |

## Architecture

### Where it lives in the app

- **Entry point:** The existing **"Package" button** in `PageActions` (appears on most pages) gets renamed to **"Bundle"** and opens the Bundle wizard modal. Same for the custom package buttons in Prompt and Test pages.
- **Route:** `/chat/deploy` — dedicated page for bundle history and management
- **Dashboard card:** "Bundle for deployment" quick-action on the Dashboard
- **No new nav item** — the button Larry already sees is the entry point; the deploy page is for history/management

The old PackageModalContext gets replaced with a new BundleModalContext that opens the real wizard instead of the fake packager.

### Backend: Bundle API endpoint

A thin FastAPI endpoint that wraps `lf bundle`:

```
POST /v1/bundle
{
  "platform": "linux",
  "arch": "x86_64",
  "accelerator": "cuda",
  "addons": ["stt", "tts"],
  "version": "v0.8.0"
}

Response: SSE stream
event: progress
data: {"step": "cli", "status": "downloading", "progress": 0.45}

event: progress
data: {"step": "cli", "status": "complete", "size": 12000000}

event: progress
data: {"step": "server", "status": "downloading", "progress": 0.12}

...

event: complete
data: {"path": "/tmp/bundles/llamafarm-v0.8.0-linux-x86_64-cuda.tar.gz", "size": 1820000000}
```

Plus:
- `GET /v1/bundles` — list past bundles (from a bundles directory on disk)
- `GET /v1/bundles/{id}/download` — download a bundle archive
- `DELETE /v1/bundles/{id}` — delete a bundle

### Frontend components

| Component | Purpose |
|-----------|---------|
| `DeployPage` | Main page at `/chat/deploy`, shows bundle history + "Create Bundle" button |
| `BundleWizard` | The create flow — platform/arch/accelerator/addons form |
| `BundleProgress` | SSE-connected progress view with per-step status |
| `BundleSuccess` | Success state with download + next steps |
| `BundleHistory` | Table of past bundles with download/delete actions |

### What we reuse from old package experience

- **Modal pattern** from PackageModalContext — the open/close/minimize/background flow
- **Progress bar UI** — same visual treatment, but wired to real SSE
- **Confetti** on success (it's fun, keep it)
- **Versions table structure** from Versions.tsx — adapted for bundle metadata

### What we delete

- `PackageModalContext.tsx` — replaced entirely
- `Versions.tsx` — replaced by DeployPage/BundleHistory
- All localStorage usage for versions/packaging state

## Phases

### Phase 1: Bundle UI (this PR)
- Deploy page + nav item
- Bundle wizard form
- Bundle API endpoint (server)
- SSE progress
- Bundle history
- Remove old package experience

### Phase 2: Deploy UI (future)
- Add "Deploy to server" flow alongside bundling
- Environment management UI
- Remote server status/health display

## Out of Scope
- Deploy command UI (Phase 2)
- Environment management in Designer
- Model push UI
- Any changes to the CLI bundle command itself
