## 1. Backend: Bundle API

- [ ] 1.1 Add `POST /v1/bundle` endpoint — accepts platform, arch, accelerator, addons, version; validates combos; returns SSE stream of progress events as it downloads each component
- [ ] 1.2 Implement bundle execution logic — reuse/shell-out to `lf bundle` or reimplement download logic in Python (download CLI binary, PyApp binaries, torch wheels, addon wheels from GitHub Releases)
- [ ] 1.3 Add `GET /v1/bundles` endpoint — list completed bundles from a bundles directory (read manifest.json from each)
- [ ] 1.4 Add `GET /v1/bundles/{id}/download` endpoint — serve bundle archive file for browser download
- [ ] 1.5 Add `DELETE /v1/bundles/{id}` endpoint — delete bundle archive and its manifest
- [ ] 1.6 Add bundle size estimation logic — return estimated size based on platform/arch/accelerator/addons selection (for the form preview)

## 2. Frontend: Navigation & Routing

- [ ] 2.1 Add "Deploy" nav item to Header.tsx navigation items (after "Test")
- [ ] 2.2 Add `/chat/deploy` route in App.tsx pointing to new DeployPage component
- [ ] 2.3 Update `/chat/versions` redirect to point to `/chat/deploy` instead of `/chat/dashboard`
- [ ] 2.4 Remove PackageModalContext.tsx and all references (App.tsx provider, PageActions trigger, etc.)
- [ ] 2.5 Remove or gut Versions.tsx (replaced by BundleHistory in DeployPage)

## 3. Frontend: Deploy Page & Bundle History

- [ ] 3.1 Create `DeployPage` component — page header, "Create Bundle" button, and BundleHistory table
- [ ] 3.2 Create `BundleHistory` component — table showing past bundles (name, target platform, size, date, download/delete actions) from `GET /v1/bundles`
- [ ] 3.3 Wire download action to `GET /v1/bundles/{id}/download` (browser download)
- [ ] 3.4 Wire delete action with confirmation dialog
- [ ] 3.5 Add empty state for when no bundles exist yet — friendly prompt to create first bundle

## 4. Frontend: Bundle Wizard (Create Flow)

- [ ] 4.1 Create `BundleWizard` component — modal or drawer with the configuration form
- [ ] 4.2 Build platform selector — visual card/pill selector for Linux, Mac, Windows with icons
- [ ] 4.3 Build architecture selector — x86_64 vs ARM pills, auto-grey invalid combos (e.g., darwin+x86_64)
- [ ] 4.4 Build accelerator selector — CUDA, ROCm, Vulkan, CPU, Metal cards with plain-language descriptions and help text; auto-filter based on platform (Metal only on Mac, etc.)
- [ ] 4.5 Build addons checkboxes — list available addons with descriptions
- [ ] 4.6 Add version selector — default to current version, allow override
- [ ] 4.7 Add dynamic size estimate — update estimated bundle size based on selections
- [ ] 4.8 Add form validation — prevent invalid combos, show clear error messages
- [ ] 4.9 Smart defaults — detect user's current platform from user-agent, pre-select it

## 5. Frontend: Bundle Progress

- [ ] 5.1 Create `BundleProgress` component — connects to SSE from `POST /v1/bundle`, shows overall progress bar + per-component step list
- [ ] 5.2 Implement per-step status indicators — checkmark (done), spinner (in progress), circle (pending) for each component
- [ ] 5.3 Add time remaining estimate based on progress rate
- [ ] 5.4 Add cancel button — aborts the SSE connection and calls a cancel endpoint
- [ ] 5.5 Add "Background" button — minimizes to floating pill (reuse pattern from old PackageModal)
- [ ] 5.6 Create floating progress pill component for minimized state

## 6. Frontend: Bundle Success

- [ ] 6.1 Create `BundleSuccess` component — shows bundle name, size, target info, and next steps
- [ ] 6.2 Add download button that triggers `GET /v1/bundles/{id}/download`
- [ ] 6.3 Add "Copy install command" button — copies `./install.sh <filename>` to clipboard
- [ ] 6.4 Add confetti animation on success (keep from old experience)
- [ ] 6.5 Add "Go to Deploy" / "Done" actions

## 7. Dashboard Integration

- [ ] 7.1 Add "Bundle for deployment" action card to Dashboard page
- [ ] 7.2 Show latest bundle info on Dashboard if bundles exist (name, date, quick download)
