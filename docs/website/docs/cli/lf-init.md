---
title: lf init
sidebar_position: 1
---

# `lf init`

Create a new LlamaFarm project and generate a validated `llamafarm.yaml` based on server templates.

## Synopsis

```
lf init [path] [flags]
```

- `path` (optional) – directory where the project should be created (default: current directory).

## Flags

| Flag | Description |
| ---- | ----------- |
| `--namespace` | Namespace to create the project in (defaults to `default`). |
| `--template` | Server-side template name (optional). |
| Global flags | `--server-url`, `--debug`, etc. |

## Behaviour

- Ensures the API server is reachable (auto-starts locally if needed).
- Creates the target directory (if missing) and writes `llamafarm.yaml` using the server’s `project.config` response.
- Fails if a LlamaFarm config already exists in the directory.

## Example

```bash
lf init customer-support
cd customer-support
cat llamafarm.yaml
```

**Output**

```
Initializing a new LlamaFarm project in customer-support
Created project default/customer-support in /path/to/customer-support
```

If you see `Error: Project already exists`, remove or rename the directory before re-running.

## See Also

- [Configuration Guide](../configuration/index.md)
- [`lf start`](./lf-start.md)
