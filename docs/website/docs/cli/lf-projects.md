---
title: lf projects
sidebar_position: 6
---

# `lf projects`

List the projects available within a namespace on the server.

## Synopsis

```
lf projects list [flags]
```

## Flags

| Flag | Description |
| ---- | ----------- |
| `--namespace` | Namespace to list (required if not set in config). |
| Global flags | `--server-url`, `--debug`, etc. |

## Example

```bash
lf projects list --namespace default
```

Output:

```
default/llamafarm-1
default/fda-assistant
default/raleigh-udo-demo
```

Use this to discover project IDs when switching between namespaces or debugging server state.

## See Also

- [`lf init`](./lf-init.md)
- [Configuration Guide](../configuration/index.md)
