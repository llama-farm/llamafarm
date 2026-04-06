---
sidebar_label: API Reference
title: Fine-Tuning API Reference
---

# Fine-Tuning API Reference

:::caution Alpha Feature
Fine-tuning is experimental. APIs may change between releases.
:::

All endpoints are on the Universal Runtime (default port `11540`).

## Endpoints

### POST /v1/finetune/sft

Start a supervised fine-tuning job.

**Request Body:**

```json
{
  "model": "google/gemma-3-4b-it",
  "dataset": [
    {
      "conversations": [
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "Hi there!"}
      ]
    }
  ],
  "dataset_format": "auto",
  "chat_template": "auto",
  "train_on_responses_only": true,
  "epochs": 3,
  "batch_size": 2,
  "learning_rate": 2e-4,
  "lora_rank": 16,
  "lora_alpha": 16,
  "max_seq_length": 2048,
  "max_steps": null,
  "warmup_steps": 10,
  "gradient_accumulation_steps": 2,
  "output_gguf": true,
  "quantization": "q8_0",
  "output_dir": null
}
```

**Response:**

```json
{
  "job_id": "a1b2c3d4",
  "status": "queued",
  "type": "sft",
  "model": "google/gemma-3-4b-it",
  "progress": 0.0,
  "metrics": null,
  "output_dir": null,
  "error": null,
  "created_at": 1708444800.0,
  "started_at": null,
  "completed_at": null
}
```

---

### POST /v1/finetune/cpt

Start a continued pre-training job.

**Request Body:**

```json
{
  "model": "Qwen/Qwen3-8B",
  "dataset": [
    {"text": "Your domain text goes here..."}
  ],
  "epochs": 1,
  "batch_size": 2,
  "learning_rate": 5e-5,
  "embedding_learning_rate": 5e-6,
  "lora_rank": 16,
  "lora_alpha": 16,
  "max_seq_length": 2048,
  "max_steps": null,
  "output_gguf": true,
  "quantization": "q8_0",
  "output_dir": null
}
```

**Response:** Same format as SFT.

---

### GET /v1/finetune/jobs/`{job_id}`

Get the status and metrics of a training job.

**Response:**

```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "type": "sft",
  "model": "google/gemma-3-4b-it",
  "progress": 1.0,
  "metrics": {
    "train_loss": 0.234,
    "val_loss": 0.281,
    "steps": 30,
    "tokens_per_second": 1247.0,
    "peak_memory_gb": 1.045,
    "wall_time_seconds": 21.0
  },
  "output_dir": "/Users/you/.llamafarm/models/llm/a1b2c3d4",
  "error": null,
  "created_at": 1708444800.0,
  "started_at": 1708444801.0,
  "completed_at": 1708444822.0
}
```

**Job Statuses:**

| Status | Description |
|---|---|
| `queued` | Waiting in the sequential job queue |
| `running` | Currently training |
| `completed` | Training finished successfully |
| `failed` | Training failed (check `error` field) |
| `cancelled` | Cancelled by user |

---

### GET /v1/finetune/jobs

List all training jobs.

**Response:**

```json
{
  "jobs": [
    {"job_id": "a1b2c3d4", "status": "completed", "type": "sft", "model": "..."},
    {"job_id": "e5f6g7h8", "status": "running", "type": "cpt", "model": "..."}
  ]
}
```

---

### DELETE /v1/finetune/jobs/`{job_id}`

Cancel a queued or running training job. Running jobs are terminated via process signal.

**Response:**

```json
{
  "job_id": "a1b2c3d4",
  "status": "cancelled"
}
```

---

### POST /v1/finetune/validate

Validate a dataset before training. Returns detected format, warnings, and errors.

**Request Body:**

```json
{
  "dataset": [
    {
      "conversations": [
        {"from": "human", "value": "test"},
        {"from": "gpt", "value": "response"}
      ]
    }
  ],
  "dataset_format": null,
  "chat_template": null
}
```

**Response:**

```json
{
  "valid": true,
  "format": "sharegpt",
  "example_count": 1,
  "warnings": [],
  "errors": []
}
```

---

### GET /v1/finetune/templates

List supported chat templates and auto-detect the best template for a model.

**Response:**

```json
{
  "templates": [
    "llama-3", "llama-2", "chatml", "gemma",
    "qwen", "phi", "mistral", "zephyr", "vicuna", "alpaca"
  ]
}
```

---

## Error Handling

All endpoints return standard HTTP error codes:

| Code | Meaning |
|---|---|
| `400` | Invalid request (bad dataset, invalid parameters) |
| `404` | Job not found |
| `422` | Validation error (empty dataset, parameter out of range) |
| `429` | Too many concurrent jobs |
| `503` | Fine-tuning service not available (dependencies not installed) |

Error response format:

```json
{
  "detail": "Dataset cannot be empty"
}
```

## Job Queue

Training jobs run **sequentially** — one at a time. This prevents OOM crashes from concurrent model loading. New jobs are queued and processed in FIFO order.

The queue has no hard limit, but each job holds a reference to its full dataset in memory while queued. For very large datasets, submit jobs one at a time.
