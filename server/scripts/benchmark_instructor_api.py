#!/usr/bin/env python3
"""Benchmark server chat completion latency with and without Instructor."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_PROMPTS = [
    "Summarize how retrieval-augmented generation works in two sentences.",
    "List three practical ways to reduce inference latency in local AI systems.",
    "Give a short explanation of what a vector database does.",
    "Write a concise checklist for debugging a failing API endpoint.",
    "Explain the trade-off between latency and quality in model selection.",
    "Provide a one-paragraph definition of structured output.",
]


@dataclass
class ModeTarget:
    label: str
    namespace: str
    project: str


@dataclass
class RequestMeasurement:
    mode: str
    request_index: int
    prompt: str
    prompt_index: int
    latency_ms: float
    status_code: int
    ok: bool
    error: str | None
    completion_id: str | None
    usage: dict[str, Any] | None
    response_bytes: int
    timestamp_utc: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure chat completion latency before and after Instructor by comparing "
            "two project targets on the same server API."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--baseline-namespace", required=True)
    parser.add_argument("--baseline-project", required=True)
    parser.add_argument("--instructor-namespace", required=True)
    parser.add_argument("--instructor-project", required=True)
    parser.add_argument("--model", default=None, help="Model config name override")
    parser.add_argument("--requests-per-mode", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--inconclusive-error-rate",
        type=float,
        default=0.05,
        help="Mark run inconclusive if mode error_rate exceeds this value.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional JSON file containing a list of prompt strings.",
    )
    parser.add_argument(
        "--output-dir",
        default="server/artifacts/benchmarks",
        help="Directory where benchmark artifacts will be written.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_prompts(prompt_file: str | None) -> list[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS.copy()
    prompt_path = Path(prompt_file)
    loaded = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("Prompt file must contain a non-empty JSON list of strings.")
    prompts = [str(item).strip() for item in loaded if str(item).strip()]
    if not prompts:
        raise ValueError("Prompt file list cannot be empty after trimming prompts.")
    return prompts


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def http_json_request(
    method: str,
    url: str,
    timeout_seconds: float,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any] | None, bytes]:
    body_bytes = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = request.Request(
        url=url,
        data=body_bytes,
        headers=req_headers,
        method=method.upper(),
    )
    raw_body = b""
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw_body = response.read()
            content = raw_body.decode("utf-8") if raw_body else "{}"
            return int(response.status), json.loads(content), raw_body
    except error.HTTPError as exc:
        raw_body = exc.read() or b""
        parsed = None
        if raw_body:
            try:
                parsed = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                parsed = {"raw_error": raw_body.decode("utf-8", errors="replace")}
        return int(exc.code), parsed, raw_body


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return (sorted_values[lower] * (1.0 - weight)) + (sorted_values[upper] * weight)


def read_project_model_signature(
    *,
    base_url: str,
    target: ModeTarget,
    timeout_seconds: float,
    model_override: str | None,
) -> dict[str, str]:
    project_url = (
        f"{base_url}/v1/projects/{target.namespace}/{target.project}"
    )
    status, data, _ = http_json_request("GET", project_url, timeout_seconds=timeout_seconds)
    if status != 200 or data is None:
        raise RuntimeError(
            f"Failed to fetch project config for {target.label} "
            f"({target.namespace}/{target.project}): status={status}, body={data}"
        )
    project = data.get("project") if isinstance(data, dict) else None
    config = project.get("config") if isinstance(project, dict) else None
    runtime = config.get("runtime") if isinstance(config, dict) else None
    models = runtime.get("models") if isinstance(runtime, dict) else None
    default_model = runtime.get("default_model") if isinstance(runtime, dict) else None

    if not isinstance(models, list) or not models:
        raise RuntimeError(
            f"Project {target.namespace}/{target.project} has no runtime.models"
        )

    selected = None
    selected_name = model_override or default_model
    if selected_name:
        for model in models:
            if isinstance(model, dict) and model.get("name") == selected_name:
                selected = model
                break
    if selected is None and isinstance(models[0], dict):
        selected = models[0]

    if not isinstance(selected, dict):
        raise RuntimeError(
            f"Could not resolve model for {target.namespace}/{target.project}"
        )

    provider = str(selected.get("provider", ""))
    model_string = str(selected.get("model", ""))
    name = str(selected.get("name", selected_name or ""))
    if not provider or not model_string:
        raise RuntimeError(
            f"Resolved model is missing provider/model for {target.namespace}/{target.project}"
        )
    return {"name": name, "provider": provider, "model": model_string}


def build_chat_payload(
    *,
    prompt: str,
    model_name: str | None,
    max_tokens: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "rag_enabled": False,
    }
    if model_name:
        payload["model"] = model_name
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def run_mode(
    *,
    base_url: str,
    target: ModeTarget,
    prompts: list[str],
    requests_per_mode: int,
    timeout_seconds: float,
    model_name: str | None,
    max_tokens: int | None,
    temperature: float | None,
    sleep_ms: int,
    verbose: bool,
) -> list[RequestMeasurement]:
    results: list[RequestMeasurement] = []
    chat_url = (
        f"{base_url}/v1/projects/{target.namespace}/{target.project}/chat/completions"
    )

    for i in range(requests_per_mode):
        prompt_index = i % len(prompts)
        prompt = prompts[prompt_index]
        payload = build_chat_payload(
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        started = time.perf_counter()
        status = 0
        data: dict[str, Any] | None = None
        raw_body = b""
        err: str | None = None
        try:
            status, data, raw_body = http_json_request(
                "POST",
                chat_url,
                timeout_seconds=timeout_seconds,
                payload=payload,
                headers={"X-No-Session": "true"},
            )
        except Exception as exc:  # noqa: BLE001
            err = f"request_exception: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if err is None and status != 200:
            detail = data.get("detail") if isinstance(data, dict) else data
            err = f"http_{status}: {detail}"

        completion_id = data.get("id") if isinstance(data, dict) else None
        usage = data.get("usage") if isinstance(data, dict) else None
        measurement = RequestMeasurement(
            mode=target.label,
            request_index=i + 1,
            prompt=prompt,
            prompt_index=prompt_index,
            latency_ms=elapsed_ms,
            status_code=status,
            ok=err is None,
            error=err,
            completion_id=str(completion_id) if completion_id is not None else None,
            usage=usage if isinstance(usage, dict) else None,
            response_bytes=len(raw_body),
            timestamp_utc=datetime.now(UTC).isoformat(),
        )
        results.append(measurement)

        if verbose:
            state = "ok" if measurement.ok else "error"
            print(
                f"[{target.label}] #{measurement.request_index:02d} "
                f"{measurement.latency_ms:.1f}ms {state}"
            )

        if sleep_ms > 0 and i < requests_per_mode - 1:
            time.sleep(sleep_ms / 1000.0)

    return results


def summarize_mode(
    mode: str,
    measurements: list[RequestMeasurement],
    warmup: int,
) -> dict[str, Any]:
    considered = measurements[warmup:]
    latencies = [m.latency_ms for m in considered]
    ok_count = sum(1 for m in considered if m.ok)
    total_count = len(considered)
    error_count = total_count - ok_count
    error_rate = (error_count / total_count) if total_count else 0.0
    return {
        "mode": mode,
        "total_requests": len(measurements),
        "warmup_excluded": min(warmup, len(measurements)),
        "samples_used": total_count,
        "ok_count": ok_count,
        "error_count": error_count,
        "error_rate": error_rate,
        "median_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "latencies_ms": latencies,
    }


def percent_delta(new: float, old: float) -> float:
    if old == 0 or math.isnan(old) or math.isnan(new):
        return float("nan")
    return ((new - old) / old) * 100.0


def save_artifact(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = output_dir / f"instructor_api_benchmark_{timestamp}.json"
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path


def main() -> int:
    args = parse_args()
    if args.requests_per_mode <= 0:
        raise ValueError("--requests-per-mode must be greater than zero.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.requests_per_mode <= args.warmup:
        raise ValueError("--warmup must be less than --requests-per-mode.")

    base_url = normalize_base_url(args.base_url)
    prompts = load_prompts(args.prompt_file)

    baseline_target = ModeTarget(
        label="baseline",
        namespace=args.baseline_namespace,
        project=args.baseline_project,
    )
    instructor_target = ModeTarget(
        label="instructor",
        namespace=args.instructor_namespace,
        project=args.instructor_project,
    )

    # Guardrail: ensure model/provider are the same.
    baseline_signature = read_project_model_signature(
        base_url=base_url,
        target=baseline_target,
        timeout_seconds=args.timeout_seconds,
        model_override=args.model,
    )
    instructor_signature = read_project_model_signature(
        base_url=base_url,
        target=instructor_target,
        timeout_seconds=args.timeout_seconds,
        model_override=args.model,
    )
    if (
        baseline_signature["provider"] != instructor_signature["provider"]
        or baseline_signature["model"] != instructor_signature["model"]
    ):
        raise RuntimeError(
            "Abort: model/provider mismatch between modes. "
            f"baseline={baseline_signature}, instructor={instructor_signature}"
        )

    model_name = args.model or baseline_signature["name"]
    print(
        "Running benchmark with model "
        f"'{model_name}' ({baseline_signature['provider']} / {baseline_signature['model']})"
    )
    print(
        f"Requests per mode: {args.requests_per_mode}, warmup excluded: {args.warmup}, "
        f"prompt count: {len(prompts)}"
    )

    baseline_measurements = run_mode(
        base_url=base_url,
        target=baseline_target,
        prompts=prompts,
        requests_per_mode=args.requests_per_mode,
        timeout_seconds=args.timeout_seconds,
        model_name=model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sleep_ms=args.sleep_ms,
        verbose=args.verbose,
    )
    instructor_measurements = run_mode(
        base_url=base_url,
        target=instructor_target,
        prompts=prompts,
        requests_per_mode=args.requests_per_mode,
        timeout_seconds=args.timeout_seconds,
        model_name=model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sleep_ms=args.sleep_ms,
        verbose=args.verbose,
    )

    baseline_summary = summarize_mode(
        "baseline",
        baseline_measurements,
        warmup=args.warmup,
    )
    instructor_summary = summarize_mode(
        "instructor",
        instructor_measurements,
        warmup=args.warmup,
    )

    median_delta = percent_delta(
        instructor_summary["median_ms"], baseline_summary["median_ms"]
    )
    p95_delta = percent_delta(instructor_summary["p95_ms"], baseline_summary["p95_ms"])

    inconclusive = (
        baseline_summary["error_rate"] > args.inconclusive_error_rate
        or instructor_summary["error_rate"] > args.inconclusive_error_rate
    )

    result = {
        "metadata": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "base_url": base_url,
            "model_name": model_name,
            "baseline_target": asdict(baseline_target),
            "instructor_target": asdict(instructor_target),
            "requests_per_mode": args.requests_per_mode,
            "warmup": args.warmup,
            "timeout_seconds": args.timeout_seconds,
            "sleep_ms": args.sleep_ms,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "prompt_count": len(prompts),
            "prompts": prompts,
            "model_signature": {
                "baseline": baseline_signature,
                "instructor": instructor_signature,
            },
            "inconclusive_error_rate_threshold": args.inconclusive_error_rate,
            "inconclusive": inconclusive,
        },
        "summary": {
            "baseline": baseline_summary,
            "instructor": instructor_summary,
            "delta_percent": {
                "median_ms": median_delta,
                "p95_ms": p95_delta,
            },
        },
        "requests": {
            "baseline": [asdict(item) for item in baseline_measurements],
            "instructor": [asdict(item) for item in instructor_measurements],
        },
    }

    artifact_path = save_artifact(Path(args.output_dir), result)

    print("\nResults:")
    print(
        f"- Baseline median/p95: {baseline_summary['median_ms']:.2f} ms / "
        f"{baseline_summary['p95_ms']:.2f} ms"
    )
    print(
        f"- Instructor median/p95: {instructor_summary['median_ms']:.2f} ms / "
        f"{instructor_summary['p95_ms']:.2f} ms"
    )
    print(
        f"- Delta median/p95: {median_delta:.2f}% / {p95_delta:.2f}% "
        "(positive means slower with instructor)"
    )
    print(
        f"- Error rates baseline/instructor: "
        f"{baseline_summary['error_rate']:.2%} / {instructor_summary['error_rate']:.2%}"
    )
    if inconclusive:
        print(
            "- Run marked inconclusive due to error-rate threshold breach "
            f"({args.inconclusive_error_rate:.2%})."
        )
    print(f"- Raw artifact: {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
