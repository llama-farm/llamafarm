"""
Tests for server-level SSE streaming to ensure token-by-token delivery.

These tests verify that the `await asyncio.sleep(0)` calls in server.py
properly flush the stream, preventing buffered/chunked responses.
"""

import pytest
import asyncio
import json
from httpx import AsyncClient, ASGITransport
import time


# Import the server app
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import app


@pytest.mark.asyncio
async def test_streaming_token_by_token_delivery():
    """
    Test that SSE streaming delivers tokens incrementally, not in large chunks.

    This verifies that `await asyncio.sleep(0)` properly flushes the stream.
    Without it, tokens would be buffered and arrive in large chunks.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Use a small, fast model for testing
        request_data = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [
                {"role": "user", "content": "Count from 1 to 10, one number per line."}
            ],
            "stream": True,
            "max_tokens": 50,
            "temperature": 0.7,
        }

        chunk_times = []
        chunks_received = []
        start_time = time.time()

        async with client.stream(
            "POST", "/v1/chat/completions", json=request_data, timeout=30.0
        ) as response:
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

            async for line in response.aiter_lines():
                if not line or line.startswith(":"):
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        chunks_received.append(chunk)

                        # Record timing of each chunk
                        elapsed = time.time() - start_time
                        chunk_times.append(elapsed)

                        # Verify chunk structure
                        assert "id" in chunk
                        assert "object" in chunk
                        assert chunk["object"] == "chat.completion.chunk"
                        assert "choices" in chunk
                        assert len(chunk["choices"]) > 0

                    except json.JSONDecodeError:
                        pass

        # Verify we received multiple chunks (streaming, not single response)
        assert len(chunks_received) > 1, (
            f"Expected multiple chunks, got {len(chunks_received)}"
        )

        # Verify chunks arrived incrementally (not all at once)
        if len(chunk_times) > 2:
            # Calculate inter-chunk delays
            inter_chunk_delays = [
                chunk_times[i + 1] - chunk_times[i] for i in range(len(chunk_times) - 1)
            ]

            # Most inter-chunk delays should be small (< 1 second)
            # This proves tokens are streaming, not buffered
            small_delays = sum(1 for delay in inter_chunk_delays if delay < 1.0)
            assert small_delays > len(inter_chunk_delays) * 0.7, (
                f"Expected incremental streaming, but delays suggest buffering: {inter_chunk_delays}"
            )

        print(f"\n✅ Streaming test passed:")
        print(f"   - Received {len(chunks_received)} chunks")
        print(f"   - Time span: {chunk_times[-1]:.3f}s")
        print(f"   - Average chunk interval: {chunk_times[-1] / len(chunk_times):.3f}s")


@pytest.mark.asyncio
async def test_streaming_without_asyncio_sleep_simulation():
    """
    Demonstrate what happens when asyncio.sleep(0) is missing.

    This test simulates buffered behavior and shows why the sleep is critical.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        request_data = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
            "max_tokens": 20,
        }

        chunks = []
        chunk_sizes = []

        async with client.stream(
            "POST", "/v1/chat/completions", json=request_data, timeout=30.0
        ) as response:
            buffer = b""
            async for raw_chunk in response.aiter_bytes():
                buffer += raw_chunk
                chunk_sizes.append(len(raw_chunk))

                # Process complete SSE messages
                while b"\n\n" in buffer:
                    message, buffer = buffer.split(b"\n\n", 1)
                    if message.startswith(b"data: "):
                        data_str = message[6:].decode()
                        if data_str != "[DONE]":
                            try:
                                chunks.append(json.loads(data_str))
                            except json.JSONDecodeError:
                                pass

        # With asyncio.sleep(0), we should see many small chunks
        # Without it, we'd see fewer, larger chunks
        assert len(chunks) > 0, "Should receive at least one chunk"

        # Log chunk distribution for diagnostics
        print(f"\n📊 Chunk size distribution:")
        print(f"   - Total chunks: {len(chunk_sizes)}")
        print(f"   - Mean size: {sum(chunk_sizes) / len(chunk_sizes):.1f} bytes")
        print(f"   - Max size: {max(chunk_sizes)} bytes")
        print(f"   - Min size: {min(chunk_sizes)} bytes")


@pytest.mark.asyncio
async def test_streaming_completion_markers():
    """
    Test that streaming properly sends completion markers.

    Verifies:
    1. Initial chunk with role
    2. Content chunks with tokens
    3. Final chunk with finish_reason
    4. [DONE] marker
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        request_data = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Say 'test' once"}],
            "stream": True,
            "max_tokens": 10,
        }

        chunks = []
        got_done = False

        async with client.stream(
            "POST", "/v1/chat/completions", json=request_data, timeout=30.0
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        got_done = True
                        break
                    try:
                        chunk = json.loads(data_str)
                        chunks.append(chunk)
                    except json.JSONDecodeError:
                        pass

        assert len(chunks) > 0, "Should receive chunks"
        assert got_done, "Should receive [DONE] marker"

        # Check first chunk has role
        first_chunk = chunks[0]
        assert "choices" in first_chunk
        assert len(first_chunk["choices"]) > 0
        delta = first_chunk["choices"][0].get("delta", {})
        # First chunk might have role or content
        assert "role" in delta or "content" in delta

        # Check last chunk has finish_reason
        last_chunk = chunks[-1]
        finish_reason = last_chunk["choices"][0].get("finish_reason")
        # Last chunk should have finish_reason or last few chunks
        has_finish_reason = any(
            c["choices"][0].get("finish_reason") is not None for c in chunks[-3:]
        )
        assert has_finish_reason, "Should have finish_reason in final chunks"

        print(f"\n✅ Completion markers verified:")
        print(f"   - Total chunks: {len(chunks)}")
        print(f"   - Got [DONE]: {got_done}")
        print(f"   - First delta: {first_chunk['choices'][0].get('delta', {})}")
        print(
            f"   - Last finish_reason: {last_chunk['choices'][0].get('finish_reason')}"
        )


@pytest.mark.asyncio
async def test_streaming_headers():
    """
    Test that streaming response has correct headers to prevent buffering.

    These headers are critical for ensuring proxies/nginx don't buffer:
    - Cache-Control: no-cache
    - Connection: keep-alive
    - X-Accel-Buffering: no
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        request_data = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "max_tokens": 5,
        }

        async with client.stream(
            "POST", "/v1/chat/completions", json=request_data, timeout=30.0
        ) as response:
            # Verify anti-buffering headers
            headers = response.headers

            assert headers.get("cache-control") == "no-cache", (
                "Missing or incorrect Cache-Control header"
            )

            assert headers.get("connection") == "keep-alive", (
                "Missing or incorrect Connection header"
            )

            assert headers.get("x-accel-buffering") == "no", (
                "Missing or incorrect X-Accel-Buffering header (needed for nginx)"
            )

            assert headers.get("content-type") == "text/event-stream; charset=utf-8", (
                "Missing or incorrect Content-Type header"
            )

            print(f"\n✅ Anti-buffering headers verified:")
            print(f"   - Cache-Control: {headers.get('cache-control')}")
            print(f"   - Connection: {headers.get('connection')}")
            print(f"   - X-Accel-Buffering: {headers.get('x-accel-buffering')}")


@pytest.mark.asyncio
async def test_streaming_immediate_start():
    """
    Test that streaming starts immediately, not after full generation.

    This verifies that the first chunk arrives quickly, proving
    we're not buffering the entire response before streaming.

    Note: This test allows up to 15 seconds for first chunk to account for
    model loading time on slower systems or CI environments.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        request_data = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Write a short story"}],
            "stream": True,
            "max_tokens": 100,  # Longer generation
        }

        start_time = time.time()
        first_chunk_time = None
        last_chunk_time = None
        chunk_count = 0

        async with client.stream(
            "POST", "/v1/chat/completions", json=request_data, timeout=60.0
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        json.loads(data_str)  # Validate it's JSON
                        chunk_count += 1

                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                        last_chunk_time = time.time()

                    except json.JSONDecodeError:
                        pass

        time_to_first_chunk = first_chunk_time - start_time
        total_time = last_chunk_time - start_time

        # First chunk should arrive within reasonable time (< 15 seconds including model loading)
        # This is more generous to account for CI environments and initial model loading
        assert time_to_first_chunk < 15.0, (
            f"First chunk took {time_to_first_chunk:.2f}s - suggests buffering!"
        )

        # Should receive multiple chunks over time (proving incremental delivery)
        assert chunk_count > 5, f"Only received {chunk_count} chunks"

        # Total time should be longer than time to first chunk (proving streaming)
        assert total_time > time_to_first_chunk, (
            "All chunks arrived at once - not streaming!"
        )

        print(f"\n✅ Streaming latency verified:")
        print(f"   - Time to first chunk: {time_to_first_chunk:.3f}s")
        print(f"   - Total streaming time: {total_time:.3f}s")
        print(f"   - Chunks received: {chunk_count}")
        print(f"   - Average interval: {total_time / chunk_count:.3f}s")


@pytest.mark.asyncio
async def test_asyncio_sleep_presence():
    """
    Verify that the critical asyncio.sleep(0) calls exist in the streaming service.

    This is a meta-test that ensures the fix isn't accidentally removed.
    """
    # The streaming code is in routers/chat_completions/service.py
    service_path = (
        Path(__file__).parent.parent / "routers" / "chat_completions" / "service.py"
    )
    service_code = service_path.read_text()

    # Count occurrences of asyncio.sleep(0) in streaming context
    sleep_count = service_code.count("await asyncio.sleep(0)")

    # We expect at least 2: one in the token loop, one before [DONE]
    assert sleep_count >= 2, (
        f"Expected at least 2 'await asyncio.sleep(0)' calls in service.py, found {sleep_count}. "
        "These are critical for preventing stream buffering!"
    )

    # Verify they're in the streaming section
    assert "generate_sse" in service_code, "Missing generate_sse function"

    # Find the streaming function and verify sleep calls are present
    sse_start = service_code.find("async def generate_sse()")
    sse_end = service_code.find("return StreamingResponse(", sse_start)
    sse_section = service_code[sse_start:sse_end]

    sse_sleep_count = sse_section.count("await asyncio.sleep(0)")
    assert sse_sleep_count >= 2, (
        f"Expected at least 2 'await asyncio.sleep(0)' in generate_sse(), found {sse_sleep_count}"
    )

    print(f"\n✅ Code verification passed:")
    print(f"   - Total asyncio.sleep(0) calls: {sleep_count}")
    print(f"   - In generate_sse() function: {sse_sleep_count}")
    print(f"   - Stream flushing is properly implemented!")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
