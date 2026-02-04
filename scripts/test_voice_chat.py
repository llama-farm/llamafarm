#!/usr/bin/env python3
"""
Test script for the Voice Chat WebSocket API.

Sends an audio file (MP3, WAV, etc.) to the voice chat endpoint and
displays the full response pipeline: STT → LLM → TTS.

Audio is converted to raw PCM 16kHz 16-bit mono for optimal performance.
The Universal Runtime's optimized path processes raw PCM directly without
file I/O or format decoding overhead.

The server automatically detects end-of-speech via VAD (Voice Activity
Detection), so no explicit "end" signal is required. This script streams
audio in chunks followed by trailing silence to trigger VAD.

Usage:
    python scripts/test_voice_chat.py path/to/audio.mp3

Options:
    --namespace, -n   Project namespace (default: "default")
    --project, -p     Project name (default: "sales-agent")
    --host            Server host (default: "localhost:14345")
    --output, -o      Save TTS audio response to file
    --no-convert      Skip ffmpeg conversion (audio must be raw PCM 16kHz mono)

Requirements:
    - pip install websockets
    - ffmpeg (for audio conversion)
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Error: websockets not installed. Run: pip install websockets")
    sys.exit(1)


def convert_audio_to_pcm(input_path: str) -> bytes:
    """Convert audio file to raw PCM 16kHz 16-bit mono using ffmpeg.

    Returns raw PCM data (no headers), which is the most efficient format
    for the Universal Runtime's optimized transcription path.
    """
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",      # 16kHz sample rate
        "-ac", "1",          # mono
        "-f", "s16le",       # Raw PCM 16-bit little-endian (no headers)
        "-",                 # output to stdout
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e.stderr.decode()}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        sys.exit(1)


async def test_voice_chat(
    audio_path: str,
    namespace: str,
    project: str,
    host: str,
    output_path: str | None = None,
    skip_convert: bool = False,
):
    """Send audio to voice chat WebSocket and display responses."""

    # Load and convert audio
    print(f"Loading audio: {audio_path}")
    if skip_convert:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        print(f"  Using raw audio: {len(audio_data):,} bytes")
    else:
        audio_data = convert_audio_to_pcm(audio_path)
        # Raw PCM: 16kHz, 16-bit mono = 2 bytes per sample
        duration_sec = len(audio_data) / (16000 * 2)
        print(f"  Converted to PCM: {len(audio_data):,} bytes ({duration_sec:.1f}s)")

    # Connect to WebSocket
    uri = f"ws://{host}/v1/{namespace}/{project}/voice/chat"
    print(f"\nConnecting to: {uri}")

    tts_chunks = []

    try:
        async with websockets.connect(uri) as ws:
            # Receive session info
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"\n✓ Session: {data.get('session_id', 'unknown')}")

            # Receive initial status
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"✓ Status: {data.get('state', 'unknown')}")

            # Stream audio in chunks (simulates real-time streaming)
            # VAD detects end-of-speech when silence follows audio
            chunk_size = 3200  # 100ms at 16kHz 16-bit mono
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            print(f"\n→ Streaming audio ({len(audio_data):,} bytes in {total_chunks} chunks)...")

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await ws.send(chunk)
                # Small delay to simulate real-time (not necessary but more realistic)
                await asyncio.sleep(0.05)

            # Append ~1 second of silence so VAD detects end-of-speech
            # PCM 16kHz 16-bit mono = 32000 bytes per second
            silence = b"\x00" * 32000
            print("→ Sending trailing silence for VAD...")
            for i in range(0, len(silence), chunk_size):
                chunk = silence[i:i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.05)

            # Receive responses
            print("\n--- Responses ---\n")

            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=60.0)

                    if isinstance(msg, bytes):
                        # TTS audio chunk
                        tts_chunks.append(msg)
                        print(f"← [TTS Audio] {len(msg):,} bytes")
                    else:
                        data = json.loads(msg)
                        msg_type = data.get("type", "unknown")

                        if msg_type == "transcription":
                            text = data.get("text", "")
                            is_final = data.get("is_final", False)
                            marker = "✓" if is_final else "..."
                            print(f"← [STT {marker}] {text}")

                        elif msg_type == "llm_text":
                            text = data.get("text", "")
                            is_final = data.get("is_final", False)
                            marker = "✓" if is_final else "..."
                            print(f"← [LLM {marker}] {text}")

                        elif msg_type == "tts_start":
                            idx = data.get("phrase_index", 0)
                            print(f"← [TTS Start] phrase {idx}")

                        elif msg_type == "tts_done":
                            idx = data.get("phrase_index", 0)
                            duration = data.get("duration", 0)
                            print(f"← [TTS Done] phrase {idx} ({duration:.2f}s)")

                        elif msg_type == "status":
                            state = data.get("state", "unknown")
                            print(f"← [Status] {state}")
                            # If we're back to idle, we're done
                            if state == "idle":
                                break

                        elif msg_type == "error":
                            print(f"← [Error] {data.get('message', 'unknown')}")
                            break

                        elif msg_type == "closed":
                            print("← [Closed]")
                            break

                        else:
                            print(f"← [{msg_type}] {data}")

            except TimeoutError:
                print("\n(Timeout waiting for response)")
            except websockets.ConnectionClosed:
                print("\n(Connection closed)")

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"Connection failed: {e}")
        return
    except ConnectionRefusedError:
        print(f"Connection refused. Is the server running at {host}?")
        return

    # Save TTS audio if requested
    if output_path and tts_chunks:
        total_audio = b"".join(tts_chunks)
        # TTS output is PCM 24kHz 16-bit mono
        with open(output_path, "wb") as f:
            f.write(total_audio)
        duration_sec = len(total_audio) / (24000 * 2)
        print(f"\n✓ Saved TTS audio: {output_path} ({len(total_audio):,} bytes, {duration_sec:.1f}s)")
        print(f"  Play with: ffplay -f s16le -ar 24000 -ac 1 {output_path}")
    elif tts_chunks:
        total_audio = b"".join(tts_chunks)
        duration_sec = len(total_audio) / (24000 * 2)
        print(f"\n✓ Received {len(tts_chunks)} TTS chunks ({len(total_audio):,} bytes, {duration_sec:.1f}s)")
        print("  Use --output/-o to save audio")


def main():
    parser = argparse.ArgumentParser(
        description="Test Voice Chat WebSocket API with an audio file"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (MP3, WAV, etc.)"
    )
    parser.add_argument(
        "-n", "--namespace",
        default="default",
        help="Project namespace (default: default)"
    )
    parser.add_argument(
        "-p", "--project",
        default="sales-agent",
        help="Project name (default: sales-agent)"
    )
    parser.add_argument(
        "--host",
        default="localhost:14345",
        help="Server host:port (default: localhost:14345)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Save TTS response audio to file (raw PCM 24kHz mono)"
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip ffmpeg conversion (audio must already be raw PCM 16kHz 16-bit mono)"
    )

    args = parser.parse_args()

    # Validate input file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    # Run test
    asyncio.run(test_voice_chat(
        audio_path=str(audio_path),
        namespace=args.namespace,
        project=args.project,
        host=args.host,
        output_path=args.output,
        skip_convert=args.no_convert,
    ))


if __name__ == "__main__":
    main()
