#!/usr/bin/env python3
"""Download voice samples for Chatterbox TTS voice cloning.

Downloads high-quality voice samples from LibriSpeech (open audiobook recordings).
Each sample is ~10 seconds of clear speech, ideal for voice cloning.

Usage:
    python download_voices.py

Or from the CLI:
    lf tts download-voices
"""

import logging
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# LibriSpeech test-clean samples - curated for voice diversity
# Format: (speaker_id, chapter_id, utterance_id, description)
# These are from dev-clean split for high quality
VOICE_SAMPLES = {
    "cb_male_calm": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "speaker_id": "1272",
        "description": "Male narrator with calm, measured delivery",
    },
    "cb_female_warm": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "speaker_id": "1462",
        "description": "Female narrator with warm, expressive tone",
    },
    "cb_male_energetic": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "speaker_id": "1988",
        "description": "Male speaker with energetic delivery",
    },
    "cb_female_professional": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "speaker_id": "1993",
        "description": "Female speaker with clear, professional tone",
    },
}

# Map voice IDs to output filenames
VOICE_FILES = {
    "cb_male_calm": "male_calm.wav",
    "cb_female_warm": "female_warm.wav",
    "cb_male_energetic": "male_energetic.wav",
    "cb_female_professional": "female_professional.wav",
}


def get_voices_dir() -> Path:
    """Get the voices directory path."""
    return Path(__file__).parent


def download_librispeech_sample(
    speaker_id: str,
    output_path: Path,
    target_duration: float = 12.0,
) -> bool:
    """Download and extract a voice sample from LibriSpeech dev-clean.

    Args:
        speaker_id: LibriSpeech speaker ID
        output_path: Path to save the WAV file
        target_duration: Target duration in seconds (will concatenate if needed)

    Returns:
        True if successful, False otherwise
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        logger.error("soundfile not installed. Run: pip install soundfile")
        return False

    # LibriSpeech dev-clean URL (smaller than full dataset)
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"

    logger.info(f"  Downloading LibriSpeech dev-clean for speaker {speaker_id}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "dev-clean.tar.gz"

        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Downloaded {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent}%)", end="", flush=True)

        try:
            urllib.request.urlretrieve(url, tar_path, reporthook=report_progress)
            print()  # Newline after progress
        except Exception as e:
            logger.error(f"  Failed to download: {e}")
            return False

        # Extract speaker's audio files
        logger.info(f"  Extracting speaker {speaker_id} audio...")
        audio_data = []
        sample_rate = None

        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                # Match speaker's FLAC files
                if f"/{speaker_id}/" in member.name and member.name.endswith(".flac"):
                    f = tar.extractfile(member)
                    if f:
                        # Read FLAC directly from stream
                        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_flac:
                            tmp_flac.write(f.read())
                            tmp_flac_path = tmp_flac.name

                        try:
                            data, sr = sf.read(tmp_flac_path)
                            if sample_rate is None:
                                sample_rate = sr
                            audio_data.append(data)
                        finally:
                            os.unlink(tmp_flac_path)

                        # Check if we have enough audio
                        total_samples = sum(len(d) for d in audio_data)
                        if sample_rate and total_samples / sample_rate >= target_duration:
                            break

        if not audio_data:
            logger.error(f"  No audio found for speaker {speaker_id}")
            return False

        # Concatenate and trim to target duration
        combined = np.concatenate(audio_data)
        target_samples = int(target_duration * sample_rate)
        if len(combined) > target_samples:
            combined = combined[:target_samples]

        # Save as WAV
        sf.write(output_path, combined, sample_rate)
        duration = len(combined) / sample_rate
        logger.info(f"  Saved {output_path.name} ({duration:.1f}s)")

        return True


def download_all_voices(force: bool = False) -> dict[str, bool]:
    """Download all built-in voice samples.

    Args:
        force: If True, re-download even if files exist

    Returns:
        Dict mapping voice IDs to success status
    """
    voices_dir = get_voices_dir()
    results = {}

    # Check which voices need downloading
    voices_to_download = {}
    for voice_id, filename in VOICE_FILES.items():
        output_path = voices_dir / filename
        if not force and output_path.exists():
            logger.info(f"✓ {voice_id} already exists: {filename}")
            results[voice_id] = True
        else:
            voices_to_download[voice_id] = output_path

    if not voices_to_download:
        logger.info("\nAll voices are already downloaded!")
        return results

    # Download missing voices
    logger.info(f"\nDownloading {len(voices_to_download)} voice sample(s)...")
    logger.info("Note: This requires downloading LibriSpeech dev-clean (~340MB)\n")

    # Download LibriSpeech once and extract all needed speakers
    for voice_id, output_path in voices_to_download.items():
        speaker_id = VOICE_SAMPLES[voice_id]["speaker_id"]
        description = VOICE_SAMPLES[voice_id]["description"]

        logger.info(f"Downloading {voice_id}: {description}")
        success = download_librispeech_sample(speaker_id, output_path)
        results[voice_id] = success

        if not success:
            logger.warning(f"  Failed to download {voice_id}")

    # Summary
    successful = sum(1 for v in results.values() if v)
    logger.info(f"\nDownloaded {successful}/{len(VOICE_FILES)} voices")

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download voice samples for Chatterbox TTS"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download even if files exist"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available voices without downloading"
    )

    args = parser.parse_args()

    if args.list:
        print("Available built-in voices for Chatterbox TTS:\n")
        voices_dir = get_voices_dir()
        for voice_id, info in VOICE_SAMPLES.items():
            filename = VOICE_FILES[voice_id]
            exists = (voices_dir / filename).exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {voice_id}: {info['description']}")
            print(f"      File: {filename}")
        return

    download_all_voices(force=args.force)


if __name__ == "__main__":
    main()
