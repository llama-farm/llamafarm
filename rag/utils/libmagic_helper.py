"""Helper for detecting and installing libmagic."""

import subprocess
import sys
import platform
import logging

logger = logging.getLogger(__name__)


def check_libmagic():
    """Check if libmagic is available and working."""
    try:
        import magic
        # Try to actually use it
        magic.Magic()
        return True
    except ImportError:
        logger.warning("python-magic not installed. Install with: uv pip install python-magic")
        return False
    except Exception as e:
        if "failed to find libmagic" in str(e).lower():
            handle_missing_libmagic()
        else:
            logger.warning(f"python-magic error: {e}")
        return False


def handle_missing_libmagic():
    """Handle missing libmagic library."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        logger.warning("=" * 60)
        logger.warning("libmagic not found! This is required for file type detection.")
        logger.warning("")
        logger.warning("To fix this on macOS, run:")
        logger.warning("  brew install libmagic")
        logger.warning("")
        logger.warning("If you don't have Homebrew, install it first from https://brew.sh")
        logger.warning("=" * 60)
        
        # Try to auto-install if brew is available
        if check_brew_available():
            response = input("\nWould you like to install libmagic automatically? [y/N]: ")
            if response.lower() == 'y':
                if install_libmagic_macos():
                    logger.info("Installation complete. Please restart your script.")
                    raise SystemExit(0)  # More explicit than sys.exit()
    
    elif system == "Linux":
        logger.warning("=" * 60)
        logger.warning("libmagic not found! This is required for file type detection.")
        logger.warning("")
        logger.warning("To fix this on Linux, run one of:")
        logger.warning("  Ubuntu/Debian: sudo apt-get install libmagic1")
        logger.warning("  Fedora/RedHat: sudo dnf install file-libs")
        logger.warning("  Arch: sudo pacman -S file")
        logger.warning("=" * 60)
    
    elif system == "Windows":
        logger.warning("=" * 60)
        logger.warning("libmagic not found! This is required for file type detection.")
        logger.warning("")
        logger.warning("To fix this on Windows:")
        logger.warning("  1. Download python-magic-bin: pip install python-magic-bin")
        logger.warning("  OR")
        logger.warning("  2. Install from https://github.com/pidydx/libmagicwin64")
        logger.warning("=" * 60)


def check_brew_available():
    """Check if Homebrew is available on macOS."""
    try:
        result = subprocess.run(['which', 'brew'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def install_libmagic_macos():
    """Attempt to install libmagic on macOS using Homebrew.
    
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        logger.info("Installing libmagic via Homebrew...")
        result = subprocess.run(['brew', 'install', 'libmagic'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ libmagic installed successfully!")
            logger.info("Please restart your Python script to use the new installation.")
            return True
        else:
            logger.error(f"Failed to install libmagic: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error installing libmagic: {e}")
        return False


# Check on import
_libmagic_available = check_libmagic()


def is_libmagic_available():
    """Check if libmagic is available."""
    return _libmagic_available