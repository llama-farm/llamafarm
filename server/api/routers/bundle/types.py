"""Bundle API types."""

from pydantic import BaseModel, Field

VALID_PLATFORMS = ["linux", "darwin", "windows"]
VALID_ARCHITECTURES = ["x86_64", "arm64"]
VALID_ACCELERATORS = ["cuda", "rocm", "vulkan", "cpu", "metal"]
VALID_ADDONS = {"stt", "tts"}

INVALID_COMBOS = {
    ("darwin", "x86_64"),  # macOS Intel not supported
}

# Metal only on macOS
ACCELERATOR_PLATFORM_RULES = {
    "metal": {"darwin"},
}

PLATFORM_TO_GOOS = {
    "linux": "linux",
    "darwin": "darwin",
    "windows": "windows",
}

# PyApp uses "macos" instead of "darwin" for macOS builds.
PLATFORM_TO_PYAPP_OS = {
    "linux": "linux",
    "darwin": "macos",
    "windows": "windows",
}

ARCH_TO_GOARCH = {
    "x86_64": "amd64",
    "arm64": "arm64",
}

# Rough size estimates in bytes
SIZE_ESTIMATES = {
    "cli": 12_000_000,
    "server": 45_000_000,
    "rag": 40_000_000,
    "runtime": 50_000_000,
    "torch_cuda": 1_200_000_000,
    "torch_rocm": 1_100_000_000,
    "torch_vulkan": 800_000_000,
    "torch_metal": 600_000_000,
    "addon_stt": 200_000_000,
    "addon_tts": 200_000_000,
    "packaging": 0,
}


class BundleRequest(BaseModel):
    platform: str
    arch: str
    accelerator: str
    addons: list[str] = Field(default_factory=list)
    version: str = ""


class BundleManifest(BaseModel):
    id: str
    version: str
    platform: str
    arch: str
    accelerator: str
    components: dict[str, str] = Field(default_factory=dict)
    addons: list[str] = Field(default_factory=list)
    size: int = 0
    filename: str = ""
    created_at: str = ""


class BundleSummary(BaseModel):
    id: str
    version: str
    platform: str
    arch: str
    accelerator: str
    addons: list[str] = Field(default_factory=list)
    size: int = 0
    filename: str = ""
    created_at: str = ""


class BundleEstimate(BaseModel):
    estimated_bytes: int
    components: dict[str, int]
