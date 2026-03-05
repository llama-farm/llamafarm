"""Tests for gpu_allocator module."""

from unittest.mock import patch

import pytest

from utils.gpu_allocator import (
    SPLIT_MODE_LAYER,
    SPLIT_MODE_NONE,
    GPUDevice,
    InsufficientVRAMError,
    allocate_gpu,
    enumerate_gpus,
    estimate_model_vram,
    get_llama_gpu_params,
)


class TestEnumerateGpus:
    """Tests for enumerate_gpus function."""

    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.is_available")
    def test_two_gpus(self, mock_available, mock_count, mock_mem_info, mock_name):
        """Test enumerating two CUDA GPUs."""
        mock_available.return_value = True
        mock_count.return_value = 2
        mock_mem_info.side_effect = [
            (20 * 1024**3, 24 * 1024**3),  # GPU 0: 20GB free / 24GB total
            (6 * 1024**3, 8 * 1024**3),  # GPU 1: 6GB free / 8GB total
        ]
        mock_name.side_effect = [
            "NVIDIA GeForce RTX 3090",
            "NVIDIA Quadro M4000",
        ]

        gpus = enumerate_gpus()

        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[0].name == "NVIDIA GeForce RTX 3090"
        assert gpus[0].free_vram == 20 * 1024**3
        assert gpus[0].total_vram == 24 * 1024**3
        assert gpus[1].index == 1
        assert gpus[1].free_vram == 6 * 1024**3

    @patch("torch.cuda.is_available")
    def test_no_cuda(self, mock_available):
        """Test when CUDA is not available."""
        mock_available.return_value = False

        gpus = enumerate_gpus()
        assert gpus == []

    def test_no_torch(self):
        """Test when torch is not installed."""
        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("builtins.__import__", side_effect=ImportError("No torch")),
        ):
            gpus = enumerate_gpus()
            assert gpus == []


class TestEstimateModelVram:
    """Tests for estimate_model_vram function."""

    def test_cpu_only(self):
        """Test that CPU-only (n_gpu_layers=0) returns 0."""
        result = estimate_model_vram(
            model_size_bytes=4 * 1024**3,
            n_ctx=4096,
            n_gpu_layers=0,
        )
        assert result == 0

    def test_full_offload_with_architecture(self):
        """Test full GPU offload with architecture metadata (Qwen3-4B)."""
        model_size = int(2.38 * 1024**3)  # 2.38 GB
        n_ctx = 8192

        result = estimate_model_vram(
            model_size_bytes=model_size,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            n_layer=36,
            n_head_kv=8,
            head_k_size=128,
            head_v_size=128,
        )

        # weights = 2.38 GB
        # KV bytes/token = 36 * 8 * (128+128) * 2 = 147,456
        # KV cache = 147,456 * 8192 = ~1.13 GB
        # total = (2.38 + 1.13) * 1.2 = ~4.21 GB
        assert result > 4 * 1024**3
        assert result < 5 * 1024**3

    def test_full_offload_fallback_kv(self):
        """Test full offload with fallback KV estimate (no architecture)."""
        model_size = 4 * 1024**3

        result = estimate_model_vram(
            model_size_bytes=model_size,
            n_ctx=4096,
            n_gpu_layers=999,
        )

        # weights = 4 GB
        # KV fallback = 256 * 1024 * 4096 = 1 GB
        # total = (4 + 1) * 1.2 = 6 GB
        assert result == int((4 * 1024**3 + 256 * 1024 * 4096) * 1.2)

    def test_partial_offload(self):
        """Test partial layer offload scales weight size."""
        model_size = 10 * 1024**3  # 10 GB model

        result = estimate_model_vram(
            model_size_bytes=model_size,
            n_ctx=2048,
            n_gpu_layers=20,
            total_layers=40,  # 50% of layers
        )

        # weights = 10 * (20/40) = 5 GB
        # KV = 256KB * 2048 = 512 MB
        # total = (5 + 0.5) * 1.2 ≈ 6.6 GB
        expected_weights = int(model_size * (20 / 40))
        expected_kv = 256 * 1024 * 2048
        expected = int((expected_weights + expected_kv) * 1.2)
        assert result == expected


class TestAllocateGpu:
    """Tests for allocate_gpu function."""

    def test_single_gpu_fits(self):
        """Test allocation when single GPU has enough VRAM."""
        gpus = [
            GPUDevice(
                index=0,
                name="RTX 3090",
                total_vram=24 * 1024**3,
                free_vram=20 * 1024**3,
            ),
            GPUDevice(
                index=1,
                name="M4000",
                total_vram=8 * 1024**3,
                free_vram=6 * 1024**3,
            ),
        ]

        allocation = allocate_gpu(estimated_vram=10 * 1024**3, gpus=gpus)

        assert allocation.split_mode == SPLIT_MODE_NONE
        assert allocation.main_gpu == 0
        assert allocation.gpu_index == 0
        assert allocation.tensor_split is None

    def test_single_gpu_picks_best(self):
        """Test that the GPU with most free VRAM is selected."""
        gpus = [
            GPUDevice(
                index=0,
                name="RTX 3090",
                total_vram=24 * 1024**3,
                free_vram=5 * 1024**3,
            ),
            GPUDevice(
                index=1,
                name="RTX 4090",
                total_vram=24 * 1024**3,
                free_vram=20 * 1024**3,
            ),
        ]

        allocation = allocate_gpu(estimated_vram=10 * 1024**3, gpus=gpus)

        assert allocation.split_mode == SPLIT_MODE_NONE
        assert allocation.main_gpu == 1
        assert allocation.gpu_index == 1

    def test_multi_gpu_split_when_no_single_fits(self):
        """Test fallback to multi-GPU split when no single GPU fits."""
        gpus = [
            GPUDevice(
                index=0,
                name="GPU0",
                total_vram=12 * 1024**3,
                free_vram=8 * 1024**3,
            ),
            GPUDevice(
                index=1,
                name="GPU1",
                total_vram=12 * 1024**3,
                free_vram=8 * 1024**3,
            ),
        ]

        # 12 GB needed - no single GPU has enough (8GB * 1.1 = 8.8 < 12*1.1=13.2)
        # But combined 16 GB > 13.2 GB
        allocation = allocate_gpu(estimated_vram=12 * 1024**3, gpus=gpus)

        assert allocation.split_mode == SPLIT_MODE_LAYER
        assert allocation.tensor_split is not None
        assert len(allocation.tensor_split) == 2
        # Equal free VRAM -> equal split
        assert abs(allocation.tensor_split[0] - 0.5) < 0.01

    def test_multi_gpu_proportional_split(self):
        """Test that multi-GPU split is proportional to free VRAM."""
        gpus = [
            GPUDevice(
                index=0,
                name="GPU0",
                total_vram=24 * 1024**3,
                free_vram=18 * 1024**3,
            ),
            GPUDevice(
                index=1,
                name="GPU1",
                total_vram=8 * 1024**3,
                free_vram=6 * 1024**3,
            ),
        ]

        allocation = allocate_gpu(estimated_vram=20 * 1024**3, gpus=gpus)

        assert allocation.split_mode == SPLIT_MODE_LAYER
        assert allocation.tensor_split is not None
        # GPU0 has 18GB, GPU1 has 6GB -> 75%/25% split
        assert abs(allocation.tensor_split[0] - 0.75) < 0.01
        assert abs(allocation.tensor_split[1] - 0.25) < 0.01

    def test_insufficient_vram_raises(self):
        """Test InsufficientVRAMError when no config can fit the model."""
        gpus = [
            GPUDevice(
                index=0,
                name="GPU0",
                total_vram=8 * 1024**3,
                free_vram=4 * 1024**3,
            ),
            GPUDevice(
                index=1,
                name="GPU1",
                total_vram=4 * 1024**3,
                free_vram=2 * 1024**3,
            ),
        ]

        with pytest.raises(InsufficientVRAMError, match="Insufficient GPU memory"):
            allocate_gpu(estimated_vram=10 * 1024**3, gpus=gpus)

    def test_no_gpus_raises(self):
        """Test InsufficientVRAMError when no GPUs available."""
        with pytest.raises(InsufficientVRAMError, match="No GPUs available"):
            allocate_gpu(estimated_vram=1 * 1024**3, gpus=[])

    def test_multi_gpu_excludes_tiny_gpus(self):
        """Test that GPUs with very little free VRAM are excluded from split."""
        gpus = [
            GPUDevice(
                index=0,
                name="Big GPU",
                total_vram=24 * 1024**3,
                free_vram=10 * 1024**3,
            ),
            GPUDevice(
                index=1,
                name="Tiny GPU",
                total_vram=4 * 1024**3,
                free_vram=256 * 1024**2,  # 256 MiB
            ),
        ]

        # 10 GB needed - GPU0 alone can't fit (10 * 1.1 = 11 > 10),
        # GPU1 is too small to participate (< 512 MiB),
        # so combined viable is only 10 GB which is < 11 GB required
        with pytest.raises(InsufficientVRAMError):
            allocate_gpu(estimated_vram=10 * 1024**3, gpus=gpus)

    def test_multi_gpu_prunes_infeasible_device(self):
        """Test that a GPU passing min_participation but failing per-device
        feasibility is pruned from the split, leaving the remaining GPUs
        to handle it.
        """
        gpus = [
            GPUDevice(
                index=0,
                name="Big GPU A",
                total_vram=24 * 1024**3,
                free_vram=10 * 1024**3,
            ),
            GPUDevice(
                index=1,
                name="Big GPU B",
                total_vram=24 * 1024**3,
                free_vram=8 * 1024**3,
            ),
            GPUDevice(
                index=2,
                name="Near-full GPU",
                total_vram=8 * 1024**3,
                free_vram=700 * 1024**2,  # 700 MiB - above 512 MiB threshold
            ),
        ]

        # 16 GiB estimated → required = 16 * 1.1 = 17.6 GiB
        # All three combined: 10 + 8 + 0.68 = 18.68 GiB >= 17.6 ✓
        # GPU 2's fraction: 700/(10240+8192+700) ≈ 3.7%
        # GPU 2's share: 3.7% * 17.6 GiB ≈ 650 MiB
        # GPU 2's share + 256 MiB overhead ≈ 906 MiB > 700 MiB → pruned
        # After pruning: GPU 0 (10) + GPU 1 (8) = 18 GiB >= 17.6 GiB ✓
        allocation = allocate_gpu(estimated_vram=16 * 1024**3, gpus=gpus)

        assert allocation.split_mode == SPLIT_MODE_LAYER
        # GPU 2 should be excluded (zero in tensor_split)
        assert allocation.tensor_split[2] == 0.0
        # GPUs 0 and 1 should have non-zero splits
        assert allocation.tensor_split[0] > 0.0
        assert allocation.tensor_split[1] > 0.0

    def test_rtx3090_m4000_scenario(self):
        """Test the exact RTX 3090 + Quadro M4000 multi-model scenario.

        Scenario: First model (20B) consumed ~9.4GB on GPU0 + ~5GB on GPU1.
        Second model (4B, ~2.38GB) should fit on GPU0 alone.
        """
        gpus = [
            GPUDevice(
                index=0,
                name="RTX 3090",
                total_vram=int(23.69 * 1024**3),
                free_vram=int(14.3 * 1024**3),  # After first model
            ),
            GPUDevice(
                index=1,
                name="Quadro M4000",
                total_vram=int(7.93 * 1024**3),
                free_vram=int(3.1 * 1024**3),  # After first model
            ),
        ]

        # Qwen3-4B: ~2.38 GB weights + ~1.13 GB KV + overhead ≈ 4.21 GB
        estimated = int(4.21 * 1024**3)

        allocation = allocate_gpu(estimated_vram=estimated, gpus=gpus)

        # Should fit on GPU 0 alone (14.3 GB free > 4.21 * 1.1 = 4.63 GB)
        assert allocation.split_mode == SPLIT_MODE_NONE
        assert allocation.main_gpu == 0
        assert allocation.gpu_index == 0
        assert allocation.tensor_split is None


class TestGetLlamaGpuParams:
    """Tests for get_llama_gpu_params convenience function."""

    @patch("utils.gpu_allocator.enumerate_gpus")
    def test_single_gpu_returns_params(self, mock_enumerate):
        """Test that params dict is returned for single GPU allocation."""
        mock_enumerate.return_value = [
            GPUDevice(
                index=0,
                name="RTX 3090",
                total_vram=24 * 1024**3,
                free_vram=20 * 1024**3,
            ),
        ]

        result = get_llama_gpu_params(
            model_size_bytes=4 * 1024**3,
            n_ctx=4096,
            n_gpu_layers=-1,
        )

        assert "main_gpu" in result
        assert "split_mode" in result
        assert result["split_mode"] == SPLIT_MODE_NONE
        assert result["main_gpu"] == 0
        assert "tensor_split" not in result

    @patch("utils.gpu_allocator.enumerate_gpus")
    def test_no_cuda_returns_empty(self, mock_enumerate):
        """Test that empty dict is returned when no CUDA GPUs available."""
        mock_enumerate.return_value = []

        result = get_llama_gpu_params(
            model_size_bytes=4 * 1024**3,
            n_ctx=4096,
            n_gpu_layers=-1,
        )

        assert result == {}

    def test_cpu_only_returns_empty(self):
        """Test that CPU-only (n_gpu_layers=0) returns empty dict."""
        result = get_llama_gpu_params(
            model_size_bytes=4 * 1024**3,
            n_ctx=4096,
            n_gpu_layers=0,
        )

        assert result == {}

    @patch("utils.gpu_allocator.enumerate_gpus")
    def test_insufficient_vram_raises(self, mock_enumerate):
        """Test that InsufficientVRAMError propagates."""
        mock_enumerate.return_value = [
            GPUDevice(
                index=0,
                name="GPU",
                total_vram=4 * 1024**3,
                free_vram=2 * 1024**3,
            ),
        ]

        with pytest.raises(InsufficientVRAMError):
            get_llama_gpu_params(
                model_size_bytes=10 * 1024**3,
                n_ctx=8192,
                n_gpu_layers=-1,
            )

    @patch("utils.gpu_allocator.enumerate_gpus")
    def test_multi_gpu_returns_tensor_split(self, mock_enumerate):
        """Test that multi-GPU allocation includes tensor_split."""
        mock_enumerate.return_value = [
            GPUDevice(
                index=0,
                name="GPU0",
                total_vram=12 * 1024**3,
                free_vram=8 * 1024**3,
            ),
            GPUDevice(
                index=1,
                name="GPU1",
                total_vram=12 * 1024**3,
                free_vram=8 * 1024**3,
            ),
        ]

        result = get_llama_gpu_params(
            model_size_bytes=10 * 1024**3,
            n_ctx=4096,
            n_gpu_layers=-1,
        )

        assert result["split_mode"] == SPLIT_MODE_LAYER
        assert "tensor_split" in result
        assert len(result["tensor_split"]) == 2
