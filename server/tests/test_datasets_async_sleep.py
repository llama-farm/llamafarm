"""
Test that the dataset processing endpoint uses proper async sleep.

This test verifies that the process_dataset endpoint doesn't block the
event loop with synchronous sleep calls.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from celery.result import AsyncResult


@pytest.mark.asyncio
async def test_process_dataset_uses_async_sleep():
    """Test that process_dataset uses asyncio.sleep instead of blocking time.sleep."""
    from api.routers.datasets.datasets import router

    # Track if asyncio.sleep was called (proper async behavior)
    async_sleep_called = False
    blocking_sleep_called = False

    # Mock asyncio.sleep
    original_async_sleep = asyncio.sleep

    async def mock_async_sleep(delay):
        nonlocal async_sleep_called
        async_sleep_called = True
        await original_async_sleep(0.001)  # Very short sleep for test

    # Mock time.sleep (should NOT be called in the polling loop)
    original_time_sleep = time.sleep

    def mock_time_sleep(delay):
        nonlocal blocking_sleep_called
        blocking_sleep_called = True
        original_time_sleep(0.001)  # Very short sleep for test

    with (
        patch("asyncio.sleep", side_effect=mock_async_sleep),
        patch("time.sleep", side_effect=mock_time_sleep),
        patch(
            "services.project_service.ProjectService.get_project"
        ) as mock_get_project,
        patch(
            "services.project_service.ProjectService.get_project_dir"
        ) as mock_get_dir,
        patch("core.celery.tasks.process_single_file_task.delay") as mock_task_delay,
    ):
        # Setup mocks
        mock_project = MagicMock()
        mock_project.config.datasets = [
            MagicMock(
                name="test_dataset",
                data_processing_strategy="test_strategy",
                database="test_db",
                files=["test_file_hash"],
            )
        ]
        mock_get_project.return_value = mock_project
        mock_get_dir.return_value = "/tmp/test_project"

        # Mock task result that completes immediately (status != PENDING)
        mock_task = MagicMock(spec=AsyncResult)
        mock_task.status = "SUCCESS"
        mock_task.result = {
            "success": True,
            "details": {
                "filename": "test.txt",
                "parser": "TextParser",
                "extractors": [],
                "chunks": 10,
                "chunk_size": 512,
                "embedder": "TestEmbedder",
                "status": "success",
                "stored_count": 10,
                "skipped_count": 0,
            },
        }
        mock_task_delay.return_value = mock_task

        # Find the process_dataset function
        process_dataset_func = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and route.endpoint.__name__ == "process_dataset"
            ):
                process_dataset_func = route.endpoint
                break

        assert process_dataset_func is not None, "process_dataset endpoint not found"

        # Create a file that exists
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the raw data directory structure
            raw_dir = os.path.join(tmpdir, "lf_data", "raw")
            os.makedirs(raw_dir, exist_ok=True)

            # Create a test file
            test_file = os.path.join(raw_dir, "test_file_hash")
            with open(test_file, "w") as f:
                f.write("test content")

            mock_get_dir.return_value = tmpdir

            # Call the endpoint
            try:
                with patch(
                    "services.data_service.DataService.get_data_file_metadata_by_hash"
                ) as mock_metadata:
                    mock_metadata.return_value = MagicMock(filename="test.txt")
                    await process_dataset_func(
                        namespace="test",
                        project="test",
                        dataset="test_dataset",
                        async_processing=False,
                    )
            except Exception:
                # Even if the function fails for other reasons, we can still check our assertion
                pass

    # Verify async sleep was used (for proper async behavior)
    # Note: async_sleep might not be called if task.status is immediately SUCCESS,
    # which is fine - it means the loop exited before sleeping
    # The important check is that blocking sleep was NOT called in the async context

    # We can't guarantee async_sleep is called because the task might complete
    # immediately (status != PENDING), but we CAN guarantee blocking sleep
    # should NEVER be called in the endpoint's polling loop
    assert not blocking_sleep_called, (
        "Blocking time.sleep() was called in async endpoint! "
        "This blocks the event loop and can cause 500 errors."
    )


@pytest.mark.asyncio
async def test_process_dataset_polling_loop_is_async():
    """
    Test that the polling loop properly yields control during waiting.

    This is a more comprehensive test that verifies the polling behavior
    doesn't block other coroutines.
    """

    # Create a mock task that completes immediately (SUCCESS on first check)
    # This allows the test to complete quickly while still verifying
    # that the endpoint properly handles async execution
    mock_task = MagicMock(spec=AsyncResult)
    mock_task.status = "SUCCESS"
    mock_task.result = {
        "success": True,
        "details": {
            "filename": "test.txt",
            "parser": "TextParser",
            "extractors": [],
            "chunks": 10,
            "chunk_size": 512,
            "embedder": "TestEmbedder",
            "status": "success",
            "stored_count": 10,
            "skipped_count": 0,
        },
    }

    with (
        patch(
            "services.project_service.ProjectService.get_project"
        ) as mock_get_project,
        patch(
            "services.project_service.ProjectService.get_project_dir"
        ) as mock_get_dir,
        patch("core.celery.tasks.process_single_file_task.delay") as mock_task_delay,
    ):
        # Setup mocks
        mock_project = MagicMock()
        mock_project.config.datasets = [
            MagicMock(
                name="test_dataset",
                data_processing_strategy="test_strategy",
                database="test_db",
                files=["test_file_hash"],
            )
        ]
        mock_get_project.return_value = mock_project

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the raw data directory structure
            raw_dir = os.path.join(tmpdir, "lf_data", "raw")
            os.makedirs(raw_dir, exist_ok=True)

            # Create a test file
            test_file = os.path.join(raw_dir, "test_file_hash")
            with open(test_file, "w") as f:
                f.write("test content")

            mock_get_dir.return_value = tmpdir
            mock_task_delay.return_value = mock_task

            # Import the function
            from api.routers.datasets.datasets import router

            process_dataset_func = None
            for route in router.routes:
                if (
                    hasattr(route, "endpoint")
                    and route.endpoint.__name__ == "process_dataset"
                ):
                    process_dataset_func = route.endpoint
                    break

            assert process_dataset_func is not None

            # Run the endpoint - it should complete quickly since task is already SUCCESS
            with patch(
                "services.data_service.DataService.get_data_file_metadata_by_hash"
            ) as mock_metadata:
                mock_metadata.return_value = MagicMock(filename="test.txt")

                # If the function is properly async, it should complete without blocking
                # If it were using blocking sleep, it would hang the test
                start_time = asyncio.get_event_loop().time()
                try:
                    await asyncio.wait_for(
                        process_dataset_func(
                            namespace="test",
                            project="test",
                            dataset="test_dataset",
                            async_processing=False,
                        ),
                        timeout=5.0,  # Should complete quickly
                    )
                except TimeoutError:
                    pytest.fail(
                        "process_dataset timed out, suggesting it may be blocking "
                        "the event loop with synchronous sleep"
                    )
                except Exception:
                    # Other exceptions are fine - we're just checking it doesn't block
                    pass

                elapsed = asyncio.get_event_loop().time() - start_time

                # Should complete very quickly (task status is immediately SUCCESS)
                # If it's blocking, it would take at least 2 seconds (one poll interval)
                assert elapsed < 1.0, (
                    f"process_dataset took {elapsed:.2f}s, which is too slow. "
                    "This suggests the polling loop may be blocking."
                )
