"""Service for cleaning up processed chunks after cancellation."""

from typing import Any

from core.celery.rag_client import delete_file_from_rag
from core.logging import FastAPIStructLogger
from services.project_service import ProjectService

logger = FastAPIStructLogger(__name__)


class DatasetCleanupService:
    """Service for cleaning up processed chunks after cancellation."""

    async def cleanup_processed_files(
        self,
        namespace: str,
        project: str,
        dataset: str,
        task_id: str,
    ) -> dict[str, Any]:
        """
        Remove chunks for successfully processed files.

        Uses Celery task via rag_client to delete chunks

        Args:
            namespace: Project namespace
            project: Project name
            dataset: Dataset name
            task_id: The cancelled task ID

        Returns:
            Dict with cleanup results:
            {
                "files_reverted": int,
                "files_failed_to_revert": int,
                "errors": [{"file_hash": str, "error": str}]
            }
        """
        logger.info(
            "Starting cleanup for processed files",
            namespace=namespace,
            project=project,
            dataset=dataset,
            task_id=task_id,
        )

        files_reverted = 0
        files_failed = 0
        errors: list[dict[str, str]] = []

        try:
            # Get task status to find successfully processed files
            successful_files = self._get_successful_files(task_id)

            if not successful_files:
                logger.info("No successfully processed files to clean up")
                return {
                    "files_reverted": 0,
                    "files_failed_to_revert": 0,
                    "errors": None,
                }

            # Get project configuration
            project_obj = ProjectService.get_project(namespace, project)
            project_dir = ProjectService.get_project_dir(namespace, project)

            # Find dataset config to get database name
            dataset_config = next(
                (ds for ds in (project_obj.config.datasets or []) if ds.name == dataset),
                None,
            )

            if not dataset_config:
                raise ValueError(f"Dataset '{dataset}' not found in project")

            database_name = dataset_config.database
            if not database_name:
                raise ValueError(f"Dataset '{dataset}' has no database configured")

            # Delete chunks for each successful file via Celery task
            for file_hash in successful_files:
                try:
                    logger.info(f"Deleting chunks for file: {file_hash[:12]}...")

                    # Use delete_file_from_rag from rag_client
                    result = await delete_file_from_rag(
                        project_dir=project_dir,
                        database_name=database_name,
                        file_hash=file_hash,
                    )

                    if result.get("status") == "error":
                        raise Exception(result.get("error", "Unknown error"))

                    deleted_count = result.get("deleted_count", 0)
                    logger.info(
                        f"Successfully deleted chunks for file {file_hash[:12]}...",
                        deleted_count=deleted_count,
                    )
                    files_reverted += 1

                except Exception as e:
                    logger.error(
                        f"Failed to delete chunks for {file_hash[:12]}...: {e}",
                        exc_info=True,
                    )
                    files_failed += 1
                    errors.append({"file_hash": file_hash, "error": str(e)})
                    # Continue with other files - don't fail entire cleanup

            logger.info(
                "Cleanup completed",
                files_reverted=files_reverted,
                files_failed=files_failed,
            )

            return {
                "files_reverted": files_reverted,
                "files_failed_to_revert": files_failed,
                "errors": errors if errors else None,
            }

        except Exception as e:
            logger.error(
                f"Error during cleanup: {e}",
                exc_info=True,
            )
            # Return partial results even if there's an error
            return {
                "files_reverted": files_reverted,
                "files_failed_to_revert": files_failed,
                "errors": errors if errors else [{"file_hash": "unknown", "error": str(e)}],
            }

    def _get_successful_files(self, task_id: str) -> list[str]:
        """
        Get list of file hashes for successfully processed files.

        Args:
            task_id: The task ID to check

        Returns:
            List of file hashes that were successfully processed
        """
        from celery.result import AsyncResult

        from core.celery import app

        successful_files = []

        try:
            # Get group task
            group_result: AsyncResult = app.AsyncResult(task_id)

            # Get stored metadata
            result_meta = None
            try:
                if group_result.state == "PENDING" and isinstance(group_result.result, dict) and group_result.result.get("type") == "group":
                    result_meta = group_result.result
            except Exception:
                pass

            if not result_meta:
                try:
                    result_value = group_result.result
                    if isinstance(result_value, dict) and result_value.get("type") == "group":
                        result_meta = result_value
                except Exception:
                    pass

            if not result_meta or result_meta.get("type") != "group":
                logger.warning(f"Could not get group metadata for task {task_id}")
                return []

            child_task_ids = result_meta.get("children", [])
            file_hashes = result_meta.get("file_hashes", [])

            # Check each child task to see if it was successful
            for i, child_id in enumerate(child_task_ids):
                try:
                    child_result = app.AsyncResult(child_id)

                    # Get child task state
                    try:
                        child_state = child_result.state
                    except Exception:
                        continue

                    # If successful, get the file hash
                    if child_state == "SUCCESS":
                        try:
                            result_data = child_result.result
                            if isinstance(result_data, dict):
                                # Get file_hash from result
                                file_hash = result_data.get("file_hash")
                                if not file_hash and i < len(file_hashes):
                                    file_hash = file_hashes[i]

                                if file_hash:
                                    # Check if it was actually processed (not skipped)
                                    details = result_data.get("details", {})
                                    status = details.get("status")
                                    if status != "skipped":
                                        successful_files.append(file_hash)
                        except Exception as e:
                            logger.debug(f"Error getting result for child {child_id}: {e}")
                            # Fallback: use file_hash from list if available
                            if i < len(file_hashes):
                                successful_files.append(file_hashes[i])

                except Exception as e:
                    logger.debug(f"Error checking child task {child_id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error getting successful files: {e}", exc_info=True)

        return successful_files
