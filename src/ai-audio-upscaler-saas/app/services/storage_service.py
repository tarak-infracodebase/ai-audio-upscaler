"""
Azure Blob Storage Service
Production-ready storage service for audio files and model artifacts
"""

import asyncio
import logging
from typing import Optional, AsyncIterator, BinaryIO
from datetime import datetime, timedelta
import os
from pathlib import Path

from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import AzureError, ResourceNotFoundError
from fastapi import UploadFile, HTTPException, status
import structlog

from app.core.config import get_settings
from app.core.monitoring import metrics

logger = structlog.get_logger(__name__)

class StorageService:
    """
    Azure Blob Storage service for managing audio files and model artifacts
    """

    def __init__(self):
        self.settings = get_settings()
        self._blob_service_client: Optional[BlobServiceClient] = None
        self._initialize_containers = False

    async def _get_blob_service_client(self) -> BlobServiceClient:
        """Get or create blob service client"""
        if self._blob_service_client is None:
            try:
                self._blob_service_client = BlobServiceClient.from_connection_string(
                    self.settings.AZURE_STORAGE_CONNECTION_STRING
                )

                # Initialize containers if needed
                if not self._initialize_containers:
                    await self._ensure_containers_exist()
                    self._initialize_containers = True

            except Exception as e:
                logger.error("Failed to initialize blob service client", error=str(e))
                raise

        return self._blob_service_client

    async def _ensure_containers_exist(self):
        """Ensure all required containers exist"""
        containers = ["audio-inputs", "audio-outputs", "models", "temp"]

        try:
            client = await self._get_blob_service_client()

            for container_name in containers:
                try:
                    container_client = client.get_container_client(container_name)
                    await container_client.create_container()
                    logger.info(f"Created container: {container_name}")
                except Exception as e:
                    if "ContainerAlreadyExists" in str(e):
                        logger.debug(f"Container already exists: {container_name}")
                    else:
                        logger.warning(f"Failed to create container {container_name}: {e}")

        except Exception as e:
            logger.error("Failed to ensure containers exist", error=str(e))
            raise

    async def upload_file(
        self,
        file: UploadFile,
        blob_path: str,
        container_name: str = "audio-inputs",
        overwrite: bool = True,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to Azure Blob Storage

        Args:
            file: FastAPI UploadFile object
            blob_path: Path within the container
            container_name: Target container name
            overwrite: Whether to overwrite existing blobs
            content_type: MIME type of the file

        Returns:
            Full blob path (container/path)
        """
        start_time = datetime.now()

        try:
            client = await self._get_blob_service_client()
            blob_client = client.get_blob_client(
                container=container_name,
                blob=blob_path
            )

            # Set content type
            if content_type is None:
                content_type = file.content_type or "application/octet-stream"

            # Read file content
            file_content = await file.read()
            file_size = len(file_content)

            # Validate file size
            max_size = 500 * 1024 * 1024  # 500MB
            if file_size > max_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large: {file_size / (1024*1024):.1f}MB. Max: {max_size / (1024*1024):.0f}MB"
                )

            # Upload with metadata
            metadata = {
                "original_filename": file.filename or "unknown",
                "upload_timestamp": start_time.isoformat(),
                "file_size": str(file_size),
                "content_type": content_type,
            }

            await blob_client.upload_blob(
                file_content,
                content_type=content_type,
                metadata=metadata,
                overwrite=overwrite
            )

            # Update metrics
            metrics.files_uploaded.inc()
            metrics.upload_size_bytes.observe(file_size)

            upload_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                "File uploaded successfully",
                blob_path=f"{container_name}/{blob_path}",
                file_size=file_size,
                upload_time=upload_time,
                content_type=content_type,
            )

            return f"{container_name}/{blob_path}"

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Failed to upload file",
                blob_path=blob_path,
                container=container_name,
                error=str(e),
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file: {str(e)}"
            )

    async def upload_file_from_path(
        self,
        local_path: str,
        blob_path: str,
        container_name: str = "audio-outputs",
        content_type: Optional[str] = None,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
    ) -> str:
        """
        Upload a file from local filesystem to blob storage

        Args:
            local_path: Path to local file
            blob_path: Target path in blob storage
            container_name: Target container
            content_type: MIME type
            chunk_size: Upload chunk size in bytes

        Returns:
            Full blob path
        """
        start_time = datetime.now()
        file_size = 0

        try:
            # Validate local file
            local_file = Path(local_path)
            if not local_file.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")

            file_size = local_file.stat().st_size

            client = await self._get_blob_service_client()
            blob_client = client.get_blob_client(
                container=container_name,
                blob=blob_path
            )

            # Determine content type
            if content_type is None:
                if local_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                    content_type = f"audio/{local_file.suffix[1:]}"
                else:
                    content_type = "application/octet-stream"

            # Upload with streaming for large files
            metadata = {
                "original_filename": local_file.name,
                "upload_timestamp": start_time.isoformat(),
                "file_size": str(file_size),
                "content_type": content_type,
            }

            with open(local_path, "rb") as file_data:
                await blob_client.upload_blob(
                    file_data,
                    content_type=content_type,
                    metadata=metadata,
                    overwrite=True,
                    max_concurrency=4,  # Parallel uploads
                )

            # Update metrics
            metrics.files_uploaded.inc()
            metrics.upload_size_bytes.observe(file_size)

            upload_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                "File uploaded from local path",
                local_path=local_path,
                blob_path=f"{container_name}/{blob_path}",
                file_size=file_size,
                upload_time=upload_time,
            )

            return f"{container_name}/{blob_path}"

        except Exception as e:
            logger.error(
                "Failed to upload file from path",
                local_path=local_path,
                blob_path=blob_path,
                container=container_name,
                error=str(e),
                exc_info=True
            )
            raise

    async def download_file(self, blob_path: str) -> AsyncIterator[bytes]:
        """
        Download a file from blob storage as async iterator

        Args:
            blob_path: Full blob path (container/path)

        Yields:
            File content chunks
        """
        try:
            container_name, blob_name = blob_path.split("/", 1)

            client = await self._get_blob_service_client()
            blob_client = client.get_blob_client(
                container=container_name,
                blob=blob_name
            )

            # Check if blob exists
            if not await blob_client.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"File not found: {blob_path}"
                )

            # Stream download
            download_stream = await blob_client.download_blob()

            # Update metrics
            metrics.files_downloaded.inc()

            async for chunk in download_stream.chunks():
                yield chunk

            logger.info("File downloaded successfully", blob_path=blob_path)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Failed to download file",
                blob_path=blob_path,
                error=str(e),
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to download file: {str(e)}"
            )

    async def download_file_to_path(
        self,
        blob_path: str,
        local_path: str,
        chunk_size: int = 1024 * 1024  # 1MB chunks
    ) -> None:
        """
        Download a file from blob storage to local filesystem

        Args:
            blob_path: Full blob path (container/path)
            local_path: Target local file path
            chunk_size: Download chunk size
        """
        start_time = datetime.now()

        try:
            container_name, blob_name = blob_path.split("/", 1)

            client = await self._get_blob_service_client()
            blob_client = client.get_blob_client(
                container=container_name,
                blob=blob_name
            )

            # Ensure local directory exists
            local_file = Path(local_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Download to file
            with open(local_path, "wb") as file_handle:
                download_stream = await blob_client.download_blob()
                async for chunk in download_stream.chunks():
                    file_handle.write(chunk)

            # Update metrics
            metrics.files_downloaded.inc()
            download_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                "File downloaded to local path",
                blob_path=blob_path,
                local_path=local_path,
                download_time=download_time,
                file_size=local_file.stat().st_size
            )

        except Exception as e:
            logger.error(
                "Failed to download file to path",
                blob_path=blob_path,
                local_path=local_path,
                error=str(e),
                exc_info=True
            )
            raise

    async def delete_file(self, blob_path: str) -> bool:
        """
        Delete a file from blob storage

        Args:
            blob_path: Full blob path (container/path)

        Returns:
            True if deleted, False if not found
        """
        try:
            container_name, blob_name = blob_path.split("/", 1)

            client = await self._get_blob_service_client()
            blob_client = client.get_blob_client(
                container=container_name,
                blob=blob_name
            )

            await blob_client.delete_blob(delete_snapshots="include")

            # Update metrics
            metrics.files_deleted.inc()

            logger.info("File deleted successfully", blob_path=blob_path)
            return True

        except ResourceNotFoundError:
            logger.warning("File not found for deletion", blob_path=blob_path)
            return False
        except Exception as e:
            logger.error(
                "Failed to delete file",
                blob_path=blob_path,
                error=str(e),
                exc_info=True
            )
            raise

    async def get_file_info(self, blob_path: str) -> Optional[dict]:
        """
        Get information about a blob

        Args:
            blob_path: Full blob path (container/path)

        Returns:
            Blob properties or None if not found
        """
        try:
            container_name, blob_name = blob_path.split("/", 1)

            client = await self._get_blob_service_client()
            blob_client = client.get_blob_client(
                container=container_name,
                blob=blob_name
            )

            properties = await blob_client.get_blob_properties()

            return {
                "name": blob_name,
                "container": container_name,
                "size": properties.size,
                "content_type": properties.content_settings.content_type,
                "last_modified": properties.last_modified,
                "etag": properties.etag,
                "metadata": properties.metadata,
            }

        except ResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(
                "Failed to get file info",
                blob_path=blob_path,
                error=str(e),
                exc_info=True
            )
            return None

    async def generate_download_url(
        self,
        blob_path: str,
        expires_in_hours: int = 24
    ) -> str:
        """
        Generate a signed URL for downloading a blob

        Args:
            blob_path: Full blob path (container/path)
            expires_in_hours: URL expiration time in hours

        Returns:
            Signed URL
        """
        try:
            container_name, blob_name = blob_path.split("/", 1)

            client = await self._get_blob_service_client()

            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=client.account_name,
                container_name=container_name,
                blob_name=blob_name,
                account_key=client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expires_in_hours),
            )

            blob_client = client.get_blob_client(
                container=container_name,
                blob=blob_name
            )

            return f"{blob_client.url}?{sas_token}"

        except Exception as e:
            logger.error(
                "Failed to generate download URL",
                blob_path=blob_path,
                error=str(e),
                exc_info=True
            )
            raise

    async def list_files(
        self,
        container_name: str,
        prefix: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        List files in a container

        Args:
            container_name: Container to list
            prefix: Optional prefix filter
            limit: Maximum number of files to return

        Returns:
            List of blob information
        """
        try:
            client = await self._get_blob_service_client()
            container_client = client.get_container_client(container_name)

            blobs = []
            async for blob in container_client.list_blobs(name_starts_with=prefix):
                blobs.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified,
                    "content_type": blob.content_settings.content_type if blob.content_settings else None,
                })

                if len(blobs) >= limit:
                    break

            return blobs

        except Exception as e:
            logger.error(
                "Failed to list files",
                container=container_name,
                prefix=prefix,
                error=str(e),
                exc_info=True
            )
            raise

    async def health_check(self) -> bool:
        """
        Health check for storage service

        Returns:
            True if healthy
        """
        try:
            client = await self._get_blob_service_client()

            # Try to list containers
            containers = []
            async for container in client.list_containers():
                containers.append(container.name)
                break  # Just check if we can list

            logger.debug("Storage health check passed", containers_found=len(containers))
            return True

        except Exception as e:
            logger.error("Storage health check failed", error=str(e))
            return False

    async def cleanup_temp_files(self, older_than_hours: int = 24) -> int:
        """
        Cleanup temporary files older than specified hours

        Args:
            older_than_hours: Delete files older than this many hours

        Returns:
            Number of files deleted
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            deleted_count = 0

            client = await self._get_blob_service_client()
            container_client = client.get_container_client("temp")

            async for blob in container_client.list_blobs():
                if blob.last_modified < cutoff_time:
                    try:
                        await container_client.delete_blob(blob.name)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to delete temp file",
                            blob_name=blob.name,
                            error=str(e)
                        )

            logger.info(f"Cleanup completed: {deleted_count} temp files deleted")
            return deleted_count

        except Exception as e:
            logger.error("Failed to cleanup temp files", error=str(e), exc_info=True)
            return 0