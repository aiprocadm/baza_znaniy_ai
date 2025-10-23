from __future__ import annotations

import logging
from dataclasses import dataclass, field

import boto3
from botocore.exceptions import ClientError

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class S3DocumentStorage:
    """Helper for storing rendered documents in S3."""

    bucket: str
    region: str
    endpoint_url: str | None
    access_key: str
    secret_key: str
    _client: boto3.client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = boto3.client(
            "s3",
            region_name=self.region,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    @classmethod
    def from_settings(cls) -> "S3DocumentStorage":
        settings = get_settings()
        return cls(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
        )

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:  # pragma: no cover - defensive path
            error_code = exc.response.get("Error", {}).get("Code", "")
            if str(error_code) not in {"404", "NoSuchBucket"}:
                raise
            params = {"Bucket": self.bucket}
            if self.region != "us-east-1":
                params["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            self._client.create_bucket(**params)

    def upload(self, key: str, data: bytes) -> str:
        self.ensure_bucket()
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        logger.info("Stored document in S3", extra={"bucket": self.bucket, "key": key})
        return key
