"""Utility for downloading and validating GGUF language models."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import httpx

try:  # pragma: no cover - optional dependency resolved at runtime
    from huggingface_hub import HfHubHTTPError, hf_hub_download, model_info
except Exception:  # pragma: no cover - fallback when the dependency is missing
    HfHubHTTPError = Exception  # type: ignore[assignment]
    hf_hub_download = None  # type: ignore[assignment]
    model_info = None  # type: ignore[assignment]

_LOGGER = logging.getLogger("download_model")

DEFAULT_MANIFEST = Path("models/model_manifest.json")
DEFAULT_OUTPUT = Path("models/model.gguf")
DEFAULT_TARGET = "default"
HUGGINGFACE_TOKEN_ENV = "HUGGINGFACE_HUB_TOKEN"


class DownloadError(RuntimeError):
    """Raised when a download or validation step fails."""


@dataclass(slots=True)
class ModelDescriptor:
    """Descriptor for a downloadable model artifact."""

    name: str
    model_id: str | None = None
    filename: str | None = None
    url: str | None = None
    sha256: str | None = None
    license: str | None = None
    description: str | None = None

    @classmethod
    def from_mapping(cls, name: str, payload: Mapping[str, Any]) -> "ModelDescriptor":
        return cls(
            name=name,
            model_id=payload.get("model_id"),
            filename=payload.get("filename"),
            url=payload.get("url"),
            sha256=payload.get("sha256"),
            license=payload.get("license"),
            description=payload.get("description"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "filename": self.filename,
            "url": self.url,
            "sha256": self.sha256,
            "license": self.license,
            "description": self.description,
        }


@dataclass(slots=True)
class DownloadResult:
    """Information about a completed download."""

    path: Path
    sha256: str
    bytes_written: int
    from_cache: bool


@dataclass(slots=True)
class DownloadRequest:
    """Parameters for the download request."""

    descriptor: ModelDescriptor
    output: Path
    overwrite: bool
    max_retries: int
    timeout: float
    dry_run: bool
    token: str | None


def load_manifest(manifest_path: Path) -> dict[str, ModelDescriptor]:
    """Load the model manifest from disk."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text("utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive guard
        raise ValueError("Manifest must contain a JSON object")

    descriptors: dict[str, ModelDescriptor] = {}
    for name, entry in payload.items():
        if not isinstance(entry, Mapping):  # pragma: no cover - defensive guard
            raise ValueError(f"Manifest entry '{name}' must be a JSON object")
        descriptors[name] = ModelDescriptor.from_mapping(str(name), entry)
    return descriptors


def dump_manifest(manifest_path: Path, descriptors: Mapping[str, ModelDescriptor]) -> None:
    """Persist manifest descriptors back to disk."""

    payload = {name: descriptor.to_mapping() for name, descriptor in descriptors.items()}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")


def compute_sha256(path: Path, chunk_size: int = 2**20) -> str:
    """Compute the SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent_dir(path: Path) -> None:
    """Create parent directories for *path* if necessary."""

    path.parent.mkdir(parents=True, exist_ok=True)


def verify_hash(path: Path, expected_sha: str) -> None:
    """Ensure that the file at *path* matches the expected SHA-256 digest."""

    actual_sha = compute_sha256(path)
    if actual_sha.lower() != expected_sha.lower():
        raise DownloadError(f"Hash mismatch for {path}: expected {expected_sha}, got {actual_sha}")


def _iter_with_backoff(max_retries: int) -> Iterator[int]:
    for attempt in range(1, max_retries + 1):
        yield attempt
        time.sleep(min(2**attempt, 30))


def download_via_http(
    url: str,
    output: Path,
    expected_sha: str | None,
    max_retries: int,
    timeout: float,
) -> DownloadResult:
    """Download the model file from an arbitrary HTTP(S) endpoint."""

    ensure_parent_dir(output)
    tmp_path = output.with_suffix(output.suffix + ".tmp")

    last_error: Exception | None = None
    for attempt in _iter_with_backoff(max_retries):
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
                response.raise_for_status()
                digest = hashlib.sha256()
                bytes_written = 0
                with tmp_path.open("wb") as temp_file:
                    for chunk in response.iter_bytes():
                        temp_file.write(chunk)
                        digest.update(chunk)
                        bytes_written += len(chunk)
            sha_value = digest.hexdigest()
            if expected_sha and sha_value.lower() != expected_sha.lower():
                raise DownloadError(
                    "Downloaded file hash mismatch: " f"expected {expected_sha}, got {sha_value}"
                )
            tmp_path.replace(output)
            return DownloadResult(
                path=output, sha256=sha_value, bytes_written=bytes_written, from_cache=False
            )
        except (httpx.HTTPError, OSError, DownloadError) as exc:  # pragma: no cover - network path
            last_error = exc
            _LOGGER.warning("Attempt %s failed to download %s: %s", attempt, url, exc)
            if attempt >= max_retries:
                break
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass
    assert last_error is not None  # pragma: no cover - defensive
    raise DownloadError(f"Failed to download model after {max_retries} attempts: {last_error}")


def download_via_hub(
    descriptor: ModelDescriptor, output: Path, expected_sha: str | None, token: str | None
) -> DownloadResult:
    """Download the model using the Hugging Face Hub client."""

    if hf_hub_download is None or model_info is None:
        raise DownloadError("huggingface_hub is not installed; cannot download from the Hub")

    if descriptor.model_id is None or descriptor.filename is None:
        raise DownloadError("Both 'model_id' and 'filename' are required for Hub downloads")

    resolved_sha = expected_sha
    if resolved_sha is None:
        try:
            info = model_info(descriptor.model_id, token=token)
        except HfHubHTTPError as exc:  # pragma: no cover - network path
            raise DownloadError(
                f"Unable to fetch metadata for {descriptor.model_id}: {exc}"
            ) from exc
        for sibling in info.siblings:
            if sibling.rfilename == descriptor.filename and sibling.sha256:
                resolved_sha = sibling.sha256
                break
        if resolved_sha is None:
            raise DownloadError(
                "SHA-256 checksum is not available from manifest or Hugging Face metadata"
            )
    ensure_parent_dir(output)
    local_path = Path(
        hf_hub_download(
            repo_id=descriptor.model_id,
            filename=descriptor.filename,
            local_dir=str(output.parent),
            local_dir_use_symlinks=False,
            token=token,
        )
    )
    # The hub may have produced a file in the same directory; if different, move it.
    if local_path.resolve() != output.resolve():
        shutil.move(str(local_path), output)
    actual_sha = compute_sha256(output)
    if actual_sha.lower() != resolved_sha.lower():
        raise DownloadError(
            f"Hash mismatch for {output}: expected {resolved_sha}, got {actual_sha}"
        )
    return DownloadResult(
        path=output, sha256=actual_sha, bytes_written=output.stat().st_size, from_cache=False
    )


def select_descriptor(manifest: Mapping[str, ModelDescriptor], target: str) -> ModelDescriptor:
    try:
        return manifest[target]
    except KeyError as exc:  # pragma: no cover - defensive
        raise DownloadError(f"Model '{target}' is not defined in the manifest") from exc


def should_skip_download(output: Path, expected_sha: str | None, overwrite: bool) -> bool:
    if not output.exists():
        return False
    if overwrite:
        return False
    if expected_sha is None:
        _LOGGER.info("No checksum available; existing file will be overwritten.")
        return False
    try:
        verify_hash(output, expected_sha)
    except DownloadError:
        _LOGGER.info("Existing model hash mismatch; re-downloading.")
        return False
    _LOGGER.info("Model already present at %s with matching checksum.", output)
    return True


def execute(request: DownloadRequest, manifest_path: Path) -> DownloadResult | None:
    descriptor = request.descriptor
    expected_sha = descriptor.sha256

    if request.dry_run:
        _LOGGER.info("Dry-run: would download %s to %s", descriptor.name, request.output)
        if descriptor.license:
            _LOGGER.info("License: %s", descriptor.license)
        if descriptor.description:
            _LOGGER.info("Description: %s", descriptor.description)
        if descriptor.url:
            _LOGGER.info("Source URL: %s", descriptor.url)
        return None

    if should_skip_download(request.output, expected_sha, request.overwrite):
        return DownloadResult(
            path=request.output,
            sha256=expected_sha or compute_sha256(request.output),
            bytes_written=request.output.stat().st_size,
            from_cache=True,
        )

    if descriptor.url:
        result = download_via_http(
            url=descriptor.url,
            output=request.output,
            expected_sha=expected_sha,
            max_retries=request.max_retries,
            timeout=request.timeout,
        )
    else:
        result = download_via_hub(
            descriptor=descriptor,
            output=request.output,
            expected_sha=expected_sha,
            token=request.token,
        )

    if descriptor.sha256 is None or descriptor.sha256.lower() != result.sha256.lower():
        descriptor.sha256 = result.sha256
        manifest = load_manifest(manifest_path)
        manifest[descriptor.name] = descriptor
        dump_manifest(manifest_path, manifest)
        _LOGGER.info("Updated manifest with SHA-256 for %s", descriptor.name)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the JSON manifest with model metadata.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Named manifest entry to download (default: 'default').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for the GGUF model file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-download even if the file already exists with a matching checksum.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of download retries for direct HTTP downloads.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout (seconds) for direct HTTP downloads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without downloading the model.",
    )
    parser.add_argument(
        "--allow-missing-hash",
        action="store_true",
        help="Allow downloads even if the manifest entry is missing an explicit sha256.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get(HUGGINGFACE_TOKEN_ENV),
        help="Optional Hugging Face token used for hub requests.",
    )
    parser.add_argument(
        "--url",
        dest="override_url",
        help="Override the manifest URL for direct HTTP download.",
    )
    parser.add_argument(
        "--sha256",
        dest="override_sha",
        help="Override the expected SHA-256 checksum.",
    )
    parser.add_argument(
        "--model-id",
        dest="override_model_id",
        help="Override the Hugging Face model ID (repo).",
    )
    parser.add_argument(
        "--filename",
        dest="override_filename",
        help="Override the filename inside the Hugging Face repo.",
    )
    return parser


def apply_overrides(descriptor: ModelDescriptor, args: argparse.Namespace) -> ModelDescriptor:
    descriptor = ModelDescriptor(**descriptor.to_mapping(), name=descriptor.name)
    if args.override_url:
        descriptor.url = args.override_url
    if args.override_sha:
        descriptor.sha256 = args.override_sha
    if args.override_model_id:
        descriptor.model_id = args.override_model_id
    if args.override_filename:
        descriptor.filename = args.override_filename
    if descriptor.url and not descriptor.url.lower().startswith(("http://", "https://")):
        raise DownloadError(f"URL must be HTTP(S): {descriptor.url}")
    if descriptor.url and descriptor.model_id:
        if descriptor.sha256:
            _LOGGER.info(
                "Both URL and Hugging Face model ID provided; using direct URL because checksum is available."
            )
            descriptor.model_id = None
            descriptor.filename = None
        else:
            _LOGGER.info(
                "Direct URL configured without checksum; falling back to Hugging Face Hub metadata."
            )
            descriptor.url = None
    if not descriptor.url and not (descriptor.model_id and descriptor.filename):
        raise DownloadError(
            "Descriptor must define either a direct 'url' or both 'model_id' and 'filename'."
        )
    return descriptor


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest_path = args.manifest
    descriptors = load_manifest(manifest_path)
    descriptor = select_descriptor(descriptors, args.target)
    descriptor = apply_overrides(descriptor, args)

    if descriptor.sha256 is None and not args.allow_missing_hash and descriptor.url:
        raise DownloadError(
            "Manifest entry is missing sha256. Use --allow-missing-hash to compute it after download"
        )

    request = DownloadRequest(
        descriptor=descriptor,
        output=args.output,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
        timeout=args.timeout,
        dry_run=args.dry_run,
        token=args.token,
    )

    try:
        result = execute(request, manifest_path)
    except DownloadError as exc:
        _LOGGER.error("%s", exc)
        return 1

    if result is None:
        return 0

    status = "cache" if result.from_cache else "downloaded"
    _LOGGER.info(
        "Model %s (%s) -> %s [%s bytes]",
        descriptor.name,
        status,
        result.path,
        result.bytes_written,
    )
    _LOGGER.info("SHA-256: %s", result.sha256)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
