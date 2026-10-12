"""Download the face-detection model files into ``models/``.

Run this once after installing the requirements::

    python download_model.py

The script contacts GitHub to fetch the OpenCV Zoo YuNet ONNX model (~230 KB)
and, optionally, the legacy ResNet-10 SSD face detector. Nothing is downloaded
unless you run this script; the URLs are listed in ``config.py`` and in the
README so you can also fetch the files manually.
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import config


class ModelSpec(NamedTuple):
    """A downloadable model file."""

    name: str
    filename: str
    url: str
    description: str
    required: bool


YUNET = ModelSpec(
    name="yunet",
    filename=config.YUNET_MODEL_FILENAME,
    url=config.YUNET_MODEL_URL,
    description="YuNet ONNX face detector (primary, recommended)",
    required=True,
)

SSD_PROTO = ModelSpec(
    name="ssd-proto",
    filename=config.SSD_PROTO_FILENAME,
    url=config.SSD_PROTO_URL,
    description="ResNet-10 SSD network definition (optional fallback)",
    required=False,
)

SSD_WEIGHTS = ModelSpec(
    name="ssd-weights",
    filename=config.SSD_WEIGHTS_FILENAME,
    url=config.SSD_WEIGHTS_URL,
    description="ResNet-10 SSD Caffe weights (optional fallback, ~10 MB)",
    required=False,
)

ALL_MODELS: tuple[ModelSpec, ...] = (YUNET, SSD_PROTO, SSD_WEIGHTS)

#: Minimum plausible size of a downloaded model, used to catch HTML error pages
#: being saved as if they were a model.
MIN_VALID_BYTES = 1024

CERT_HELP = """
TLS certificate verification failed. Your Python installation cannot find a
trusted certificate store. Fix it with any one of the following:

  * macOS (python.org build): run
        /Applications/Python 3.x/Install Certificates.command
  * any platform: install the certificate bundle
        pip install certifi
  * or download the model files manually with a browser or curl and place them
    in the models/ folder (URLs are printed above and listed in README.md).

As a last resort, re-run this script with --insecure to skip verification.
"""


class DownloadError(RuntimeError):
    """Raised when a model file could not be downloaded."""


def _format_size(num_bytes: int) -> str:
    """Return a human-readable file size."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def build_ssl_context(*, insecure: bool = False) -> ssl.SSLContext:
    """Build the TLS context used for downloads.

    Prefers the system trust store and falls back to the :mod:`certifi` bundle
    when it is installed, which covers Python builds that ship without root
    certificates configured.

    Args:
        insecure: Disable certificate verification entirely. Only set this if
            the user explicitly asked for it.

    Returns:
        A configured :class:`ssl.SSLContext`.
    """
    if insecure:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    context = ssl.create_default_context()
    if ssl.get_default_verify_paths().cafile is None:
        try:
            import certifi
        except ImportError:
            pass
        else:
            context.load_verify_locations(cafile=certifi.where())
    return context


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: int = 60,
    context: ssl.SSLContext | None = None,
) -> None:
    """Download ``url`` to ``destination`` atomically.

    The payload is written to a temporary ``.part`` file and only moved into
    place once the transfer succeeds, so an interrupted download can never
    leave a truncated model behind.

    Args:
        url: Source URL.
        destination: Target file path. Parent directories are created.
        timeout: Socket timeout in seconds.
        context: TLS context. Built with :func:`build_ssl_context` if omitted.

    Raises:
        DownloadError: On any network error or if the payload is implausibly
            small (which usually means an HTML error page was served).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "face-image-sorter/1.0"}
        )
        with urllib.request.urlopen(
            request, timeout=timeout, context=context or build_ssl_context()
        ) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
        raise DownloadError(f"Download failed for {url}: {error}") from error

    if len(payload) < MIN_VALID_BYTES:
        raise DownloadError(
            f"Downloaded file from {url} is only {len(payload)} bytes; "
            "this is almost certainly an error page rather than a model."
        )

    try:
        temp_path.write_bytes(payload)
        temp_path.replace(destination)
    except OSError as error:
        raise DownloadError(f"Could not write {destination}: {error}") from error


def ensure_model(
    spec: ModelSpec,
    models_dir: Path,
    *,
    force: bool = False,
    context: ssl.SSLContext | None = None,
) -> tuple[bool, str]:
    """Download one model unless it is already present.

    Args:
        spec: Model to fetch.
        models_dir: Directory the model is stored in.
        force: Re-download even if the file already exists.
        context: TLS context to reuse across downloads.

    Returns:
        A ``(success, message)`` pair suitable for printing.
    """
    destination = models_dir / spec.filename

    if destination.exists() and not force:
        return True, f"already present ({_format_size(destination.stat().st_size)})"

    try:
        download_file(spec.url, destination, context=context)
    except DownloadError as error:
        return False, str(error)

    return True, f"downloaded ({_format_size(destination.stat().st_size)})"


def verify_models(models_dir: Path = config.MODELS_DIR) -> dict[str, bool]:
    """Report which model files exist in ``models_dir``.

    Returns:
        Mapping of model name to whether its file is present.
    """
    return {
        spec.name: (models_dir / spec.filename).is_file() for spec in ALL_MODELS
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the downloader."""
    parser = argparse.ArgumentParser(
        prog="download_model.py",
        description=(
            "Download face-detection model files into the models/ folder. "
            "This contacts github.com."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=config.MODELS_DIR,
        help="Directory to download into (default: %(default)s)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also download the optional ResNet-10 SSD fallback model (~10 MB)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Skip TLS certificate verification. Use only if your Python "
            "installation has no working certificate store."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    models_dir: Path = args.models_dir
    selected = ALL_MODELS if args.all else (YUNET,)

    print(f"Model directory: {models_dir}")
    print("Source: github.com (opencv/opencv_zoo, opencv/opencv_3rdparty)\n")
    if args.insecure:
        print("WARNING: TLS certificate verification is disabled (--insecure).\n")

    context = build_ssl_context(insecure=args.insecure)

    failures: list[str] = []
    cert_problem = False
    for spec in selected:
        print(f"  {spec.filename}")
        print(f"    {spec.description}")
        print(f"    {spec.url}")
        success, message = ensure_model(
            spec, models_dir, force=args.force, context=context
        )
        print(f"    -> {message}\n")
        if not success:
            cert_problem = cert_problem or "CERTIFICATE_VERIFY_FAILED" in message
            if spec.required:
                failures.append(spec.filename)

    if cert_problem:
        print(CERT_HELP)

    if failures:
        print("ERROR: required model file(s) could not be downloaded:")
        for name in failures:
            print(f"  - {name}")
        print(
            "\nDownload them manually and place them in the models/ folder. "
            "URLs are listed in README.md."
        )
        return 1

    print("Model setup complete.")
    if not args.all:
        print("Tip: run with --all to also fetch the optional SSD fallback model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
