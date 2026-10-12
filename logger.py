"""Logging setup for the face image sorter.

The application writes a detailed, timestamped log to ``logs/processing.log``.
A console handler is always installed but is normally configured to stay
silent, because ``main.py`` prints per-image outcomes and the final summary to
stdout itself; ``--verbose`` lowers the console threshold so log records appear
there as well.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

LOGGER_NAME: Final[str] = "face_image_sorter"

#: Matches the format requested in the specification, e.g.
#: ``2026-10-01 16:30:21 | INFO | Processing: family.jpg``
FILE_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(message)s"
CONSOLE_FORMAT: Final[str] = "%(levelname)s: %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    log_file: Path,
    *,
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
    name: str = LOGGER_NAME,
) -> logging.Logger:
    """Create (or reconfigure) the application logger.

    Args:
        log_file: Path of the log file. Parent directories are created and, if
            the file cannot be opened, logging degrades to console only instead
            of aborting the run.
        console_level: Minimum level echoed to stderr. Pass ``logging.CRITICAL``
            to effectively silence the console handler.
        file_level: Minimum level written to the log file.
        name: Logger name, exposed mainly so tests can use isolated loggers.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    # Re-running setup (e.g. in tests) must not duplicate handlers.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    logger.addHandler(console_handler)

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
    except OSError as error:
        # Reported directly rather than through the logger, which may be
        # configured to stay silent on the console.
        print(
            f"Warning: could not open log file {log_file}: {error}\n"
            "Continuing without a log file.",
            file=sys.stderr,
        )
    else:
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def configure_opencv_logging(*, verbose: bool = False) -> None:
    """Quieten OpenCV's internal logger so it does not interleave with output.

    OpenCV writes its own messages straight to stderr, including unconditional
    notices such as the OpenCV 5 "Targets are not supported by the new graph
    engine" warning emitted whenever a DNN target is selected. Those would
    scroll through the per-image progress lines without telling the user
    anything actionable, and this application reports decode and detection
    failures itself through :func:`setup_logger`.

    Args:
        verbose: Keep OpenCV's warnings visible instead of hiding them.
    """
    try:
        cv_logging = __import__("cv2").utils.logging
    except (ImportError, AttributeError):
        return

    level = (
        cv_logging.LOG_LEVEL_WARNING if verbose else cv_logging.LOG_LEVEL_ERROR
    )
    try:
        cv_logging.setLogLevel(level)
    except Exception as error:  # noqa: BLE001 - purely cosmetic, never fatal
        get_logger().debug("Could not set OpenCV log level: %s", error)


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the application logger without reconfiguring its handlers.

    Modules use this at import time so they always log through whatever
    handlers :func:`setup_logger` installed.
    """
    return logging.getLogger(name)
