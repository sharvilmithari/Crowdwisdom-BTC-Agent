"""
Secure logging configuration with automated secret redaction.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional


class SecretRedactingFormatter(logging.Formatter):
    """Custom logging formatter that strips API tokens and sensitive patterns."""

    PATTERNS = [
        re.compile(r'(?i)(apify_api_token|api_token|token|key|secret|password)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{8,})["\']?'),
        re.compile(r'tvly-[a-zA-Z0-9_\-]{16,}'),
        re.compile(r'apify_api_[a-zA-Z0-9]{20,}'),
    ]

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for pattern in self.PATTERNS:
            msg = pattern.sub(r'\1: [REDACTED]', msg) if '\1' in pattern.pattern else pattern.sub('[REDACTED_KEY]', msg)
        return msg


def setup_logger(name: str = "crowdwisdom", log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = SecretRedactingFormatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if log_dir provided
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

