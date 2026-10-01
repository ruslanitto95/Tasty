"""PHI-safe logging.

Technical log messages are written in English. Any Cyrillic text, e-mail address or
phone-like number reaching a log record is treated as potential patient data and
redacted *after* formatting, so neither arguments nor exception texts can leak it.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
from pathlib import Path

_CYRILLIC = re.compile(r"[А-Яа-яЁё][А-Яа-яЁё\w\-]*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<![\w.:-])\+?\d[\d\-\s()]{8,}\d(?![\w.:])")
# Long quoted spans may be (English) transcript text in tests; redact them as well.
_QUOTED = re.compile(r"(['\"«])([^'\"»\n]{40,})(['\"»])")
_SECRET = re.compile(r"(?i)(bearer\s+|api[_-]?key[\"'=:\s]+|sk-)[A-Za-z0-9_\-\.]{6,}")


def redact(text: str) -> str:
    text = _SECRET.sub(r"\1[SECRET]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _CYRILLIC.sub("[RU]", text)
    text = _QUOTED.sub(r"\1[TEXT]\3", text)
    return text


class RedactingFormatter(logging.Formatter):
    """Redacts message, exception and stack text; leaves the timestamp/level prefix intact."""

    def format(self, record: logging.LogRecord) -> str:
        safe = logging.makeLogRecord(record.__dict__)
        safe.msg = redact(record.getMessage())
        safe.args = None
        if record.exc_info:
            safe.exc_text = redact(self.formatException(record.exc_info))
            safe.exc_info = None
        elif record.exc_text:
            safe.exc_text = redact(record.exc_text)
        if record.stack_info:
            safe.stack_info = redact(record.stack_info)
        return super().format(safe)


class RedactingFilter(logging.Filter):
    """Collapses args into the message so downstream handlers never see raw objects."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = redact(message)
        record.args = None
        return True


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"


def configure_logging(log_dir: Path | None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)
    handlers: list[logging.Handler] = []
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    if log_dir is not None:
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_dir / "mva.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
        )
    for handler in handlers:
        handler.setFormatter(RedactingFormatter(_LOG_FORMAT))
        handler.addFilter(RedactingFilter())
        root.addHandler(handler)
    logging.captureWarnings(True)
