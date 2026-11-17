from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import TracebackType
from typing import Any, TextIO


def _utc_timestamp() -> str:
    """
    Returns:
        str: Current UTC timestamp in ISO 8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


class Logger:
    """
    Text and JSONL run logger with an optional console stream.
    """

    def __init__(
        self,
        logfile: str | Path | None,
        metrics_file: str | Path | None = None,
        *,
        enabled: bool = True,
        clear_metrics: bool = True,
        console: TextIO | None = None,
    ) -> None:
        """
        Args:
            logfile (str | Path | None): Optional text-log path, opened for appending.
            metrics_file (str | Path | None): Optional JSONL metrics path.
            enabled (bool): Enable file and console output.
            clear_metrics (bool): Truncate the metrics file during initialization.
            console (TextIO | None): Console stream; None selects sys.stdout.
        """
        self._enabled = bool(enabled)
        self._console = sys.stdout if console is None else console
        self._closed = False
        self._log: TextIO | None = None
        self.metrics_file = (
            Path(metrics_file) if metrics_file is not None and enabled else None
        )

        if self._enabled and logfile is not None:
            path = Path(logfile)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log = path.open("a", encoding="utf-8")
        if self.metrics_file is not None:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            if clear_metrics:
                self.metrics_file.write_text("", encoding="utf-8")

    def _append_json(self, payload: dict[str, Any]) -> None:
        """
        Append a JSON record when the metrics file is enabled.

        Args:
            payload (dict[str, Any]): JSON-serializable fields with finite numbers.
        """

        if self.metrics_file is None:
            return
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Append metrics with a UTC timestamp.

        Args:
            metrics (dict[str, Any]): JSON-serializable metric fields.
        """

        self._append_json({**metrics, "timestamp": _utc_timestamp()})

    def log_config(self, config: dict[str, Any]) -> None:
        """
        Append a configuration record with a UTC timestamp.

        Args:
            config (dict[str, Any]): JSON-serializable configuration fields.
        """

        self._append_json({"type": "config", "timestamp": _utc_timestamp(), **config})

    def flush(self) -> None:
        """
        Flush the open text log and enabled console stream.
        """

        if self._log is not None and not self._log.closed:
            self._log.flush()
        if self._enabled:
            self._console.flush()

    def write(self, message: str) -> int:
        """
        Write text to the enabled run log and console.

        Args:
            message (str): Text to write verbatim.

        Returns:
            int: Console character count, or zero when disabled or closed.
        """

        if not self._enabled or self._closed:
            return 0
        if self._log is not None:
            self._log.write(message)
        written = self._console.write(message)
        return int(written) if isinstance(written, int) else len(message)

    def writelines(self, lines: list[str] | tuple[str, ...]) -> int:
        """
        Write a sequence of text fragments.

        Args:
            lines (list[str] | tuple[str, ...]): Strings to write in order.

        Returns:
            int: Total character count reported by write.
        """

        return sum(self.write(line) for line in lines)

    @property
    def closed(self) -> bool:
        """
        Returns:
            bool: Whether the logger has been closed.
        """

        return self._closed

    def __enter__(self) -> "Logger":
        """
        Returns:
            Logger: This logger for use within a context manager.
        """

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """
        Close the text log when leaving the context.

        Args:
            exc_type (type[BaseException] | None): Active exception type, if any.
            exc (BaseException | None): Active exception instance, if any.
            tb (TracebackType | None): Active traceback, if any.
        """

        self.close()

    def close(self) -> None:
        """
        Close the text log and disable subsequent text writes.
        """

        if self._closed:
            return
        if self._log is not None and not self._log.closed:
            self._log.close()
        self._closed = True
