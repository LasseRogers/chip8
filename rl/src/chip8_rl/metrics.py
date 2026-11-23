"""Incremental reader for a run's metrics.jsonl."""

import json
from pathlib import Path


class MetricsReader:
    """Returns the records appended to a JSON-lines file since the last read.

    Only complete lines are parsed: a line the trainer is still writing is
    left for the next read.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0

    def read_new(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        complete = data[: data.rfind(b"\n") + 1]
        self._offset += len(complete)
        return [json.loads(line) for line in complete.splitlines()]


def format_info(info: dict) -> str:
    """A game's info dict as `key=value` pairs, for logs and status lines."""
    return " ".join(f"{key}={value}" for key, value in info.items())
