"""Tests for MetricsReader."""

from chip8_rl.metrics import MetricsReader, format_info


def test_reads_only_new_complete_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    reader = MetricsReader(path)
    assert reader.read_new() == []  # File doesn't exist yet.

    path.write_text('{"episode": 1}\n{"episode": 2}\n{"epi')  # Third line half-written.
    assert [r["episode"] for r in reader.read_new()] == [1, 2]

    with open(path, "a") as f:
        f.write('sode": 3}\n')
    assert [r["episode"] for r in reader.read_new()] == [3]
    assert reader.read_new() == []


def test_format_info_lists_key_value_pairs():
    assert format_info({"agent_score": 9, "opponent_score": 3}) == "agent_score=9 opponent_score=3"
