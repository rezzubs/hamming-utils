"""Tests for `encoded_memory.results.load_results`."""

from encoded_memory.results import load_results


def test_load_results_flattens_across_paths(tmp_path, save_result):
    save_result(tmp_path / "a" / "result.json", faults=1)
    save_result(tmp_path / "b" / "result.json", faults=2)

    loaded = load_results([tmp_path / "a", tmp_path / "b"])
    assert len(loaded) == 2


def test_load_results_empty_for_nothing_found(tmp_path):
    assert load_results([tmp_path]) == []


def test_load_results_skips_invalid_files_with_a_warning(tmp_path, save_result, caplog):
    save_result(tmp_path / "good.json", faults=1)
    (tmp_path / "not_a_result.json").write_text("this is not a result file")

    with caplog.at_level("WARNING"):
        loaded = load_results([tmp_path])

    assert len(loaded) == 1
    assert loaded[0][0].name == "good.json"
    assert any("skipping" in message for message in caplog.messages)
