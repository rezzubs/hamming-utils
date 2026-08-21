"""Tests for `encoded_memory.results.discover_result_files`."""

from encoded_memory.results import discover_result_files


def test_discover_result_files_single_file(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("{}")
    assert discover_result_files(path) == [path]


def test_discover_result_files_recurses_directories(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.json").write_text("{}")
    (tmp_path / "b" / "nested").mkdir(parents=True)
    (tmp_path / "b" / "nested" / "two.json").write_text("{}")

    found = discover_result_files(tmp_path)
    assert found == sorted(found)
    assert {p.name for p in found} == {"one.json", "two.json"}
