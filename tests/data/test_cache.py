# tests/data/test_cache.py
from pathlib import Path
from fpl_model.data.cache import FileCache


class TestFileCache:
    def test_cache_miss(self, tmp_path):
        cache = FileCache(tmp_path)
        assert cache.get("vaastav", "2024-25", "players_raw.csv") is None

    def test_cache_put_and_get(self, tmp_path):
        cache = FileCache(tmp_path)
        content = b"id,name\n1,Salah"
        cache.put("vaastav", "2024-25", "players_raw.csv", content)
        result = cache.get("vaastav", "2024-25", "players_raw.csv")
        assert result == content

    def test_cache_path_structure(self, tmp_path):
        cache = FileCache(tmp_path)
        cache.put("vaastav", "2024-25", "players_raw.csv", b"data")
        expected = tmp_path / "vaastav" / "2024-25" / "players_raw.csv"
        assert expected.exists()

    def test_cache_clear_season(self, tmp_path):
        cache = FileCache(tmp_path)
        cache.put("vaastav", "2024-25", "players_raw.csv", b"data")
        cache.clear("vaastav", "2024-25")
        assert cache.get("vaastav", "2024-25", "players_raw.csv") is None

    def test_has(self, tmp_path):
        cache = FileCache(tmp_path)
        assert not cache.has("vaastav", "2024-25", "players_raw.csv")
        cache.put("vaastav", "2024-25", "players_raw.csv", b"data")
        assert cache.has("vaastav", "2024-25", "players_raw.csv")
