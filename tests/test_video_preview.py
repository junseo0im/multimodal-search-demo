from pathlib import Path

from src.ui import gradio_demo


def test_make_full_video_html_handles_missing_path():
    html = gradio_demo.make_full_video_html({"video_path": "", "video_id": "short_001"})
    assert "unavailable" in html


def test_make_full_video_html_uses_cached_tmp_video(tmp_path, monkeypatch):
    source = tmp_path / "short_001.mp4"
    source.write_bytes(b"fake mp4")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(gradio_demo, "FULL_VIDEO_CACHE_DIR", cache_dir)

    html = gradio_demo.make_full_video_html(
        {
            "video_path": str(source),
            "video_id": "short_001",
            "recipe_name": "test",
            "time": "5.0s - 7.0s",
            "start_time": 5.0,
        }
    )

    cached = Path(cache_dir / "short_001.mp4")
    assert cached.exists()
    assert "/file=" in html
    assert "#t=5.0" in html
