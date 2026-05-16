import pandas as pd

from src.index.build_index import make_search_text


def test_make_search_text_falls_back_to_caption():
    row = pd.Series({"recipe_name": "\uae40\uce58\ucc0c\uac1c", "caption": "\ub300\ud30c\ub97c \ub123\ub294 \uc7a5\uba74"})
    text = make_search_text(row)
    assert "\uae40\uce58\ucc0c\uac1c" in text
    assert "\ub300\ud30c" in text


def test_make_search_text_includes_optional_sources():
    row = pd.Series(
        {
            "recipe_name": "\ub5a1\ubcf6\uc774",
            "caption": "\uc591\ub150\uc744 \ub123\ub294 \uc7a5\uba74",
            "title_text": "\ub9e4\uc6b4 \ub5a1\ubcf6\uc774",
            "asr_text": "\uace0\ucd94\uc7a5\uc744 \ub123\uc5b4\uc694",
            "ocr_text": "\uace0\ucd94\uc7a5 2\uc2a4\ud47c",
            "scene_caption": "\ud504\ub77c\uc774\ud32c\uc5d0 \uc591\ub150\uc744 \ub123\ub294 \uc7a5\uba74",
        }
    )
    text = make_search_text(row)
    assert "\ub9e4\uc6b4 \ub5a1\ubcf6\uc774" in text
    assert "\uace0\ucd94\uc7a5 2\uc2a4\ud47c" in text
    assert "\ud504\ub77c\uc774\ud32c" in text
