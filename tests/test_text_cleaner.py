import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.text_cleaner import clean_text, PROTECTED_KEYWORDS


def test_removes_url():
    assert "http" not in clean_text("check this out https://example.com/x")


def test_keeps_critical_keywords():
    out = clean_text("People are trapped and injured, urgent rescue needed")
    for kw in ["trapped", "injured", "urgent", "rescue"]:
        assert kw in out


def test_hashtag_kept_as_word():
    out = clean_text("#SendHelp now")
    assert "sendhelp" in out


def test_mention_keeps_username_by_default():
    out = clean_text("@RedCross please help")
    assert "redcross" in out


def test_mention_removed_when_requested():
    out = clean_text("@RedCross please help", remove_mentions=True)
    assert "redcross" not in out


def test_none_and_empty_safe():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_html_entities_decoded():
    out = clean_text("water &amp; food needed")
    assert "&amp;" not in out


def test_protected_keywords_all_lowercase_single_words():
    # sanity check on the constant itself
    for kw in PROTECTED_KEYWORDS:
        assert kw == kw.lower()
        assert " " not in kw


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
