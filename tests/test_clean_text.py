import pytest

from audio8_tts_data import clean_text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("你\n好", "你好"),
        ("你 \t\n 好", "你好"),
        ("你 好", "你 好"),
        ("こんにちは。\n世界", "こんにちは。世界"),
        ("안녕하세요\n세계", "안녕하세요세계"),
        ("안녕하세요 세계", "안녕하세요 세계"),
        ("hello\nworld", "hello world"),
        ("你好\nworld", "你好 world"),
        ("こんにちは\nworld", "こんにちは world"),
        ("안녕하세요\nworld", "안녕하세요 world"),
        ("hello\n世界", "hello 世界"),
        ("甲\U00020000乙", "甲\U00020000乙"),
    ],
)
def test_clean_text_normalizes_whitespace_by_script(value: str, expected: str) -> None:
    assert clean_text(value) == expected


def test_clean_text_rejects_whitespace_only_input() -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        clean_text(" \n\t")
