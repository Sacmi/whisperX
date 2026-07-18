import pytest

from whisperx.utils import WriteSRT, WriteVTT


SUBTITLE_OPTIONS = {
    "highlight_words": False,
    "max_line_count": None,
    "max_line_width": None,
}


@pytest.mark.parametrize(
    ("writer_class", "expected_start", "expected_end"),
    [
        (WriteSRT, "00:00:01,250", "00:00:02,500"),
        (WriteVTT, "00:01.250", "00:02.500"),
    ],
)
def test_subtitle_cue_uses_word_timestamps(
    writer_class, expected_start, expected_end, tmp_path
):
    result = {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 1.25, "end": 1.75},
                    {"word": "world", "start": 2.0, "end": 2.5},
                ],
            }
        ],
    }

    cues = list(writer_class(str(tmp_path)).iterate_result(result, SUBTITLE_OPTIONS))

    assert cues == [(expected_start, expected_end, "Hello world")]


@pytest.mark.parametrize(
    ("writer_class", "expected_start", "expected_end"),
    [
        (WriteSRT, "00:00:00,250", "00:00:05,750"),
        (WriteVTT, "00:00.250", "00:05.750"),
    ],
)
def test_subtitle_cue_falls_back_to_segment_timestamps(
    writer_class, expected_start, expected_end, tmp_path
):
    result = {
        "language": "en",
        "segments": [
            {
                "start": 0.25,
                "end": 5.75,
                "text": "Hello",
                "words": [{"word": "Hello"}],
            }
        ],
    }

    cues = list(writer_class(str(tmp_path)).iterate_result(result, SUBTITLE_OPTIONS))

    assert cues == [(expected_start, expected_end, "Hello")]
