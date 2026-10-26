from asrgui.m0.transcript import normalize_model_result, surface_tokens, timestamped_tokens
from asrgui.m0.validation import validate_transcript


def test_chinese_english_and_number_tokenization():
    text = "我们使用 Python 3.11。"
    tokens = surface_tokens(text)
    assert "".join(token["text"] for token in tokens) == text
    assert [token["text"] for token in tokens if token["clickable"]] == [
        "我",
        "们",
        "使",
        "用",
        "Python",
        "3.11",
    ]
    assert [token["text"] for token in tokens if not token["clickable"]] == [" ", " ", "。"]


def test_token_timestamps_and_punctuation():
    tokens, precision = timestamped_tokens(
        "你好。",
        [[0, 100], [100, 200]],
        0,
        220,
    )
    assert precision == "token"
    assert [(item["start_ms"], item["end_ms"]) for item in tokens] == [
        (0, 100),
        (100, 200),
        (200, 200),
    ]
    assert tokens[-1]["clickable"] is False


def test_partial_timestamps_are_resampled_without_time_reversal():
    tokens, precision = timestamped_tokens(
        "你好吗？",
        [[0, 100], [100, 200]],
        0,
        300,
    )
    assert precision == "interpolated"
    assert [(item["time_precision"], item["clickable"]) for item in tokens] == [
        ("interpolated", True),
        ("interpolated", True),
        ("interpolated", True),
        ("segment", False),
    ]
    clickable_times = [(item["start_ms"], item["end_ms"]) for item in tokens if item["clickable"]]
    assert clickable_times == [(0, 100), (100, 200), (200, 300)]


def test_normalize_speaker_segments():
    result = normalize_model_result(
        {
            "text": "你好。",
            "sentence_info": [
                {
                    "text": "你好。",
                    "start": 100,
                    "end": 500,
                    "spk": 0,
                    "timestamp": [[100, 200], [200, 300]],
                }
            ],
        },
        transcript_id="tr_test",
        meeting_id="mtg_test",
        audio={},
        models={},
        effective_settings={},
    )
    assert result["speakers"][0]["id"] == "spk_1"
    assert result["segments"][0]["speaker_id"] == "spk_1"
    assert result["segments"][0]["tokens"][0]["time_precision"] == "token"


def test_normalize_drops_punctuation_only_artifact():
    result = normalize_model_result(
        {
            "text": "你好。",
            "sentence_info": [
                {
                    "text": "你好。",
                    "start": 100,
                    "end": 500,
                    "spk": 0,
                    "timestamp": [[100, 200], [200, 300]],
                },
                {
                    "text": "，",
                    "start": 500,
                    "end": 900,
                    "spk": 0,
                    "timestamp": [[500, 600], [600, 900]],
                },
            ],
        },
        transcript_id="tr_test",
        meeting_id="mtg_test",
        audio={},
        models={},
        effective_settings={},
    )
    assert len(result["segments"]) == 1
    assert result["diagnostics"]["dropped_non_text_segments"] == 1


def test_validation_rejects_broken_token_times():
    transcript = {
        "audio": {"duration_ms": 1000},
        "speakers": [{"id": "spk_1"}],
        "segments": [
            {
                "id": "seg_0001",
                "speaker_id": "spk_1",
                "start_ms": 100,
                "end_ms": 500,
                "text": "好",
                "tokens": [
                    {
                        "id": "tok_00001",
                        "text": "好",
                        "start_ms": 400,
                        "end_ms": 300,
                        "time_precision": "token",
                        "clickable": True,
                    }
                ],
            }
        ],
    }
    try:
        validate_transcript(transcript)
    except ValueError as exc:
        assert "clickable token has no duration" in str(exc)
    else:
        raise AssertionError("invalid transcript was accepted")
