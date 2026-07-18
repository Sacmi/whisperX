import io
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import whisperx.alignment as alignment_module
import whisperx.asr as asr_module
import whisperx.transcribe as transcribe_module
import whisperx.utils as utils_module
from whisperx.utils import ProgressEmitter, str2bool


class FlushCountingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


def test_progress_emitter_writes_compact_jsonl_and_flushes():
    stream = FlushCountingStream()
    emit = ProgressEmitter(stream, min_interval=0)

    emit("stage_start", "transcribe", file="sample.wav")
    emit("progress", "transcribe", done=1, total=2, file="sample.wav")
    emit("stage_end", "transcribe", file="sample.wav")

    assert stream.getvalue() == (
        '{"event":"stage_start","stage":"transcribe","file":"sample.wav"}\n'
        '{"event":"progress","stage":"transcribe","done":1,"total":2,'
        '"file":"sample.wav"}\n'
        '{"event":"stage_end","stage":"transcribe","file":"sample.wav"}\n'
    )
    assert stream.flush_count == 3
    assert [json.loads(line) for line in stream.getvalue().splitlines()][1] == {
        "event": "progress",
        "stage": "transcribe",
        "done": 1,
        "total": 2,
        "file": "sample.wav",
    }


def test_progress_emitter_throttles_but_emits_final_once(monkeypatch):
    times = iter([10.0, 10.05, 10.05])
    monkeypatch.setattr(utils_module.time, "monotonic", lambda: next(times))
    stream = FlushCountingStream()
    emit = ProgressEmitter(stream, min_interval=0.1)

    emit("progress", "align", done=1, total=3)
    emit("progress", "align", done=2, total=3)
    emit("progress", "align", done=3, total=3)
    emit("progress", "align", done=3, total=3)

    assert [json.loads(line) for line in stream.getvalue().splitlines()] == [
        {"event": "progress", "stage": "align", "done": 1, "total": 3},
        {"event": "progress", "stage": "align", "done": 3, "total": 3},
    ]
    assert stream.flush_count == 2


def test_cli_progress_json_routes_human_output_and_reports_all_stages(
    monkeypatch, capfd, tmp_path
):
    audio_paths = ["first.wav", "second.wav"]
    loaded_audio = []
    aligned_audio = []
    written = []
    diarized = []
    diarization_configs = []
    cleanup = []

    class FakeASR:
        def transcribe(self, audio, **kwargs):
            print(f"fake ASR: {audio}")
            assert kwargs["print_progress"] is True
            callback = kwargs["progress_callback"]
            callback("stage_start", "vad")
            callback("stage_end", "vad")
            callback("stage_start", "transcribe")
            callback("progress", "transcribe", done=1, total=1)
            callback("progress", "transcribe", done=1, total=1)
            callback("stage_end", "transcribe")
            return {
                "segments": [{"start": 0.0, "end": 1.0, "text": "text"}],
                "language": "en",
            }

    def fake_load_model(*args, **kwargs):
        print("loading fake ASR")
        os.write(1, b"native ASR output\n")
        return FakeASR()

    def fake_load_audio(path):
        loaded_audio.append(path)
        return f"audio:{path}"

    def fake_load_align_model(*args, **kwargs):
        print("loading fake aligner")
        return object(), {"language": "en"}

    def fake_align(segments, model, metadata, audio, device, **kwargs):
        print(f"fake align: {audio}")
        aligned_audio.append(audio)
        callback = kwargs["progress_callback"]
        callback("stage_start", "align")
        callback("progress", "align", done=1, total=1)
        callback("stage_end", "align")
        return {"segments": segments, "word_segments": []}

    class FakeDiarizationPipeline:
        def __init__(self, model_name, token, device):
            print("loading fake diarizer")
            diarization_configs.append((model_name, token, device))

        def __call__(self, audio_path, **kwargs):
            print(f"fake diarize: {audio_path}")
            diarized.append(audio_path)
            return []

    def fake_assign_word_speakers(diarize_segments, result):
        return result

    def fake_get_writer(output_format, output_dir):
        def writer(result, audio_path, options):
            print(f"fake writer: {audio_path}")
            written.append(audio_path)

        return writer

    monkeypatch.setattr(transcribe_module, "load_model", fake_load_model)
    monkeypatch.setattr(transcribe_module, "load_audio", fake_load_audio)
    monkeypatch.setattr(transcribe_module, "load_align_model", fake_load_align_model)
    monkeypatch.setattr(transcribe_module, "align", fake_align)
    monkeypatch.setattr(
        transcribe_module, "DiarizationPipeline", FakeDiarizationPipeline
    )
    monkeypatch.setattr(
        transcribe_module, "assign_word_speakers", fake_assign_word_speakers
    )
    monkeypatch.setattr(transcribe_module, "get_writer", fake_get_writer)
    monkeypatch.setattr(
        transcribe_module.gc, "collect", lambda: cleanup.append("gc")
    )
    monkeypatch.setattr(
        transcribe_module.torch.cuda,
        "empty_cache",
        lambda: cleanup.append("cuda"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "whisperx",
            *audio_paths,
            "--device",
            "cpu",
            "--output_dir",
            str(tmp_path),
            "--output_format",
            "json",
            "--progress_json",
            "true",
            "--print_progress",
            "true",
            "--diarize",
            "--diarize_model",
            "custom/diarization-model",
            "--hf_token",
            "token",
        ],
    )

    transcribe_module.cli()

    captured = capfd.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert events
    global_stages = [
        (event["event"], event["stage"])
        for event in events
        if "file" not in event
    ]
    assert global_stages == [
        ("stage_start", "load_asr"),
        ("stage_end", "load_asr"),
        ("stage_start", "load_align"),
        ("stage_end", "load_align"),
        ("stage_start", "load_diarize"),
        ("stage_end", "load_diarize"),
    ]

    for audio_path in audio_paths:
        file_events = [event for event in events if event.get("file") == audio_path]
        assert [(event["event"], event["stage"]) for event in file_events] == [
            ("stage_start", "vad"),
            ("stage_end", "vad"),
            ("stage_start", "transcribe"),
            ("progress", "transcribe"),
            ("stage_end", "transcribe"),
            ("stage_start", "align"),
            ("progress", "align"),
            ("stage_end", "align"),
            ("stage_start", "diarize"),
            ("stage_end", "diarize"),
        ]
        final_events = [
            event for event in file_events if event["event"] == "progress"
        ]
        assert all(event["done"] == event["total"] == 1 for event in final_events)

    assert ">>Performing transcription..." in captured.err
    assert ">>Performing alignment..." in captured.err
    assert ">>Performing diarization..." in captured.err
    assert "loading fake ASR" in captured.err
    assert "native ASR output" in captured.err
    assert "fake writer: first.wav" in captured.err
    assert loaded_audio == audio_paths
    assert aligned_audio == audio_paths
    assert diarized == audio_paths
    assert diarization_configs == [("custom/diarization-model", "token", "cpu")]
    assert written == audio_paths
    assert cleanup == ["gc", "cuda", "gc", "cuda"]


def test_cli_without_progress_json_keeps_human_output_on_stdout(
    monkeypatch, capsys, tmp_path
):
    class FakeASR:
        def transcribe(self, audio, **kwargs):
            assert kwargs["progress_callback"] is None
            print("fake transcription")
            return {"segments": [], "language": "en"}

    def fake_load_model(*args, **kwargs):
        print("loading fake ASR")
        return FakeASR()

    def fake_get_writer(output_format, output_dir):
        return lambda result, audio_path, options: print("fake writer")

    monkeypatch.setattr(transcribe_module, "load_model", fake_load_model)
    monkeypatch.setattr(transcribe_module, "load_audio", lambda path: "audio")
    monkeypatch.setattr(transcribe_module, "get_writer", fake_get_writer)
    monkeypatch.setattr(transcribe_module.gc, "collect", lambda: None)
    monkeypatch.setattr(transcribe_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "whisperx",
            "sample.wav",
            "--device",
            "cpu",
            "--output_dir",
            str(tmp_path),
            "--output_format",
            "json",
            "--no_align",
            "--progress_json",
            "false",
        ],
    )

    transcribe_module.cli()

    captured = capsys.readouterr()
    assert ">>Performing transcription..." in captured.out
    assert "loading fake ASR" in captured.out
    assert "fake transcription" in captured.out
    assert '"event":' not in captured.out
    assert captured.err == ""


def test_qwen_pipeline_reports_batch_progress(monkeypatch):
    vad_segments = [
        {"start": 0.0, "end": 0.25},
        {"start": 0.25, "end": 0.5},
        {"start": 0.5, "end": 0.75},
    ]
    monkeypatch.setattr(asr_module, "merge_chunks", lambda *args, **kwargs: vad_segments)

    class FakeModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, language):
            self.calls.append(len(audio))
            return [
                SimpleNamespace(text="text", language="English") for _ in audio
            ]

    fake_model = FakeModel()
    pipeline = asr_module.Qwen3ASRPipeline(
        fake_model, lambda audio: object(), {}, "cpu", language="en"
    )
    events = []

    def callback(event, stage, **values):
        events.append({"event": event, "stage": stage, **values})

    result = pipeline.transcribe(
        np.zeros(16000, dtype=np.float32),
        batch_size=2,
        progress_callback=callback,
    )

    assert len(result["segments"]) == 3
    assert fake_model.calls == [3]
    assert fake_model.max_inference_batch_size == 2
    assert events == [
        {"event": "stage_start", "stage": "vad"},
        {"event": "stage_end", "stage": "vad"},
        {"event": "stage_start", "stage": "transcribe"},
        {"event": "progress", "stage": "transcribe", "done": 3, "total": 3},
        {"event": "stage_end", "stage": "transcribe"},
    ]


def test_empty_alignment_reports_zero_progress():
    events = []

    def callback(event, stage, **values):
        events.append({"event": event, "stage": stage, **values})

    result = alignment_module.align(
        [],
        object(),
        {"language": "en", "type": "qwen3"},
        np.zeros(16000, dtype=np.float32),
        "cpu",
        progress_callback=callback,
    )

    assert result == {"segments": [], "word_segments": []}
    assert events == [
        {"event": "stage_start", "stage": "align"},
        {"event": "progress", "stage": "align", "done": 0, "total": 0},
        {"event": "stage_end", "stage": "align"},
    ]


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_str2bool_accepts_lowercase(value, expected):
    assert str2bool(value) is expected


def test_cli_help_lists_community_diarization_model(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["whisperx", "--help"])

    with pytest.raises(SystemExit, match="0"):
        transcribe_module.cli()

    help_text = capsys.readouterr().out
    assert "--diarize_model" in help_text
    assert "pyannote/speaker-diarization-community-1" in help_text
