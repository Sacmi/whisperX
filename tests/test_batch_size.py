import sys
from types import SimpleNamespace

import numpy as np
import pytest

import whisperx.asr as asr_module
import whisperx.transcribe as transcribe_module


def test_qwen_pipeline_delegates_batching_to_qwen(monkeypatch):
    vad_segments = [
        {"start": index / 10, "end": (index + 1) / 10} for index in range(10)
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
    audio = np.zeros(16000, dtype=np.float32)

    result = pipeline.transcribe(audio, batch_size=16)

    assert len(result["segments"]) == 10
    assert fake_model.calls == [10]
    assert fake_model.max_inference_batch_size == 16

    pipeline.transcribe(audio)

    assert fake_model.calls == [10, 10]
    assert fake_model.max_inference_batch_size == 8


@pytest.mark.parametrize("batch_size", [0, -1])
def test_qwen_pipeline_rejects_invalid_batch_size(batch_size):
    pipeline = asr_module.Qwen3ASRPipeline(
        object(), lambda audio: pytest.fail("VAD should not run"), {}, "cpu"
    )

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        pipeline.transcribe(np.zeros(16000, dtype=np.float32), batch_size=batch_size)


def test_cohere_pipeline_uses_batch_size(monkeypatch):
    vad_segments = [
        {"start": 0.0, "end": 0.25},
        {"start": 0.25, "end": 0.5},
        {"start": 0.5, "end": 0.75},
    ]
    monkeypatch.setattr(asr_module, "merge_chunks", lambda *args, **kwargs: vad_segments)

    class FakeModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, processor, audio_arrays, sample_rates, language):
            self.calls.append(len(audio_arrays))
            return ["text"] * len(audio_arrays)

    fake_model = FakeModel()
    pipeline = asr_module.CohereASRPipeline(
        object(), fake_model, lambda audio: object(), {}, "cpu", language="en"
    )

    result = pipeline.transcribe(
        np.zeros(16000, dtype=np.float32), batch_size=2
    )

    assert len(result["segments"]) == 3
    assert fake_model.calls == [2, 1]


def test_load_model_accepts_deprecated_batch_size_alias(monkeypatch):
    loaded = {}

    def fake_from_pretrained(*args, **kwargs):
        loaded.update(kwargs)
        return object()

    monkeypatch.setattr(
        asr_module.Qwen3ASRModel, "from_pretrained", fake_from_pretrained
    )

    with pytest.warns(DeprecationWarning, match="max_inference_batch_size"):
        pipeline = asr_module.load_model(
            "Qwen/Qwen3-ASR-1.7B",
            device="cpu",
            vad_model=lambda audio: object(),
            max_inference_batch_size=12,
        )

    assert pipeline.default_batch_size == 12
    assert loaded["max_inference_batch_size"] == 12


def test_cli_maps_deprecated_batch_size_alias(monkeypatch, tmp_path):
    transcribe_batch_sizes = []

    class FakeASR:
        def transcribe(self, audio, **kwargs):
            transcribe_batch_sizes.append(kwargs["batch_size"])
            return {"segments": [], "language": "en"}

    monkeypatch.setattr(transcribe_module, "load_model", lambda *args, **kwargs: FakeASR())
    monkeypatch.setattr(transcribe_module, "load_audio", lambda path: "audio")
    monkeypatch.setattr(
        transcribe_module, "get_writer", lambda *args: lambda *writer_args: None
    )
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
            "--max_inference_batch_size",
            "12",
        ],
    )

    with pytest.warns(FutureWarning, match="max_inference_batch_size"):
        transcribe_module.cli()

    assert transcribe_batch_sizes == [12]


def test_cli_rejects_conflicting_batch_size_options(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "whisperx",
            "sample.wav",
            "--batch_size",
            "8",
            "--max_inference_batch_size",
            "16",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        transcribe_module.cli()

    assert "cannot be used together" in capsys.readouterr().err


def test_cli_help_hides_deprecated_batch_size_alias(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["whisperx", "--help"])

    with pytest.raises(SystemExit, match="0"):
        transcribe_module.cli()

    help_text = capsys.readouterr().out
    assert "--batch_size" in help_text
    assert "--max_inference_batch_size" not in help_text
