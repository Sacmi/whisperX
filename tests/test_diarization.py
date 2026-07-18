from types import SimpleNamespace

import numpy as np
import pytest
import torch

import whisperx.diarize as diarize_module


def test_diarization_pipeline_uses_community_model_by_default(monkeypatch):
    loaded = []

    class FakeLoadedPipeline:
        def to(self, device):
            loaded.append(("device", device))
            return self

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_name, token=None):
            loaded.append(("model", model_name, token))
            return FakeLoadedPipeline()

    monkeypatch.setattr(diarize_module, "Pipeline", FakePipeline)

    pipeline = diarize_module.DiarizationPipeline(token="token", device="cpu")

    assert isinstance(pipeline.model, FakeLoadedPipeline)
    assert loaded == [
        ("model", "pyannote/speaker-diarization-community-1", "token"),
        ("device", torch.device("cpu")),
    ]


@pytest.mark.parametrize("structured_output", [True, False])
def test_diarization_pipeline_accepts_pyannote_and_direct_outputs(
    structured_output,
):
    segment = SimpleNamespace(start=0.25, end=0.75)

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            return iter([(segment, "track", "SPEAKER_00")])

    annotation = FakeAnnotation()
    output = (
        SimpleNamespace(speaker_diarization=annotation)
        if structured_output
        else annotation
    )
    calls = []

    class FakeModel:
        def __call__(self, audio_data, **kwargs):
            calls.append((audio_data, kwargs))
            return output

    pipeline = object.__new__(diarize_module.DiarizationPipeline)
    pipeline.model = FakeModel()

    result = pipeline(
        np.zeros(16000, dtype=np.float32), min_speakers=1, max_speakers=2
    )

    audio_data, kwargs = calls[0]
    assert audio_data["waveform"].shape == (1, 16000)
    assert audio_data["sample_rate"] == 16000
    assert kwargs == {"num_speakers": None, "min_speakers": 1, "max_speakers": 2}
    assert result.columns.tolist() == ["segment", "label", "speaker", "start", "end"]
    assert result.iloc[0].to_dict() == {
        "segment": segment,
        "label": "track",
        "speaker": "SPEAKER_00",
        "start": 0.25,
        "end": 0.75,
    }
