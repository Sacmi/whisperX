# Repository Guide

## Setup And Verification

- Use the locked environment: `uv sync --extra dev`. Python 3.10+ is required; `pytest` is only in the `dev` extra.
- Run all tests with `uv run --extra dev pytest`; run one file or test with `uv run --extra dev pytest tests/test_progress_json.py` or `uv run --extra dev pytest tests/test_batch_size.py::test_cohere_pipeline_uses_batch_size`.
- Build both distributions with `uv build`. There is no configured lint, format, or type-check command.
- Unit tests import the full Torch/qwen-asr/Transformers/pyannote stack but fake model inference. They must not need model downloads, real audio, ffmpeg, a GPU, or Hugging Face credentials.

## Runtime Map

- `pyproject.toml` installs `whisperx.transcribe:cli`; `python -m whisperx` calls the same entrypoint. Public library exports are in `whisperx/__init__.py`.
- The CLI deliberately processes every input through VAD+ASR, frees that model, then aligns all results, frees the aligner, optionally diarizes, and finally writes outputs. Preserve this ordering because it limits peak model memory.
- `whisperx/asr.py` selects Cohere only when the model name contains `cohere-transcribe`; every other name takes the Qwen path. Both pipelines always run the bundled pyannote VAD first.
- Alignment is enabled by default and uses `Qwen/Qwen3-ForcedAligner-0.6B`; use `--no_align` to skip it. Per-segment alignment failures warn and produce uniformly estimated word timings rather than aborting.
- `whisperx/utils.py` contains the active TXT/VTT/SRT/TSV/JSON/Audacity writers. `whisperx/SubtitlesProcessor.py` is disconnected from the CLI and public API.
- File-path audio loading shells out to `ffmpeg` and returns mono 16 kHz float32. Library APIs can instead receive already-decoded 16 kHz NumPy arrays.
- Do not alter `whisperx/assets/pytorch_model.bin` casually: VAD loading checks it against the SHA-256 embedded in `whisperx/vad.py`. `mel_filters.npz` is also a runtime package asset.

## Compatibility Traps

- `--forced_aligner` is accepted and stored but does not select or run alignment. `--align_model` selects the alignment model; README examples using `--forced_aligner` conflict with the implementation.
- Qwen submits all VAD chunks in one call and sets `model.max_inference_batch_size`; Cohere explicitly loops over batches. `max_inference_batch_size` remains a deprecated API and hidden CLI alias for `batch_size`.
- Cohere requires one of its 14 language codes; omission warns and defaults to Japanese (`ja`), not auto-detection. `--hf_token` is passed only to diarization, so it does not authenticate gated Cohere ASR loading.
- `--task translate` is still parsed for compatibility, but it warns, transcribes instead, and disables alignment. The current Qwen alignment path accepts but does not use `interpolate_method` or `return_char_alignments`.
- The CLI currently overwrites each output result's language with the initially selected alignment language (`en` when omitted). Do not assume detected ASR language survives in output files.
- With `--progress_json true`, stdout is reserved for flushed compact JSONL and all human/native output is redirected to stderr. Preserve stage names and final `done == total` events; `tests/test_progress_json.py` defines this protocol.
- Treat `EXAMPLES.md` as stale inherited Whisper/Wav2Vec2 documentation. Prefer executable code and `README.md` where they differ, while noting the `--forced_aligner` conflict above.
