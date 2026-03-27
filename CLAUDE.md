# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WhisperX is a fork of m-bain/whisperX modified to use **Qwen3-ASR** instead of Whisper for automatic speech recognition. It provides word-level timestamps and speaker diarization with multilingual support (52 languages).

## Installation & Development

```bash
# Development installation
git clone https://github.com/Sacmi/whisperX.git
cd whisperX
pip install -e .

# Install from git
pip install git+https://github.com/Sacmi/whisperX.git
```

Requires Python 3.10+, PyTorch 2, ffmpeg CLI, and optionally Hugging Face token for speaker diarization.

## Core Architecture

The codebase follows a pipeline architecture with three main stages:

### 1. Transcription (`whisperx/asr.py`)
- **`load_model()`** - Loads Qwen3-ASR model (`Qwen3ASRModel` from `qwen-asr` package)
- **`Qwen3ASRPipeline`** - Wraps ASR model with VAD preprocessing
- Uses pyannote.audio's VAD to detect speech segments before ASR
- Language mapping: ISO 639-1 codes (e.g., "en", "zh") to Qwen's full names (e.g., "English", "Chinese")

### 2. Alignment (`whisperx/alignment.py`)
- **`load_align_model()`** - Loads Qwen3-ForcedAligner for word-level timestamps
- **`align()`** - Aligns transcription to audio for word/character timestamps
- Supports 11 languages: en, zh, de, es, fr, ja, ko, pt, ru, tr, ar
- Falls back to uniform time distribution if alignment fails
- Sentence splitting via NLTK's Punkt tokenizer

### 3. Diarization (`whisperx/diarize.py`)
- **`DiarizationPipeline`** - Wraps pyannote.audio's speaker-diarization-3.1
- **`assign_word_speakers()`** - Assigns speaker labels to segments and words
- Requires Hugging Face token with access to pyannote gated models

## Key Modules

| Module | Purpose |
|--------|---------|
| `asr.py` | ASR transcription with Qwen3-ASR, VAD integration |
| `alignment.py` | Forced alignment with Qwen3-ForcedAligner |
| `vad.py` | Voice Activity Detection via pyannote.audio |
| `diarize.py` | Speaker diarization and speaker assignment |
| `audio.py` | Audio loading (ffmpeg), mel spectrogram computation |
| `transcribe.py` | CLI entry point, orchestrates the full pipeline |
| `types.py` | TypedDict definitions for all result types |
| `utils.py` | Helper functions, output writers (srt, vtt, json, etc.) |

## Model Names

**ASR Models (Hugging Face):**
- `Qwen/Qwen3-ASR-1.7B` - Full model (recommended)
- `Qwen/Qwen3-ASR-0.6B` - Smaller/faster

**Alignment Models:**
- `Qwen/Qwen3-ForcedAligner-0.6B` - Default aligner

## Breaking Changes from Original WhisperX

1. **Model names:** Use Hugging Face paths (e.g., `Qwen/Qwen3-ASR-1.7B`) instead of `large-v2`, etc.
2. **No translation:** Qwen3-ASR only supports transcription (not `--task translate`)
3. **New parameters:** `--forced_aligner`, `--max_inference_batch_size`, `--dtype`
4. **Removed parameters:** `suppress_numerals`, `task="translate"`

## Common Commands

```bash
# Basic transcription
whisperx audio.wav --model Qwen/Qwen3-ASR-1.7B

# With word-level alignment
whisperx audio.wav --model Qwen/Qwen3-ASR-1.7B --forced_aligner Qwen/Qwen3-ForcedAligner-0.6B

# With speaker diarization
whisperx audio.wav --model Qwen/Qwen3-ASR-1.7B --diarize --hf_token YOUR_TOKEN

# Full pipeline
whisperx audio.wav \
  --model Qwen/Qwen3-ASR-1.7B \
  --language en \
  --forced_aligner Qwen/Qwen3-ForcedAligner-0.6B \
  --max_inference_batch_size 32 \
  --dtype bfloat16 \
  --output_format json \
  --diarize \
  --hf_token YOUR_TOKEN
```

## CLI Pipeline Flow

The `transcribe.py` CLI executes stages sequentially with memory cleanup between stages:

1. **VAD + ASR:** Transcribe all audio files
2. **Unload model** (gc + cuda.empty_cache)
3. **Alignment:** Load aligner, align segments
4. **Unload aligner**
5. **Diarization:** Load diarization model, assign speakers
6. **Write output**

## Language Support

**ASR (52 languages):** en, zh, de, es, ru, ko, fr, ja, pt, tr, ar, it, nl, pl, cs, hi, fa, uk, vi, ro, th, el, hu, da, fi, no, sv, he, id, ms, ca, sr, hr, bg, sk, lt, lv, et, sl, bn, ta, te, mr, ur, kn, ml, gu, pa, si, km, lo

**Alignment (11 languages):** en, zh, de, es, fr, ja, ko, pt, ru, tr, ar

## Output Formats

Supported via `utils.py`: `all`, `srt`, `vtt`, `txt`, `tsv`, `json`, `aud`

Output files contain `segments` array with `start`, `end`, `text`, `words` (optional), `speaker` (optional).

## Important Constants

- `SAMPLE_RATE = 16000` (audio.py)
- `CHUNK_LENGTH = 30` seconds (max VAD chunk size)
- VAD default thresholds: `vad_onset=0.500`, `vad_offset=0.363`
