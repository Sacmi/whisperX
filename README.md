# WhisperX with Qwen3-ASR and Cohere Transcribe

> [!NOTE]
> This is a modified version of WhisperX that supports **Qwen3-ASR** and **Cohere Transcribe** instead of Whisper for automatic speech recognition. Both are state-of-the-art multilingual ASR models. The model is selected via `--model`.

<p align="center">
  <a href="https://github.com/m-bain/whisperX/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/m-bain/whisperX.svg"
             alt="GitHub license">
  </a>
</p>

This repository provides fast automatic speech recognition with word-level timestamps and speaker diarization.

## Key Features

- 🎯 **Accurate word-level timestamps** using Qwen3-ForcedAligner
- 👯‍♂️ **Multispeaker ASR** using speaker diarization from [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- 🗣️ **VAD preprocessing**, reduces hallucination & improves transcription quality
- ⚡️ **Efficient batch processing** for fast transcription
- 🌍 **52 languages supported** (Qwen3-ASR) / **14 languages** (Cohere Transcribe)
- 🤗 **Direct integration** with Hugging Face models

## Supported ASR Models

### Qwen3-ASR

**Qwen3-ASR** is an advanced ASR model developed by Alibaba's Qwen team. It offers:
- Support for 52 languages with high accuracy
- Language auto-detection
- Efficient inference with batch processing
- Word-level timestamps via Qwen3-ForcedAligner (11 languages)

### Cohere Transcribe

**Cohere Transcribe** (`cohere-transcribe-03-2026`) is a 2B-parameter Conformer-based ASR model by Cohere. It offers:
- 14 languages: en, de, fr, it, es, pt, el, nl, pl, zh, ja, ko, vi, ar
- #1 accuracy on the Open ASR Leaderboard (5.42% average WER)
- Requires language to be specified explicitly (no auto-detection)
- Requires accepting the [model license](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) on Hugging Face

**Voice Activity Detection (VAD)** detects the presence or absence of human speech, reducing hallucinations.

**Forced Alignment** (via Qwen3-ForcedAligner) aligns transcriptions to audio to generate word-level timestamps. Works with both ASR models.

**Speaker Diarization** partitions audio into segments by speaker identity.

## Setup ⚙️

Tested with Python 3.10 and PyTorch 2.

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Make sure you have uv installed.

### Install WhisperX

For regular installation:
```bash
uv pip install git+https://github.com/Sacmi/whisperX.git
```

For development (with uv.lock):
```bash
git clone https://github.com/Sacmi/whisperX.git
cd whisperX
uv sync  # Automatically creates venv and installs dependencies
source .venv/bin/activate
```

For editable development install:
```bash
git clone https://github.com/Sacmi/whisperX.git
cd whisperX
uv pip install -e .
```

You may also need to install ffmpeg. Follow instructions from [OpenAI Whisper setup](https://github.com/openai/whisper#setup).

### 3. Speaker Diarization (Optional)

To **enable Speaker Diarization**, include your Hugging Face access token (read) that you can generate from [here](https://huggingface.co/settings/tokens) after the `--hf_token` argument. Accept the user agreements for:
- [Segmentation](https://huggingface.co/pyannote/segmentation-3.0)
- [Speaker-Diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

## Usage 💬

### Command Line

Basic transcription with Qwen3-ASR:
```bash
whisperx audio.wav --model Qwen/Qwen3-ASR-1.7B
```

Basic transcription with Cohere Transcribe:
```bash
whisperx audio.wav --model CohereLabs/cohere-transcribe-03-2026 --language en
```

With word-level alignment (works with both models):
```bash
whisperx audio.wav --model Qwen/Qwen3-ASR-1.7B --forced_aligner Qwen/Qwen3-ForcedAligner-0.6B
whisperx audio.wav --model CohereLabs/cohere-transcribe-03-2026 --language ja --forced_aligner Qwen/Qwen3-ForcedAligner-0.6B
```

With speaker diarization:
```bash
whisperx audio.wav --model Qwen/Qwen3-ASR-1.7B --diarize --hf_token YOUR_TOKEN
```

Full example with all features:
```bash
whisperx audio.wav \
  --model Qwen/Qwen3-ASR-1.7B \
  --language en \
  --forced_aligner Qwen/Qwen3-ForcedAligner-0.6B \
  --batch_size 32 \
  --dtype bfloat16 \
  --output_format json \
  --diarize \
  --hf_token YOUR_TOKEN
```

Machine-readable progress is available as JSON Lines on stdout:
```bash
whisperx audio.wav --progress_json true --output_format json
```

Each line is a flushed JSON object. Human-readable output is redirected to
stderr while this option is enabled, and `--print_progress` can be enabled at
the same time. Example events:
```json
{"event":"stage_start","stage":"vad"}
{"event":"progress","stage":"transcribe","done":8,"total":12}
{"event":"stage_end","stage":"transcribe"}
```

Pipeline stages are `vad`, `transcribe`, `align`, and `diarize`. Model loading
uses the non-countable stages `load_asr`, `load_align`, and `load_diarize`.
Countable stages always emit a final event where `done` equals `total`,
including `0/0` for an empty stage. With multiple input files, pipeline events
contain the original audio path in the `file` field and totals are per file.

### Python Usage

Basic transcription:
```python
import whisperx
import gc

device = "cuda"
audio_file = "audio.mp3"
batch_size = 16

# 1. Transcribe with Qwen3-ASR
model = whisperx.load_model(
    "Qwen/Qwen3-ASR-1.7B",
    device=device,
    language="en",  # Optional: specify language
    dtype="bfloat16"
)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"])  # segments with start/end times

# 2. Align with Qwen3-ForcedAligner for word-level timestamps
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=device,
    model_name="Qwen/Qwen3-ForcedAligner-0.6B"
)
result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device,
    return_char_alignments=False
)

print(result["segments"])  # segments with word-level timestamps

# Optional: Clean up memory
gc.collect()
torch.cuda.empty_cache()
del model_a

# 3. Assign speaker labels (optional)
import torch
diarize_model = whisperx.DiarizationPipeline(token=YOUR_HF_TOKEN, device=device)
diarize_segments = diarize_model(audio_file)
result = whisperx.assign_word_speakers(diarize_segments, result)
print(result["segments"])  # segments with speaker IDs
```

### Available Models

**ASR Models:**
- `Qwen/Qwen3-ASR-1.7B` - Full-sized Qwen3 model (recommended for accuracy, 52 languages)
- `Qwen/Qwen3-ASR-0.6B` - Smaller Qwen3 model (faster, slightly lower accuracy)
- `CohereLabs/cohere-transcribe-03-2026` - Cohere Transcribe (14 languages, requires HF login and license acceptance)

**Aligner Models:**
- `Qwen/Qwen3-ForcedAligner-0.6B` - Word-level timestamps (supports 11 languages, works with both ASR models)

### Supported Languages

**Qwen3-ASR** supports 52 languages:

**Major Languages:** English (en), Chinese (zh), German (de), Spanish (es), Russian (ru), Korean (ko), French (fr), Japanese (ja), Portuguese (pt), Turkish (tr), Arabic (ar), Italian (it), Dutch (nl), Polish (pl), Czech (cs), Hindi (hi), Persian (fa), Ukrainian (uk), Vietnamese (vi)

**Additional Languages:** Romanian (ro), Thai (th), Greek (el), Hungarian (hu), Danish (da), Finnish (fi), Norwegian (no), Swedish (sv), Hebrew (he), Indonesian (id), Malay (ms), Catalan (ca), Serbian (sr), Croatian (hr), Bulgarian (bg), Slovak (sk), Lithuanian (lt), Latvian (lv), Estonian (et), Slovenian (sl), Bengali (bn), Tamil (ta), Telugu (te), Marathi (mr), Urdu (ur), Kannada (kn), Malayalam (ml), Gujarati (gu), Punjabi (pa), Sinhala (si), Khmer (km), Lao (lo)

**Cohere Transcribe** supports 14 languages: en, de, fr, it, es, pt, el, nl, pl, zh, ja, ko, vi, ar. Language must be specified explicitly.

**Qwen3-ForcedAligner** supports 11 languages: en, zh, de, es, fr, ja, ko, pt, ru, tr, ar

## Breaking Changes from Previous Version ⚠️

This version introduces **breaking changes** from the Transformers-based Whisper implementation:

### 1. Model Names Changed

**Before (Whisper):**
```python
model = whisperx.load_model("large-v2", device="cuda")
```

**After (Qwen3-ASR):**
```python
model = whisperx.load_model("Qwen/Qwen3-ASR-1.7B", device="cuda")
```

### 2. Translation Task Removed

**Before:**
```bash
whisperx audio.wav --task translate  # Translated to English
```

**After:**
```bash
# Translation is NOT supported
# Only transcription in the original language is available
```

### 3. Removed Parameters

The following parameters are no longer supported:
- `suppress_numerals` - Not available in Qwen3-ASR
- `task="translate"` - Only transcription supported

### 4. Alignment Changes

**Before:** Used language-specific Wav2Vec2 models

**After:** Uses unified Qwen3-ForcedAligner (supports 11 languages)

```python
# The API is the same, but uses Qwen3-ForcedAligner internally
model_a, metadata = whisperx.load_align_model(
    language_code="en",
    device=device,
    model_name="Qwen/Qwen3-ForcedAligner-0.6B"  # Optional
)
```

### 5. New Parameters

New parameters added for Qwen3-ASR:
- `forced_aligner` - Specify aligner model
- `dtype` - Model precision: "bfloat16" (recommended), "float16", or "float32"

Use `batch_size` in `transcribe()` or `--batch_size` on the command line to
limit the number of segments in each inference batch. The default is 8.
`max_inference_batch_size` is accepted as a deprecated compatibility alias.

## Migration Guide

### For Command Line Users

**Old:**
```bash
whisperx audio.wav --model large-v2 --task translate
```

**New:**
```bash
# Translation removed - transcription only
whisperx audio.wav --model Qwen/Qwen3-ASR-1.7B --language en
```

### For Python API Users

**Old:**
```python
model = whisperx.load_model("large-v2", device="cuda", task="translate")
result = model.transcribe("audio.wav")
```

**New:**
```python
model = whisperx.load_model(
    "Qwen/Qwen3-ASR-1.7B",  # Changed: new model name
    device="cuda",
    # task parameter removed
    language="en",  # Optional: specify language
    dtype="bfloat16"  # New: control precision
)
result = model.transcribe("audio.wav", batch_size=16)
```

## Advantages of Qwen3-ASR

✅ **More Languages**: 52 languages vs Whisper's ~100 (but better quality for supported ones)
✅ **Better Accuracy**: Improved performance on diverse audio conditions
✅ **Unified Architecture**: Single model architecture for all languages
✅ **Active Development**: Regularly updated by Alibaba's Qwen team
✅ **Efficient Inference**: Optimized for batch processing

## Limitations ⚠️

- **No Translation**: Qwen3-ASR only supports transcription (not translation)
- **Alignment Coverage**: Qwen3-ForcedAligner supports 11 languages (vs 52 for ASR)
- **Model Names**: Breaking change - old Whisper model names won't work
- **Memory Requirements**: Large models require significant GPU memory
- Transcript words not recognized by the aligner may have estimated timestamps
- Overlapping speech handling is limited
- Diarization accuracy may vary depending on audio quality

## Performance Tips

1. **Use bfloat16 precision** for best balance of speed and accuracy
2. **Adjust batch size** based on your GPU memory:
   - 8GB VRAM: `--batch_size 16`
   - 16GB+ VRAM: `--batch_size 32` or higher
3. **Specify language** when known to improve accuracy and speed
4. **Use VAD parameters** (`--vad_onset`, `--vad_offset`) to tune speech detection

## Credits

- Original WhisperX: [m-bain/whisperX](https://github.com/m-bain/whisperX)
- Qwen3-ASR: [Alibaba Qwen Team](https://huggingface.co/Qwen)
- Cohere Transcribe: [CohereLabs](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- OpenAI Whisper: [openai/whisper](https://github.com/openai/whisper)
- pyannote.audio: [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)

## License

This project inherits the license from the original WhisperX repository.

## Citation

If you use this work, please cite the original WhisperX paper:

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2303.00747},
  year={2023}
}
```
