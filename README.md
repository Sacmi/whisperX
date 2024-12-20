# WhisperX
> [!NOTE]
> This is a modified version of WhisperX that uses the transformers library instead of faster-whisper and ctranslate2, which provides a simpler installation process and better compatibility with other Transformers-based models.

<p align="center">
  <a href="https://github.com/m-bain/whisperX/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/m-bain/whisperX.svg"
             alt="GitHub license">
  </a>
</p>

This repository provides fast automatic speech recognition (70x realtime with large-v2) with word-level timestamps and speaker diarization.

Key Features:
- 🎯 Accurate word-level timestamps using wav2vec2 alignment
- 👯‍♂️ Multispeaker ASR using speaker diarization from [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- 🗣️ VAD preprocessing, reduces hallucination & improves transcription quality
- ⚡️ Efficient batch processing for fast transcription
- 🤗 Direct integration with Hugging Face Transformers library

**Whisper** is an ASR model [developed by OpenAI](https://github.com/openai/whisper), trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI's whisper does not natively support batching.

**Phoneme-Based ASR** A suite of models finetuned to recognise the smallest unit of speech distinguishing one word from another, e.g. the element p in "tap". A popular example model is [wav2vec2.0](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).

**Forced Alignment** refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.

**Voice Activity Detection (VAD)** is the detection of the presence or absence of human speech.

**Speaker Diarization** is the process of partitioning an audio stream containing human speech into homogeneous segments according to the identity of each speaker.

## Setup ⚙️

Tested with Python 3.10 and PyTorch 2.

### 1. Create Python Environment

```bash
micromamba create --name whisperx python=3.10
micromamba activate whisperx
```

### 2. Install WhisperX

```bash
pip install git+https://github.com/Sacmi/whisperX.git
```

For development:
```bash
git clone https://github.com/Sacmi/whisperX.git
cd whisperX
pip install -e .
```

You may also need to install ffmpeg, rust etc. Follow openAI instructions here https://github.com/openai/whisper#setup.

### Speaker Diarization
To **enable Speaker Diarization**, include your Hugging Face access token (read) that you can generate from [Here](https://huggingface.co/settings/tokens) after the `--hf_token` argument and accept the user agreement for the following models: [Segmentation](https://huggingface.co/pyannote/segmentation-3.0) and [Speaker-Diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (if you choose to use Speaker-Diarization 2.x, follow requirements [here](https://huggingface.co/pyannote/speaker-diarization) instead.)

## Usage 💬

### Command Line

Basic usage:
```bash
whisperx examples/sample01.wav
```

With word highlighting in subtitles:
```bash
whisperx examples/sample01.wav --highlight_words True
```

With speaker diarization:
```bash
whisperx examples/sample01.wav --diarize --highlight_words True
```

### Python Usage

```python
import whisperx
import gc 

device = "cuda" 
audio_file = "audio.mp3"
batch_size = 16 

# 1. Transcribe with whisperx
model = whisperx.load_model("large-v2", device)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# Optional: Clean up memory
gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)
print(result["segments"]) # segments with speaker IDs
```

## Important Changes in This Version

1. Using Transformers ASR pipeline instead of faster-whisper
2. Removed dependency on ctranslate2
3. Simplified model loading parameters
4. Updated version requirements for dependencies
5. Better integration with other Transformers-based models

## Limitations ⚠️

- Transcript words which do not contain characters in the alignment models dictionary cannot be aligned
- Overlapping speech handling is limited 
- Diarization accuracy may vary
- Language specific wav2vec2 model is needed for alignment