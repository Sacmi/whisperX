from typing import Callable, List, Union, Optional
import logging
import warnings

# Suppress HF transformers logging warning about pad_token_id.
# Must target the specific child logger — parent logger filters are skipped during propagation.
logging.getLogger("transformers.generation.utils").addFilter(
    type("_PadTokenFilter", (logging.Filter,), {
        "filter": lambda self, r: "Setting `pad_token_id` to `eos_token_id`" not in r.getMessage()
    })()
)

import numpy as np
import torch
from tqdm import tqdm
from qwen_asr import Qwen3ASRModel
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from .audio import SAMPLE_RATE, load_audio
from .vad import load_vad_model, merge_chunks
from .types import TranscriptionResult, SingleSegment

# WhisperX uses ISO 639-1 codes, Qwen3-ASR uses full language names
WHISPERX_TO_QWEN_LANGUAGE = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "ar": "Arabic",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "hi": "Hindi",
    "fa": "Persian",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "ro": "Romanian",
    "th": "Thai",
    "el": "Greek",
    "hu": "Hungarian",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "sv": "Swedish",
    "he": "Hebrew",
    "id": "Indonesian",
    "ms": "Malay",
    "ca": "Catalan",
    "sr": "Serbian",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "sk": "Slovak",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "sl": "Slovenian",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ur": "Urdu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "lo": "Lao",
}

# Reverse mapping for language detection
QWEN_TO_WHISPERX_LANGUAGE = {v: k for k, v in WHISPERX_TO_QWEN_LANGUAGE.items()}

# Cohere Transcribe supported languages (ISO 639-1 codes)
COHERE_SUPPORTED_LANGUAGES = {
    "en", "de", "fr", "it", "es", "pt", "el", "nl", "pl", "zh", "ja", "ko", "vi", "ar"
}


class Qwen3ASRPipeline:
    """
    Qwen3-ASR model wrapper for WhisperX compatibility.
    """

    def __init__(
        self,
        asr_model: Qwen3ASRModel,
        vad,
        vad_params: dict,
        device: Union[int, str, "torch.device"],
        language: Optional[str] = None,
        forced_aligner: Optional[str] = None,
        default_batch_size: int = 8,
    ):
        self.asr_model = asr_model
        self.vad_model = vad
        self.vad_params = vad_params
        self.device = device
        self.language = language
        self.forced_aligner = forced_aligner
        self.default_batch_size = default_batch_size

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size=None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Qwen3-ASR with VAD segmentation.

        Args:
            audio: Audio file path or numpy array
            batch_size: Maximum number of segments in a Qwen inference batch
            chunk_size: Maximum chunk size for VAD merging
            print_progress: Whether to print progress
            combined_progress: Whether this is part of a combined progress (affects percentage)
            progress_callback: Optional machine-readable progress callback

        Returns:
            TranscriptionResult with segments and detected language
        """
        process_batch_size = (
            self.default_batch_size if batch_size is None else batch_size
        )
        if process_batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        # Load audio if string path provided
        if isinstance(audio, str):
            audio = load_audio(audio)

        # Apply VAD segmentation
        if progress_callback:
            progress_callback("stage_start", "vad")
        vad_segments = self.vad_model(
            {
                "waveform": torch.from_numpy(audio).unsqueeze(0),
                "sample_rate": SAMPLE_RATE,
            }
        )
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self.vad_params.get("vad_onset", 0.500),
            offset=self.vad_params.get("vad_offset", 0.363),
        )
        if progress_callback:
            progress_callback("stage_end", "vad")

        # Extract audio chunks for each VAD segment
        # Qwen3-ASR expects tuples of (audio_array, sample_rate)
        audio_chunks = []
        for seg in vad_segments:
            f1 = int(seg["start"] * SAMPLE_RATE)
            f2 = int(seg["end"] * SAMPLE_RATE)
            chunk = audio[f1:f2]
            # Pass as tuple (audio, sample_rate) for Qwen3-ASR
            audio_chunks.append((chunk, SAMPLE_RATE))

        total_segments = len(audio_chunks)
        if progress_callback:
            progress_callback("stage_start", "transcribe")

        # Handle empty segments
        if total_segments == 0:
            if progress_callback:
                progress_callback("progress", "transcribe", done=0, total=0)
                progress_callback("stage_end", "transcribe")
            return {"segments": [], "language": self.language or "en"}

        # Map language to Qwen format
        qwen_language = None
        if self.language:
            qwen_language = WHISPERX_TO_QWEN_LANGUAGE.get(self.language)
            if qwen_language is None:
                warnings.warn(
                    f"Language '{self.language}' not supported by Qwen3-ASR. "
                    f"Supported languages: {list(WHISPERX_TO_QWEN_LANGUAGE.keys())}. "
                    f"Falling back to auto-detection."
                )

        segments: List[SingleSegment] = []
        detected_language = self.language  # Default to specified language

        # Create progress bar
        pbar = None
        if print_progress:
            pbar_desc = "Transcribing (50%)" if combined_progress else "Transcribing"
            pbar = tqdm(total=total_segments, desc=pbar_desc, unit="segment")

        # qwen-asr owns inference chunking; submitting all segments here avoids
        # a second, conflicting batching layer in WhisperX.
        self.asr_model.max_inference_batch_size = process_batch_size
        batch_results = self.asr_model.transcribe(
            audio=audio_chunks,
            language=qwen_language,
        )

        for idx, result in enumerate(batch_results):
            text = result.text.strip()

            if result.language:
                detected_language = QWEN_TO_WHISPERX_LANGUAGE.get(
                    result.language, detected_language or "en"
                )

            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]["start"], 3),
                    "end": round(vad_segments[idx]["end"], 3),
                }
            )

            if pbar:
                pbar.update(1)

        if progress_callback:
            progress_callback(
                "progress", "transcribe", done=total_segments, total=total_segments
            )

        # Close progress bar
        if pbar:
            pbar.close()
        if progress_callback:
            progress_callback("stage_end", "transcribe")

        return {"segments": segments, "language": detected_language or "en"}


class CohereASRPipeline:
    """
    Cohere Transcribe model wrapper for WhisperX compatibility.
    """

    def __init__(
        self,
        processor,
        asr_model,
        vad,
        vad_params: dict,
        device: Union[int, str, "torch.device"],
        language: Optional[str] = None,
        default_batch_size: int = 8,
    ):
        self.processor = processor
        self.asr_model = asr_model
        self.vad_model = vad
        self.vad_params = vad_params
        self.device = device
        self.language = language
        self.default_batch_size = default_batch_size

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size=None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Cohere Transcribe with VAD segmentation.
        """
        process_batch_size = (
            self.default_batch_size if batch_size is None else batch_size
        )
        if process_batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        if isinstance(audio, str):
            audio = load_audio(audio)

        # Apply VAD segmentation
        if progress_callback:
            progress_callback("stage_start", "vad")
        vad_segments = self.vad_model(
            {
                "waveform": torch.from_numpy(audio).unsqueeze(0),
                "sample_rate": SAMPLE_RATE,
            }
        )
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self.vad_params.get("vad_onset", 0.500),
            offset=self.vad_params.get("vad_offset", 0.363),
        )
        if progress_callback:
            progress_callback("stage_end", "vad")

        # Extract audio chunks as numpy arrays
        audio_chunks = []
        for seg in vad_segments:
            f1 = int(seg["start"] * SAMPLE_RATE)
            f2 = int(seg["end"] * SAMPLE_RATE)
            audio_chunks.append(audio[f1:f2])

        total_segments = len(audio_chunks)
        if progress_callback:
            progress_callback("stage_start", "transcribe")

        if total_segments == 0:
            if progress_callback:
                progress_callback("progress", "transcribe", done=0, total=0)
                progress_callback("stage_end", "transcribe")
            return {"segments": [], "language": self.language or "ja"}

        segments: List[SingleSegment] = []
        pbar = None
        if print_progress:
            pbar_desc = "Transcribing (50%)" if combined_progress else "Transcribing"
            pbar = tqdm(total=total_segments, desc=pbar_desc, unit="segment")

        for batch_start in range(0, total_segments, process_batch_size):
            batch_end = min(batch_start + process_batch_size, total_segments)
            batch_chunks = audio_chunks[batch_start:batch_end]
            batch_vad_segments = vad_segments[batch_start:batch_end]

            batch_results = self.asr_model.transcribe(
                processor=self.processor,
                audio_arrays=batch_chunks,
                sample_rates=[SAMPLE_RATE] * len(batch_chunks),
                language=self.language,
            )

            for idx, text in enumerate(batch_results):
                segments.append(
                    {
                        "text": text.strip(),
                        "start": round(batch_vad_segments[idx]["start"], 3),
                        "end": round(batch_vad_segments[idx]["end"], 3),
                    }
                )
                if pbar:
                    pbar.update(1)

            if progress_callback:
                progress_callback(
                    "progress", "transcribe", done=batch_end, total=total_segments
                )

        if pbar:
            pbar.close()
        if progress_callback:
            progress_callback("stage_end", "transcribe")

        return {"segments": segments, "language": self.language or "ja"}


def _load_cohere_model(
    model_name: str,
    vad,
    vad_params: dict,
    device,
    language: Optional[str],
    dtype,
    default_batch_size: int,
    **kwargs,
):
    """Load Cohere Transcribe model."""
    if language is not None and language not in COHERE_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Language '{language}' is not supported by Cohere Transcribe. "
            f"Supported: {sorted(COHERE_SUPPORTED_LANGUAGES)}"
        )
    if language is None:
        warnings.warn(
            "Cohere Transcribe does not support language auto-detection. "
            "Defaulting to 'ja'. Set --language explicitly for other languages."
        )
        language = "ja"

    # Map device
    if isinstance(device, int):
        device_map = f"cuda:{device}" if device >= 0 else "cpu"
    elif device == "cuda":
        device_map = "cuda:0"
    else:
        device_map = device

    # Convert string dtype
    if isinstance(dtype, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"Loading Cohere Transcribe model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, trust_remote_code=True, dtype=dtype, device_map=device_map
    )
    model.eval()

    return CohereASRPipeline(
        processor=processor,
        asr_model=model,
        vad=vad,
        vad_params=vad_params,
        device=device,
        language=language,
        default_batch_size=default_batch_size,
    )


def load_model(
    model_name: str,
    device,
    vad_model=None,
    vad_options=None,
    language: Optional[str] = None,
    forced_aligner: Optional[str] = None,
    max_inference_batch_size: Optional[int] = None,
    dtype=torch.bfloat16,
    **kwargs,
):
    """
    Load ASR model for inference.

    Automatically detects the model type (Qwen3-ASR or Cohere Transcribe)
    based on the model name and returns the appropriate pipeline.

    Args:
        model_name: Name/path of the ASR model
            (e.g., "Qwen/Qwen3-ASR-1.7B" or "CohereLabs/cohere-transcribe-03-2026")
        device: Device for loading model ("cuda", "cpu", or device index)
        vad_model: Optional pre-loaded VAD model
        vad_options: Optional dict of VAD options (vad_onset, vad_offset)
        language: Optional language code (ISO 639-1, e.g., "en", "zh")
        forced_aligner: Optional Qwen3-ForcedAligner model name for word-level timestamps
        max_inference_batch_size: Deprecated default batch size alias
        dtype: Model precision (torch.bfloat16, torch.float16, or torch.float32)
        **kwargs: Additional arguments passed to model loading

    Returns:
        ASR pipeline instance ready for transcription
    """
    is_cohere = "cohere-transcribe" in model_name.lower()

    default_batch_size = 8
    if max_inference_batch_size is not None:
        if max_inference_batch_size < 1:
            raise ValueError("max_inference_batch_size must be a positive integer")
        warnings.warn(
            "max_inference_batch_size is deprecated; pass batch_size to "
            "transcribe() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        default_batch_size = max_inference_batch_size

    # Set up VAD options
    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363,
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    # Load or use provided VAD model
    if vad_model is not None:
        vad = vad_model
    else:
        vad = load_vad_model(
            torch.device(
                device
                if isinstance(device, str)
                else f"cuda:{device}"
                if device >= 0
                else "cpu"
            ),
            token=None,
            **default_vad_options,
        )

    if is_cohere:
        return _load_cohere_model(
            model_name=model_name,
            vad=vad,
            vad_params=default_vad_options,
            device=device,
            language=language,
            dtype=dtype,
            default_batch_size=default_batch_size,
            **kwargs,
        )

    # Qwen3-ASR path
    # Map device to device_map format
    if isinstance(device, int):
        if device >= 0:
            device_map = f"cuda:{device}"
        else:
            device_map = "cpu"
    elif device == "cuda":
        device_map = "cuda:0"
    else:
        device_map = device

    # Convert string dtype to torch dtype
    if isinstance(dtype, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"Loading Qwen3-ASR model: {model_name}")
    asr_model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=default_batch_size,
        **kwargs,
    )

    # Warn about removed parameters if present in kwargs
    if kwargs.get("task") and kwargs["task"] != "transcribe":
        warnings.warn(
            "Translation task is not supported in Qwen3-ASR. Only transcription is available."
        )

    if kwargs.get("suppress_numerals"):
        warnings.warn(
            "suppress_numerals parameter is not supported in Qwen3-ASR and will be ignored."
        )

    return Qwen3ASRPipeline(
        asr_model=asr_model,
        vad=vad,
        vad_params=default_vad_options,
        device=device,
        language=language,
        forced_aligner=forced_aligner,
        default_batch_size=default_batch_size,
    )
