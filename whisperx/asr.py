from typing import List, Union, Optional
import warnings

import numpy as np
import torch
from tqdm import tqdm
from qwen_asr import Qwen3ASRModel

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
    ):
        self.asr_model = asr_model
        self.vad_model = vad
        self.vad_params = vad_params
        self.device = device
        self.language = language
        self.forced_aligner = forced_aligner

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size=None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Qwen3-ASR with VAD segmentation.

        Args:
            audio: Audio file path or numpy array
            batch_size: Batch size for inference (not used, controlled by model)
            chunk_size: Maximum chunk size for VAD merging
            print_progress: Whether to print progress
            combined_progress: Whether this is part of a combined progress (affects percentage)

        Returns:
            TranscriptionResult with segments and detected language
        """
        # Load audio if string path provided
        if isinstance(audio, str):
            audio = load_audio(audio)

        # Apply VAD segmentation
        vad_segments = self.vad_model(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE}
        )
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self.vad_params.get("vad_onset", 0.500),
            offset=self.vad_params.get("vad_offset", 0.363),
        )

        # Extract audio chunks for each VAD segment
        # Qwen3-ASR expects tuples of (audio_array, sample_rate)
        audio_chunks = []
        for seg in vad_segments:
            f1 = int(seg["start"] * SAMPLE_RATE)
            f2 = int(seg["end"] * SAMPLE_RATE)
            chunk = audio[f1:f2]
            # Pass as tuple (audio, sample_rate) for Qwen3-ASR
            audio_chunks.append((chunk, SAMPLE_RATE))

        # Handle empty segments
        if len(audio_chunks) == 0:
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

        # Transcribe segments in smaller batches to avoid OOM
        # The batch_size parameter is not used by Qwen3-ASR directly,
        # but we use it to control how many segments we process at once
        total_segments = len(audio_chunks)
        segments: List[SingleSegment] = []
        detected_language = self.language  # Default to specified language

        # Process in batches to avoid memory issues
        # Use a reasonable batch size (default to 8 segments at a time)
        process_batch_size = min(batch_size or 8, 8)

        # Create progress bar
        pbar = None
        if print_progress:
            pbar_desc = "Transcribing (50%)" if combined_progress else "Transcribing"
            pbar = tqdm(total=total_segments, desc=pbar_desc, unit="segment")

        for batch_start in range(0, total_segments, process_batch_size):
            batch_end = min(batch_start + process_batch_size, total_segments)
            batch_chunks = audio_chunks[batch_start:batch_end]
            batch_vad_segments = vad_segments[batch_start:batch_end]

            # Transcribe this batch
            batch_results = self.asr_model.transcribe(
                audio=batch_chunks,
                language=qwen_language,
            )

            # Format results for this batch
            for idx, result in enumerate(batch_results):
                # Extract text and language from result
                text = result.text.strip()

                # Map detected language back to ISO code
                if result.language:
                    detected_language = QWEN_TO_WHISPERX_LANGUAGE.get(
                        result.language, detected_language or "en"
                    )

                # Create segment with timestamps from VAD
                segments.append(
                    {
                        "text": text,
                        "start": round(batch_vad_segments[idx]["start"], 3),
                        "end": round(batch_vad_segments[idx]["end"], 3),
                    }
                )

                # Update progress bar
                if pbar:
                    pbar.update(1)

        # Close progress bar
        if pbar:
            pbar.close()

        return {
            "segments": segments,
            "language": detected_language or "en"
        }


def load_model(
    model_name: str,
    device,
    vad_model=None,
    vad_options=None,
    language: Optional[str] = None,
    forced_aligner: Optional[str] = None,
    max_inference_batch_size: int = 32,
    dtype=torch.bfloat16,
    **kwargs
):
    """
    Load Qwen3-ASR model for inference.

    Args:
        model_name: Name/path of the Qwen3-ASR model (e.g., "Qwen/Qwen3-ASR-1.7B")
        device: Device for loading model ("cuda", "cpu", or device index)
        vad_model: Optional pre-loaded VAD model
        vad_options: Optional dict of VAD options (vad_onset, vad_offset)
        language: Optional language code (ISO 639-1, e.g., "en", "zh")
        forced_aligner: Optional Qwen3-ForcedAligner model name for word-level timestamps
        max_inference_batch_size: Maximum batch size for Qwen3-ASR inference
        dtype: Model precision (torch.bfloat16, torch.float16, or torch.float32)
        **kwargs: Additional arguments passed to Qwen3ASRModel.from_pretrained()

    Returns:
        Qwen3ASRPipeline instance ready for transcription
    """
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
            torch.device(device if isinstance(device, str) else f"cuda:{device}" if device >= 0 else "cpu"),
            token=None,
            **default_vad_options
        )

    # Map device to device_map format expected by Qwen3-ASR
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

    # Initialize Qwen3-ASR model
    print(f"Loading Qwen3-ASR model: {model_name}")
    asr_model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=max_inference_batch_size,
        **kwargs
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
    )
