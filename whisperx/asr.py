from typing import List, Union, Optional

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer

from .audio import SAMPLE_RATE, load_audio
from .vad import load_vad_model, merge_chunks
from .types import TranscriptionResult, SingleSegment

def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(len(tokenizer)):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens

class TransformersASRPipeline:
    """
    Huggingface Pipeline wrapper for Transformers ASR models.
    """

    def __init__(
            self,
            model_name: str,
            vad,
            vad_params: dict,
            device: Union[int, str, "torch.device"] = -1,
            tokenizer=None,
            language: Optional[str] = None,
            suppress_numerals: bool = False,
            task: Optional[str] = None,
            **kwargs
    ):
        self.vad_model = vad
        self.vad_params = vad_params
        self.suppress_numerals = suppress_numerals
        self.language = language
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.task = task

        # Инициализация transformers ASR pipeline
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=0 if (isinstance(device, int) and device >= 0) or device == "cuda" else -1,
            **kwargs
        )

        if self.suppress_numerals and self.asr_pipeline.tokenizer is not None:
            self.numeral_symbol_tokens = find_numeral_symbol_tokens(self.asr_pipeline.tokenizer)

    def transcribe(
            self,
            audio: Union[str, np.ndarray],
            batch_size=None,
            chunk_size=30,
            print_progress=False,
            combined_progress=False
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                yield audio[f1:f2]

        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self.vad_params.get("vad_onset", 0.500),
            offset=self.vad_params.get("vad_offset", 0.363),
        )

        segments: List[SingleSegment] = []
        batch_size = batch_size or 1
        total_segments = len(vad_segments)
        audio_chunks = list(data(audio, vad_segments))

        generate_kwargs = {}

        # TODO
        if self.model_name == "litagin/anime-whisper":
            generate_kwargs["no_repeat_ngram_size"] = 0
            generate_kwargs["repetition_penalty"] = 1.0

        if self.task is not None:
            generate_kwargs["task"] = self.task
        if self.language is not None:
            generate_kwargs["language"] = self.language

        # Использование transformers pipeline в режиме пакетной обработки
        results = self.asr_pipeline(audio_chunks, batch_size=batch_size, generate_kwargs=generate_kwargs)

        for idx, out in enumerate(results):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            if self.suppress_numerals and self.tokenizer is not None:
                tokens = self.tokenizer.encode(text)
                tokens = [tok for tok in tokens if tok not in self.numeral_symbol_tokens]
                text = self.tokenizer.decode(tokens)
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )

        return {"segments": segments, "language": self.language}

def load_model(
        model_name: str,
        device,
        vad_model=None,
        vad_options=None,
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        **kwargs
):
    '''Загружает модель ASR для инференса с использованием transformers.
    Args:
        model_name: str - Название модели ASR для загрузки.
        device: str или int - Устройство для загрузки модели (например, "cuda" или -1 для CPU).
        vad_model: Optional - Модель VAD для использования.
        vad_options: Optional[dict] - Словарь опций для VAD.
        language: Optional[str] - Язык модели.
        suppress_numerals: bool - Нужно ли подавлять числительные и символы в выводе.
        kwargs: Дополнительные аргументы для transformers pipeline.
    Returns:
        Экземпляр TransformersASRPipeline.
    '''

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is not None:
        vad = vad_model
    else:
        vad = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options)

    # Инициализация токенизатора, если требуется подавление числительных
    tokenizer = None
    if suppress_numerals:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return TransformersASRPipeline(
        model_name=model_name,
        vad=vad,
        vad_params=default_vad_options,
        device=device,
        language=language,
        suppress_numerals=suppress_numerals,
        tokenizer=tokenizer,
        **kwargs
    )
