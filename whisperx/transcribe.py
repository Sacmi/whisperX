import argparse
import gc
import os
import sys
import warnings
from contextlib import contextmanager, redirect_stdout
from functools import partial

import torch

from .alignment import align, load_align_model
from .asr import load_model
from .audio import load_audio
from .diarize import DiarizationPipeline, assign_word_speakers
from .utils import (
    LANGUAGES,
    ProgressEmitter,
    TO_LANGUAGE_CODE,
    get_writer,
    optional_float,
    optional_int,
    str2bool,
)


def cli():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="name of the ASR model (e.g., Qwen/Qwen3-ASR-1.7B or CohereLabs/cohere-transcribe-03-2026)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--batch_size", default=argparse.SUPPRESS, type=int, help="maximum number of segments per inference batch (default: 8)")

    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="[DEPRECATED] Qwen3-ASR only supports transcription. Translation is not available.")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    # ASR model params
    parser.add_argument("--forced_aligner", type=str, default=None, help="Qwen3-ForcedAligner model for word-level timestamps (e.g., 'Qwen/Qwen3-ForcedAligner-0.6B')")
    parser.add_argument("--max_inference_batch_size", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model precision (bfloat16 recommended for Qwen3-ASR)")

    # alignment params
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")
    parser.add_argument("--return_char_alignments", action='store_true', help="Return character-level alignments in the output json file")

    # vad params
    parser.add_argument("--vad_onset", type=float, default=0.500, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    parser.add_argument("--chunk_size", type=int, default=30, help="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.")

    # diarization params
    parser.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--min_speakers", default=None, type=int, help="Minimum number of speakers to in audio file")
    parser.add_argument("--max_speakers", default=None, type=int, help="Maximum number of speakers to in audio file")
    parser.add_argument("--diarize_model", default="pyannote/speaker-diarization-community-1", type=str, help="Name of the speaker diarization model to use")

    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")

    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of lines in a segment")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--segment_resolution", type=str, default="sentence", choices=["sentence", "chunk"], help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")

    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")

    parser.add_argument("--print_progress", type=str2bool, default = False, help = "if True, progress will be printed in transcribe() and align() methods.")
    parser.add_argument("--progress_json", type=str2bool, default=False, help="emit machine-readable progress as JSONL to stdout")

    parser.add_argument("--no_repeat_ngram_size", type=optional_int, default=None)
    parser.add_argument("--repetition_penalty", type=optional_float, default=None)
    # fmt: on

    args = parser.parse_args().__dict__
    progress_json: bool = args.pop("progress_json")

    if progress_json:
        with _progress_output() as progress_emitter:
            return _run(args, parser, progress_emitter)
    return _run(args, parser, None)


@contextmanager
def _progress_output():
    original_stdout = sys.stdout
    try:
        stdout_fd = original_stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        with redirect_stdout(sys.stderr):
            yield ProgressEmitter(original_stdout)
        return

    original_stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    progress_stream = os.fdopen(
        os.dup(stdout_fd),
        "w",
        buffering=1,
        encoding=getattr(original_stdout, "encoding", None) or "utf-8",
    )
    os.dup2(stderr_fd, 1)
    try:
        with redirect_stdout(sys.stderr):
            yield ProgressEmitter(progress_stream)
    finally:
        try:
            progress_stream.close()
        finally:
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)


def _run(args, parser, progress_emitter):
    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size", None)
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")

    # ASR model params
    forced_aligner: str = args.pop("forced_aligner")
    max_inference_batch_size: int = args.pop("max_inference_batch_size", None)
    dtype: str = args.pop("dtype")

    if batch_size is not None and max_inference_batch_size is not None:
        parser.error(
            "--batch_size and deprecated --max_inference_batch_size cannot be used together"
        )
    if max_inference_batch_size is not None:
        warnings.warn(
            "--max_inference_batch_size is deprecated; use --batch_size instead.",
            FutureWarning,
            stacklevel=2,
        )
        batch_size = max_inference_batch_size
    if batch_size is None:
        batch_size = 8
    if batch_size < 1:
        parser.error("--batch_size must be a positive integer")

    # model_flush: bool = args.pop("model_flush")
    os.makedirs(output_dir, exist_ok=True)

    align_model: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    task: str = args.pop("task")
    if task == "translate":
        warnings.warn(
            "Translation task is not supported in Qwen3-ASR. Only transcription is available. "
            "Proceeding with transcription."
        )
        task = "transcribe"
        # translation cannot be aligned
        no_align = True

    return_char_alignments: bool = args.pop("return_char_alignments")

    hf_token: str = args.pop("hf_token")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")

    chunk_size: int = args.pop("chunk_size")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")
    diarize_model_name: str = args.pop("diarize_model")
    print_progress: bool = args.pop("print_progress")

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")

    if model_name.endswith(".en") and args["language"] != "en":
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{args['language']}'; using English instead."
            )
        args["language"] = "en"
    align_language = (
        args["language"] if args["language"] is not None else "en"
    )  # default to loading english if not specified

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}

    audio_paths = args.pop("audio")
    multiple_files = len(audio_paths) > 1

    def progress_callback(audio_path):
        if progress_emitter is None:
            return None
        event_file = audio_path if multiple_files else None
        return partial(progress_emitter, file=event_file)

    # Part 1: VAD & ASR Loop
    results = []
    tmp_results = []
    # model = load_model(model_name, device=device, download_root=model_dir)
    if progress_emitter:
        progress_emitter("stage_start", "load_asr")
    model = load_model(
        model_name,
        device=device,
        language=args["language"],
        forced_aligner=forced_aligner,
        dtype=dtype,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
    )
    if progress_emitter:
        progress_emitter("stage_end", "load_asr")

    for audio_path in audio_paths:
        audio = load_audio(audio_path)
        # >> VAD & ASR
        print(">>Performing transcription...")
        result = model.transcribe(
            audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            print_progress=print_progress,
            progress_callback=progress_callback(audio_path),
        )
        results.append((result, audio_path))

    # Unload Whisper and VAD
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Part 2: Align Loop
    if not no_align:
        tmp_results = results
        results = []
        if progress_emitter:
            progress_emitter("stage_start", "load_align")
        align_model, align_metadata = load_align_model(
            align_language, device, model_name=align_model
        )
        if progress_emitter:
            progress_emitter("stage_end", "load_align")
        for result, audio_path in tmp_results:
            align_progress = progress_callback(audio_path)
            # >> Align
            if len(tmp_results) > 1:
                input_audio = audio_path
            else:
                # lazily load audio from part 1
                input_audio = audio

            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language", "en") != align_metadata["language"]:
                    # load new language
                    print(
                        f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language..."
                    )
                    if align_progress:
                        align_progress("stage_start", "load_align")
                    align_model, align_metadata = load_align_model(
                        result["language"], device
                    )
                    if align_progress:
                        align_progress("stage_end", "load_align")
                print(">>Performing alignment...")
                result = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    input_audio,
                    device,
                    interpolate_method=interpolate_method,
                    return_char_alignments=return_char_alignments,
                    print_progress=print_progress,
                    progress_callback=align_progress,
                )
            elif align_progress and len(result["segments"]) == 0:
                align_progress("stage_start", "align")
                align_progress("progress", "align", done=0, total=0)
                align_progress("stage_end", "align")

            results.append((result, audio_path))

        # Unload align model
        del align_model
        gc.collect()
        torch.cuda.empty_cache()

    # >> Diarize
    if diarize:
        if hf_token is None:
            print(
                "Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model..."
            )
        tmp_results = results
        print(">>Performing diarization...")
        print(f">>Using model: {diarize_model_name}")
        results = []
        if progress_emitter:
            progress_emitter("stage_start", "load_diarize")
        diarize_model = DiarizationPipeline(
            model_name=diarize_model_name, token=hf_token, device=device
        )
        if progress_emitter:
            progress_emitter("stage_end", "load_diarize")
        for result, input_audio_path in tmp_results:
            diarize_progress = progress_callback(input_audio_path)
            if diarize_progress:
                diarize_progress("stage_start", "diarize")
            diarize_segments = diarize_model(
                input_audio_path, min_speakers=min_speakers, max_speakers=max_speakers
            )
            result = assign_word_speakers(diarize_segments, result)
            if diarize_progress:
                diarize_progress("stage_end", "diarize")
            results.append((result, input_audio_path))
    # >> Write
    for result, audio_path in results:
        result["language"] = align_language
        writer(result, audio_path, writer_args)


if __name__ == "__main__":
    cli()
