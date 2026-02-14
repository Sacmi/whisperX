""""
Forced Alignment with Qwen3-ForcedAligner
Based on WhisperX by C. Max Bain
Modified to use Qwen3-ForcedAligner
"""
from dataclasses import dataclass
from typing import Iterable, Union, List
import warnings

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from qwen_asr import Qwen3ForcedAligner

from .audio import SAMPLE_RATE, load_audio
from .utils import interpolate_nans
from .types import AlignedTranscriptionResult, SingleSegment, SingleAlignedSegment, SingleWordSegment
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

PUNKT_ABBREVIATIONS = ['dr', 'vs', 'mr', 'mrs', 'prof']

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

# Qwen3-ForcedAligner supported languages (11 languages)
QWEN_ALIGNER_SUPPORTED_LANGUAGES = [
    "en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru", "tr", "ar"
]

DEFAULT_QWEN_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"

# Language code to full name mapping for Qwen3-ForcedAligner
LANGUAGE_CODE_TO_NAME = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "ar": "Arabic",
}


def load_align_model(language_code, device, model_name=None, model_dir=None):
    """
    Load Qwen3-ForcedAligner for word-level timestamps.

    Args:
        language_code: ISO 639-1 language code
        device: Device to load model on
        model_name: Optional custom aligner model name
        model_dir: Unused, kept for API compatibility

    Returns:
        Tuple of (aligner_model, metadata)
    """
    # Check if language is supported
    if language_code not in QWEN_ALIGNER_SUPPORTED_LANGUAGES:
        warnings.warn(
            f"Qwen3-ForcedAligner does not officially support language '{language_code}'. "
            f"Supported languages: {QWEN_ALIGNER_SUPPORTED_LANGUAGES}. "
            f"Alignment may not work correctly."
        )

    # Use default model if none specified
    if model_name is None:
        model_name = DEFAULT_QWEN_ALIGNER

    # Map device to device_map format
    if isinstance(device, str):
        if device.startswith("cuda"):
            device_map = device
        elif device == "cpu":
            device_map = "cpu"
        else:
            device_map = "cuda:0"
    else:
        # Assume it's a torch.device
        device_map = str(device)

    print(f"Loading Qwen3-ForcedAligner: {model_name}")
    try:
        aligner = Qwen3ForcedAligner.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=torch.bfloat16,
        )
    except Exception as e:
        print(f"Error loading Qwen3-ForcedAligner: {e}")
        raise ValueError(f"Failed to load Qwen3-ForcedAligner '{model_name}'")

    align_metadata = {
        "language": language_code,
        "type": "qwen3",
        "model_name": model_name
    }

    return aligner, align_metadata


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """
    Align transcription using Qwen3-ForcedAligner to get word-level timestamps.

    Args:
        transcript: Segments with text and timestamps
        model: Qwen3-ForcedAligner model
        align_model_metadata: Metadata including language and model type
        audio: Audio data (path, numpy array, or torch tensor)
        device: Device for computation
        interpolate_method: Method to handle missing timestamps
        return_char_alignments: Whether to return character-level alignments
        print_progress: Whether to print progress
        combined_progress: Whether this is part of combined progress

    Returns:
        Aligned transcription with word-level timestamps
    """

    # Load audio if needed
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata.get("type", "qwen3")

    # For Qwen3-ForcedAligner, use a different alignment approach
    if model_type == "qwen3":
        return _align_with_qwen3(
            transcript,
            model,
            model_lang,
            audio,
            MAX_DURATION,
            interpolate_method,
            return_char_alignments,
            print_progress,
            combined_progress,
        )

    # Fall back to original Wav2Vec2 alignment for compatibility
    # (This is the legacy path, kept for reference)
    model_dictionary = align_model_metadata.get("dictionary", {})

    # 1. Preprocess to keep only characters in dictionary
    total_segments = len(transcript)


def _align_with_qwen3(
    transcript: Iterable[SingleSegment],
    model: Qwen3ForcedAligner,
    language: str,
    audio: torch.Tensor,
    max_duration: float,
    interpolate_method: str,
    return_char_alignments: bool,
    print_progress: bool,
    combined_progress: bool,
) -> AlignedTranscriptionResult:
    """
    Align transcription using Qwen3-ForcedAligner.

    This function uses Qwen3-ForcedAligner to generate word-level timestamps
    for each segment in the transcript.
    """
    aligned_segments: List[SingleAlignedSegment] = []
    transcript_list = list(transcript)
    total_segments = len(transcript_list)

    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
    sentence_splitter = PunktSentenceTokenizer(punkt_param)

    # Create progress bar
    pbar = None
    if print_progress:
        pbar_desc = "Aligning (100%)" if combined_progress else "Aligning"
        pbar = tqdm(total=total_segments, desc=pbar_desc, unit="segment")

    for sdx, segment in enumerate(transcript_list):
        if pbar:
            pbar.update(1)

        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        # Skip empty segments
        if not text or len(text.strip()) == 0:
            aligned_segments.append({
                "start": t1,
                "end": t2,
                "text": text,
                "words": [],
            })
            continue

        # Check segment validity
        if t1 >= max_duration:
            print(f'Skipping segment ("{text}"): start time beyond audio duration')
            aligned_segments.append({
                "start": t1,
                "end": t2,
                "text": text,
                "words": [],
            })
            continue

        # Extract audio segment
        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)
        audio_segment = audio[:, f1:f2].squeeze(0)

        # Use Qwen3-ForcedAligner to get word-level timestamps
        try:
            # Convert audio to numpy array and pass as tuple (waveform, sample_rate)
            audio_np = audio_segment.cpu().numpy()

            # Convert language code to full name (e.g., "ja" -> "Japanese")
            lang_name = LANGUAGE_CODE_TO_NAME.get(language, "English")

            # Call the align method with text
            align_results = model.align(
                audio=(audio_np, SAMPLE_RATE),
                text=text,
                language=lang_name,
            )

            # Parse alignment results
            if align_results and len(align_results) > 0:
                result = align_results[0]
                # Extract word timings from ForcedAlignResult
                words = _parse_word_timings(result, text, t1, t2, language)
            else:
                # Fallback: split text into words without precise timing
                words = _create_fallback_words(text, t1, t2, language)

        except Exception as e:
            warnings.warn(f"Alignment failed for segment '{text}': {e}. Using fallback.")
            words = _create_fallback_words(text, t1, t2, language)

        # Split into sentences
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        # Create aligned subsegments for each sentence
        aligned_subsegments = []
        for sstart, send in sentence_spans:
            sentence_text = text[sstart:send]

            # Find words that belong to this sentence
            sentence_words = []
            for word in words:
                word_text = word.get("word", "")
                # Simple heuristic: word belongs to sentence if it's in the sentence text
                if word_text in sentence_text:
                    sentence_words.append(word)

            # Calculate sentence start/end from word timings
            sentence_start = min([w["start"] for w in sentence_words]) if sentence_words else t1
            sentence_end = max([w["end"] for w in sentence_words]) if sentence_words else t2

            aligned_subsegments.append({
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": sentence_words,
            })

        aligned_segments += aligned_subsegments

    # Close progress bar
    if pbar:
        pbar.close()

    # Create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment.get("words", [])

    return {"segments": aligned_segments, "word_segments": word_segments}


def _restore_punctuation(words: List[dict], original_text: str, language: str) -> List[dict]:
    """
    Restore punctuation from original text to aligned words.

    The forced aligner removes punctuation during tokenization, so we need to
    add it back by matching words to the original text.

    Args:
        words: List of word dictionaries from aligner (without punctuation)
        original_text: Original text with punctuation
        language: Language code

    Returns:
        List of word dictionaries with punctuation restored
    """
    if not words:
        return words

    # Build clean text from aligned words (without punctuation)
    aligned_chars = [w["word"] for w in words]

    # Match each aligned word/char to position in original text
    result_words = []
    orig_idx = 0

    for i, word_dict in enumerate(words):
        word_clean = word_dict["word"]

        # Find this word/char in original text starting from orig_idx
        start_search = orig_idx
        word_start = original_text.find(word_clean, start_search)

        if word_start == -1:
            # If not found, keep word as-is
            result_words.append(word_dict)
            continue

        # Collect any punctuation before this word (shouldn't happen usually)
        prefix_punct = original_text[orig_idx:word_start]

        # The word itself
        word_end = word_start + len(word_clean)

        # Collect punctuation after this word (before next word)
        punct_end = word_end
        # Look ahead to see where next word starts (or end of text)
        if i + 1 < len(words):
            next_word = words[i + 1]["word"]
            next_pos = original_text.find(next_word, word_end)
            if next_pos != -1:
                punct_end = next_pos
            else:
                # Next word not found, take all remaining punctuation
                punct_end = len(original_text)
        else:
            # Last word, take all remaining text
            punct_end = len(original_text)

        # Extract the word with following punctuation
        word_with_punct = original_text[word_start:punct_end]

        # For languages without spaces, be more conservative
        # Only include punctuation that directly follows (no spaces)
        if language in LANGUAGES_WITHOUT_SPACES:
            # Take word + immediately following punctuation
            punct_idx = word_end
            while punct_idx < len(original_text):
                ch = original_text[punct_idx]
                if _is_cjk_char(ch) or ch.isalnum():
                    break
                punct_idx += 1
            word_with_punct = original_text[word_start:punct_idx]
            orig_idx = punct_idx
        else:
            orig_idx = punct_end

        result_words.append({
            "word": word_with_punct.strip() if language not in LANGUAGES_WITHOUT_SPACES else word_with_punct,
            "start": word_dict["start"],
            "end": word_dict["end"],
            "score": word_dict.get("score", 1.0),
        })

    return result_words


def _is_cjk_char(ch: str) -> bool:
    """Check if character is CJK (Chinese/Japanese/Korean)"""
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or     # CJK Unified Ideographs
        0x3400 <= code <= 0x4DBF or     # Extension A
        0x20000 <= code <= 0x2A6DF or   # Extension B
        0x3040 <= code <= 0x309F or     # Hiragana
        0x30A0 <= code <= 0x30FF or     # Katakana
        0xAC00 <= code <= 0xD7AF        # Hangul
    )


def _parse_word_timings(
    result,
    text: str,
    segment_start: float,
    segment_end: float,
    language: str,
) -> List[dict]:
    """
    Parse word-level timings from Qwen3-ForcedAligner result.

    Args:
        result: ForcedAlignResult object with items attribute
        text: Original segment text
        segment_start: Segment start time in the full audio
        segment_end: Segment end time in the full audio
        language: Language code

    Returns:
        List of word dictionaries with timestamps
    """
    words = []

    # Parse ForcedAlignResult items
    if hasattr(result, 'items') and result.items:
        for item in result.items:
            words.append({
                "word": item.text,
                "start": segment_start + item.start_time,
                "end": segment_start + item.end_time,
                "score": 1.0,
            })

        # Restore punctuation from original text
        words = _restore_punctuation(words, text, language)
    else:
        # Fallback: create uniform word distribution
        words = _create_fallback_words(text, segment_start, segment_end, language)

    return words


def _create_fallback_words(
    text: str,
    start: float,
    end: float,
    language: str,
) -> List[dict]:
    """
    Create word segments with uniform time distribution when alignment fails.

    Args:
        text: Text to split into words
        start: Segment start time
        end: Segment end time
        language: Language code

    Returns:
        List of word dictionaries with estimated timestamps
    """
    # Split text into words
    if language not in LANGUAGES_WITHOUT_SPACES:
        words_list = text.split()
    else:
        # For languages without spaces (Chinese, Japanese), treat each character as a unit
        words_list = list(text)

    if not words_list:
        return []

    # Distribute time uniformly across words
    duration = end - start
    time_per_word = duration / len(words_list)

    words = []
    for i, word in enumerate(words_list):
        word_start = start + i * time_per_word
        word_end = start + (i + 1) * time_per_word
        words.append({
            "word": word,
            "start": round(word_start, 3),
            "end": round(word_end, 3),
            "score": 1.0,  # Default score
        })

    return words


# Legacy Wav2Vec2 alignment code below (for reference, not used with Qwen3)
def _legacy_wav2vec2_align(transcript, model_dictionary, model_type, model_lang):
    """Legacy alignment code for Wav2Vec2 models. Not used with Qwen3."""
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")
            
        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")
            
            # ignore whitespace at beginning and end of transcript
            if cdx < num_leading:
                pass
            elif cdx > len(text) - num_trailing - 1:
                pass
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd]):
                clean_wdx.append(wdx)

                
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment["clean_char"] = clean_char
        segment["clean_cdx"] = clean_cdx
        segment["clean_wdx"] = clean_wdx
        segment["sentence_spans"] = sentence_spans
    
    aligned_segments: List[SingleAlignedSegment] = []
    
    # 2. Get prediction matrix from alignment model & align
    for sdx, segment in enumerate(transcript):
        
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
        }

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment["clean_char"]) == 0:
            print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            print(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping...')
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment["clean_char"])
        tokens = [model_dictionary[c] for c in text_clean]

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        # TODO: Probably can get some speedup gain with batched inference here
        waveform_segment = audio[:, f1:f2]
        # Handle the minimum input length for wav2vec2 models
        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None
            
        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device), lengths=lengths)
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)

        if path is None:
            print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)

        duration = t2 -t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment["clean_cdx"]:
                char_seg = char_segments[segment["clean_cdx"].index(cdx)]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                }
            )

            # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx+1] == " ":
                word_idx += 1
            
        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        # assign sentence_idx to each character index
        char_segments_arr["sentence-idx"] = None
        for sdx, (sstart, send) in enumerate(segment["sentence_spans"]):
            curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
            char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx
        
            sentence_text = text[sstart:send]
            sentence_start = curr_chars["start"].min()
            end_chars = curr_chars[curr_chars["char"] != ' ']
            sentence_end = end_chars["end"].max()
            sentence_words = []

            for word_idx in curr_chars["word-idx"].unique():
                word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if len(word_text) == 0:
                    continue

                # dont use space character for alignment
                word_chars = word_chars[word_chars["char"] != " "]

                word_start = word_chars["start"].min()
                word_end = word_chars["end"].max()
                word_score = round(word_chars["score"].mean(), 3)

                # -1 indicates unalignable 
                word_segment = {"word": word_text}

                if not np.isnan(word_start):
                    word_segment["start"] = word_start
                if not np.isnan(word_end):
                    word_segment["end"] = word_end
                if not np.isnan(word_score):
                    word_segment["score"] = word_score

                sentence_words.append(word_segment)
            
            aligned_subsegments.append({
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": sentence_words,
            })

            if return_char_alignments:
                curr_chars = curr_chars[["char", "start", "end", "score"]]
                curr_chars.fillna(-1, inplace=True)
                curr_chars = curr_chars.to_dict("records")
                curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                aligned_subsegments[-1]["chars"] = curr_chars

        aligned_subsegments = pd.DataFrame(aligned_subsegments)
        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
        # concatenate sentences with same timestamps
        agg_dict = {"text": " ".join, "words": "sum"}
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"
        aligned_subsegments= aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
        aligned_subsegments = aligned_subsegments.to_dict('records')
        aligned_segments += aligned_subsegments

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}

"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""
def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        # failed
        return None
    return path[::-1]

# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words
