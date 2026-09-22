
# ============================================================
# Standard Library
# ============================================================
import os
import re
import sys
import math
import time
import json
import subprocess
from pathlib import Path
from collections import Counter
from typing import List, Set, Tuple, Dict, Any
from difflib import SequenceMatcher
import pandas as pd

from semanticsubgraph import normalized_levenshtein


import hashlib
import unicodedata

# ============================================================
# Audio Processing
# ============================================================
import torch
import torchaudio
import torchaudio.functional as AF
from pydub import AudioSegment

# ============================================================
# Whisper / MLX
# ============================================================
import mlx_whisper

# ============================================================
# Fuzzy Matching
# ============================================================
from rapidfuzz import process, fuzz

# ============================================================
# Video Editing
# ============================================================
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    ColorClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

# ============================================================
# Demucs (Offline Loader)
# ============================================================
from demucs.apply import apply_model
from demucs.htdemucs import HTDemucs
from safetensors.torch import load_file as safetensor_load



# ============================================================
# SPEECH ENHANCER
# ============================================================

def enhance_speech(vocals_mono: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Lightweight speech enhancer:
    - High-pass to remove rumble
    - Gentle presence boost (3 kHz)
    - Soft-knee compressor (manual implementation)
    """
    x = vocals_mono

    # 1. High-pass at 80 Hz
    x = AF.highpass_biquad(x, sr, cutoff_freq=80.0)

    # 2. Presence boost around 3 kHz
    x = AF.equalizer_biquad(
        x, sr,
        center_freq=3000.0,
        gain=3.0,
        Q=0.7
    )

    # 3. Soft-knee compressor
    threshold = 10 ** (-18 / 20)
    knee = 10 ** (6 / 20)
    ratio = 3.0

    abs_x = x.abs()
    over = abs_x - threshold
    soft = torch.clamp(over / knee, min=0.0, max=1.0)
    gain = 1.0 - soft * (1.0 - 1.0 / ratio)

    return x * gain


# ============================================================
# LOCAL HTDEMUCS LOADER (NO NETWORK)
# ============================================================

_DEMUCS_MODEL = None


def get_or_run_transcription(audio_path, fpath_transcribed, fs, overwrite=False):
    """
    Deterministic transcription wrapper:
    - Coerces paths to Path objects.
    - If JSON exists, return it.
    - Otherwise run Whisper on chunks and write JSON.
    """

    # Coerce to Path if needed
    audio_path = Path(audio_path)
    fpath_transcribed = Path(fpath_transcribed)

    if(overwrite is False):
        
        # If transcription already exists, skip Whisper
        if fpath_transcribed.exists():
            return fpath_transcribed

    # Ensure directory exists
    fpath_transcribed.parent.mkdir(parents=True, exist_ok=True)

    # Run Whisper on chunks
    merged_segments = run_whisper_on_chunks(audio_path)

    # Write JSON
    fs.write_json(fpath_transcribed, merged_segments)

    return fpath_transcribed


def get_demucs_model_local():
    """
    Load HTDemucs entirely from local JSON + safetensors.
    No HuggingFace requests.
    """
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is not None:
        return _DEMUCS_MODEL

    model_dir = Path("/Users/pense/models/demucs/htdemucs")

    # Load JSON config
    with open(model_dir / "955717e8.json", "r") as f:
        raw = json.load(f)

    # Parse kwargs (stored as a JSON string)
    kwargs = json.loads(raw["kwargs"])

    # Convert segment fraction → float
    seg = kwargs.get("segment")
    if isinstance(seg, dict) and seg.get("_type") == "fraction":
        kwargs["segment"] = seg["numerator"] / seg["denominator"]

    # Instantiate model
    model = HTDemucs(**kwargs)

    # Load safetensors weights
    weights = safetensor_load(model_dir / "955717e8.safetensors")
    model.load_state_dict(weights)

    model.eval()
    model.to("cpu")

    _DEMUCS_MODEL = model
    return model


# ============================================================
# UTILS
# ============================================================

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_")


# ============================================================
# MAIN CLEANER
# ============================================================

def clean_mp3_demucs(input_mp3: str, output_dir: str, title: str) -> Path:
    """
    Max-quality offline Demucs speech cleaner.
    - Uses local HTDemucs (no network)
    - High overlap
    - Float32 processing
    - Optional resampling to 44.1kHz
    - Speech enhancement stage
    """

    input_mp3 = Path(input_mp3)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_title = sanitize_filename(title)
    cleaned_wav = out_dir / f"{safe_title}_clean.wav"
    cleaned_mp3 = out_dir / f"{safe_title}_clean.mp3"

    # Skip if already processed
    if cleaned_wav.exists():
        return cleaned_wav
    if cleaned_mp3.exists():
        return cleaned_mp3

    # CPU config
    total_cores = os.cpu_count() or 4
    torch.set_num_threads(total_cores)
    torch.manual_seed(0)

    # Load audio
    try:
        wav, sr = torchaudio.load(str(input_mp3))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    if wav.ndim != 2:
        raise RuntimeError("Audio must be 2-D (channels, samples).")

    wav = wav.to(torch.float32)

    if wav.shape[-1] < sr:
        raise RuntimeError("Audio too short for Demucs processing.")

    # Stereo
    if wav.shape[0] == 1:
        wav = torch.stack([wav[0], wav[0]], dim=0)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # Resample to 44.1kHz
    target_sr = 44100
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    # Avoid normalization unless clipping
    peak = wav.abs().max()
    if peak > 1.0:
        wav = wav / peak

    wav_tensor = wav.unsqueeze(0)

    # Load offline model
    model = get_demucs_model_local()

    # Apply model
    with torch.no_grad():
        out_tensor = apply_model(
            model,
            wav_tensor,
            shifts=10,
            split=True,
            overlap=0.15,
            num_workers=total_cores,
        )

    # Extract vocals
    vocals_idx = model.sources.index("vocals")
    vocals = out_tensor[0, vocals_idx]

    # Downmix
    vocals_mono = vocals.mean(dim=0, keepdim=True)

    # Enhance speech
    vocals_enhanced = enhance_speech(vocals_mono, sr)

    # Save WAV
    torchaudio.save(str(cleaned_wav), vocals_enhanced, sr)

    # Save MP3
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(cleaned_wav),
            "-acodec", "libmp3lame",
            "-b:a", "128k",
            "-ar", str(sr),
            "-ac", "1",
            str(cleaned_mp3),
        ],
        check=True,
        capture_output=True,
    )

    # Cleanup
    del wav, wav_tensor, out_tensor, vocals, vocals_mono
    torch.cuda.empty_cache()

    return cleaned_wav


def create_folder_if_not_exists(dpath):

    if(type(dpath) is str):
        dpath = Path(dpath)
    try:
        if(not dpath.exists()):
            os.makedirs(dpath)
    except Exception as e:
        print('create folder failed')
        print(e)

def anchor_alignment(words, anchor_text, gap_penalty=-0.3, mismatch_penalty=-0.5):
    """
    Dynamic programming alignment of anchor_text to transcript words.
    Allows gaps, missing words, insertions, deletions, paraphrasing.

    Returns:
        (start_ms, end_ms, best_score, (start_idx, end_idx))
    """
    
    def normalize_token(t):
        return re.sub(r'[^a-z0-9]', '', t.lower())
    
    def token_similarity(a, b):
        """Fuzzy similarity between two tokens."""
        return SequenceMatcher(None, a, b).ratio()

    # Normalize anchor tokens
    anchor_tokens = [normalize_token(t) for t in anchor_text.split()]
    A = len(anchor_tokens)

    # Normalize transcript tokens
    transcript_tokens = [normalize_token(w["text"]) for w in words]
    T = len(transcript_tokens)

    # DP matrix: score[i][j] = best alignment score ending at anchor i, transcript j
    score = [[0.0] * (T + 1) for _ in range(A + 1)]
    back = [[None] * (T + 1) for _ in range(A + 1)]

    best_score = float("-inf")
    best_pos = (None, None)

    for i in range(1, A + 1):
        for j in range(1, T + 1):

            sim = token_similarity(anchor_tokens[i-1], transcript_tokens[j-1])

            # Match / substitution
            match_score = score[i-1][j-1] + sim

            # Gap in anchor (transcript has extra word)
            gap_in_anchor = score[i][j-1] + gap_penalty

            # Gap in transcript (anchor has extra word)
            gap_in_transcript = score[i-1][j] + gap_penalty

            # Choose best
            best = max(match_score, gap_in_anchor, gap_in_transcript)

            score[i][j] = best

            if best == match_score:
                back[i][j] = (i-1, j-1)
            elif best == gap_in_anchor:
                back[i][j] = (i, j-1)
            else:
                back[i][j] = (i-1, j)

            # Track best endpoint
            if best > best_score:
                best_score = best
                best_pos = (i, j)

    # Backtrack to find transcript span
    i, j = best_pos
    used_transcript_indices = []

    while i > 0 and j > 0:
        pi, pj = back[i][j]
        if pj != j:  # transcript index used
            used_transcript_indices.append(j-1)
        i, j = pi, pj

    if not used_transcript_indices:
        return None, None, best_score, None

    start_idx = min(used_transcript_indices)
    end_idx   = max(used_transcript_indices)

    start_ms = int(words[start_idx]["start"] * 1000)
    end_ms   = int(words[end_idx]["end"] * 1000)

    return start_ms, end_ms, best_score, (start_idx, end_idx)



def convert_to_wav(input_file, temp_dir, sr=16000):
    
    os.makedirs(temp_dir, exist_ok=True)
    out = os.path.join(temp_dir, os.path.splitext(os.path.basename(input_file))[0] + ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_file, "-ar", str(sr), "-ac", "1", out],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out

def decide_edge(edge_score: float, best_anchor_score: float, side: str, delta: float = 0.15, verbose=False):
    """
    Decide contract/extend/ok based on differential between edge score and best anchor score.
    delta = tolerance margin.
    """
    diff = edge_score - best_anchor_score

    if(verbose):
        print(f"[DIAG] {side} edge vs anchor diff={diff:.3f} (edge={edge_score:.3f}, anchor={best_anchor_score:.3f})")

    if diff < -delta:
        action = f"contract_{side}"
        if(verbose):
            print(f"[DIAG] {side.capitalize()} edge much worse than anchor → {action}.")
    elif diff > delta:
        action = f"extend_{side}"
        if(verbose):
            print(f"[DIAG] {side.capitalize()} edge much better than anchor → {action}.")
    else:
        action = "ok"
        if(verbose):
            print(f"[DIAG] {side.capitalize()} edge close to anchor → ok.")
    return action


def best_anchor_match(candidate: str, anchor: str, window: int) -> Tuple[float, int]:
    if len(candidate) <= window:
        _, score = qual_fuzzy_match(candidate, anchor)
        return score, 0
    best_score, best_pos = 0.0, -1
    for i in range(0, len(candidate) - window + 1):
        slice_ = candidate[i:i+window]
        _, score = qual_fuzzy_match(slice_, anchor)
        if score > best_score:
            best_score, best_pos = score, i
    return best_score, best_pos

def get_clip_timestamps(fpath_audio, seed_quote, add_secs = None):

    words = run_whisper_on_chunks(
        fpath_audio, 
        sleep_duration=5, 
        word_timestamps=True
    )

    start_secs, end_secs, quote = find_string_timestamps(
        words, 
        seed_quote, 
    )

    final_start, final_end, last_result = iterative_alignment(
        words,
        quote,
        start_secs*1000,
        end_secs*1000,
        max_iters=20,
    )
    
    if(add_secs is not None):
        
        final_end = final_end + add_secs*1000
    
    return(final_start, final_end, last_result)



def find_string_timestamps(words, target_string, min_score=70, max_duration_per_word=1500):
    """
    Find most likely start and end timestamps for target_string in transcript words.
    Uses fuzzy matching on variable-length windows.
    """
    
    def normalize(text):
        return re.sub(r"[^\w\s]", "", text.lower())
    
    target_norm = normalize(target_string)
    best_match = None
    
    # Try all possible subsequences
    for i in range(len(words)):
        for j in range(i+1, len(words)+1):
            window = words[i:j]
            window_text = " ".join([w['text'] for w in window])
            window_norm = normalize(window_text)
            
            score = fuzz.ratio(target_norm, window_norm)
            if score >= min_score:
                start = window[0]['start']
                end = window[-1]['end']
                duration_ms = (end - start) * 1000
                n_words = len(window)
                
                # Guardrail: duration per word
                if duration_ms <= n_words * max_duration_per_word:
                    if best_match is None or score > best_match['score']:
                        best_match = {
                            'start': start,
                            'end': end,
                            'score': score,
                            'text': window_text
                        }
    
    if best_match:
        return best_match['start'], best_match['end'], best_match['text']
    else:
        return None, None, None


def qual_fuzzy_match(str1: str, str2: str, threshold: float = 0.7):
    
    def normalize(text: str) -> str:
        """
        Remove spaces and special characters, keep only alphanumerics.
        Lowercase for consistency.
        """
        return re.sub(r'[^a-z0-9]', '', text.lower())
    """
    Compare two strings ignoring spaces and special characters.
    Returns (match_decision, similarity_score).
    """
    s1 = normalize(str1)
    s2 = normalize(str2)
    score = SequenceMatcher(None, s1, s2).ratio()
    
    return (score >= threshold, score)


def evaluate_alignment(candidate: str, target: str,
                       threshold_overall: float = 0.85,
                       window: int = 80,
                       delta: float = 0.1, 
                       verbose=False):


    _, overall = qual_fuzzy_match(candidate, target)

    if(verbose):
        print(f"[DIAG] Overall similarity: {overall:.3f} (threshold={threshold_overall})")

    cand_start, cand_end = candidate[:window], candidate[-window:]
    targ_start, targ_end = target[:window], target[-window:]

    _, start_edge_score = qual_fuzzy_match(cand_start, targ_start)
    _, end_edge_score   = qual_fuzzy_match(cand_end, targ_end)

    start_anchor_score, start_anchor_pos = best_anchor_match(candidate, targ_start, window)
    end_anchor_score,   end_anchor_pos   = best_anchor_match(candidate, targ_end, window)

    if(verbose):
        print(f"[DIAG] Start edge score={start_edge_score:.3f}, best anchor score={start_anchor_score:.3f}")
        print(f"[DIAG] End edge score={end_edge_score:.3f}, best anchor score={end_anchor_score:.3f}")

    start_action = decide_edge(start_edge_score, start_anchor_score, "start", delta)
    end_action   = decide_edge(end_edge_score, end_anchor_score, "end", delta)

    return {
        "start": start_action,
        "end": end_action,
        "similarity": overall,
        "start_edge_score": start_edge_score,
        "end_edge_score": end_edge_score,
        "start_anchor_score": start_anchor_score,
        "end_anchor_score": end_anchor_score,
        "start_anchor_pos": start_anchor_pos,
        "end_anchor_pos": end_anchor_pos,
    }



def iterative_alignment(
    words,
    anchor_text,
    time_start_milli,
    time_end_milli,
    max_iters=20,
    min_ratio=0.8,
    max_ratio=1.2,
    threshold_overall=0.8,
    window=200,
    base_step=1500,   # minimum adjustment in ms
    frac_clip=0.05,   # fraction of clip length for step size
    verbose=False
):
    """
    Iteratively align transcript words to anchor text by adjusting time boundaries.
    Step size is dynamic: proportional to clip length and shrinks each iteration.
    Returns updated boundaries and diagnostics.
    """
    result = None

    for i in range(max_iters):
        # Slice words within current boundaries
        cand_words = [
            w for w in words
            if time_start_milli <= w["start"] * 1000
            and w["end"] * 1000 <= time_end_milli
        ]
        whisper_result_text = " ".join([w["text"] for w in cand_words])

        # Evaluate alignment
        result = evaluate_alignment(
            whisper_result_text,
            anchor_text,
            threshold_overall=threshold_overall,
            window=window,
        )

        # Length ratio
        cand_len = len(whisper_result_text)
        targ_len = len(anchor_text)
        length_ratio = cand_len / max(1, targ_len)

        # Edge differentials
        start_diff = abs(result["start_edge_score"] - result["start_anchor_score"])
        end_diff   = abs(result["end_edge_score"]   - result["end_anchor_score"])

        # --- Dynamic step size ---
        clip_len = time_end_milli - time_start_milli
        step_size = int(max(base_step, clip_len * frac_clip))
        step_size = int(step_size * (1 - i / max_iters))

        # Decide movement mode
        if length_ratio < min_ratio:
            mode = "LEN_TOO_SHORT"
        elif length_ratio > max_ratio:
            mode = "LEN_TOO_LONG"
            # smart rule: use anchor position to decide which side has extra text
            start_pos = result.get("start_anchor_pos", 0)
            end_pos   = result.get("end_anchor_pos", 0)
        else:
            mode = "EDGE_ALIGN"

        # Compute actual movements
        start_before = time_start_milli
        end_before   = time_end_milli
        start_move = 0
        end_move = 0

        if mode == "LEN_TOO_SHORT":
            # extend the side whose anchor is closer
            if start_diff <= end_diff:
                time_start_milli = max(0, time_start_milli - step_size)
                start_move = -step_size
            else:
                time_end_milli += step_size
                end_move = step_size

        elif mode == "LEN_TOO_LONG":
            # smart contraction: anchor closer to end → extra text at start
            if start_pos > end_pos:
                # CONTRACT START = move start earlier (subtract)
                time_start_milli = max(0, time_start_milli - step_size)
                start_move = -step_size
            else:
                # CONTRACT END = move end earlier (subtract)
                time_end_milli -= step_size
                end_move = -step_size

        else:  # EDGE_ALIGN
            if result["start"] == "contract_start":
                # CONTRACT START = move start earlier
                time_start_milli = max(0, time_start_milli - step_size)
                start_move = -step_size
            elif result["start"] == "extend_start":
                # EXTEND START = move start later
                time_start_milli += step_size
                start_move = step_size

            if result["end"] == "contract_end":
                # CONTRACT END = move end earlier
                time_end_milli -= step_size
                end_move = -step_size
            elif result["end"] == "extend_end":
                # EXTEND END = move end later
                time_end_milli += step_size
                end_move = step_size

        if verbose:
            print(f"\n=== Iteration {i+1} ===")
            print(
                f"[STATE] sim={result['similarity']:.3f} "
                f"len_ratio={length_ratio:.3f} "
                f"mode={mode}"
            )
            print(
                f"[ANCHOR] "
                f"start_edge={result['start_edge_score']:.3f} "
                f"start_anchor={result['start_anchor_score']:.3f} "
                f"end_edge={result['end_edge_score']:.3f} "
                f"end_anchor={result['end_anchor_score']:.3f} "
                f"start_pos={result.get('start_anchor_pos', 0)} "
                f"end_pos={result.get('end_anchor_pos', 0)}"
            )
            print(
                f"[STEP] step_size={step_size}ms "
                f"start_diff={start_diff:.3f} "
                f"end_diff={end_diff:.3f}"
            )
            print(
                f"[APPLY] start_ms: {start_before} -> {time_start_milli} "
                f"(Δ={start_move}), "
                f"end_ms: {end_before} -> {time_end_milli} "
                f"(Δ={end_move})"
            )
            print(f"[LEN] cand={cand_len} targ={targ_len}")

        # Stop early if both edges ok AND length ratio acceptable
        if (
            result["start"] == "ok"
            and result["end"] == "ok"
            and min_ratio <= length_ratio <= max_ratio
        ):
            if verbose:
                print("[LOOP] Both edges aligned and candidate length acceptable → stopping early.")
            break

    return time_start_milli, time_end_milli, result



import subprocess, os, gc, time
from pydub.utils import mediainfo

def run_whisper_on_chunks(
    fpath,
    model_path='/Users/pense/.lmstudio/models/mlx-community/whisper-large-v3-turbo',
    chunk_duration=60000,
    overlap=2000,
    sleep_duration=30,
    word_timestamps=False,
    verbose=False
):

    info = mediainfo(fpath)
    audio_duration_ms = float(info["duration"]) * 1000
    num_splits = math.ceil(audio_duration_ms / chunk_duration)

    tempdir = os.path.join(os.path.dirname(fpath), "temp")
    create_folder_if_not_exists(tempdir)

    merged = []
    last_end = 0.0
    last_word_text = None

    for n in range(num_splits):

        start = n * chunk_duration
        stop = (n + 1) * chunk_duration + overlap

        outpath = os.path.join(tempdir, f"short_test_clip_{n}.wav")

        # Extract chunk WITHOUT loading whole file
        subprocess.run([
            "ffmpeg", "-y",
            "-i", fpath,
            "-ss", str(start / 1000.0),
            "-t", str((stop - start) / 1000.0),
            outpath
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if verbose:
            print(f"Processing chunk {n+1}/{num_splits}")

        offset = start / 1000.0

        result = mlx_whisper.transcribe(
            outpath,
            path_or_hf_repo=model_path,
            word_timestamps=word_timestamps,
        )

        time.sleep(sleep_duration)

        if not word_timestamps:
            for seg in result["segments"]:
                seg_start = max(seg["start"] + offset, last_end)
                seg_end = max(seg["end"] + offset, seg_start + 0.01)

                merged.append({
                    "text": seg["text"].strip(),
                    "start": seg_start,
                    "end": seg_end,
                })
                last_end = seg_end

        else:
            for seg in result["segments"]:
                for w in seg["words"]:
                    text = w["word"].strip()
                    if not text or text == last_word_text:
                        continue

                    start_ts = max(w["start"] + offset, last_end)
                    end_ts = max(w["end"] + offset, start_ts + 0.01)

                    merged.append({
                        "text": text,
                        "start": start_ts,
                        "end": end_ts,
                    })

                    last_end = end_ts
                    last_word_text = text

        # 🔥 Free memory aggressively
        del result
        gc.collect()

    return merged

def clip_audio(
    fpath_audio,
    time_start_milli,
    time_end_milli,
    dpath_audio_clip,
    file_prefix="clip",
    output_audio_format="wav",
    fname=None,
    verbose=False
    ):
    """
    Create an audio clip from a source file between start and end times (ms).
    Returns the full path to the exported clip.
    """

    # Load audio using format autodetection
    audio = AudioSegment.from_file(fpath_audio)

    # Slice the audio
    clip = audio[time_start_milli:time_end_milli]

    # Determine output filename
    if fname is None:
        start_s = int(time_start_milli / 1000)
        end_s = int(time_end_milli / 1000)
        fname = f"{file_prefix}_{start_s}_{end_s}.{output_audio_format}"

    # Ensure output directory exists
    os.makedirs(dpath_audio_clip, exist_ok=True)

    # Build full output path
    fpath_out = os.path.join(dpath_audio_clip, fname)

    # Export clip
    clip.export(fpath_out, format=output_audio_format)

    return fpath_out

def clean_text_preserve_phrases(s: str) -> str:
    """
    Clean text but preserve acronyms (all caps) and hyphenated phrases.
    """
    s = re.sub(r"[^A-Za-z0-9\s\-]", " ", s)  # keep letters, numbers, spaces, hyphens
    return re.sub(r"\s+", " ", s).strip()

def build_frequency(transcript_df, text_col="text"):
    """Build frequency dictionary across transcript."""
    all_tokens = []
    for txt in transcript_df[text_col]:
        all_tokens.extend(clean_text_preserve_phrases(str(txt)).split())
    return Counter(all_tokens)


def count_non_filler_words(text: str) -> int:
    """Count words longer than 5 characters (non-filler heuristic)."""
    return sum(1 for w in text.split() if len(w) > 5)


def uncommon_tokens(context, freq_dict, rarity_threshold=2, min_length=4):
    """
    Identify rare tokens in context:
    - Rare by frequency (<= rarity_threshold)
    - Long enough (min_length)
    - Acronyms/hyphenated only if also rare
    - Return tokens sorted by ascending frequency (rarest first)
    """
    tokens = set(clean_text_preserve_phrases(context).split())
    rare = []

    for t in sorted((tokens)):
        freq = freq_dict.get(t, 0)

        if freq <= rarity_threshold and len(t) >= min_length:
            rare.append((t, freq))
        elif ("-" in t or t.isupper()) and freq <= rarity_threshold and len(t) == min_length:
            rare.append((t, freq))

    # Sort by frequency (rarest first), then by length (longer anchors first)
    rare.sort(key=lambda x: (x[1], - len(x[0])))
    return [t for t, _ in rare]

def best_guess_time_for_context(
    context,
    transcript_df,
    text_col="text",
    start_col="start",
    end_col="end",
    threshold=0.6,
    weak_threshold=0.3,
    allow_gap=1,
    min_words_start=12,
    min_words_floor=6,
    min_chars=15
):
    """
    Align a context string against a transcript DataFrame with [text, start, end].
    Tiered system:
      - Strong match (threshold + contiguity)
      - Rare-anchor match (distinctive tokens + substantive overlap)
      - Weak similarity match
      - Very weak fallback
    """
    context_clean = clean_text_preserve_phrases(context)
    context_words = len(context_clean.split())
    freq_dict = build_frequency(transcript_df, text_col)

    matched_rows = []
    min_words = min_words_start

    # Tier 1: Strong match
    while not matched_rows and min_words >= min_words_floor:
        for idx, row in transcript_df.iterrows():
            row_text = clean_text_preserve_phrases(str(row[text_col]))
            row_words = count_non_filler_words(row_text)
            row_chars = len(row_text)

            if row_words < min_words or row_chars < min_chars:
                continue

            ratio = SequenceMatcher(None, row_text, context_clean).ratio()
            is_match = (ratio >= threshold) or (row_text in context_clean)

            if is_match:
                matched_rows.append((idx, row[start_col], row[end_col], ratio))

        if not matched_rows:
            min_words -= 2

    if matched_rows:
        matched_rows.sort(key=lambda x: x[0])
        contiguous_block = [matched_rows[0]]
        for r in matched_rows[1:]:
            if r[0] - contiguous_block[-1][0] <= allow_gap:
                contiguous_block.append(r)
            else:
                break
        start_time = min(r[1] for r in contiguous_block)
        end_time = max(r[2] for r in contiguous_block)
        indices = [r[0] for r in contiguous_block]
        return {
            "context": context,
            "start": start_time,
            "end": end_time,
            "matched_indices": indices,
            "final_min_words": min_words,
            "fallback": False,
            "confidence": "strong"
        }

    # Tier 2: Rare-anchor match
    rare_tokens = uncommon_tokens(context, freq_dict)
    rare_candidates = []
    
    for idx, row in transcript_df.iterrows():
        row_text = clean_text_preserve_phrases(str(row[text_col]))
        ratio = SequenceMatcher(None, row_text, context_clean).ratio()
        words = row_text.split()
        rare_overlap = set(words) & set(rare_tokens)
        substantive_overlap = [w for w in words if len(w) > 5 and w in context_clean.split() and w != w.upper()]
        if rare_overlap and substantive_overlap:
            # Weighted scoring: similarity + boosted rare anchors
            acronym_boost = sum(1 for w in rare_overlap if w.isupper())
            hyphen_boost = sum(1 for w in rare_overlap if "-" in w)
            score = 0.8 * ratio + 0.3 * (len(rare_overlap) + 2*acronym_boost + 2*hyphen_boost)
            rare_candidates.append((
                idx, score, row[start_col], row[end_col], rare_overlap, substantive_overlap
            ))
            
    if rare_candidates:
        rare_candidates.sort(key=lambda x: x[1], reverse=True)
        best = rare_candidates[0]
        return {
            "context": context,
            "start": best[2],
            "end": best[3],
            "matched_indices": [best[0]],
            "fallback": True,
            "confidence": "rare-anchor",
            "score": best[1],
            "rare_overlap": list(best[4]),
            "substantive_overlap": list(best[5])
        }

    # Tier 3: Weak similarity match
    candidates = []
    for idx, row in transcript_df.iterrows():
        row_text = clean_text_preserve_phrases(str(row[text_col]))
        ratio = SequenceMatcher(None, row_text, context_clean).ratio()
        non_filler = count_non_filler_words(row_text)
        candidates.append((idx, ratio, non_filler, row[start_col], row[end_col]))
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_idx, best_score, best_non_filler, start, end = candidates[0]

    if best_score >= weak_threshold and best_non_filler >= 3:
        return {
            "context": context,
            "start": start,
            "end": end,
            "matched_indices": [best_idx],
            "fallback": True,
            "confidence": "weak",
            "score": best_score,
            "non_filler_words": best_non_filler
        }

    # Tier 4: Very weak match
    return {
        "context": context,
        "start": start,
        "end": end,
        "matched_indices": [best_idx],
        "fallback": True,
        "confidence": "very weak",
        "score": best_score,
        "non_filler_words": best_non_filler
    }


def export_clip(final_clip, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_clip.export(output_path, format="mp3")
    return output_path


def add_suffix(path, suffix):
    path = Path(path)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def normalize(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    """Normalize and split into tokens."""
    return normalize(text).split()

def token_similarity(tokens_a, tokens_b):
    """Token-level fuzzy similarity."""
    return SequenceMatcher(None, tokens_a, tokens_b).ratio()

def closest_candidate(reference, candidates: dict):
    """
    Given a reference string and a dict of {name: candidate_string},
    return the key whose candidate is closest to the reference.
    """

    ref_tokens = tokenize(reference)

    best_key = None
    best_score = -1
    all_scores = {}

    for name, cand in candidates.items():
        cand_tokens = tokenize(cand)
        score = token_similarity(ref_tokens, cand_tokens)
        all_scores[name] = score

        if score > best_score:
            best_score = score
            best_key = name

    return best_key, best_score, all_scores



def extract_quote_clip(audio_path, quote_info, transcript_json, verbose=False):
    """
    Pure logic version of gen_quote_audio:
    - No project-specific paths
    - No channel/ticker/title
    - No final export
    - Returns final AudioSegment + metadata
    - Alignment logic identical to original (including fuzzy comparison)
    """

    audio_path = Path(audio_path)

    # --- Validate quote_info ---
    if not isinstance(quote_info, dict):
        raise ValueError("quote_info must be a dict containing 'text' and 'name'.")
    quote = quote_info.get("text")
    quote_name = quote_info.get("name")
    if not quote or not quote_name:
        raise ValueError("quote_info must contain non-empty 'text' and 'name'.")

    # --- Validate transcript ---
    if not transcript_json or not isinstance(transcript_json, (list, tuple)):
        raise ValueError("Transcript JSON is empty or malformed.")
    dft = pd.DataFrame(transcript_json)
    required_cols = {"text", "start", "end"}
    if not required_cols.issubset(dft.columns):
        raise ValueError(f"Transcript missing required columns: {required_cols}")

    # --- Estimate quote location ---
    dialogue_time_response = best_guess_time_for_context(quote, dft)
    if not dialogue_time_response:
        raise RuntimeError("Could not locate quote context in transcript.")

    start_sec = dialogue_time_response["start"]
    end_sec   = dialogue_time_response["end"]

    # --- Build window ---
    seconds_pre = 30
    seconds_post = 30
    start_sec = max(0, start_sec - seconds_pre)
    end_sec   = end_sec + seconds_post

    start_ms = int(start_sec * 1000)
    end_ms   = int(end_sec * 1000)

    # --- Extended window ---
    clip_len = end_ms - start_ms
    extend_amt = int(clip_len * 2)
    ext_start = max(0, start_ms - extend_amt)
    ext_end   = end_ms + extend_amt

    # --- Load full audio ---
    audio_full = AudioSegment.from_mp3(audio_path)
    audio_duration = len(audio_full)

    ext_start = max(0, min(ext_start, audio_duration - 1))
    ext_end   = max(ext_start + 1, min(ext_end, audio_duration))

    if verbose:
        print(f"Extended window (ms): {ext_start} → {ext_end}")

    # --- Extract extended clip ---
    extended_clip = audio_full[ext_start:ext_end]

    # --- Temp dir + export extended clip to file ---
    temp_dir = audio_path.parent / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    extended_path = temp_dir / f"extended_{ext_start}_{ext_end}.wav"
    extended_clip.export(extended_path, format="wav")

    # --- Run Whisper ---
    clip_words = run_whisper_on_chunks(
        extended_path,
        sleep_duration=5,
        word_timestamps=True
    )
    if not clip_words:
        raise RuntimeError("Whisper returned no word-level timestamps.")

    # --- Adjust timestamps ---
    clip_words_secs = []
    for c in clip_words:
        if "start" in c and "end" in c:
            clip_words_secs.append({
                "text": c.get("text", ""),
                "start": c["start"] + ext_start / 1000,
                "end":   c["end"]   + ext_start / 1000,
            })

    if not clip_words_secs:
        raise RuntimeError("Whisper produced no usable word timestamps.")

    # --- Alignment method 1 ---
    try:
        aligned_start0, aligned_end0, _ = iterative_alignment(
            clip_words_secs,
            quote,
            start_ms,
            end_ms,
            min_ratio=0.95,
            max_ratio=1.05,
            max_iters=20,
            verbose=False
        )
    except Exception:
        aligned_start0 = aligned_end0 = None

    aligned_quote0 = ""
    if aligned_start0 is not None and aligned_end0 is not None:
        aligned_quote0 = " ".join(
            c["text"] for c in clip_words_secs
            if aligned_start0 <= c["start"] * 1000 <= aligned_end0
        )

    # --- Alignment method 2 ---
    try:
        aligned_start1, aligned_end1, _, _ = anchor_alignment(
            clip_words_secs, quote
        )
    except Exception:
        aligned_start1 = aligned_end1 = None

    aligned_quote1 = ""
    if aligned_start1 is not None and aligned_end1 is not None:
        aligned_quote1 = " ".join(
            c["text"] for c in clip_words_secs
            if aligned_start1 <= c["start"] * 1000 <= aligned_end1
        )

    # --- Fuzzy comparison (IDENTICAL to original) ---
    candidates = {
        0: aligned_quote0,
        1: aligned_quote1,
    }

    best_key, _, _ = closest_candidate(quote, candidates)

    if best_key == 0:
        final_start = aligned_start0
        final_end   = aligned_end0
    else:
        final_start = aligned_start1
        final_end   = aligned_end1

    if final_start is None or final_end is None:
        raise RuntimeError("No valid alignment produced.")

    # --- Extract final clip (+500ms tail) ---
    final_clip = audio_full[final_start:final_end + 500]

    final_info = {
        "quote_name": quote_name,
        "start_ms": final_start,
        "end_ms": final_end,
        "extended_window": (ext_start, ext_end),
        "temp_extended_path": extended_path,
        "method": best_key,
    }

    return final_clip, final_info


"""
def gen_audio_clips(
        audio_path, 
        fpath_transcribed, 
        clip_title, 
        quote_block, 
        quote_hook, 
        fs
    ):

    name = slugify(clip_title)
    audio_path = audio_path
    fpath_transcribed = fpath_transcribed

    quote_block = quote_block

    quote_str = ' '.join(
        [
            c['content'] for c in quote_block
        ]
    )

    full_clip_info = {
        'name':name,
        'text':quote_str,
    }

    transcript_json = fs.read_json(fpath_transcribed)
    
    final_clip, final_clip_info = extract_quote_clip(
        audio_path, 
        full_clip_info, 
        transcript_json, 
        verbose=True
    )
    
    fpath_final_clip = add_suffix(
        audio_path, 
        name
    )
    
    export_clip(
        final_clip, 
        fpath_final_clip
    )
    
    fpath_transcribed_clip = add_suffix(fpath_transcribed, name)
    
    fpath_transcribed_clip = get_or_run_transcription(
        fpath_final_clip, 
        fpath_transcribed_clip, 
        fs, 
        overwrite=True
    )
    
    transcript_clip_json = fs.read_json(fpath_transcribed_clip)
    
    final_clip_hook, final_clip_info = extract_quote_clip(
        fpath_final_clip, 
        {
            'name':name,
            'text':quote_hook
        }, 
        transcript_clip_json, 
        verbose=True
    )
    
    fpath_final_clip_hook = add_suffix(
        audio_path, 
        name+'_hook',
    )
    
    export_clip(
        final_clip_hook, 
        fpath_final_clip_hook
    )

    return(fpath_final_clip, fpath_final_clip_hook)

def slugify(title: str) -> str:
    # lowercase
    s = title.lower()

    # remove punctuation
    s = re.sub(r'[^a-z0-9\s-]', '', s)

    # replace whitespace with hyphens
    s = re.sub(r'\s+', '-', s)

    # collapse multiple hyphens
    s = re.sub(r'-+', '-', s)

    # trim hyphens
    return s.strip('-')
"""

def gen_audio_clips(
    audio_path,
    transcript_path,
    quote_block: list,
    quote_hook: str | None,
    full_clip_out,
    hook_clip_out,
    fs,
    verbose: bool = False
):
    """
    Pure clip generator:
    - Always generates a full clip from quote_block
    - Optionally generates a hook clip if quote_hook is provided
    - Caller provides explicit output paths
    - Transcript JSON is saved next to the audio clip using *_transcript.json
    - All path inputs are defensively cast to Path objects
    """

    # ---------------------------------------------------------
    # Defensive path normalization
    # ---------------------------------------------------------
    audio_path = Path(audio_path) if not isinstance(audio_path, Path) else audio_path
    transcript_path = Path(transcript_path) if not isinstance(transcript_path, Path) else transcript_path
    full_clip_out = Path(full_clip_out) if not isinstance(full_clip_out, Path) else full_clip_out

    if hook_clip_out is not None:
        hook_clip_out = Path(hook_clip_out) if not isinstance(hook_clip_out, Path) else hook_clip_out

    # ---------------------------------------------------------
    # Build full-clip text
    # ---------------------------------------------------------
    full_text = " ".join(c["content"] for c in quote_block)
    full_name = full_clip_out.stem

    # ---------------------------------------------------------
    # Load transcript JSON for the *source audio*
    # ---------------------------------------------------------
    transcript_json = fs.read_json(transcript_path)

    # ---------------------------------------------------------
    # Extract full clip
    # ---------------------------------------------------------
    full_clip_audio, full_clip_info = extract_quote_clip(
        audio_path,
        {"text": full_text, "name": full_name},
        transcript_json,
        verbose=verbose
    )

    # ---------------------------------------------------------
    # Export full clip
    # ---------------------------------------------------------
    export_clip(full_clip_audio, full_clip_out)

    # ---------------------------------------------------------
    # Build transcript output path (no conflict with metadata)
    # ---------------------------------------------------------
    full_transcript_out = full_clip_out.with_name(f"{full_name}_transcript.json")

    # ---------------------------------------------------------
    # Transcribe full clip
    # ---------------------------------------------------------
    full_transcript_out = get_or_run_transcription(
        full_clip_out,
        full_transcript_out,
        fs,
        overwrite=True
    )

    # ---------------------------------------------------------
    # If no hook clip requested, return only full clip
    # ---------------------------------------------------------
    if quote_hook is None or hook_clip_out is None:
        return {
            "full_clip_path": full_clip_out,
            "hook_clip_path": None,
            "full_transcript_path": full_transcript_out
        }

    # ---------------------------------------------------------
    # Build hook clip
    # ---------------------------------------------------------
    transcript_clip_json = fs.read_json(full_transcript_out)
    hook_name = hook_clip_out.stem

    hook_clip_audio, hook_clip_info = extract_quote_clip(
        full_clip_out,
        {"text": quote_hook, "name": hook_name},
        transcript_clip_json,
        verbose=verbose
    )

    # ---------------------------------------------------------
    # Export hook clip
    # ---------------------------------------------------------
    export_clip(hook_clip_audio, hook_clip_out)

    # ---------------------------------------------------------
    # Return both
    # ---------------------------------------------------------
    return {
        "full_clip_path": full_clip_out,
        "hook_clip_path": hook_clip_out,
        "full_transcript_path": full_transcript_out
    }

def normalize_text_for_hash(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)   # strip punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_clip_id(
    data_source: str,
    dialogue_id: str | int,
    raw_text: str,
    max_id_length: int = 32
) -> str:
    """
    Create a stable, bounded-length clip ID based on:
    - data_source (e.g., 'congress', 'scotus', 'podcast')
    - dialogue_id (generalized event_id)
    - raw_text (cleaned transcript text)
    - max_id_length (hard cap on final ID length)
    """

    clean = normalize_text_for_hash(raw_text)

    # Build canonical payload
    payload = f"{data_source}|{dialogue_id}|{clean}"

    # Full SHA-256 hash
    full_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Final ID is truncated hash
    clip_id = full_hash[:max_id_length]

    return clip_id

def get_clip_paths(
    root_dir,
    data_source: str,
    dialogue_id: str | int,
    raw_text: str,
    max_id_length: int = 32
):
    """
    Resolve the canonical filesystem paths for a clip.
    - root_dir: base directory (string or Path)
    - data_source: domain (e.g., 'congress', 'scotus', 'podcast')
    - dialogue_id: generalized event ID
    - raw_text: cleaned transcript text
    - max_id_length: hard cap for clip_id length
    """

    # ---------------------------------------------------------
    # Defensive path normalization
    # ---------------------------------------------------------
    root_dir = Path(root_dir) if not isinstance(root_dir, Path) else root_dir

    # ---------------------------------------------------------
    # Compute deterministic clip_id
    # ---------------------------------------------------------
    clip_id = make_clip_id(
        data_source=data_source,
        dialogue_id=dialogue_id,
        raw_text=raw_text,
        max_id_length=max_id_length
    )

    # ---------------------------------------------------------
    # Directory structure:
    #   <root>/<data_source>/<dialogue_id>/<clip_id>.wav
    #   <root>/<data_source>/<dialogue_id>/<clip_id>.json
    # ---------------------------------------------------------
    dialogue_dir = root_dir / data_source / str(dialogue_id)
    dialogue_dir.mkdir(parents=True, exist_ok=True)

    audio_path = dialogue_dir / f"{clip_id}.wav"
    meta_path  = dialogue_dir / f"{clip_id}.json"

    return clip_id, audio_path, meta_path



def get_or_create_clip(
    root_dir,
    data_source: str,
    dialogue_id: str | int,
    quote_block: list,
    quote_hook: str | None,
    audio_path,
    transcript_path,
    fs,
    verbose=False
):
    """
    High-level clip manager:
    - Computes clip_id for full clip (based on quote_block)
    - Computes clip_id for hook clip (based on quote_hook)
    - Computes clip paths
    - Checks cache for each clip independently
    - Calls gen_audio_clips once to generate both clips
    - Writes metadata for both clips
    - All path inputs are defensively cast to Path
    """

    # ---------------------------------------------------------
    # Defensive path normalization
    # ---------------------------------------------------------
    root_dir = Path(root_dir)
    audio_path = Path(audio_path)
    transcript_path = Path(transcript_path)

    # ---------------------------------------------------------
    # Build raw_text for full clip identity
    # ---------------------------------------------------------
    full_text = " ".join(c["content"] for c in quote_block)

    # ---------------------------------------------------------
    # Compute FULL clip paths
    # ---------------------------------------------------------
    full_clip_id, full_clip_out, full_meta_path = get_clip_paths(
        root_dir=root_dir,
        data_source=data_source,
        dialogue_id=dialogue_id,
        raw_text=full_text,
        max_id_length=32
    )

    # ---------------------------------------------------------
    # Compute HOOK clip paths (if hook exists)
    # ---------------------------------------------------------
    if quote_hook:
        hook_clip_id, hook_clip_out, hook_meta_path = get_clip_paths(
            root_dir=root_dir,
            data_source=data_source,
            dialogue_id=dialogue_id,
            raw_text=quote_hook,
            max_id_length=32
        )
    else:
        hook_clip_id = None
        hook_clip_out = None
        hook_meta_path = None

    # ---------------------------------------------------------
    # Cache checks (independent)
    # ---------------------------------------------------------
    full_cached = full_clip_out.exists() and full_meta_path.exists()
    hook_cached = (
        hook_clip_out is None or
        (hook_clip_out.exists() and hook_meta_path.exists())
    )

    # ---------------------------------------------------------
    # Case 1: Both clips cached → return immediately
    # ---------------------------------------------------------
    if full_cached and hook_cached:
        return {
            "full_clip_id": full_clip_id,
            "hook_clip_id": hook_clip_id,
            "full_clip_path": full_clip_out,
            "hook_clip_path": hook_clip_out,
            "full_metadata_path": full_meta_path,
            "hook_metadata_path": hook_meta_path
        }

    # ---------------------------------------------------------
    # Case 2: Full cached, hook missing → generate hook only
    # ---------------------------------------------------------
    if full_cached and not hook_cached:
        result = gen_audio_clips(
            audio_path=audio_path,
            transcript_path=transcript_path,
            quote_block=quote_block,
            quote_hook=quote_hook,
            full_clip_out=full_clip_out,   # full clip already exists
            hook_clip_out=hook_clip_out,
            fs=fs,
            verbose=verbose
        )

        # Write hook metadata
        hook_metadata = {
            "clip_id": hook_clip_id,
            "type": "hook",
            "data_source": data_source,
            "dialogue_id": dialogue_id,
            "raw_text": quote_hook,
            "quote_block": quote_block,
            "audio_path": str(hook_clip_out),
            "transcript_path": str(result["full_transcript_path"]),
            "active":True,
        }
        fs.write_json(hook_meta_path, hook_metadata)

        return {
            "full_clip_id": full_clip_id,
            "hook_clip_id": hook_clip_id,
            "full_clip_path": full_clip_out,
            "hook_clip_path": hook_clip_out,
            "full_metadata_path": full_meta_path,
            "hook_metadata_path": hook_meta_path
        }

    # ---------------------------------------------------------
    # Case 3: Full missing → generate both clips
    # ---------------------------------------------------------
    result = gen_audio_clips(
        audio_path=audio_path,
        transcript_path=transcript_path,
        quote_block=quote_block,
        quote_hook=quote_hook,
        full_clip_out=full_clip_out,
        hook_clip_out=hook_clip_out,
        fs=fs,
        verbose=verbose
    )

    fpath_full_transcript = str(result["full_transcript_path"])

    c_transcript = fs.read_json(
        fpath_full_transcript
    )

    c_transcript_text = ' '.join(
        [c['text'] for c in c_transcript if len(c['text'])]
    )

    text_distance_score = normalized_levenshtein(
        full_text, 
        c_transcript_text
    )

    is_close = (text_distance_score>0.8)



    # Write FULL metadata
    full_metadata = {
        "clip_id": full_clip_id,
        "type": "full",
        "data_source": data_source,
        "dialogue_id": dialogue_id,
        "raw_text": full_text,
        "quote_block": quote_block,
        "audio_path": str(full_clip_out),
        "transcript_path": fpath_full_transcript,
        "active":is_close,
    }


    fs.write_json(full_meta_path, full_metadata)

    # Write HOOK metadata (if applicable)
    if hook_clip_out:
        hook_metadata = {
            "clip_id": hook_clip_id,
            "type": "hook",
            "data_source": data_source,
            "dialogue_id": dialogue_id,
            "raw_text": quote_hook,
            "quote_block": quote_block,
            "audio_path": str(hook_clip_out),
            "transcript_path": str(result["full_transcript_path"]),
            "active":is_close,
        }
        fs.write_json(hook_meta_path, hook_metadata)

    return {
        "full_clip_id": full_clip_id,
        "hook_clip_id": hook_clip_id,
        "full_clip_path": full_clip_out,
        "hook_clip_path": hook_clip_out,
        "full_metadata_path": full_meta_path,
        "hook_metadata_path": hook_meta_path
    }
