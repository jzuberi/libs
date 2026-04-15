
import re, os, sys, math, time, subprocess
from collections import Counter
from difflib import SequenceMatcher
from pydub import AudioSegment
import mlx_whisper
from rapidfuzz import process, fuzz
from typing import List, Set, Tuple, Dict, Any


from moviepy.editor import AudioFileClip, ImageClip, ColorClip,CompositeVideoClip, concatenate_videoclips



sys.path.append('/Users/pense/projects/pensiveturtles_data/pfuncs/utils')

import utils as f


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
        if(verbose):
            print(f"\n=== Iteration {i+1} ===")

        # Slice words within current boundaries
        cand_words = [
            w for w in words
            if time_start_milli <= w["start"] * 1000
            and w["end"] * 1000 <= time_end_milli
        ]
        whisper_result_text = ' '.join([w['text'] for w in cand_words])

        # Evaluate alignment
        result = evaluate_alignment(
            whisper_result_text,
            anchor_text,
            threshold_overall=threshold_overall,
            window=window,
        )
        if(verbose):
            print(result)

        # Length ratio
        cand_len = len(whisper_result_text)
        targ_len = len(anchor_text)
        length_ratio = cand_len / max(1, targ_len)
        if(verbose):
            print(f"[DIAG] Candidate length={cand_len}, Target length={targ_len}, Ratio={length_ratio:.2f}")

        # Edge differentials
        start_diff = abs(result["start_edge_score"] - result["start_anchor_score"])
        end_diff   = abs(result["end_edge_score"] - result["end_anchor_score"])

        if(verbose):
            print(f"[DIAG] Start diff={start_diff:.3f}, End diff={end_diff:.3f}")

        # --- Dynamic step size ---
        clip_len = time_end_milli - time_start_milli
        # proportional to clip length
        step_size = int(max(base_step, clip_len * frac_clip))
        # shrink as iterations progress
        step_size = int(step_size * (1 - i / max_iters))
        
        if(verbose):
            print(f"[DIAG] Dynamic step_size={step_size} ms")

        # --- OVERRIDE BASED ON LENGTH ---
        if length_ratio < min_ratio:
            if start_diff <= end_diff:
                if(verbose):
                    print("[DIAG] Candidate too short → extending start.")
                time_start_milli = max(0, time_start_milli - step_size)
            else:
                if(verbose):
                    print("[DIAG] Candidate too short → extending end.")
                time_end_milli += step_size

        elif length_ratio > max_ratio:
            if start_diff >= end_diff:
                if(verbose):
                    print("[DIAG] Candidate too long → contracting start.")
                time_start_milli += step_size
            else:
                if(verbose):
                    print("[DIAG] Candidate too long → contracting end.")
                time_end_milli -= step_size

        else:
            if result["start"] == "contract_start":
                time_start_milli += step_size
            elif result["start"] == "extend_start":
                time_start_milli = max(0, time_start_milli - step_size)

            if result["end"] == "contract_end":
                time_end_milli -= step_size
            elif result["end"] == "extend_end":
                time_end_milli += step_size

        # Stop early if both edges ok AND length ratio acceptable
        if result["start"] == "ok" and result["end"] == "ok" and min_ratio <= length_ratio <= max_ratio:

            if(verbose):
                print("[LOOP] Both edges aligned and candidate length acceptable → stopping early.")
            break

    return time_start_milli, time_end_milli, result


def run_whisper_on_chunks(
    fpath, 
    model_path='/Users/pense/.lmstudio/models/mlx-community/whisper-large-v3-turbo', 
    chunk_duration=60000, 
    overlap=2000,
    sleep_duration=5,
    word_timestamps=False,
    verbose=False
    ):

    audio = AudioSegment.from_mp3(fpath)
    audio_duration_ms = audio.duration_seconds * 1000
    num_splits = math.ceil(audio_duration_ms / chunk_duration)

    tempdir = os.path.join(os.path.dirname(fpath), "temp")
    f.create_folder_if_not_exists(tempdir)

    # Export chunks
    for n in range(num_splits):
        start = n * chunk_duration
        stop = (n + 1) * chunk_duration + overlap
        clip = audio[start:stop]
        clip.export(os.path.join(tempdir, f"short_test_clip_{n}.wav"), format="wav")

    merged = []
    last_end = 0.0
    last_word_text = None

    for n in range(num_splits):

        if verbose:
            print(f"Processing chunk {n+1}/{num_splits}")

        offset = n * (chunk_duration / 1000.0)

        result = mlx_whisper.transcribe(
            os.path.join(tempdir, f"short_test_clip_{n}.wav"),
            path_or_hf_repo=model_path,
            word_timestamps=word_timestamps,
        )

        time.sleep(sleep_duration)

        if not word_timestamps:
            # Segment-level mode (unchanged)
            for seg in result["segments"]:
                seg_start = seg["start"] + offset
                seg_end = seg["end"] + offset

                # enforce monotonicity
                if seg_start < last_end:
                    seg_start = last_end + 0.01
                if seg_end < seg_start:
                    seg_end = seg_start + 0.01

                merged.append({
                    "text": seg["text"].strip(),
                    "start": seg_start,
                    "end": seg_end,
                })
                last_end = seg_end

            continue

        # WORD TIMESTAMPS MODE
        for seg in result["segments"]:
            for w in seg["words"]:

                text = w["word"].strip()
                if not text:
                    continue

                # dedupe based on text, not timestamp jitter
                if text == last_word_text:
                    continue

                start = w["start"] + offset
                end = w["end"] + offset

                # enforce monotonic timestamps
                if start < last_end:
                    start = last_end + 0.01
                if end < start:
                    end = start + 0.01

                merged.append({
                    "text": text,
                    "start": start,
                    "end": end,
                })

                last_end = end
                last_word_text = text

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
    tokens = clean_text_preserve_phrases(context).split()
    rare = []
    for t in tokens:
        freq = freq_dict.get(t, 0)
        if freq <= rarity_threshold and len(t) >= min_length:
            rare.append((t, freq))
        elif ("-" in t or t.isupper()) and freq <= rarity_threshold and len(t) >= min_length:
            rare.append((t, freq))

    # Sort by frequency (rarest first), then by length (longer anchors first)
    rare.sort(key=lambda x: (x[1], -len(x[0])))
    return [t for t, _ in rare]

def best_guess_time_for_context(
    context,
    transcript_df,
    text_col="text",
    start_col="start",
    end_col="end",
    threshold=0.6,
    weak_threshold=0.4,
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
            score = 0.7 * ratio + 0.3 * (len(rare_overlap) + 2*acronym_boost + 2*hyphen_boost)
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

