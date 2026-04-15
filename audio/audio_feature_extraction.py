import librosa
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from itertools import chain

import math
from collections import defaultdict

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
model.eval()


def load_audio(input_file, sr=16000, mono=True):
    """
    Load audio directly with librosa. Handles mp3, mp4, wav, etc.
    Always resamples to 16 kHz for AST.
    """
    waveform, sr = librosa.load(input_file, sr=sr, mono=mono)
    return waveform, sr


def detect_annotation_timestamps(
    waveform,
    annotation_groupings,
    sr=16000,
    window_size=1.0,
    hop_size=1.0,
    verbose=False
    ):

    """
    Slide a window across the waveform and return ALL class scores
    for each window, without filtering.
    """

    annotation_label_list = list(chain(*[c['labels'] for c in annotation_groupings]))

    labels = model.config.id2label
    class_indices = list(labels.keys())  # all classes

    win_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    detections = []

    if(verbose):
        print(len(range(0, len(waveform) - win_samples, hop_samples)))

    num=0

    for start in range(0, len(waveform) - win_samples, hop_samples):

        num+=1

        if(verbose):
            print(num)

        end = start + win_samples
        chunk = waveform[start:end]

        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # produce all class scores for this window
        for idx in class_indices:

            if(labels[idx] in annotation_label_list):

                detections.append({
                    "start": start / sr,
                    "end": end / sr,
                    "label": labels[idx],
                    "score": float(probs[idx])
                })

    return detections


def find_annotation_segments(
    input_file,
    window_size=1.0,
    hop_size=1.0,
    verbose=False
    ):

    """
    Public API: load audio, run windowed inference, return raw segments.
    """
    waveform, sr = load_audio(input_file, sr=16000, mono=True)

    segments = detect_annotation_timestamps(
        waveform,
        sr=sr,
        window_size=window_size,
        hop_size=hop_size,
        verbose=verbose
    )

    return segments

def scaled_threshold(base_threshold, window_size_sec, mode="sqrt"):
    if mode == "sqrt":
        return base_threshold / math.sqrt(window_size_sec)
    elif mode == "linear":
        return base_threshold / window_size_sec
    else:
        return base_threshold


def aggregate_by_window_and_group(detections, annotation_groupings):

    # Build mappings
    label_to_group = {}
    for g in annotation_groupings:
        for lab in g['labels']:
            label_to_group[lab] = g['name']

    # (start, end, group_name) -> max_score
    agg = defaultdict(float)

    for d in detections:
        label = d['label']
        if label not in label_to_group:
            continue

        group_name = label_to_group[label]
        key = (d['start'], d['end'], group_name)
        if d['score'] > agg[key]:
            agg[key] = d['score']

    # Turn into a list of dicts
    out = []
    for (start, end, group_name), max_score in agg.items():
        out.append({
            "start": start,
            "end": end,
            "group": group_name,
            "score": max_score,
        })
    return out


def hierarchical_find_annotation_segments(
    input_file,
    annotation_groupings,
    coarse_window_size=60.0,
    coarse_hop_size=60.0,
    fine_window_size=1.0,
    fine_hop_size=1.0,
    verbose=False
):
    

    # 1) Load audio once
    waveform, sr = load_audio(input_file, sr=16000, mono=True)

    # 2) Coarse scan (big windows)
    coarse_detections = detect_annotation_timestamps(
        waveform,
        annotation_groupings,
        sr=sr,
        window_size=coarse_window_size,
        hop_size=coarse_hop_size,
        verbose=verbose
    )

    # 3) Aggregate by window + group (max over labels in group)
    coarse_group_windows = aggregate_by_window_and_group(
        coarse_detections,
        annotation_groupings
    )

    # Build group -> base threshold map
    group_thresholds = {g['name']: g['threshold'] for g in annotation_groupings}

    # 4) Decide which coarse windows to refine
    flagged_windows = []
    for w in coarse_group_windows:
        group_name = w['group']
        base_th = group_thresholds[group_name]
        th = scaled_threshold(base_th, coarse_window_size, mode="linear")

        if w['score'] >= th:
            flagged_windows.append(w)

    if verbose:
        print(f"Flagged {len(flagged_windows)} coarse windows for fine scan")

    # 5) Fine scan only inside flagged windows
    fine_results = []

    for win in flagged_windows:
        start_s = int(win['start'] * sr)
        end_s = int(win['end'] * sr)
        chunk = waveform[start_s:end_s]

        # Run the same detector but with 1s windows
        fine_dets = detect_annotation_timestamps(
            chunk,
            annotation_groupings,
            sr=sr,
            window_size=fine_window_size,
            hop_size=fine_hop_size,
            verbose=verbose
        )

        # Adjust times back to global timeline
        for d in fine_dets:
            d['start'] += win['start']
            d['end'] += win['start']
            fine_results.append(d)

    return fine_results

