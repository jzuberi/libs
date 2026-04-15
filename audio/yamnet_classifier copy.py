"""
yamnet_classifier.py
--------------------
Reusable library for running YAMNet audio classification, with focus on laughter detection.
"""

import os
import subprocess
import librosa
import numpy as np
import tensorflow as tf
import csv

def load_model(model_path: str):
    """Load a YAMNet SavedModel from disk."""
    return tf.saved_model.load(model_path)

def convert_to_wav(input_file: str, temp_dir: str, sr: int = 16000) -> str:
    """Convert mp4 to wav using ffmpeg, return path to wav file."""
    os.makedirs(temp_dir, exist_ok=True)
    output_file = os.path.join(
        temp_dir, os.path.splitext(os.path.basename(input_file))[0] + ".wav"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_file, "-ar", str(sr), "-ac", "1", output_file],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_file

def load_audio(input_file: str, sr: int = 16000, mono: bool = True, temp_dir: str = None):
    """Load audio, converting mp4 to wav if necessary."""
    cleanup = False
    audio_file = input_file
    if input_file.lower().endswith(".mp4"):
        if temp_dir is None:
            raise ValueError("temp_dir must be provided for mp4 conversion")
        audio_file = convert_to_wav(input_file, temp_dir, sr)
        cleanup = True
    waveform, sr = librosa.load(audio_file, sr=sr, mono=mono)
    return waveform, sr, audio_file, cleanup

def get_class_names(model) -> list:
    """Extract class names from YAMNet model."""
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, "r") as f:
        reader = csv.DictReader(f)
        return [row["display_name"] for row in reader]

def detect_laughter(model, waveform, class_names, threshold: float = 0.01):
    """
    Return frames where laughter classes exceed threshold.
    Each result includes (frame_idx, start_sec, end_sec, class_name, score).
    """
    scores, _, _ = model(waveform)
    scores_np = scores.numpy()

    total_duration = len(waveform) / 16000.0  # YAMNet expects 16 kHz
    num_frames = scores_np.shape[0]
    stride = total_duration / num_frames       # ~0.48s in practice
    patch_duration = 0.975                     # YAMNet patch coverage (~0.975s)

    laughter_indices = [
        i for i, name in enumerate(class_names)
        if "laugh" in name.lower() or "giggle" in name.lower() or "snicker" in name.lower()
    ]

    results = []
    for frame_idx in range(num_frames):
        start_sec = frame_idx * stride
        end_sec = start_sec + patch_duration
        for class_idx in laughter_indices:
            score = scores_np[frame_idx, class_idx]
            if score > threshold:
                results.append((frame_idx, start_sec, end_sec, class_names[class_idx], score))
    return results

def has_laughter(model_path: str, input_file: str, threshold: float = 0.01, temp_dir: str = "./temp"):
    """
    Returns (flag, results) where:
      - flag: True if any laughter detected
      - results: list of (frame_idx, start_sec, end_sec, class_name, score)
    """
    model = load_model(model_path)
    waveform, sr, audio_file, cleanup = load_audio(input_file, sr=16000, mono=True, temp_dir=temp_dir)
    class_names = get_class_names(model)
    results = detect_laughter(model, waveform, class_names, threshold)

    if cleanup and os.path.exists(audio_file):
        os.remove(audio_file)

    return results

