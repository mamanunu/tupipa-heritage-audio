#!/usr/bin/env python3
"""Extract onset-aligned acoustic features from the Tupipa WAV recordings.

The script intentionally uses only the Python standard library and NumPy so
it can run without extra installation beyond NumPy.

Usage:
    python analyze_tupipa_features.py AUDIO_DIRECTORY OUTPUT_CSV
"""

from __future__ import annotations

import csv
import argparse
import json
import math
import re
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np


FILENAME_RE = re.compile(
    r"^session(?P<session>\d+)_(?P<position>\d+)_(?P<intensity>P|MF|F)(?:_(?P<take>\d+))?\.wav$"
)


def read_pcm(path: Path) -> tuple[int, int, int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values | ~0xFFFFFF, values)
        data = values.astype(np.float64) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {width}: {path}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return sample_rate, channels, width, data


def frame_rms(audio: np.ndarray, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) < frame:
        padded = np.pad(audio, (0, frame - len(audio)))
    else:
        padded = audio
    starts = np.arange(0, max(1, len(padded) - frame + 1), hop)
    values = np.empty(len(starts), dtype=np.float64)
    for i, start in enumerate(starts):
        x = padded[start : start + frame]
        values[i] = math.sqrt(float(np.mean(x * x)) + 1e-18)
    return starts.astype(np.int64), values


def detect_onset(sr: int, audio: np.ndarray) -> tuple[float, float, float]:
    frame = max(256, int(0.020 * sr))
    hop = max(64, int(0.005 * sr))
    starts, rms = frame_rms(audio, frame, hop)
    if len(rms) == 0:
        return 0.0, 0.0, 0.0

    peak_idx = int(np.argmax(rms))
    peak_rms = float(rms[peak_idx])
    baseline_n = max(1, min(len(rms), int(0.10 * sr / hop)))
    baseline = float(np.median(rms[:baseline_n]))
    noise = float(np.median(np.abs(rms[:baseline_n] - baseline))) + 1e-12
    threshold = max(baseline * 3.0, baseline + 8.0 * noise, peak_rms * 0.08)
    candidates = np.where(rms[: peak_idx + 1] >= threshold)[0]
    onset_idx = int(candidates[0]) if len(candidates) else max(0, peak_idx - 1)
    onset = float((starts[onset_idx] + frame / 2) / sr)
    peak = float((starts[peak_idx] + frame / 2) / sr)
    return onset, peak, peak_rms


def parabolic_peak(freqs: np.ndarray, magnitude: np.ndarray, index: int) -> tuple[float, float]:
    if index <= 0 or index >= len(magnitude) - 1:
        return float(freqs[index]), float(magnitude[index])
    alpha = magnitude[index - 1]
    beta = magnitude[index]
    gamma = magnitude[index + 1]
    denom = alpha - 2.0 * beta + gamma
    shift = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-18 else 0.0
    return float(freqs[index] + shift * (freqs[1] - freqs[0])), float(
        max(0.0, beta - 0.25 * (alpha - gamma) * shift)
    )


def spectrum_features(
    x: np.ndarray, sr: int, f0: float | None = None
) -> dict[str, float]:
    if len(x) < 256:
        return {
            "centroid_hz": 0.0,
            "bandwidth_hz": 0.0,
            "rolloff_hz": 0.0,
            "flatness": 0.0,
            "harmonic_ratio": 0.0,
            "relative_h2_db": 0.0,
            "relative_h3_db": 0.0,
            "relative_h4_db": 0.0,
            "relative_h5_db": 0.0,
        }
    n = min(16384, max(4096, 2 ** int(math.ceil(math.log2(len(x))))))
    window = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * window, n=n)) + 1e-12
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = spec * spec
    total = float(np.sum(power))
    centroid = float(np.sum(freqs * power) / total)
    bandwidth = float(math.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total))
    cumulative = np.cumsum(power)
    roll_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])
    geometric_mean = float(np.exp(np.mean(np.log(spec))))
    flatness = geometric_mean / max(float(np.mean(spec)), 1e-12)

    band = (freqs >= 80.0) & (freqs <= min(8000.0, sr / 2.0))
    band_power = float(np.sum(power[band])) + 1e-12
    harmonic_ratio = 0.0
    ratios: dict[str, float] = {}
    if f0 and f0 > 0:
        harmonic_power = 0.0
        amps = []
        for harmonic in range(1, 11):
            target = harmonic * f0
            mask = np.abs(freqs - target) <= max(10.0, f0 * 0.08)
            if np.any(mask):
                amp = float(np.max(spec[mask]))
                harmonic_power += float(np.sum(power[mask]))
            else:
                amp = 0.0
            amps.append(amp)
        harmonic_ratio = harmonic_power / band_power
        h1 = max(amps[0], 1e-12)
        for harmonic in range(2, 6):
            ratios[f"relative_h{harmonic}_db"] = float(
                20.0 * math.log10(max(amps[harmonic - 1], 1e-12) / h1)
            )
    return {
        "centroid_hz": centroid,
        "bandwidth_hz": bandwidth,
        "rolloff_hz": rolloff,
        "flatness": flatness,
        "harmonic_ratio": harmonic_ratio,
        **{f"relative_h{i}_db": ratios.get(f"relative_h{i}_db", 0.0) for i in range(2, 6)},
    }


def spectrum_feature_names() -> tuple[str, ...]:
    return (
        "centroid_hz",
        "bandwidth_hz",
        "rolloff_hz",
        "flatness",
        "harmonic_ratio",
        "relative_h2_db",
        "relative_h3_db",
        "relative_h4_db",
        "relative_h5_db",
    )


def spectrum_features_multiframe(
    x: np.ndarray,
    sr: int,
    f0: float | None = None,
    frame_length: int = 16384,
    hop_length: int = 4096,
) -> tuple[dict[str, float], int]:
    """Summarize a complete region with fixed-window short-time spectra.

    The decay region is much longer than one FFT window. Complete frames are
    analyzed with a fixed Hann window and the feature values are aggregated by
    median, so the reported decay descriptors use the whole region rather than
    the prefix that fits in a single FFT.
    """
    if len(x) < 256:
        return spectrum_features(x, sr, f0), 0
    if len(x) < frame_length:
        padded = np.pad(x, (0, frame_length - len(x)))
        starts = np.asarray([0], dtype=np.int64)
    else:
        starts = np.arange(0, len(x) - frame_length + 1, hop_length, dtype=np.int64)
        padded = x
    frame_values: list[dict[str, float]] = []
    for start in starts:
        frame = padded[int(start) : int(start) + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        spec = spectrum_features(frame, sr, f0)
        frame_values.append(spec)
    aggregated = {
        name: float(np.median([values[name] for values in frame_values]))
        for name in spectrum_feature_names()
    }
    return aggregated, len(frame_values)


def estimate_f0(sr: int, audio: np.ndarray, onset: float) -> tuple[float, float]:
    start = int((onset + 0.08) * sr)
    duration = min(len(audio) - start, int(0.35 * sr))
    if duration < 2048:
        return 0.0, 0.0
    x = audio[start : start + duration]
    x = x - float(np.mean(x))
    window = np.hanning(len(x))
    n = max(32768, 2 ** int(math.ceil(math.log2(len(x)))))
    spec = np.abs(np.fft.rfft(x * window, n=n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = (freqs >= 250.0) & (freqs <= 500.0)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return 0.0, 0.0
    idx = int(idxs[np.argmax(spec[idxs])])
    f0, amp = parabolic_peak(freqs, spec, idx)
    return f0, amp


def duration_to_threshold(
    sr: int, audio: np.ndarray, peak_sample: int, threshold_ratio: float
) -> float:
    envelope_frame = max(256, int(0.010 * sr))
    hop = max(64, int(0.0025 * sr))
    starts, rms = frame_rms(audio[peak_sample:], envelope_frame, hop)
    if len(rms) == 0:
        return 0.0
    peak = float(np.max(rms))
    target = peak * threshold_ratio
    below = np.where(rms <= target)[0]
    if len(below) == 0:
        return float((len(audio) - peak_sample) / sr)
    return float((starts[int(below[0])] + envelope_frame / 2) / sr)


def analyze_one(path: Path) -> dict[str, object]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized filename: {path.name}")
    meta = match.groupdict()
    take = int(meta["take"]) if meta["take"] is not None else 0
    sr, channels, width, audio = read_pcm(path)
    audio = audio - float(np.mean(audio[: max(1, int(0.05 * sr))]))
    onset, peak, peak_rms = detect_onset(sr, audio)
    peak_sample = min(len(audio) - 1, max(0, int(peak * sr)))
    f0, f0_amp = estimate_f0(sr, audio, onset)

    onset_sample = int(onset * sr)
    windows = {
        "onset": audio[onset_sample : onset_sample + int(0.05 * sr)],
        "body": audio[onset_sample + int(0.08 * sr) : onset_sample + int(0.30 * sr)],
        "decay": audio[onset_sample + int(0.30 * sr) : onset_sample + int(1.80 * sr)],
    }
    rms_values = {
        f"rms_{name}": float(math.sqrt(np.mean(x * x) + 1e-18))
        if len(x)
        else 0.0
        for name, x in windows.items()
    }
    onset_features = spectrum_features(windows["onset"], sr, f0)
    body_features = spectrum_features(windows["body"], sr, f0)
    decay_features, decay_fft_frame_count = spectrum_features_multiframe(
        windows["decay"], sr, f0
    )

    return {
        "filename": path.name,
        "session": int(meta["session"]),
        "position": int(meta["position"]),
        "intensity": meta["intensity"],
        "take": take,
        "sample_rate": sr,
        "channels": channels,
        "sample_width_bytes": width,
        "duration_s": len(audio) / sr,
        "onset_s": onset,
        "peak_s": peak,
        "peak_rms": peak_rms,
        "peak_abs": float(np.max(np.abs(audio))),
        "decay_fft_frame_count": decay_fft_frame_count,
        "f0_hz": f0,
        "f0_peak_amplitude": f0_amp,
        "decay_6db_s": duration_to_threshold(sr, audio, peak_sample, 0.5),
        "decay_20db_s": duration_to_threshold(sr, audio, peak_sample, 0.1),
        **rms_values,
        **{f"onset_{key}": value for key, value in onset_features.items()},
        **{f"body_{key}": value for key, value in body_features.items()},
        **{f"decay_{key}": value for key, value in decay_features.items()},
    }


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def summarize(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    numeric_keys = [
        "duration_s",
        "onset_s",
        "peak_s",
        "peak_rms",
        "peak_abs",
        "f0_hz",
        "rms_onset",
        "rms_body",
        "rms_decay",
        "onset_centroid_hz",
        "body_centroid_hz",
        "body_harmonic_ratio",
        "body_relative_h2_db",
        "body_relative_h3_db",
        "body_relative_h4_db",
        "body_relative_h5_db",
        "decay_6db_s",
        "decay_20db_s",
        "decay_fft_frame_count",
    ]
    output = []
    for group, members in sorted(groups.items()):
        item: dict[str, object] = {key: value for key, value in zip(keys, group)}
        item["n"] = len(members)
        for key in numeric_keys:
            values = [float(member[key]) for member in members]
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_median"] = median(values)
            item[f"{key}_iqr"] = float(
                np.percentile(np.asarray(values, dtype=np.float64), 75)
                - np.percentile(np.asarray(values, dtype=np.float64), 25)
            ) if values else 0.0
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the per-file acoustic features used in the Tupipa study."
    )
    parser.add_argument("audio_directory", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    files = sorted(args.audio_directory.rglob("*.wav"))
    if not files:
        raise SystemExit(f"No WAV files found in: {args.audio_directory}")
    rows = [analyze_one(path) for path in files]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
