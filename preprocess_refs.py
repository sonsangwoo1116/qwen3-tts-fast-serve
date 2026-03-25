"""
Reference audio preprocessing pipeline for Qwen3-TTS voice cloning.

Steps:
1. Noise reduction (stationary noise, noisereduce library)
2. Leading/trailing silence trim (threshold-based)
3. Peak normalization to 28000/32768
4. Append 0.5s silence padding (prevent phoneme bleed)

Usage: python preprocess_refs.py --input_dir /refs --output_dir /refs_processed
"""

import argparse
import os
import glob
import numpy as np
import soundfile as sf
import noisereduce as nr


def trim_silence(audio, sr, threshold=0.01, min_silence_ms=50):
    """Trim leading and trailing silence."""
    min_samples = int(min_silence_ms / 1000 * sr)
    abs_audio = np.abs(audio)

    # Find first non-silent sample
    start = 0
    for i in range(len(audio)):
        if abs_audio[i] > threshold:
            start = max(0, i - int(0.01 * sr))  # keep 10ms before speech
            break

    # Find last non-silent sample
    end = len(audio)
    for i in range(len(audio) - 1, -1, -1):
        if abs_audio[i] > threshold:
            end = min(len(audio), i + int(0.01 * sr))  # keep 10ms after speech
            break

    return audio[start:end]


def peak_normalize(audio, target_peak=28000/32768):
    """Normalize audio to target peak level."""
    current_peak = np.max(np.abs(audio))
    if current_peak < 1e-6:
        return audio
    gain = target_peak / current_peak
    return audio * gain


def add_silence_padding(audio, sr, duration_s=0.5):
    """Append silence padding at the end."""
    silence = np.zeros(int(duration_s * sr), dtype=audio.dtype)
    return np.concatenate([audio, silence])


def process_reference(input_path, output_path, sr_target=None):
    """Full preprocessing pipeline for one reference file."""
    audio, sr = sf.read(input_path)

    # Mono conversion if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    original_rms = np.sqrt(np.mean(audio**2)) * 32768
    original_peak = np.max(np.abs(audio)) * 32768

    # 1. Noise reduction
    audio_denoised = nr.reduce_noise(
        y=audio, sr=sr,
        stationary=True,
        prop_decrease=0.75,  # moderate reduction to preserve naturalness
        n_fft=2048,
        hop_length=512,
    )

    # 2. Trim silence
    audio_trimmed = trim_silence(audio_denoised, sr)

    # 3. Peak normalize
    audio_normalized = peak_normalize(audio_trimmed)

    # 4. Add 0.5s silence padding
    audio_padded = add_silence_padding(audio_normalized, sr, 0.5)

    final_rms = np.sqrt(np.mean(audio_padded**2)) * 32768
    final_peak = np.max(np.abs(audio_padded)) * 32768

    sf.write(output_path, audio_padded, sr, subtype='PCM_16')

    return {
        'sr': sr,
        'original_dur': len(audio) / sr,
        'final_dur': len(audio_padded) / sr,
        'original_peak': int(original_peak),
        'final_peak': int(final_peak),
        'original_rms': int(original_rms),
        'final_rms': int(final_rms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    voices = sorted(os.listdir(args.input_dir))
    for voice in voices:
        voice_dir = os.path.join(args.input_dir, voice)
        if not os.path.isdir(voice_dir):
            continue

        wavs = glob.glob(os.path.join(voice_dir, '*.wav'))
        txt_files = glob.glob(os.path.join(voice_dir, '*.txt'))

        if not wavs:
            continue

        out_voice_dir = os.path.join(args.output_dir, voice)
        os.makedirs(out_voice_dir, exist_ok=True)

        # Process WAV
        wav_path = wavs[0]
        out_wav = os.path.join(out_voice_dir, os.path.basename(wav_path))
        stats = process_reference(wav_path, out_wav)

        # Copy text files
        for txt in txt_files:
            import shutil
            shutil.copy2(txt, os.path.join(out_voice_dir, os.path.basename(txt)))

        print(f"{voice}: {stats['original_dur']:.1f}s→{stats['final_dur']:.1f}s, "
              f"peak {stats['original_peak']}→{stats['final_peak']}, "
              f"rms {stats['original_rms']}→{stats['final_rms']}")


if __name__ == '__main__':
    main()
