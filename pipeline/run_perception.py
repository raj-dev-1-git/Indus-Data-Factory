import json
from pathlib import Path

import librosa
import numpy as np
import torch
import whisper
import laion_clap

BASE = Path(__file__).resolve().parent.parent

SCENES = BASE / "scenes" / "audio"
METADATA = BASE / "scenes" / "metadata"
OUT = BASE / "perception"
OUT.mkdir(parents=True, exist_ok=True)

EVENT_LABELS = [
    "airplane flying overhead",
    "dog barking",
    "emergency siren",
    "road traffic noise",
    "crowd of people talking",
]

# map CLAP descriptive labels to canonical dataset event_folder names
LABEL_TO_EVENT = {
    "airplane flying overhead": "airplane",
    "dog barking":              "dog_bark",
    "emergency siren":          "siren",
    "road traffic noise":       "traffic",
    "crowd of people talking":  "crowd",
}

# map folder names to whisper language codes
LANG_MAP = {
    "marathi": "mr",
    "tamil": "ta",
    "bangla": "bn",
    "telugu": "te",
}

WINDOW_SECONDS = 1
STRIDE_SECONDS = 0.5
EVENT_THRESHOLD = 0.22

INITIAL_PROMPTS = {
    "mr": "कृपया सर्व आकडे शब्दांत लिहा.",
    "ta": "தயவுசெய்து அனைத்து எண்களையும் வார்த்தைகளில் எழுதவும்.",
    "bn": "অনুগ্রহ করে সব সংখ্যা শব্দে লিখুন।",
    "te": "దయచేసి అన్ని సంఖ్యలను పదాలలో రాయండి."
}

print("Loading Whisper large-v3...")
whisper_model = whisper.load_model("large-v3")

print("Loading CLAP...")
clap_model = laion_clap.CLAP_Module(enable_fusion=False)
clap_model.load_ckpt()


scene_files = sorted(SCENES.glob("*.wav"))

for audio_path in scene_files:
    print(f"\nProcessing: {audio_path.name}")

    # read scene metadata to get the actual language
    meta_file = METADATA / f"{audio_path.stem}.json"
    lang_code = None
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        lang_code = LANG_MAP.get(meta.get("language", ""), None)

    # ASR with forced language (the key fix)
    try:
        import noisereduce as nr
        # Load and denoise audio specifically for Whisper at 16kHz
        whisper_audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        denoised_audio = nr.reduce_noise(y=whisper_audio, sr=16000)
        
        if lang_code:
            prompt = INITIAL_PROMPTS.get(lang_code, "")
            result = whisper_model.transcribe(
                denoised_audio, language=lang_code, task="transcribe", condition_on_previous_text=False, initial_prompt=prompt
            )
            safe_text = result['text'][:80].encode('ascii', 'backslashreplace').decode('ascii')
            print(f"Forced lang={lang_code} | Text: {safe_text}...")
        else:
            result = whisper_model.transcribe(denoised_audio, condition_on_previous_text=False)
            safe_text = result['text'][:80].encode('ascii', 'backslashreplace').decode('ascii')
            print(f"Auto-detect lang={result.get('language')} | Text: {safe_text}...")
    except Exception as e:
        print(f"ASR failed: {e}")
        continue

    transcript = result["text"]

    # speech segments from whisper output
    speech_segments = []
    for seg in result.get("segments", []):
        speech_segments.append(
            {
                "type": "speech",
                "content": seg["text"].strip(),
                "time": f"{seg['start']:.2f}-{seg['end']:.2f}",
                "confidence": round(float(1.0 - seg.get("no_speech_prob", 0)), 3),
            }
        )

    # CLAP event detection at 48kHz
    audio, sr = librosa.load(audio_path, sr=48000, mono=True)

    sound_events = []
    window_size = int(WINDOW_SECONDS * sr)
    stride_size = int(STRIDE_SECONDS * sr)
    total_length = len(audio)

    for start_idx in range(0, total_length - window_size, stride_size):
        end_idx = start_idx + window_size
        chunk = audio[start_idx:end_idx]

        if np.max(np.abs(chunk)) < 0.01:
            continue

        chunk_tensor = torch.from_numpy(chunk).float()

        audio_embed = clap_model.get_audio_embedding_from_data(
            x=[chunk_tensor], use_tensor=True
        )
        text_embed = clap_model.get_text_embedding(EVENT_LABELS, use_tensor=True)
        similarity = torch.softmax(audio_embed @ text_embed.T, dim=-1)
        similarity = similarity[0].detach().cpu().numpy()

        for label, score in zip(EVENT_LABELS, similarity):
            if score > EVENT_THRESHOLD:
                start_time = start_idx / sr
                end_time = end_idx / sr
                sound_events.append(
                    {
                        "type": "sound_event",
                        "content": label,
                        "canonical_event": LABEL_TO_EVENT.get(label, "unknown"),
                        "time": f"{start_time:.2f}-{end_time:.2f}",
                        "confidence": round(float(score), 3),
                    }
                )

    perception = {
        "audio_file": audio_path.name,
        "detected_language": lang_code or result.get("language", "unknown"),
        "full_transcript": transcript,
        "speech": speech_segments,
        "events": sound_events,
    }

    out_file = OUT / f"{audio_path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(perception, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_file.name}")

print("\nPerception complete.")
