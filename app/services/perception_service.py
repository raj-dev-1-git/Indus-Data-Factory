import librosa
import numpy as np
import torch
import whisper
import laion_clap
import noisereduce as nr

whisper_model = None
clap_model = None

def get_models():
    global whisper_model, clap_model
    if whisper_model is None:
        print("Loading Whisper large-v3...")
        whisper_model = whisper.load_model("large-v3")
    if clap_model is None:
        print("Loading CLAP...")
        clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        clap_model.load_ckpt()
    return whisper_model, clap_model

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

LANG_MAP = {
    "marathi": "mr",
    "tamil": "ta",
    "bangla": "bn",
    "telugu": "te",
}

WINDOW_SECONDS = 1
STRIDE_SECONDS = 0.5
EVENT_THRESHOLD = 0.18

INITIAL_PROMPTS = {
    "mr": "कृपया सर्व आकडे शब्दांत लिहा.",
    "ta": "தயவுசெய்து அனைத்து எண்களையும் வார்த்தைகளில் எழுதவும்.",
    "bn": "অনুগ্রহ করে সব সংখ্যা শব্দে লিখুন।",
    "te": "దయచేసి అన్ని సంఖ్యలను పదాలలో రాయండి."
}


def run_perception(audio_path, language_hint=None):
    """
    Run ASR + event detection on an audio file.
    """
    whisper_model, clap_model = get_models()
    
    lang_code = None
    if language_hint:
        lang_code = LANG_MAP.get(language_hint, language_hint)

    # ASR
    try:
        # Load and denoise audio specifically for Whisper at 16kHz
        whisper_audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        denoised_audio = nr.reduce_noise(y=whisper_audio, sr=16000)

        if lang_code:
            prompt = INITIAL_PROMPTS.get(lang_code, "")
            result = whisper_model.transcribe(
                denoised_audio, language=lang_code, task="transcribe", condition_on_previous_text=False, initial_prompt=prompt
            )
        else:
            result = whisper_model.transcribe(denoised_audio, condition_on_previous_text=False)
    except Exception as e:
        print(f"ASR failed: {e}")
        return {
            "detected_language": "unknown",
            "full_transcript": "Transcription failed.",
            "speech": [],
            "events": [],
        }

    transcript = result["text"]
    detected_lang = lang_code or result.get("language", "unknown")

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
    audio_48k, sr = librosa.load(audio_path, sr=48000, mono=True)

    sound_events = []
    window_size = int(WINDOW_SECONDS * sr)
    stride_size = int(STRIDE_SECONDS * sr)
    total_length = len(audio_48k)
    
    # Compute text embeddings once
    text_embed = clap_model.get_text_embedding(EVENT_LABELS, use_tensor=True)
    
    # Store max confidence per event type
    event_max_scores = {label: 0.0 for label in EVENT_LABELS}

    for start_idx in range(0, total_length - window_size, stride_size):
        end_idx = start_idx + window_size
        chunk = audio_48k[start_idx:end_idx]

        if np.max(np.abs(chunk)) < 0.01:
            continue

        chunk_tensor = torch.from_numpy(chunk).float()

        audio_embed = clap_model.get_audio_embedding_from_data(
            x=[chunk_tensor], use_tensor=True
        )
        similarity = torch.softmax(audio_embed @ text_embed.T, dim=-1)
        similarity = similarity[0].detach().cpu().numpy()

        for label, score in zip(EVENT_LABELS, similarity):
            if score > event_max_scores[label]:
                event_max_scores[label] = float(score)

    # Add events that exceed threshold
    for label, max_score in event_max_scores.items():
        if max_score > EVENT_THRESHOLD:
            sound_events.append(
                {
                    "type": "sound_event",
                    "content": label,
                    "canonical_event": LABEL_TO_EVENT.get(label, "unknown"),
                    "time": f"0.00-10.00",
                    "confidence": round(max_score, 3),
                }
            )

    perception = {
        "detected_language": detected_lang,
        "full_transcript": transcript,
        "speech": speech_segments,
        "events": sound_events,
    }

    return perception
