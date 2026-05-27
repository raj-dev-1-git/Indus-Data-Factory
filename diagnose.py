"""
Indus Data Factory -- Pipeline Alignment Diagnostic
Run: python diagnose.py
Checks everything is wired correctly across all 3 stages.
"""

import json
import sys
import os
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.system("")  # enable ANSI escape codes on Windows
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent

# -- Colors for terminal --
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed = 0
failed = 0
warnings = 0

def ok(msg):
    global passed
    passed += 1
    print(f"  {GREEN}[PASS]{RESET}  {msg}")

def fail(msg):
    global failed
    failed += 1
    print(f"  {RED}[FAIL]{RESET}  {msg}")

def warn(msg):
    global warnings
    warnings += 1
    print(f"  {YELLOW}[WARN]{RESET}  {msg}")

def header(title):
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{RESET}")


# ══════════════════════════════════════════════════════════
# CHECK 1: Mixer event types vs dataset folders
# ══════════════════════════════════════════════════════════
header("1. MIXER → DATASET FOLDER ALIGNMENT")

EVENTS_DIR = BASE / "dataset" / "events"

# What the mixer tries to use (from generate_mixer_logs.py)
MIXER_EVENT_TYPES = ["civil_defense_siren", "dog_bark", "gunfire", "subway_train"]

# What folders actually exist
existing_folders = [d.name for d in EVENTS_DIR.iterdir() if d.is_dir()] if EVENTS_DIR.exists() else []

print(f"\n  Mixer EVENT_TYPES:     {MIXER_EVENT_TYPES}")
print(f"  dataset/events/ has:   {sorted(existing_folders)}\n")

for etype in MIXER_EVENT_TYPES:
    folder = EVENTS_DIR / etype
    if not folder.exists():
        fail(f"Mixer uses '{etype}' but folder dataset/events/{etype}/ DOES NOT EXIST → events silently skipped")
    else:
        wav_files = list(folder.glob("*.wav"))
        if len(wav_files) == 0:
            fail(f"Folder dataset/events/{etype}/ exists but has 0 .wav files → events silently skipped")
        else:
            ok(f"dataset/events/{etype}/ → {len(wav_files)} .wav files found")

unused_folders = set(existing_folders) - set(MIXER_EVENT_TYPES)
if unused_folders:
    for f in sorted(unused_folders):
        wav_count = len(list((EVENTS_DIR / f).glob("*.wav")))
        warn(f"Folder dataset/events/{f}/ exists ({wav_count} .wav) but mixer NEVER uses it")


# ══════════════════════════════════════════════════════════
# CHECK 2: PANNs labels vs dataset folder names
# ══════════════════════════════════════════════════════════
header("2. PANNs DETECTOR LABELS → DATASET FOLDER ALIGNMENT")

# What PANNs searches for (from run_perception.py)
PANNS_LABELS = ["civil defense siren", "dog bark", "gunfire", "subway train"]
PANNS_NORMALIZED = [l.replace(" ", "_") for l in PANNS_LABELS]

print(f"\n  PANNs EVENT_LABELS:      {PANNS_LABELS}")
print(f"  Normalized (underscore): {PANNS_NORMALIZED}\n")

for label, normalized in zip(PANNS_LABELS, PANNS_NORMALIZED):
    if normalized in existing_folders:
        ok(f"PANNs label '{label}' → folder '{normalized}/' exists")
    else:
        warn(f"PANNs label '{label}' → no matching folder (not necessarily a bug, PANNs detects from audio)")


# ══════════════════════════════════════════════════════════
# CHECK 3: Mixer labels vs PANNs labels (THE KEY CHECK)
# ══════════════════════════════════════════════════════════
header("3. MIXER LABELS ↔ PANNs LABELS (F1 alignment)")

print(f"\n  Mixer writes event_type as:  {MIXER_EVENT_TYPES}")
print(f"  PANNs detects and outputs as: {PANNS_NORMALIZED}")
print(f"  Evaluator compares these two sets for Precision/Recall/F1\n")

mixer_set = set(MIXER_EVENT_TYPES)
panns_set = set(PANNS_NORMALIZED)

matching = mixer_set & panns_set
only_mixer = mixer_set - panns_set
only_panns = panns_set - mixer_set

for m in sorted(matching):
    ok(f"'{m}' exists in BOTH mixer and PANNs → F1 can match ✓")

for m in sorted(only_mixer):
    fail(f"'{m}' in mixer ground truth but NOT in PANNs labels → F1 will always miss this (false negative)")

for c in sorted(only_panns):
    warn(f"'{c}' in PANNs labels but NOT in mixer → if detected, counts as false positive in F1")


# ══════════════════════════════════════════════════════════
# CHECK 4: Scene metadata — did events actually get mixed?
# ══════════════════════════════════════════════════════════
header("4. SCENE METADATA — Events actually mixed in?")

META_DIR = BASE / "scenes" / "metadata"
if META_DIR.exists():
    scene_files = sorted(META_DIR.glob("*.json"))
    total_events = 0
    for sf in scene_files:
        with open(sf, "r", encoding="utf-8") as f:
            meta = json.load(f)
        events = [t for t in meta.get("tracks", []) if t.get("role") == "event"]
        total_events += len(events)
        if events:
            event_descs = [f"{e['event_type']} @ {e['start']}s ({e['gain_db']}dB)" for e in events]
            ok(f"{sf.name}: {len(events)} event(s) → {', '.join(event_descs)}")
        else:
            warn(f"{sf.name}: 0 events mixed in (speech-only scene)")

    if total_events == 0:
        fail(f"ALL {len(scene_files)} scenes have 0 events — nothing for PANNs to detect!")
else:
    fail("scenes/metadata/ directory not found — run generate_mixer_logs.py first")


# ══════════════════════════════════════════════════════════
# CHECK 5: Perception outputs — did PANNs detect anything?
# ══════════════════════════════════════════════════════════
header("5. PERCEPTION OUTPUTS — PANNs detections")

PERC_DIR = BASE / "perception"
if PERC_DIR.exists():
    perc_files = sorted(PERC_DIR.glob("*.json"))
    total_detections = 0
    for pf in perc_files:
        with open(pf, "r", encoding="utf-8") as f:
            perc = json.load(f)
        events = perc.get("events", [])
        total_detections += len(events)
        if events:
            descs = [f"{e['content']} ({e['confidence']:.3f})" for e in events]
            ok(f"{pf.name}: {len(events)} detection(s) → {', '.join(descs)}")
        else:
            warn(f"{pf.name}: 0 events detected by PANNs")

    if total_detections == 0:
        fail(f"PANNs detected 0 events across ALL {len(perc_files)} scenes")
else:
    fail("perception/ directory not found — run the pipeline first")


# ══════════════════════════════════════════════════════════
# CHECK 6: evaluate_asr.py — CER aggregation bug
# ══════════════════════════════════════════════════════════
header("6. EVALUATE_ASR.PY — Code checks")

eval_file = BASE / "pipeline" / "evaluate_asr.py"
if eval_file.exists():
    code = eval_file.read_text(encoding="utf-8")

    if "all_cer" in code and "all_cer.append" in code:
        ok("CER is being aggregated into a list")
    else:
        fail("CER computed per-file but never aggregated (no 'all_cer' list) — average CER is missing")

    if "all_wer" in code and "all_wer.append" in code:
        ok("WER is being aggregated ✓")

    if "all_f1" in code and "all_f1.append" in code:
        ok("Event F1 is being aggregated ✓")
else:
    fail("pipeline/evaluate_asr.py not found")


# ══════════════════════════════════════════════════════════
# CHECK 7: Speech dataset
# ══════════════════════════════════════════════════════════
header("7. SPEECH DATASET")

CLEAN_DIR = BASE / "clean"
LANGUAGES = ["marathi", "tamil", "telugu"]

for lang in LANGUAGES:
    lang_dir = CLEAN_DIR / lang
    if not lang_dir.exists():
        fail(f"clean/{lang}/ directory missing")
        continue
    wav_files = list(lang_dir.glob("*.wav"))
    tsv_file = lang_dir / "line_index.tsv"
    if len(wav_files) == 0:
        fail(f"clean/{lang}/ has 0 .wav files")
    else:
        ok(f"clean/{lang}/ → {len(wav_files)} .wav files")
    if tsv_file.exists():
        lines = len(tsv_file.read_text(encoding="utf-8").strip().split("\n"))
        ok(f"clean/{lang}/line_index.tsv → {lines} transcript entries")
    else:
        fail(f"clean/{lang}/line_index.tsv missing — no ground truth transcripts")


# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
header("SUMMARY")
print(f"""
  {GREEN}✔ {passed} passed{RESET}
  {RED}✘ {failed} failed{RESET}
  {YELLOW}⚠ {warnings} warnings{RESET}
""")

if failed > 0:
    print(f"  {RED}{BOLD}Pipeline has alignment issues. Fix the FAILs above before running.{RESET}")
else:
    print(f"  {GREEN}{BOLD}Everything is aligned! Pipeline should work correctly.{RESET}")
