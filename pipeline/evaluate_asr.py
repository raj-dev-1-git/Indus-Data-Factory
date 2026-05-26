import json
from pathlib import Path
import re
from whisper.normalizers import BasicTextNormalizer

BASE = Path(__file__).resolve().parent.parent

PERCEPTION = BASE / "perception"

METADATA = BASE / "scenes" / "metadata"


def word_error_rate(reference, hypothesis):
    # Use Whisper's built-in normalizer
    normalizer = BasicTextNormalizer()
    reference = normalizer(reference)
    hypothesis = normalizer(hypothesis)
    
    # Remove digits
    reference = re.sub(r'\d+(\.\d+)?', '', reference)
    hypothesis = re.sub(r'\d+(\.\d+)?', '', hypothesis)
    
    # Remove common Marathi number words to normalize digits-vs-words discrepancy
    num_words = ["एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ", "दहा", "हजार", "शे", "शंभर", "पन्नास", "पूर्णांक", "टक्के", "टक्क्यांनी"]
    for word in num_words:
        reference = re.sub(fr'(?<!\S){word}(?!\S)', '', reference)
        hypothesis = re.sub(fr'(?<!\S){word}(?!\S)', '', hypothesis)

    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # DP TABLE

    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):

        dp[i][0] = i

    for j in range(len(hyp_words) + 1):

        dp[0][j] = j

        # LEVENSHTEIN

    for i in range(1, len(ref_words) + 1):

        for j in range(1, len(hyp_words) + 1):

            if ref_words[i - 1] == hyp_words[j - 1]:

                cost = 0

            else:

                cost = 1

            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    edits = dp[len(ref_words)][len(hyp_words)]

    if len(ref_words) == 0:
        return 0.0

    return edits / len(ref_words)

def character_error_rate(reference, hypothesis):
    """
    Calculates Character Error Rate (CER), which is much better for languages 
    like Tamil and Marathi that have heavy agglutination and spacing differences.
    """
    import re
    # Remove all spaces and punctuation for a pure character match
    ref_chars = list(re.sub(r'[\s\W_]+', '', reference.lower()))
    hyp_chars = list(re.sub(r'[\s\W_]+', '', hypothesis.lower()))
    
    dp = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        dp[i][0] = i
    for j in range(len(hyp_chars) + 1):
        dp[0][j] = j
        
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    edits = dp[len(ref_chars)][len(hyp_chars)]
    if len(ref_chars) == 0:
        return 0.0
    return edits / len(ref_chars)


def calculate_event_metrics(true_events, pred_events):
    # If there are no ground truth events and no predicted events, it's a perfect match
    if not true_events and not pred_events:
        return 1.0, 1.0, 1.0
        
    true_set = set(true_events)
    pred_set = set(pred_events)
    
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


perception_files = sorted(PERCEPTION.glob("*.json"))

all_wer = []

all_f1 = []

for perception_file in perception_files:

    # LOAD PERCEPTION

    with open(perception_file, "r", encoding="utf-8") as f:

        perception_data = json.load(f)

    predicted = perception_data.get("full_transcript", "").strip()

    # LOAD GROUND TRUTH

    metadata_file = METADATA / perception_file.name

    with open(metadata_file, "r", encoding="utf-8") as f:

        metadata = json.load(f)

    ground_truth = ""

    for track in metadata["tracks"]:

        if track["role"] == "speech":

            ground_truth = track.get("transcript", "").strip()

            break

        # COMPUTE METRICS

    wer = word_error_rate(ground_truth, predicted)
    
    pred_events = [e.get("canonical_event", "") for e in perception_data.get("events", []) if "sound_event" in e.get("type", "")]
    # Fallback to the old method if canonical_event is missing (for older json results)
    if not any(pred_events):
        pred_events = [e.get("content", "").lower().replace(" ", "_") for e in perception_data.get("events", []) if "sound_event" in e.get("type", "")]
    
    true_events = [track.get("event_type", "").lower() for track in metadata.get("tracks", []) if track.get("role") == "event"]
    
    precision, recall, f1 = calculate_event_metrics(true_events, pred_events)

    all_wer.append(wer)
    
    # Calculate CER
    cer = character_error_rate(ground_truth, predicted)
    
    all_f1.append(f1)

    print(f"{perception_file.name}")

    print(f"WER: {wer:.3f} | CER: {cer:.3f} | F1: {f1:.3f}")

    safe_gt = ground_truth.encode('ascii', 'backslashreplace').decode('ascii')
    safe_pred = predicted.encode('ascii', 'backslashreplace').decode('ascii')
    print(f"GT : {safe_gt}")
    print(f"PRED: {safe_pred}")
    print("-" * 40)

    

if all_wer:

    avg_wer = sum(all_wer) / len(all_wer)
    
    # Optional: also compute average CER if you're saving it to a list
    # But for now we just show it per file to see the difference.
    
    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0

    print(f"\nAverage WER: {avg_wer:.3f}")
    
    print(f"Average Event F1: {avg_f1:.3f}")

else:

    print("No files found.")
