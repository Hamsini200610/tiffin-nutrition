import os
import json
from collections import defaultdict
from detector import analyze_food_image

TUNING_DIR = os.path.join('static', 'uploads', 'tuning')
# prefer groundtruth/labels.json when present (created by label_from_folders.py)
LABELS_FILE = os.path.join(TUNING_DIR, 'labels.json')
GT_LABELS_FILE = os.path.join(TUNING_DIR, 'groundtruth', 'labels.json')
if os.path.exists(GT_LABELS_FILE):
    LABELS_FILE = GT_LABELS_FILE

def load_labels(path):
    with open(path, 'r') as f:
        return json.load(f)

def normalize(name):
    return name.strip().lower()

def evaluate(labels_map):
    stats = defaultdict(lambda: {'tp':0,'fp':0,'fn':0,'pred_count':0,'gt_count':0})
    files = list(labels_map.keys())
    total_files = 0
    for fname in files:
        # labels file may contain paths relative to its own directory (e.g., groundtruth/..)
        labels_dir = os.path.dirname(LABELS_FILE)
        img_path = os.path.join(labels_dir, fname)
        if not os.path.exists(img_path):
            img_path = os.path.join(TUNING_DIR, fname)
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path} - skipping")
            continue
        total_files += 1
        gt = [normalize(x) for x in labels_map[fname]]
        for g in gt:
            stats[g]['gt_count'] += 1
        res = analyze_food_image(img_path)
        if 'items' not in res:
            print(f"No items detected for {fname}: {res.get('error')}")
            preds = []
        else:
            preds = [normalize(x) for x in res['items']]
        # count preds
        for p in preds:
            stats[p]['pred_count'] += 1
        # mark tp
        matched = set()
        for p in preds:
            if p in gt and p not in matched:
                stats[p]['tp'] += 1
                matched.add(p)
            else:
                stats[p]['fp'] += 1
        # mark fn for any gt not matched
        for g in gt:
            if g not in matched:
                stats[g]['fn'] += 1
    return stats, total_files


def compute_metrics(stats):
    metrics = {}
    for label, s in stats.items():
        tp = s['tp']
        fp = s['fp']
        fn = s['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[label] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(prec, 3), 'recall': round(rec, 3), 'f1': round(f1, 3),
            'pred_count': s['pred_count'], 'gt_count': s['gt_count']
        }
    return metrics


def suggest_changes(metrics, precision_threshold=0.6, recall_threshold=0.6):
    suggestions = {}
    for label, m in metrics.items():
        sug = []
        if m['precision'] < precision_threshold:
            sug.append('precision_low: consider lowering boosts or increasing penalty for false-positives')
        if m['recall'] < recall_threshold:
            sug.append('recall_low: consider increasing score boost or relaxing floor for this label')
        if not sug:
            sug.append('ok')
        suggestions[label] = sug
    return suggestions


if __name__ == '__main__':
    if not os.path.exists(TUNING_DIR):
        print(f"Tuning directory {TUNING_DIR} does not exist. Create it and add images + labels.json")
        exit(1)
    if not os.path.exists(LABELS_FILE):
        print(f"Labels file not found at {LABELS_FILE}. Create labels.json mapping filenames to lists of labels.")
        exit(1)
    labels_map = load_labels(LABELS_FILE)
    stats, total = evaluate(labels_map)
    metrics = compute_metrics(stats)
    suggestions = suggest_changes(metrics)
    out = {
        'total_files': total,
        'metrics': metrics,
        'suggestions': suggestions
    }
    with open(os.path.join(TUNING_DIR, 'tuning_report.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Tuning report written to {os.path.join(TUNING_DIR, 'tuning_report.json')}")
    print("Summary:")
    for label, m in metrics.items():
        print(f"{label}: prec={m['precision']} rec={m['recall']} f1={m['f1']} (gt={m['gt_count']} pred={m['pred_count']})")
    print("Suggestions:")
    for label, s in suggestions.items():
        print(f"{label}: {s}")
