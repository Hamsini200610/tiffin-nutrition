import os
import json
from collections import defaultdict
from detector import analyze_food_image

DATASET_DIR = os.path.join('static', 'uploads', 'tuning', 'Indian Food Dataset')
REPORT_PATH = os.path.join('static', 'uploads', 'tuning', 'tuning_report_folders.json')


def normalize(x):
    return x.strip().lower()


def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset dir {DATASET_DIR} not found")
        return
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    stats = defaultdict(lambda: {'tp':0,'fp':0,'fn':0,'pred_count':0,'gt_count':0, 'examples': []})
    total_files = 0
    per_image = {}
    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for im in imgs:
            total_files += 1
            path = os.path.join(cls_dir, im)
            gt = [normalize(cls)]
            stats[normalize(cls)]['gt_count'] += 1
            print(f"Processing {path} ...")
            res = analyze_food_image(path)
            preds = []
            if 'items' in res:
                preds = [normalize(x) for x in res['items']]
            # record
            per_image[path] = {'ground_truth': gt, 'predictions': preds, 'result': res}
            for p in preds:
                stats[p]['pred_count'] += 1
            matched = set()
            for p in preds:
                if p in gt and p not in matched:
                    stats[p]['tp'] += 1
                    matched.add(p)
                else:
                    stats[p]['fp'] += 1
            for g in gt:
                if g not in matched:
                    stats[g]['fn'] += 1
            # keep an example
            for p in preds:
                if len(stats[p]['examples']) < 5:
                    stats[p]['examples'].append(path)
            if len(preds) == 0:
                if len(stats[normalize(cls)]['examples']) < 5:
                    stats[normalize(cls)]['examples'].append(path)
    # compute metrics
    metrics = {}
    for label, s in stats.items():
        tp = s['tp']
        fp = s['fp']
        fn = s['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[label] = {'tp':tp,'fp':fp,'fn':fn,'precision':prec,'recall':rec,'f1':f1,'pred_count':s['pred_count'],'gt_count':s['gt_count'],'examples':s['examples']}
    out = {'total_images': total_files, 'metrics': metrics, 'per_image': per_image}
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Done. Report written to {REPORT_PATH}")

if __name__ == '__main__':
    main()
