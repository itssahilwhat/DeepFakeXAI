# scripts/generate_manifest.py

import os
import csv
import random
from collections import defaultdict
from src.config import Config
from tqdm import tqdm


def find_all_samples(root_path, source_name):
    """Recursively finds all image samples and their corresponding masks."""
    samples = []
    if not os.path.exists(root_path):
        print(f"[WARNING] Path not found, skipping: {root_path}")
        return samples

    print(f"Scanning {source_name} in {root_path}...")
    for dirpath, _, filenames in tqdm(os.walk(root_path), desc=f"Scanning {source_name}"):

        # --- CRITICAL FIX: Skip any directory named 'masks' ---
        if 'masks' in dirpath.split(os.sep):
            continue
        # ----------------------------------------------------

        for fname in filenames:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dirpath, fname)
                label = 1 if 'fake' in dirpath.split(os.sep) else 0

                if f'{os.sep}valid{os.sep}' in img_path:
                    split = 'valid'
                elif f'{os.sep}test{os.sep}' in img_path:
                    split = 'test'
                else:
                    split = 'train'

                mask_path = ''
                if label == 1:
                    mask_candidate = img_path.replace(f'{os.sep}images{os.sep}', f'{os.sep}masks{os.sep}')
                    if os.path.exists(mask_candidate):
                        mask_path = mask_candidate

                samples.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'label': label,
                    'split': split,
                    'source': source_name
                })
    return samples


def main():
    """Generates a manifest CSV of all data without any balancing."""
    random.seed(Config.RANDOM_SEED)

    all_samples = []
    all_samples.extend(find_all_samples(Config.CELEBAHQ_PATH, 'celebahq'))
    all_samples.extend(find_all_samples(Config.FFHQ_PATH, 'ffhq'))

    out_csv_path = os.path.join(Config.DATA_ROOT, 'manifest.csv')
    print(f"\nWriting full manifest to {out_csv_path}...")

    with open(out_csv_path, 'w', newline='') as f:
        fieldnames = ['img_path', 'mask_path', 'label', 'split', 'source']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)

    print("\n--- Full Dataset Manifest Summary ---")
    final_counts = defaultdict(lambda: defaultdict(int))
    for sample in all_samples:
        label_name = 'Real' if sample['label'] == 0 else 'Fake'
        final_counts[sample['split']][label_name] += 1

    total_samples = 0
    for split_name in ['train', 'valid', 'test']:
        counts = final_counts[split_name]
        total = counts.get('Real', 0) + counts.get('Fake', 0)
        total_samples += total
        print(
            f"{split_name.title():<12}: Real: {counts.get('Real', 0):<7} | Fake: {counts.get('Fake', 0):<7} | Total: {total}")

    print(f"\nTotal samples in manifest: {total_samples}")
    print("--- Generation Complete ---")


if __name__ == "__main__":
    main()