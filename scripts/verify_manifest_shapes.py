import os
import cv2
import pandas as pd
from tqdm import tqdm
from src.config import Config

def main():
    """
    Scans the manifest.csv file to find image-mask pairs with mismatched dimensions.
    """
    manifest_path = os.path.join(Config.DATA_ROOT, 'manifest.csv')
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest file not found at: {manifest_path}")
        return

    print(f"Verifying shapes in {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    mismatched_files = []
    errors_found = 0
    max_errors_to_find = 10

    # We only need to check fake images, as real ones don't have masks.
    fake_df = df[df['label'] == 1].copy()
    
    # Add a check for NaN or empty mask paths
    fake_df.dropna(subset=['mask_path'], inplace=True)
    fake_df = fake_df[fake_df['mask_path'] != '']

    pbar = tqdm(fake_df.itertuples(), total=len(fake_df), desc="Checking fake samples")
    for row in pbar:
        img = cv2.imread(row.img_path)
        mask = cv2.imread(row.mask_path)

        if img is None:
            print(f"\n[WARNING] Could not read image: {row.img_path}")
            continue
        
        if mask is None:
            # This case shouldn't happen if the path exists, but is a good safeguard.
            continue

        if img.shape[:2] != mask.shape[:2]:
            errors_found += 1
            mismatch_info = {
                'image_path': row.img_path,
                'image_shape': img.shape,
                'mask_path': row.mask_path,
                'mask_shape': mask.shape,
                'split': row.split
            }
            mismatched_files.append(mismatch_info)
            pbar.set_description(f"Found {errors_found} mismatches...")
        
        if errors_found >= max_errors_to_find:
            break
            
    if not mismatched_files:
        print("\n✅ All checked image-mask pairs have matching dimensions!")
    else:
        print(f"\n❌ Found {len(mismatched_files)} mismatched image-mask pairs:")
        for info in mismatched_files:
            print("-" * 20)
            print(f"  Split: {info['split']}")
            print(f"  Image: {info['image_path']} (Shape: {info['image_shape']})")
            print(f"  Mask:  {info['mask_path']} (Shape: {info['mask_shape']})")

if __name__ == "__main__":
    main() 