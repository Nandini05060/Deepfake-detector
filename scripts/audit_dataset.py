import os
import json
import hashlib
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import yaml
from collections import Counter, defaultdict

def compute_md5(file_path: Path, block_size=65536) -> str:
    """Computes exact MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(image: Image.Image, hash_size=8) -> str:
    """
    Computes difference hash (perceptual hash) of an image.
    Difference hash works on gradients between adjacent pixels.
    """
    try:
        gray = ImageOps.grayscale(image)
        resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.int32)
        # Calculate difference between adjacent pixels
        diff = pixels[:, 1:] > pixels[:, :-1]
        # Flatten boolean array to hex string
        return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))[2:].zfill(hash_size * hash_size // 4)
    except Exception:
        return ""

def audit_dataset(dataset_dir: str, output_dir: str):
    dataset_path = Path(dataset_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[AUDIT] Starting dataset audit for: {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist!")
    
    # 1. Detect REAL and FAKE folders automatically
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"[AUDIT] Found subdirectories: {[d.name for d in subdirs]}")
    
    class_mapping = {}
    for d in subdirs:
        lname = d.name.lower()
        if 'real' in lname:
            class_mapping[d.name] = 'REAL'
        elif 'fake' in lname or 'deepfake' in lname or 'spoof' in lname:
            class_mapping[d.name] = 'FAKE'
        else:
            class_mapping[d.name] = d.name.upper()

    image_records = []
    md5_hash_map = defaultdict(list)
    dhash_map = defaultdict(list)
    
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    total_scanned = 0
    corrupted_files = []
    
    for folder, label in class_mapping.items():
        folder_path = dataset_path / folder
        print(f"[AUDIT] Scanning class '{label}' in {folder_path}...")
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext not in supported_exts:
                    continue
                
                total_scanned += 1
                rel_path = str(file_path.relative_to(dataset_path))
                
                file_size = file_path.stat().st_size
                md5_hash = compute_md5(file_path)
                md5_hash_map[md5_hash].append(rel_path)
                
                # Check image integrity & resolution
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Verify image integrity
                    
                    with Image.open(file_path) as img:
                        width, height = img.size
                        mode = img.mode
                        dhash = compute_dhash(img)
                        dhash_map[dhash].append(rel_path)
                        
                        record = {
                            "file_path": str(file_path),
                            "relative_path": rel_path,
                            "label": label,
                            "folder": folder,
                            "format": ext[1:],
                            "width": width,
                            "height": height,
                            "aspect_ratio": round(width / height, 3),
                            "file_size_bytes": file_size,
                            "md5": md5_hash,
                            "dhash": dhash,
                            "corrupted": False
                        }
                        image_records.append(record)
                except Exception as e:
                    corrupted_files.append({
                        "file_path": str(file_path),
                        "relative_path": rel_path,
                        "label": label,
                        "error": str(e)
                    })
                    image_records.append({
                        "file_path": str(file_path),
                        "relative_path": rel_path,
                        "label": label,
                        "folder": folder,
                        "format": ext[1:],
                        "width": None,
                        "height": None,
                        "aspect_ratio": None,
                        "file_size_bytes": file_size,
                        "md5": md5_hash,
                        "dhash": "",
                        "corrupted": True
                    })

    df = pd.DataFrame(image_records)
    
    # 2. Duplicate Detection
    exact_duplicates = {k: v for k, v in md5_hash_map.items() if len(v) > 1}
    num_exact_dup_files = sum(len(v) - 1 for v in exact_duplicates.values())
    
    perceptual_duplicates = {k: v for k, v in dhash_map.items() if len(v) > 1 and k != ""}
    num_perceptual_dup_files = sum(len(v) - 1 for v in perceptual_duplicates.values())
    
    # 3. Class Counts & Metrics
    label_counts = df['label'].value_counts().to_dict()
    valid_df = df[~df['corrupted']]
    
    resolution_counts = valid_df.groupby(['width', 'height']).size().reset_index(name='count').to_dict('records')
    format_counts = df['format'].value_counts().to_dict()
    
    audit_summary = {
        "dataset_root": str(dataset_path),
        "total_images": len(df),
        "total_valid_images": len(valid_df),
        "class_distribution": label_counts,
        "class_imbalance_ratio": {
            k: round(v / len(valid_df), 4) for k, v in label_counts.items()
        },
        "file_formats": format_counts,
        "corrupted_files_count": len(corrupted_files),
        "exact_duplicates_unique_hashes": len(exact_duplicates),
        "exact_duplicate_files_count": num_exact_dup_files,
        "perceptual_duplicates_unique_hashes": len(perceptual_duplicates),
        "perceptual_duplicate_files_count": num_perceptual_dup_files,
        "resolution_distribution": resolution_counts[:10]  # top resolutions
    }
    
    # 4. Save JSON Report
    json_path = out_path / "dataset_audit.json"
    with open(json_path, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"[AUDIT] Saved JSON audit report to {json_path}")
    
    # 5. Save CSV Report
    csv_path = out_path / "dataset_audit.csv"
    df.to_csv(csv_path, index=False)
    print(f"[AUDIT] Saved CSV file table to {csv_path}")
    
    # 6. Save Markdown Report
    md_path = out_path / "dataset_audit.md"
    with open(md_path, "w") as f:
        f.write("# LX-DFD Dataset Audit Report\n\n")
        f.write(f"**Dataset Path**: `{dataset_path}`\n\n")
        f.write("## Overview Statistics\n\n")
        f.write(f"- **Total Images**: `{audit_summary['total_images']}`\n")
        f.write(f"- **Valid Images**: `{audit_summary['total_valid_images']}`\n")
        f.write(f"- **Corrupted Files**: `{audit_summary['corrupted_files_count']}`\n")
        f.write(f"- **Exact Duplicate Files (MD5)**: `{audit_summary['exact_duplicate_files_count']}`\n")
        f.write(f"- **Perceptual Duplicate Files (dHash)**: `{audit_summary['perceptual_duplicate_files_count']}`\n\n")
        
        f.write("## Class Distribution\n\n")
        f.write("| Class | Image Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for cls, count in label_counts.items():
            pct = audit_summary['class_imbalance_ratio'].get(cls, 0) * 100
            f.write(f"| **{cls}** | {count} | {pct:.2f}% |\n")
        
        f.write("\n## File Formats\n\n")
        f.write("| Format | Count |\n")
        f.write("|---|---|\n")
        for fmt, cnt in format_counts.items():
            f.write(f"| `.{fmt}` | {cnt} |\n")
            
        f.write("\n## Primary Image Resolutions\n\n")
        f.write("| Width x Height | Image Count |\n")
        f.write("|---|---|\n")
        for res in resolution_counts[:5]:
            f.write(f"| {res['width']} x {res['height']} | {res['count']} |\n")
            
    print(f"[AUDIT] Saved Markdown summary to {md_path}")
    return audit_summary

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    audit_dataset(config['data']['dataset_dir'], config['data']['reports_audit_dir'])
