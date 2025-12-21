#!/usr/bin/env python3
"""
複数のデータセットのtrainとvalidationを統合するプログラム
- 画像ファイルをそのままの名前で新しいフォルダに統合
- メタデータJSONファイルもそのまま統合
"""

import os
import json
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

def load_json(file_path):
    """JSONファイルを読み込み"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def save_json(data, file_path):
    """JSONファイルを保存"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {file_path} ({len(data)} entries)")

def merge_datasets(source_dirs, output_dir, dataset_name="merged_dataset"):
    """
    複数のデータセットを統合
    
    Args:
        source_dirs: ソースデータセットのディレクトリリスト
        output_dir: 出力先ディレクトリ
        dataset_name: 統合後のデータセット名
    """
    
    print(f"🔄 Starting dataset merger: {dataset_name}")
    print(f"Source datasets: {len(source_dirs)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリの作成
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "validation").mkdir(exist_ok=True)
    
    # 統合されたメタデータ
    merged_train_data = []
    merged_validation_data = []
    
    # 統計情報
    stats = {
        "total_datasets": len(source_dirs),
        "train_images": 0,
        "validation_images": 0,
        "errors": 0,
        "skipped": 0
    }
    
    # 各データセットを処理
    for i, source_dir in enumerate(source_dirs):
        source_path = Path(source_dir)
        
        print(f"📂 Processing dataset {i+1}/{len(source_dirs)}: {source_path.name}")
        
        # train と validation の処理
        for split in ["train", "validation"]:
            # メタデータファイルのパスを探す
            metadata_files = list(source_path.glob(f"{split}_metadata*.json"))
            if not metadata_files:
                print(f"  ⚠️  No {split} metadata file found")
                continue
            
            metadata_file = metadata_files[0]  # 最初に見つかったファイルを使用
            print(f"  📄 Loading {split} metadata: {metadata_file.name}")
            
            # メタデータの読み込み
            metadata = load_json(metadata_file)
            if not metadata:
                continue
            
            # 画像フォルダ
            images_dir = source_path / split
            if not images_dir.exists():
                print(f"  ⚠️  Images directory not found: {images_dir}")
                continue
            
            processed_count = 0
            
            # 画像ファイルをすべてコピー
            for image_file in images_dir.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    try:
                        # 同名ファイルがある場合はスキップ
                        output_image_path = output_path / split / image_file.name
                        if output_image_path.exists():
                            stats["skipped"] += 1
                            continue
                        
                        # ファイルをそのままコピー
                        shutil.copy2(image_file, output_image_path)
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"    ✗ Error copying {image_file.name}: {e}")
                        stats["errors"] += 1
            
            # メタデータをそのまま統合
            if split == "train":
                merged_train_data.extend(metadata)
                stats["train_images"] += processed_count
            else:
                merged_validation_data.extend(metadata)
                stats["validation_images"] += processed_count
            
            print(f"  ✓ {split}: {processed_count} images copied")
        
        print()
    
    # メタデータファイルの保存
    print("💾 Saving merged metadata files...")
    save_json(merged_train_data, output_path / "train_metadata.json")
    save_json(merged_validation_data, output_path / "validation_metadata.json")
    
    # 統計情報の保存
    stats_data = {
        "dataset_name": dataset_name,
        "source_datasets": [str(Path(d).name) for d in source_dirs],
        "statistics": stats,
        "output_structure": {
            "train_images": stats["train_images"],
            "validation_images": stats["validation_images"],
            "total_images": stats["train_images"] + stats["validation_images"]
        }
    }
    
    save_json(stats_data, output_path / "dataset_info.json")
    
    # 結果表示
    print("\n" + "="*60)
    print("🎉 DATASET MERGER COMPLETED!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"  Source datasets: {stats['total_datasets']}")
    print(f"  Train images: {stats['train_images']:,}")
    print(f"  Validation images: {stats['validation_images']:,}")
    print(f"  Total images: {stats['train_images'] + stats['validation_images']:,}")
    print(f"  Skipped (duplicates): {stats['skipped']:,}")
    print(f"  Errors: {stats['errors']:,}")
    print(f"\n📁 Output directory: {output_path}")
    print(f"  📄 train_metadata.json")
    print(f"  📄 validation_metadata.json")
    print(f"  📄 dataset_info.json")
    print(f"  📂 train/ ({stats['train_images']} images)")
    print(f"  📂 validation/ ({stats['validation_images']} images)")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple datasets into one")
    parser.add_argument(
        '--input', '-i', 
        nargs='+', 
        required=True,
        help='Input dataset directories'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--name', '-n',
        default="merged_dataset",
        help='Dataset name (default: merged_dataset)'
    )
    
    args = parser.parse_args()
    
    # 入力ディレクトリの確認
    print("🔍 Checking input directories...")
    valid_dirs = []
    
    for input_dir in args.input:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  ✗ Directory not found: {input_dir}")
            continue
        
        # train または validation フォルダの存在確認
        has_train = (input_path / "train").exists()
        has_validation = (input_path / "validation").exists()
        
        if not (has_train or has_validation):
            print(f"  ⚠️  No train/validation folders in: {input_dir}")
            continue
        
        valid_dirs.append(str(input_path))
        print(f"  ✓ {input_path.name}")
    
    if not valid_dirs:
        print("❌ No valid input directories found!")
        return
    
    print(f"\n📝 Configuration:")
    print(f"  Input datasets: {len(valid_dirs)}")
    print(f"  Output directory: {args.output}")
    print(f"  Dataset name: {args.name}")
    
    # 確認
    response = input("\nProceed with merging? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled by user")
        return
    
    # データセットの統合実行
    merge_datasets(valid_dirs, args.output, args.name)

if __name__ == "__main__":
    main()