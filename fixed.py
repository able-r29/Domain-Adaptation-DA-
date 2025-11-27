#!/usr/bin/env python3
import json

# all_data_162.jsonを読み込み（リスト形式）
with open('all_data_162.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)

# filenameをキーとするインデックスを作成
filename_to_entry = {}
for entry in all_data:
    filename = entry['filename']
    filename_to_entry[filename] = entry

print(f"Loaded {len(all_data)} entries from all_data_162.json")

# 各ファイルを修正
files = ['test1', 'test2']
for file_name in files:
    input_file = f'{file_name}_metadata.json'
    output_file = f'{file_name}_metadata_standardized.json'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 辞書形式から正しいリスト形式に変換
    fixed_data = []
    found_count = 0
    missing_count = 0
    
    for image_name in data.keys():
        if image_name in filename_to_entry:
            # all_data_162.jsonから完全なエントリをコピー
            entry = filename_to_entry[image_name].copy()
            fixed_data.append(entry)
            found_count += 1
        else:
            print(f"Warning: {image_name} not found")
            missing_count += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {output_file} with {len(fixed_data)} entries (found: {found_count}, missing: {missing_count})")

print("All files standardized!")