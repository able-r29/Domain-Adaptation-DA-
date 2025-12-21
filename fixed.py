#!/usr/bin/env python3
import json
import os

# all_data_162.jsonを読み込み（リスト形式）
with open('all_data_162.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)

# filenameをキーとするインデックスを作成
filename_to_entry = {}
for entry in all_data:
    filename = entry['filename']
    filename_to_entry[filename] = entry

print(f"Loaded {len(all_data)} entries from all_data_162.json")

# デバッグ: all_data_162.jsonのfilenameサンプルを表示
print("Sample filenames from all_data_162.json:")
sample_filenames = list(filename_to_entry.keys())[:5]
for fname in sample_filenames:
    print(f"  '{fname}'")

# 各ファイルを修正
files = ['train_metadata', 'validation_metadata']
for file_name in files:
    input_file = f'{file_name}.json'
    output_file = f'{file_name}_fixed.json'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\nProcessing {input_file}...")
        print(f"Data type: {type(data)}")
        
        fixed_data = []
        found_count = 0
        missing_count = 0
        missing_files = []
        
        # データの形式を判定して処理
        if isinstance(data, dict):
            # 辞書形式の場合（キーが画像名）
            print(f"  Format: Dictionary with {len(data)} keys")
            for image_name in data.keys():
                if image_name in filename_to_entry:
                    # all_data_162.jsonから完全なエントリをコピー
                    entry = filename_to_entry[image_name].copy()
                    fixed_data.append(entry)
                    found_count += 1
                else:
                    print(f"  Warning: {image_name} not found in all_data_162.json")
                    missing_files.append(image_name)
                    missing_count += 1
                    
        elif isinstance(data, list):
            # リスト形式の場合
            print(f"  Format: List with {len(data)} items")
            
            # サンプルアイテムを表示
            if data:
                sample_item = data[0]
                if isinstance(sample_item, dict):
                    print(f"  Sample item keys: {list(sample_item.keys())[:10]}")
                    print(f"  Sample original_filename: '{sample_item.get('original_filename', 'NOT FOUND')}'")
                    print(f"  Sample filename: '{sample_item.get('filename', 'NOT FOUND')}'")
            
            for i, item in enumerate(data):
                image_name = None
                
                if isinstance(item, dict):
                    # 優先順位を変更: original_filenameを最優先に
                    if 'original_filename' in item and item['original_filename']:
                        image_name = item['original_filename']
                        print(f"  Using original_filename: '{image_name}' for item {i}")
                    elif 'filename' in item and item['filename']:
                        image_name = item['filename']
                        print(f"  Using filename: '{image_name}' for item {i}")
                    elif 'copied_filename' in item and item['copied_filename']:
                        image_name = item['copied_filename']
                        print(f"  Using copied_filename: '{image_name}' for item {i}")
                    else:
                        print(f"  Warning: No suitable filename found in item {i}")
                        missing_count += 1
                        continue
                        
                elif isinstance(item, str):
                    # 文字列の場合（ファイル名のみ）
                    image_name = item
                    print(f"  Using string: '{image_name}' for item {i}")
                else:
                    print(f"  Warning: Unknown item format at index {i}: {type(item)}")
                    missing_count += 1
                    continue
                
                # マッチング処理
                if image_name in filename_to_entry:
                    # all_data_162.jsonから完全なエントリをコピー
                    entry = filename_to_entry[image_name].copy()
                    fixed_data.append(entry)
                    found_count += 1
                    if i < 5:  # 最初の5件のみ詳細ログ
                        print(f"    ✅ Match found for: '{image_name}'")
                else:
                    # マッチしない場合の詳細ログ
                    print(f"    ❌ No match for: '{image_name}'")
                    missing_files.append(image_name)
                    missing_count += 1
                    
                    # ベースネーム（パスなし）でも試行
                    base_name = os.path.basename(image_name)
                    if base_name != image_name and base_name in filename_to_entry:
                        entry = filename_to_entry[base_name].copy()
                        fixed_data.append(entry)
                        found_count += 1
                        missing_count -= 1  # カウンタ修正
                        missing_files.pop()  # 最後に追加したファイルを削除
                        print(f"    ✅ Match found with basename: '{base_name}'")
        
        else:
            print(f"  Error: Unknown data format: {type(data)}")
            continue
        
        # 見つからなかったファイルのサンプルを表示
        if missing_files:
            print(f"\n  Missing files sample (first 5):")
            for missing_file in missing_files[:5]:
                print(f"    '{missing_file}'")
            
            # missing_files.txtに保存
            missing_file_path = f'{file_name}_missing.txt'
            with open(missing_file_path, 'w', encoding='utf-8') as f:
                for missing_file in missing_files:
                    f.write(f"{missing_file}\n")
            print(f"  Full missing list saved to: {missing_file_path}")
        
        # 結果を保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Created {output_file}: {len(fixed_data)} entries (found: {found_count}, missing: {missing_count})")
        
    except FileNotFoundError:
        print(f"  ⚠️ {input_file} not found, skipping...")
    except Exception as e:
        print(f"  ❌ Error processing {input_file}: {e}")
        import traceback
        traceback.print_exc()

print("\n🎉 All available files processed!")

# 統計情報を表示
print("\n📊 Final Statistics:")
for file_name in files:
    output_file = f'{file_name}_fixed.json'
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            fixed_data = json.load(f)
        
        if not fixed_data:
            print(f"\n  📄 {output_file}: EMPTY ❌")
            continue
        
        # 簡単な統計
        class_counts = {}
        part_counts = {}
        age_counts = {'under_30': 0, 'over_30': 0, 'unknown': 0}
        facility_counts = {}
        
        for entry in fixed_data:
            # クラス統計
            label = entry.get('LABEL', 'unknown')
            class_counts[label] = class_counts.get(label, 0) + 1
            
            # 部位統計  
            part = entry.get('part', ['unknown'])
            if isinstance(part, list):
                part = part[0] if part else 'unknown'
            part_counts[part] = part_counts.get(part, 0) + 1
            
            # 施設統計
            facility = entry.get('univ_ID', 'unknown')
            facility_counts[facility] = facility_counts.get(facility, 0) + 1
            
            # 年齢統計
            age = entry.get('age', [0])
            if isinstance(age, list):
                age_val = age[0] if age else 0
            else:
                age_val = age
            
            try:
                age_num = int(float(age_val)) if age_val != 0 else 0
                if age_num == 0 or age_num > 200:  # 異常な年齢値
                    age_counts['unknown'] += 1
                elif age_num < 30:
                    age_counts['under_30'] += 1
                else:
                    age_counts['over_30'] += 1
            except:
                age_counts['unknown'] += 1
        
        print(f"\n  📄 {output_file}:")
        print(f"    Total: {len(fixed_data)} entries ✅")
        print(f"    Classes: {dict(class_counts)}")
        print(f"    Parts: {dict(part_counts)}")
        print(f"    Facilities: {dict(facility_counts)}")
        print(f"    Ages: {dict(age_counts)}")
        
    except FileNotFoundError:
        print(f"    ⚠️ {output_file} not found")
    except Exception as e:
        print(f"    ❌ Error reading {output_file}: {e}")