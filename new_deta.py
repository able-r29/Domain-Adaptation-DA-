import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

class JSONBasedStratifiedDatasetCreator:
    """JSONファイルベース層化データセット作成クラス"""
    
    def __init__(self, json_path, image_dir, output_path, train_ratio=0.6, val_ratio=0.1, test1_ratio=0.15, test2_ratio=0.15, age_threshold=30):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.output_path = Path(output_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test1_ratio = test1_ratio
        self.test2_ratio = test2_ratio
        self.age_threshold = age_threshold
        
        # 比率の合計が1.0になるかチェック
        total_ratio = train_ratio + val_ratio + test1_ratio + test2_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比率の合計が1.0ではありません: {total_ratio}")
        
        # 出力ディレクトリ作成
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 JSONファイル: {self.json_path}")
        print(f"📁 画像ディレクトリ: {self.image_dir}")
        print(f"📁 出力ディレクトリ: {self.output_path}")
        print(f"🎯 年齢閾値: {self.age_threshold}歳")
        print(f"📊 分割比率: Train={train_ratio:.1f} Val={val_ratio:.1f} Test1={test1_ratio:.1f} Test2={test2_ratio:.1f}")
    
    def load_json_data(self):
        """JSONファイルからメタデータを読み込み"""
        print(f"\n📋 JSONファイル読み込み: {self.json_path}")
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            print(f"✅ JSONデータ読み込み成功: {len(json_data)}件")
            return json_data
            
        except FileNotFoundError:
            print(f"❌ JSONファイルが見つかりません: {self.json_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析エラー: {e}")
            raise
        except Exception as e:
            print(f"❌ JSONファイル読み込みエラー: {e}")
            raise
    
    def normalize_class_label(self, label_data):
        """クラスラベルを正規化（0: 良性, 1: 悪性）"""
        if isinstance(label_data, list):
            label = label_data[0] if label_data else None
        else:
            label = label_data
        
        # 様々な形式のクラスラベルを0/1に変換
        if label in [0, '0', 'benign', 'nevus', 'naevus']:
            return 0  # 良性（母斑）
        elif label in [1, '1', 31, '31', 'malignant', 'melanoma']:
            return 1  # 悪性（メラノーマ）
        else:
            print(f"⚠️ 未知のクラスラベル: {label}")
            return None
    
    def normalize_age(self, age_data):
        """年齢を正規化"""
        if isinstance(age_data, list):
            age = age_data[0] if age_data else None
        else:
            age = age_data
        
        try:
            age_num = int(float(age)) if age is not None else None
            return age_num
        except (ValueError, TypeError):
            print(f"⚠️ 年齢変換エラー: {age}")
            return None
    
    def normalize_body_part(self, part_data):
        """疾患部位を正規化"""
        if isinstance(part_data, list):
            part = part_data[0] if part_data else None
        else:
            part = part_data
        
        if not part:
            return 'Unknown'
        
        # 部位名の正規化
        part_str = str(part).lower().strip()
        
        # 部位マッピング
        part_mapping = {
            'leg': 'Leg',
            'legs': 'Leg',
            'lower_limb': 'Leg',
            'lower limb': 'Leg',
            'trunk': 'Trunk',
            'torso': 'Trunk',
            'body': 'Trunk',
            'chest': 'Trunk',
            'back': 'Trunk',
            'upperarm': 'Upper_Arm',
            'upper_arm': 'Upper_Arm',
            'upper arm': 'Upper_Arm',
            'arm': 'Upper_Arm',
            'arms': 'Upper_Arm',
            'upper_limb': 'Upper_Arm',
            'upper limb': 'Upper_Arm'
        }
        
        normalized_part = part_mapping.get(part_str, part)
        return normalized_part
    
    def get_facility_id(self, item):
        """施設IDを取得"""
        # 複数のフィールドから施設IDを取得
        for key in ['univ_ID', 'facility', 'institution', 'site']:
            if key in item:
                facility = item[key]
                if isinstance(facility, list):
                    return facility[0] if facility else 'Unknown'
                return str(facility) if facility else 'Unknown'
        return 'Unknown'
    
    def process_json_data(self, json_data):
        """JSONデータを処理して画像データリストを作成"""
        print(f"\n🔄 JSONデータ処理開始")
        
        processed_data = []
        skipped_count = 0
        missing_files = []
        
        for i, item in enumerate(json_data):
            # ファイル名取得
            filename = item.get('filename')
            if not filename:
                # filenameがない場合、jpg_srcから取得を試す
                jpg_src = item.get('jpg_src', '')
                if jpg_src:
                    filename = os.path.basename(jpg_src)
                else:
                    print(f"⚠️ ファイル名が見つかりません: item {i}")
                    skipped_count += 1
                    continue
            
            # 画像ファイルの存在確認
            image_path = self.image_dir / filename
            if not image_path.exists():
                missing_files.append(filename)
                continue
            
            # クラスラベル正規化
            class_label = None
            for key in ['LABEL', 'class', 'label', 'class_label']:
                if key in item:
                    class_label = self.normalize_class_label(item[key])
                    break
            
            if class_label is None:
                print(f"⚠️ クラスラベルが見つかりません: {filename}")
                skipped_count += 1
                continue
            
            # 年齢正規化
            age = self.normalize_age(item.get('age'))
            if age is None:
                print(f"⚠️ 年齢情報が見つかりません: {filename}")
                skipped_count += 1
                continue
            
            # 年齢層分類
            age_group = 'over_30' if age >= self.age_threshold else 'under_30'
            
            # 疾患部位正規化
            body_part = self.normalize_body_part(item.get('part'))
            if body_part == 'Unknown':
                print(f"⚠️ 疾患部位が見つかりません: {filename}")
                skipped_count += 1
                continue
            
            # 施設ID取得
            facility = self.get_facility_id(item)
            
            # 処理済みデータに追加
            processed_item = {
                'filename': filename,
                'image_path': str(image_path),
                'class_label': class_label,
                'age': age,
                'age_group': age_group,
                'body_part': body_part,
                'facility': facility,
                'original_item': item  # 元のJSONアイテムを保持
            }
            processed_data.append(processed_item)
            
            # 進捗表示
            if (i + 1) % 1000 == 0:
                print(f"   処理済み: {i + 1}/{len(json_data)} 件")
        
        print(f"\n📊 JSON処理結果:")
        print(f"   総アイテム数: {len(json_data)}")
        print(f"   処理成功: {len(processed_data)}")
        print(f"   スキップ: {skipped_count}")
        print(f"   画像ファイル欠損: {len(missing_files)}")
        
        if missing_files:
            # 欠損ファイルリストを保存
            missing_log = self.output_path / "missing_files.txt"
            with open(missing_log, 'w', encoding='utf-8') as f:
                for missing_file in missing_files:
                    f.write(f"{missing_file}\n")
            print(f"   📄 欠損ファイルログ: {missing_log}")
        
        return processed_data
    
    def analyze_data_distribution(self, processed_data):
        """データ分布を分析"""
        print(f"\n📈 データ分布分析")
        
        # 各カテゴリ別の集計
        class_counts = Counter()
        age_group_counts = Counter()
        body_part_counts = Counter()
        facility_counts = Counter()
        combination_counts = Counter()
        
        for item in processed_data:
            class_counts[item['class_label']] += 1
            age_group_counts[item['age_group']] += 1
            body_part_counts[item['body_part']] += 1
            facility_counts[item['facility']] += 1
            
            # 3要素の組み合わせ
            combo = (item['class_label'], item['age_group'], item['body_part'])
            combination_counts[combo] += 1
        
        # 結果表示
        print(f"\nクラス分布:")
        for class_label, count in sorted(class_counts.items()):
            class_name = '良性（母斑）' if class_label == 0 else '悪性（メラノーマ）'
            percentage = (count / len(processed_data)) * 100
            print(f"   Class {class_label} ({class_name}): {count}件 ({percentage:.1f}%)")
        
        print(f"\n年齢層分布:")
        for age_group, count in sorted(age_group_counts.items()):
            age_desc = f'{self.age_threshold}歳未満' if age_group == 'under_30' else f'{self.age_threshold}歳以上'
            percentage = (count / len(processed_data)) * 100
            print(f"   {age_group} ({age_desc}): {count}件 ({percentage:.1f}%)")
        
        print(f"\n疾患部位分布:")
        for body_part, count in sorted(body_part_counts.items()):
            percentage = (count / len(processed_data)) * 100
            print(f"   {body_part}: {count}件 ({percentage:.1f}%)")
        
        print(f"\n施設分布:")
        for facility, count in sorted(facility_counts.items()):
            percentage = (count / len(processed_data)) * 100
            print(f"   {facility}: {count}件 ({percentage:.1f}%)")
        
        print(f"\nクラス×年齢層×疾患部位の組み合わせ:")
        for combo, count in sorted(combination_counts.items()):
            class_label, age_group, body_part = combo
            class_name = '良性' if class_label == 0 else '悪性'
            age_desc = f'{self.age_threshold}歳未満' if age_group == 'under_30' else f'{self.age_threshold}歳以上'
            print(f"   {class_name} × {age_desc} × {body_part}: {count}件")
        
        return {
            'class_counts': class_counts,
            'age_group_counts': age_group_counts,
            'body_part_counts': body_part_counts,
            'facility_counts': facility_counts,
            'combination_counts': combination_counts
        }
    
    def stratified_split_4way(self, processed_data):
        """クラス・年齢層・疾患部位を考慮した4分割層化分割"""
        print(f"\n📂 4分割層化データセット分割")
        print(f"🎯 Train:{self.train_ratio:.1f} Val:{self.val_ratio:.1f} Test1:{self.test1_ratio:.1f} Test2:{self.test2_ratio:.1f}")
        print("🎯 各分割でクラス・年齢層・疾患部位の割合を保持")
        
        # データを組み合わせ別にグループ化
        grouped_data = defaultdict(list)
        for data in processed_data:
            key = (data['class_label'], data['age_group'], data['body_part'])
            grouped_data[key].append(data)
        
        print(f"\n📊 組み合わせ別データ数:")
        for key, data_list in grouped_data.items():
            class_label, age_group, body_part = key
            class_name = '良性' if class_label == 0 else '悪性'
            age_desc = f'{self.age_threshold}歳未満' if age_group == 'under_30' else f'{self.age_threshold}歳以上'
            print(f"   {class_name} × {age_desc} × {body_part}: {len(data_list)}件")
        
        # 各組み合わせを4分割
        splits = {
            'train': [],
            'validation': [], 
            'test1': [],
            'test2': []
        }
        
        split_counts = {
            'train': {'class': Counter(), 'age_group': Counter(), 'bodypart': Counter()},
            'validation': {'class': Counter(), 'age_group': Counter(), 'bodypart': Counter()},
            'test1': {'class': Counter(), 'age_group': Counter(), 'bodypart': Counter()},
            'test2': {'class': Counter(), 'age_group': Counter(), 'bodypart': Counter()}
        }
        
        for key, data_list in grouped_data.items():
            if not data_list:
                continue
            
            class_label, age_group, body_part = key
            
            # シャッフル
            random.seed(42)
            shuffled_data = data_list.copy()
            random.shuffle(shuffled_data)
            
            # 分割点計算
            total_count = len(shuffled_data)
            train_count = int(total_count * self.train_ratio)
            val_count = int(total_count * self.val_ratio)
            
            # 残りをtest1とtest2で分割（均等化）
            remaining_count = total_count - train_count - val_count
            test1_count = remaining_count // 2
            test2_count = remaining_count - test1_count
            
            # 分割実行
            train_data = shuffled_data[:train_count]
            val_data = shuffled_data[train_count:train_count + val_count]
            test1_data = shuffled_data[train_count + val_count:train_count + val_count + test1_count]
            test2_data = shuffled_data[train_count + val_count + test1_count:]
            
            # 結果に追加
            for data in train_data:
                splits['train'].append(data)
                split_counts['train']['class'][data['class_label']] += 1
                split_counts['train']['age_group'][data['age_group']] += 1
                split_counts['train']['bodypart'][data['body_part']] += 1
            
            for data in val_data:
                splits['validation'].append(data)
                split_counts['validation']['class'][data['class_label']] += 1
                split_counts['validation']['age_group'][data['age_group']] += 1
                split_counts['validation']['bodypart'][data['body_part']] += 1
            
            for data in test1_data:
                splits['test1'].append(data)
                split_counts['test1']['class'][data['class_label']] += 1
                split_counts['test1']['age_group'][data['age_group']] += 1
                split_counts['test1']['bodypart'][data['body_part']] += 1
            
            for data in test2_data:
                splits['test2'].append(data)
                split_counts['test2']['class'][data['class_label']] += 1
                split_counts['test2']['age_group'][data['age_group']] += 1
                split_counts['test2']['bodypart'][data['body_part']] += 1
            
            class_name = '良性' if class_label == 0 else '悪性'
            age_desc = f'{self.age_threshold}歳未満' if age_group == 'under_30' else f'{self.age_threshold}歳以上'
            print(f"   {class_name}-{age_desc}-{body_part}: Train={len(train_data)} Val={len(val_data)} Test1={len(test1_data)} Test2={len(test2_data)}")
        
        # test1とtest2の枚数差を確認
        test1_total = len(splits['test1'])
        test2_total = len(splits['test2'])
        test_diff = abs(test1_total - test2_total)
        
        print(f"\n🎯 Test分割均等化結果:")
        print(f"   Test1: {test1_total}件")
        print(f"   Test2: {test2_total}件")
        print(f"   差分: {test_diff}件")
        
        # 各分割の統計表示
        for split_name in ['train', 'validation', 'test1', 'test2']:
            total_images = len(splits[split_name])
            print(f"\n📊 {split_name.upper()}: {total_images}件")
            
            # クラス別統計
            print(f"    クラス別:")
            for class_label, count in sorted(split_counts[split_name]['class'].items()):
                percentage = (count / total_images) * 100 if total_images > 0 else 0
                class_name = '良性' if class_label == 0 else '悪性'
                print(f"      Class {class_label} ({class_name}): {count}件 ({percentage:.1f}%)")
            
            # 年齢層別統計
            print(f"    年齢層別:")
            for age_group, count in sorted(split_counts[split_name]['age_group'].items()):
                percentage = (count / total_images) * 100 if total_images > 0 else 0
                age_desc = f'{self.age_threshold}歳未満' if age_group == 'under_30' else f'{self.age_threshold}歳以上'
                print(f"      {age_group} ({age_desc}): {count}件 ({percentage:.1f}%)")
            
            # 疾患部位別統計
            print(f"    疾患部位別:")
            for body_part, count in sorted(split_counts[split_name]['bodypart'].items()):
                percentage = (count / total_images) * 100 if total_images > 0 else 0
                print(f"      {body_part}: {count}件 ({percentage:.1f}%)")
        
        return splits, split_counts
    
    def copy_images_to_splits(self, splits):
        """各分割に画像ファイルをコピー"""
        print(f"\n📁 画像ファイルのコピー開始")
        
        copy_summary = {
            'train': 0,
            'validation': 0,
            'test1': 0,
            'test2': 0
        }
        
        for split_name, data_list in splits.items():
            print(f"\n🔄 {split_name.upper()} 分割への画像コピー: {len(data_list)}件")
            
            # 出力ディレクトリ作成
            split_dir = self.output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            successful_copies = 0
            failed_copies = []
            
            for i, data in enumerate(data_list):
                source_path = Path(data['image_path'])
                
                # ファイルが存在するかチェック
                if not source_path.exists():
                    print(f"⚠️ ファイルが見つかりません: {source_path}")
                    failed_copies.append(str(source_path))
                    continue
                
                # 出力ファイル名を決定（重複回避）
                dest_path = split_dir / data['filename']
                
                # ファイル名重複の場合は連番を追加
                counter = 1
                while dest_path.exists():
                    stem = Path(data['filename']).stem
                    suffix = Path(data['filename']).suffix
                    new_name = f"{stem}_{counter:03d}{suffix}"
                    dest_path = split_dir / new_name
                    data['copied_filename'] = new_name  # コピー後のファイル名を記録
                    counter += 1
                
                if 'copied_filename' not in data:
                    data['copied_filename'] = data['filename']
                
                # ファイルコピー実行
                try:
                    shutil.copy2(source_path, dest_path)
                    successful_copies += 1
                    
                    # 進捗表示（100枚ごと）
                    if (i + 1) % 100 == 0:
                        print(f"   進捗: {i + 1}/{len(data_list)} 枚完了")
                        
                except Exception as e:
                    print(f"❌ コピーエラー: {source_path} → {dest_path}: {e}")
                    failed_copies.append(f"{source_path} (エラー: {e})")
            
            copy_summary[split_name] = successful_copies
            
            print(f"   ✅ {split_name.upper()}: {successful_copies}/{len(data_list)} 枚コピー完了")
            
            if failed_copies:
                print(f"   ⚠️ 失敗: {len(failed_copies)} 枚")
                # 失敗したファイルのリストを保存
                failed_log_path = self.output_path / f"{split_name}_failed_copies.txt"
                with open(failed_log_path, 'w', encoding='utf-8') as f:
                    for failed_file in failed_copies:
                        f.write(f"{failed_file}\n")
                print(f"   📄 失敗ログ保存: {failed_log_path}")
        
        return copy_summary
    
    def create_metadata_files(self, splits):
        """各分割のメタデータファイルを作成"""
        print(f"\n📄 メタデータファイル作成")
        
        for split_name, data_list in splits.items():
            metadata_list = []
            
            for data in data_list:
                # メタデータ作成
                metadata_item = {
                    'filename': data.get('copied_filename', data['filename']),
                    'original_filename': data['filename'],
                    'class_label': data['class_label'],
                    'age': data['age'],
                    'age_group': data['age_group'],
                    'body_part': data['body_part'],
                    'facility': data['facility'],
                    'image_path': data['image_path'],
                    'LABEL': data['class_label'],  # 既存フォーマットとの互換性
                    'part': [data['body_part']],   # リスト形式での互換性
                    'age': [data['age']],          # リスト形式での互換性
                    'univ_ID': data['facility']    # 既存フォーマットとの互換性
                }
                
                # 元のJSONアイテムから他の情報もコピー
                original_item = data.get('original_item', {})
                for key, value in original_item.items():
                    if key not in metadata_item:
                        metadata_item[key] = value
                
                metadata_list.append(metadata_item)
            
            # JSONファイル保存
            metadata_file = self.output_path / f"{split_name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ {split_name}_metadata.json: {len(metadata_list)} 件")
        
        print(f"📄 全メタデータファイル作成完了")
    
    def create_dataset_info(self, split_counts, data_stats, copy_summary):
        """データセット情報を詳細に記録"""
        # data_statsのcombination_countsのタプルキーを文字列に変換
        processed_data_stats = {}
        for key, value in data_stats.items():
            if key == 'combination_counts':
                # タプルキーを文字列に変換
                string_keyed_combinations = {}
                for combo_tuple, count in value.items():
                    class_label, age_group, body_part = combo_tuple
                    class_name = '良性' if class_label == 0 else '悪性'
                    combo_str = f"{class_name}_{age_group}_{body_part}"
                    string_keyed_combinations[combo_str] = count
                processed_data_stats[key] = string_keyed_combinations
            else:
                # Counter オブジェクトを辞書に変換
                if hasattr(value, 'items'):
                    processed_data_stats[key] = dict(value)
                else:
                    processed_data_stats[key] = value
        
        dataset_info = {
            'source_description': f'JSONベース層化分割データセット (年齢閾値: {self.age_threshold}歳)',
            'source_json': str(self.json_path),
            'source_image_dir': str(self.image_dir),
            'age_threshold': self.age_threshold,
            'classes': [0, 1],
            'class_names': {0: '良性（母斑）', 1: '悪性（メラノーマ）'},
            'age_groups': ['under_30', 'over_30'],
            'split_ratios': {
                'train': self.train_ratio,
                'validation': self.val_ratio,
                'test1': self.test1_ratio,
                'test2': self.test2_ratio
            },
            'data_statistics': processed_data_stats,
            'split_statistics': {},
            'copy_summary': copy_summary
        }
        
        # 各分割の統計情報
        for split_name in ['train', 'validation', 'test1', 'test2']:
            dataset_info['split_statistics'][split_name] = {
                'class_counts': dict(split_counts[split_name]['class']),
                'age_group_counts': dict(split_counts[split_name]['age_group']),
                'bodypart_counts': dict(split_counts[split_name]['bodypart']),
                'total_images': sum(split_counts[split_name]['class'].values())
            }
        
        # JSONファイル保存
        info_file = self.output_path / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"📄 データセット情報保存: {info_file}")
        return dataset_info
    
    def create_dataset(self):
        """JSONベース層化データセット作成メイン処理"""
        print(f"\n{'='*80}")
        print(f"📋 JSONベース 4分割層化データセット作成")
        print(f"🎯 Train:{self.train_ratio:.1f} Val:{self.val_ratio:.1f} Test1:{self.test1_ratio:.1f} Test2:{self.test2_ratio:.1f}")
        print(f"🎯 クラス・年齢層・疾患部位均等分割")
        print(f"{'='*80}")
        
        # 1. JSONデータ読み込み
        json_data = self.load_json_data()
        
        # 2. JSONデータ処理
        processed_data = self.process_json_data(json_data)
        
        if not processed_data:
            print(f"❌ 処理可能なデータが見つかりません")
            return None
        
        # 3. データ分布分析
        data_stats = self.analyze_data_distribution(processed_data)
        
        # 4. 4分割層化分割
        splits, split_counts = self.stratified_split_4way(processed_data)
        
        # 5. 画像ファイルコピー
        copy_summary = self.copy_images_to_splits(splits)
        
        # 6. メタデータファイル作成
        self.create_metadata_files(splits)
        
        # 7. データセット情報保存
        dataset_info = self.create_dataset_info(split_counts, data_stats, copy_summary)
        
        print(f"\n🎉 JSONベース層化データセット作成完了!")
        print(f"📁 出力先: {self.output_path}")
        
        return {
            'splits': splits,
            'split_counts': split_counts,
            'copy_summary': copy_summary,
            'dataset_info': dataset_info,
            'data_stats': data_stats
        }

def main():
    """メイン実行関数"""
    print("=== JSONベース 4分割層化データセット作成ツール ===")
    
    # 設定
    json_path = "./YN_Under_30_Body_dataset_fixed.json"  # JSONファイルのパス
    image_dir = "./dataset_merged_body/YN/Under_30/Body"              # 画像ファイルのディレクトリ
    output_path = "./dataset_before_YN_U30body"
    train_ratio = 0.6
    val_ratio = 0.2
    test1_ratio = 0.1
    test2_ratio = 0.1
    age_threshold = 30  # 年齢の閾値
    
    print(f"📋 入力JSONファイル: {json_path}")
    print(f"📁 画像ディレクトリ: {image_dir}")
    print(f"📁 出力ディレクトリ: {output_path}")
    print(f"🎯 年齢閾値: {age_threshold}歳")
    print(f"📊 分割比率: Train={train_ratio:.1f} Val={val_ratio:.1f} Test1={test1_ratio:.1f} Test2={test2_ratio:.1f}")
    print(f"🎯 層化分割: クラス・年齢層・疾患部位の割合を全分割で保持")
    
    # データセット作成実行
    creator = JSONBasedStratifiedDatasetCreator(
        json_path=json_path,
        image_dir=image_dir,
        output_path=output_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test1_ratio=test1_ratio,
        test2_ratio=test2_ratio,
        age_threshold=age_threshold
    )
    
    try:
        result = creator.create_dataset()
        
        if result:
            print(f"\n🎉 JSONベース層化データセット作成完了!")
            print(f"📁 出力先: {output_path}")
            print(f"\n📊 各分割の特徴:")
            print(f"   ✅ 全分割でクラス比率（良性 vs 悪性）が等しい")
            print(f"   ✅ 全分割で年齢層比率が等しい")
            print(f"   ✅ 全分割で疾患部位比率が等しい")
            print(f"   ✅ Test1とTest2が均等分割されモデル評価の信頼性向上")
            print(f"   🎯 JSONメタデータに基づく高品質層化データセット")
        else:
            print(f"\n❌ データセット作成失敗")
            
    except Exception as e:
        print(f"\n❌ メイン処理でエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()