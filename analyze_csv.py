import pandas as pd
import argparse
import os
from itertools import combinations

def load_csv_files(csv_dir, before_correct_name="before_correct.csv", 
                   before_incorrect_name="before_incorrect.csv",
                   after_correct_name="after_correct.csv", 
                   after_incorrect_name="after_incorrect.csv"):
    """
    指定ディレクトリから4つのCSVファイルを読み込み
    """
    files = {
        'before_correct': os.path.join(csv_dir, before_correct_name),
        'before_incorrect': os.path.join(csv_dir, before_incorrect_name),
        'after_correct': os.path.join(csv_dir, after_correct_name),
        'after_incorrect': os.path.join(csv_dir, after_incorrect_name)
    }
    
    print(f"CSVディレクトリ: {csv_dir}")
    
    dfs = {}
    for name, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['category'] = name
            # 年齢を数値に変換
            df['age_numeric'] = pd.to_numeric(df['age'], errors='coerce')
            # 年齢層を分類（30歳を基準）
            df['age_group'] = df['age_numeric'].apply(lambda x: 'under_30' if x < 30 else 'over_30' if pd.notnull(x) else 'unknown')
            dfs[name] = df
            print(f"✓ {name}: {len(df)}件読み込み ({os.path.basename(path)})")
        else:
            print(f"✗ 警告: {path} が見つかりません")
            dfs[name] = pd.DataFrame()
    
    return dfs

def create_summary_table_a(dfs):
    """
    A. 施設だけが異なる市販前後の画像枚数
    （年齢層と疾患部位は同じ。2×3＝6通り）
    """
    print("\n" + "="*60)
    print("A. 施設だけが異なる市販前後の画像枚数")
    print("="*60)
    
    # 全データを結合
    all_data = pd.concat([df for df in dfs.values() if not df.empty], ignore_index=True)
    
    if all_data.empty:
        print("データが見つかりません")
        return None
    
    # 部位の種類を確認
    parts = sorted(all_data['part'].unique())
    age_groups = ['under_30', 'over_30']
    
    print(f"検出された部位: {parts}")
    print(f"年齢層: {age_groups}")
    
    # 施設別の集計
    before_data = pd.concat([dfs['before_correct'], dfs['before_incorrect']], ignore_index=True)
    after_data = pd.concat([dfs['after_correct'], dfs['after_incorrect']], ignore_index=True)
    
    results_a = []
    
    for age_group in age_groups:
        for part in parts:
            # Before（市販前）のデータ
            before_filtered = before_data[
                (before_data['age_group'] == age_group) & 
                (before_data['part'] == part)
            ]
            before_count = len(before_filtered)
            before_univ = before_filtered['univ_ID'].unique() if before_count > 0 else []
            
            # After（市販後）のデータ
            after_filtered = after_data[
                (after_data['age_group'] == age_group) & 
                (after_data['part'] == part)
            ]
            after_count = len(after_filtered)
            after_univ = after_filtered['univ_ID'].unique() if after_count > 0 else []
            
            # 施設が異なるかチェック
            facilities_different = len(set(before_univ) & set(after_univ)) == 0
            
            result = {
                'age_group': age_group,
                'part': part,
                'before_count': before_count,
                'after_count': after_count,
                'before_facilities': ', '.join(before_univ),
                'after_facilities': ', '.join(after_univ),
                'facilities_different': facilities_different
            }
            results_a.append(result)
            
            print(f"{age_group:10s} × {part:10s}: Before={before_count:3d}件({', '.join(before_univ)}), "
                  f"After={after_count:3d}件({', '.join(after_univ)}), "
                  f"施設異なる={facilities_different}")
    
    return pd.DataFrame(results_a)

def create_summary_table_b(dfs):
    """
    B. 施設、年齢層、疾患部位が異なる市販前後の画像の枚数
    （6C2=15通り：年齢層2×部位3の組み合わせから2つを選ぶ）
    """
    print("\n" + "="*60)
    print("B. 施設、年齢層、疾患部位が異なる市販前後の画像の枚数")
    print("="*60)
    
    # 全データを結合
    all_data = pd.concat([df for df in dfs.values() if not df.empty], ignore_index=True)
    
    if all_data.empty:
        print("データが見つかりません")
        return None
    
    # Before（市販前）とAfter（市販後）のデータ
    before_data = pd.concat([dfs['before_correct'], dfs['before_incorrect']], ignore_index=True)
    after_data = pd.concat([dfs['after_correct'], dfs['after_incorrect']], ignore_index=True)
    
    # 年齢層と部位の組み合わせを作成
    age_groups = ['under_30', 'over_30']
    parts = sorted(all_data['part'].unique())
    
    # 6通りの組み合わせ（年齢層2 × 部位3）
    combinations_list = []
    for age in age_groups:
        for part in parts:
            combinations_list.append((age, part))
    
    print(f"組み合わせ一覧: {combinations_list}")
    
    # 15通りの比較（6C2 = 15）
    results_b = []
    comparison_pairs = list(combinations(combinations_list, 2))
    
    print(f"\n比較ペア数: {len(comparison_pairs)}")
    
    for i, (combo1, combo2) in enumerate(comparison_pairs, 1):
        age1, part1 = combo1
        age2, part2 = combo2
        
        # Before側：combo1の条件
        before_filtered = before_data[
            (before_data['age_group'] == age1) & 
            (before_data['part'] == part1)
        ]
        before_count = len(before_filtered)
        before_univ = before_filtered['univ_ID'].unique() if before_count > 0 else []
        
        # After側：combo2の条件  
        after_filtered = after_data[
            (after_data['age_group'] == age2) & 
            (after_data['part'] == part2)
        ]
        after_count = len(after_filtered)
        after_univ = after_filtered['univ_ID'].unique() if after_count > 0 else []
        
        # 条件チェック
        age_different = age1 != age2
        part_different = part1 != part2
        facilities_different = len(set(before_univ) & set(after_univ)) == 0
        
        # 少なくとも1つ以上の条件が異なる場合のみ記録
        conditions_different = age_different or part_different or facilities_different
        
        result = {
            'comparison_id': i,
            'before_condition': f"{age1}_{part1}",
            'after_condition': f"{age2}_{part2}",
            'before_count': before_count,
            'after_count': after_count,
            'before_facilities': ', '.join(before_univ),
            'after_facilities': ', '.join(after_univ),
            'age_different': age_different,
            'part_different': part_different,
            'facilities_different': facilities_different,
            'conditions_different': conditions_different
        }
        results_b.append(result)
        
        print(f"{i:2d}. {age1:10s}×{part1:10s} vs {age2:10s}×{part2:10s}: "
              f"Before={before_count:3d}件, After={after_count:3d}件, "
              f"年齢={'○' if age_different else '×'}, "
              f"部位={'○' if part_different else '×'}, "
              f"施設={'○' if facilities_different else '×'}")
    
    return pd.DataFrame(results_b)

def create_detailed_statistics(dfs):
    """
    詳細な統計情報を作成
    """
    print("\n" + "="*60)
    print("詳細統計情報")
    print("="*60)
    
    for name, df in dfs.items():
        if df.empty:
            continue
            
        print(f"\n[{name}]")
        print(f"総データ数: {len(df)}")
        
        # 施設別
        if 'univ_ID' in df.columns:
            univ_counts = df['univ_ID'].value_counts()
            print(f"施設別: {dict(univ_counts)}")
        
        # 年齢層別
        if 'age_group' in df.columns:
            age_counts = df['age_group'].value_counts()
            print(f"年齢層別: {dict(age_counts)}")
        
        # 部位別
        if 'part' in df.columns:
            part_counts = df['part'].value_counts()
            print(f"部位別: {dict(part_counts)}")

def save_results(results_a, results_b, output_dir):
    """
    結果をCSVファイルとして保存
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if results_a is not None:
        output_a = os.path.join(output_dir, "analysis_a_facility_different.csv")
        results_a.to_csv(output_a, index=False)
        print(f"\n結果Aを保存: {output_a}")
    
    if results_b is not None:
        output_b = os.path.join(output_dir, "analysis_b_multiple_different.csv")
        results_b.to_csv(output_b, index=False)
        print(f"結果Bを保存: {output_b}")

def main():
    parser = argparse.ArgumentParser(
        description="市販前後のデータ比較分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  python analyze_csv.py --csv_dir ./results

  # カスタムファイル名を指定
  python analyze_csv.py --csv_dir ./results \\
    --before_correct predict_results_before_correct.csv \\
    --before_incorrect predict_results_before_incorrect.csv \\
    --after_correct predict_results_after_correct.csv \\
    --after_incorrect predict_results_after_incorrect.csv

  # 出力ディレクトリを指定
  python analyze_csv.py --csv_dir ./results --output_dir ./my_analysis
        """
    )
    
    # 必須引数
    parser.add_argument('--csv_dir', '-d', required=True, type=str,
                        help='CSVファイルが格納されているディレクトリのパス')
    
    # オプション引数（ファイル名）
    parser.add_argument('--before_correct', type=str, default='before_correct.csv',
                        help='before_correct.csvのファイル名（デフォルト: before_correct.csv）')
    parser.add_argument('--before_incorrect', type=str, default='before_incorrect.csv',
                        help='before_incorrect.csvのファイル名（デフォルト: before_incorrect.csv）')
    parser.add_argument('--after_correct', type=str, default='after_correct.csv',
                        help='after_correct.csvのファイル名（デフォルト: after_correct.csv）')
    parser.add_argument('--after_incorrect', type=str, default='after_incorrect.csv',
                        help='after_incorrect.csvのファイル名（デフォルト: after_incorrect.csv）')
    
    # 出力ディレクトリ
    parser.add_argument('--output_dir', '-o', type=str, default='./analysis_results',
                        help='結果出力ディレクトリ（デフォルト: ./analysis_results）')
    
    args = parser.parse_args()
    
    # ディレクトリの存在確認
    if not os.path.exists(args.csv_dir):
        print(f"エラー: 指定されたディレクトリ '{args.csv_dir}' が存在しません")
        return
    
    print("市販前後データ比較分析を開始します...")
    print(f"対象ディレクトリ: {os.path.abspath(args.csv_dir)}")
    
    # CSVファイルを読み込み
    dfs = load_csv_files(
        args.csv_dir,
        args.before_correct,
        args.before_incorrect,
        args.after_correct,
        args.after_incorrect
    )
    
    # データが存在するかチェック
    total_data = sum(len(df) for df in dfs.values())
    if total_data == 0:
        print("エラー: 有効なデータが見つかりませんでした")
        return
    
    print(f"合計データ数: {total_data}件")
    
    # 詳細統計情報
    create_detailed_statistics(dfs)
    
    # A. 施設だけが異なる市販前後の比較
    results_a = create_summary_table_a(dfs)
    
    # B. 施設、年齢層、疾患部位が異なる市販前後の比較
    results_b = create_summary_table_b(dfs)
    
    # 結果を保存
    save_results(results_a, results_b, args.output_dir)
    
    print(f"\n分析完了！結果は {os.path.abspath(args.output_dir)} に保存されました。")

if __name__ == '__main__':
    main()