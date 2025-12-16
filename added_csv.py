import pandas as pd
import json
import argparse
import os
import sys

def load_metadata(json_path):
    """
    all_data_162.jsonから追加のメタデータを読み込み
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # filenameをキーとした辞書を作成
        metadata_dict = {}
        for item in metadata:
            filename = item.get('filename')
            if filename:
                metadata_dict[filename] = {
                    'part': item.get('part', ['unknown'])[0] if isinstance(item.get('part'), list) else item.get('part', 'unknown'),
                    'age': item.get('age', ['unknown'])[0] if isinstance(item.get('age'), list) else item.get('age', 'unknown'),
                    'univ_ID': item.get('univ_ID', 'unknown')
                }
        
        print(f"メタデータを {len(metadata_dict)} 件読み込みました")
        return metadata_dict
        
    except FileNotFoundError:
        print(f"エラー: {json_path} が見つかりません")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: メタデータ読み込み中にエラーが発生しました: {e}")
        sys.exit(1)

def merge_csv_with_metadata(csv_path, metadata_dict, output_dir=None):
    """
    CSVファイルにメタデータをマージし、正解・不正解別に分けて保存
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(csv_path)
        print(f"CSVファイルを読み込みました: {len(df)} 件")
        
        # 出力ディレクトリを設定
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # メタデータを追加
        df['part'] = df['filename'].map(lambda x: metadata_dict.get(x, {}).get('part', 'unknown'))
        df['age'] = df['filename'].map(lambda x: metadata_dict.get(x, {}).get('age', 'unknown'))
        df['univ_ID'] = df['filename'].map(lambda x: metadata_dict.get(x, {}).get('univ_ID', 'unknown'))
        
        # マッチング統計
        matched_count = sum(1 for _, row in df.iterrows() if row['part'] != 'unknown')
        print(f"メタデータマッチング: {matched_count}/{len(df)} 件 ({matched_count/len(df)*100:.1f}%)")
        
        # 正解・不正解で分ける
        correct_df = df[df['correct'] == 'correct'].copy()
        incorrect_df = df[df['correct'] == 'incorrect'].copy()
        
        # ファイル名を準備
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        
        # 全体のCSVファイル（メタデータ追加版）
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.csv")
        df.to_csv(enhanced_path, index=False)
        print(f"拡張CSVファイル保存: {enhanced_path} ({len(df)}件)")
        
        # 正解のCSVファイル
        correct_path = os.path.join(output_dir, f"{base_name}_correct.csv")
        correct_df.to_csv(correct_path, index=False)
        print(f"正解CSVファイル保存: {correct_path} ({len(correct_df)}件)")
        
        # 不正解のCSVファイル
        incorrect_path = os.path.join(output_dir, f"{base_name}_incorrect.csv")
        incorrect_df.to_csv(incorrect_path, index=False)
        print(f"不正解CSVファイル保存: {incorrect_path} ({len(incorrect_df)}件)")
        
        return df, correct_df, incorrect_df, enhanced_path, correct_path, incorrect_path
        
    except FileNotFoundError:
        print(f"エラー: CSVファイル {csv_path} が見つかりません")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: CSV処理中にエラーが発生しました: {e}")
        sys.exit(1)

def print_statistics(df):
    """
    統計情報を表示
    """
    print(f"\n{'='*50}")
    print(f"予測結果統計")
    print(f"{'='*50}")
    
    # 基本統計
    total_count = len(df)
    correct_count = len(df[df['correct'] == 'correct'])
    incorrect_count = len(df[df['correct'] == 'incorrect'])
    
    print(f"全体: {total_count}件")
    print(f"正解: {correct_count}件 ({correct_count/total_count*100:.1f}%)")
    print(f"不正解: {incorrect_count}件 ({incorrect_count/total_count*100:.1f}%)")
    
    # クラス別統計
    print(f"\n{'='*30}")
    print(f"クラス別統計")
    print(f"{'='*30}")
    class_stats = df.groupby(['true_class', 'predicted_class']).size().reset_index(name='count')
    for _, row in class_stats.iterrows():
        print(f"真のクラス{row['true_class']} → 予測クラス{row['predicted_class']}: {row['count']}件")
    
    # 部位別統計
    if 'part' in df.columns and df['part'].nunique() > 1:
        print(f"\n{'='*30}")
        print(f"部位別統計")
        print(f"{'='*30}")
        part_stats = df.groupby(['part', 'correct']).size().unstack(fill_value=0)
        
        for part in part_stats.index:
            if part == 'unknown':
                continue
                
            correct_count = part_stats.loc[part, 'correct'] if 'correct' in part_stats.columns else 0
            incorrect_count = part_stats.loc[part, 'incorrect'] if 'incorrect' in part_stats.columns else 0
            total_count = correct_count + incorrect_count
            accuracy = correct_count / total_count * 100 if total_count > 0 else 0
            print(f"{part:15s}: {correct_count:3d}/{total_count:3d} ({accuracy:5.1f}%)")
    
    # 大学ID別統計
    if 'univ_ID' in df.columns and df['univ_ID'].nunique() > 1:
        print(f"\n{'='*30}")
        print(f"大学ID別統計")
        print(f"{'='*30}")
        univ_stats = df.groupby(['univ_ID', 'correct']).size().unstack(fill_value=0)
        
        for univ_id in univ_stats.index:
            if univ_id == 'unknown':
                continue
                
            correct_count = univ_stats.loc[univ_id, 'correct'] if 'correct' in univ_stats.columns else 0
            incorrect_count = univ_stats.loc[univ_id, 'incorrect'] if 'incorrect' in univ_stats.columns else 0
            total_count = correct_count + incorrect_count
            accuracy = correct_count / total_count * 100 if total_count > 0 else 0
            print(f"{univ_id:10s}: {correct_count:3d}/{total_count:3d} ({accuracy:5.1f}%)")
    
    # 年齢別統計（年齢が数値の場合）
    if 'age' in df.columns:
        print(f"\n{'='*30}")
        print(f"年齢別統計")
        print(f"{'='*30}")
        
        # 年齢を数値に変換
        df_age = df.copy()
        df_age['age_numeric'] = pd.to_numeric(df_age['age'], errors='coerce')
        df_age = df_age.dropna(subset=['age_numeric'])
        
        if len(df_age) > 0:
            # 年齢を10歳刻みでグループ化
            df_age['age_group'] = (df_age['age_numeric'] // 10) * 10
            df_age['age_group'] = df_age['age_group'].astype(int).astype(str) + '代'
            
            age_stats = df_age.groupby(['age_group', 'correct']).size().unstack(fill_value=0)
            
            for age_group in sorted(age_stats.index):
                correct_count = age_stats.loc[age_group, 'correct'] if 'correct' in age_stats.columns else 0
                incorrect_count = age_stats.loc[age_group, 'incorrect'] if 'incorrect' in age_stats.columns else 0
                total_count = correct_count + incorrect_count
                accuracy = correct_count / total_count * 100 if total_count > 0 else 0
                print(f"{age_group:6s}: {correct_count:3d}/{total_count:3d} ({accuracy:5.1f}%)")

def create_summary_report(df, output_path):
    """
    サマリーレポートをテキストファイルで作成
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("予測結果分析レポート\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本統計
        total_count = len(df)
        correct_count = len(df[df['correct'] == 'correct'])
        incorrect_count = len(df[df['correct'] == 'incorrect'])
        
        f.write("基本統計:\n")
        f.write(f"  全体データ数: {total_count}\n")
        f.write(f"  正解数: {correct_count} ({correct_count/total_count*100:.2f}%)\n")
        f.write(f"  不正解数: {incorrect_count} ({incorrect_count/total_count*100:.2f}%)\n\n")
        
        # 各統計の詳細を書き込み
        if 'part' in df.columns:
            f.write("部位別統計:\n")
            part_stats = df.groupby(['part', 'correct']).size().unstack(fill_value=0)
            for part in part_stats.index:
                if part != 'unknown':
                    correct = part_stats.loc[part, 'correct'] if 'correct' in part_stats.columns else 0
                    incorrect = part_stats.loc[part, 'incorrect'] if 'incorrect' in part_stats.columns else 0
                    total = correct + incorrect
                    acc = correct / total * 100 if total > 0 else 0
                    f.write(f"  {part}: {correct}/{total} ({acc:.1f}%)\n")
            f.write("\n")
        
        if 'univ_ID' in df.columns:
            f.write("大学ID別統計:\n")
            univ_stats = df.groupby(['univ_ID', 'correct']).size().unstack(fill_value=0)
            for univ_id in univ_stats.index:
                if univ_id != 'unknown':
                    correct = univ_stats.loc[univ_id, 'correct'] if 'correct' in univ_stats.columns else 0
                    incorrect = univ_stats.loc[univ_id, 'incorrect'] if 'incorrect' in univ_stats.columns else 0
                    total = correct + incorrect
                    acc = correct / total * 100 if total > 0 else 0
                    f.write(f"  {univ_id}: {correct}/{total} ({acc:.1f}%)\n")
    
    print(f"サマリーレポート保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="CSVファイルにメタデータをマージし、正解・不正解別に分けて出力"
    )
    parser.add_argument('--csv', '-c', required=True, type=str,
                        help='入力CSVファイルのパス')
    parser.add_argument('--metadata', '-m', required=True, type=str,
                        help='メタデータJSONファイルのパス (e.g., all_data_162.json)')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help='出力ディレクトリ（デフォルト: CSVファイルと同じディレクトリ）')
    parser.add_argument('--report', '-r', action='store_true',
                        help='サマリーレポート（テキストファイル）を作成')
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.csv):
        print(f"エラー: CSVファイル {args.csv} が見つかりません")
        sys.exit(1)
    
    if not os.path.exists(args.metadata):
        print(f"エラー: メタデータファイル {args.metadata} が見つかりません")
        sys.exit(1)
    
    print("CSVファイルとメタデータのマージを開始します...")
    print(f"入力CSV: {args.csv}")
    print(f"メタデータ: {args.metadata}")
    
    # メタデータを読み込み
    metadata_dict = load_metadata(args.metadata)
    
    # CSVファイルとマージ
    df, correct_df, incorrect_df, enhanced_path, correct_path, incorrect_path = merge_csv_with_metadata(
        args.csv, metadata_dict, args.output_dir
    )
    
    # 統計情報を表示
    print_statistics(df)
    
    # サマリーレポート作成
    if args.report:
        report_path = enhanced_path.replace('.csv', '_report.txt')
        create_summary_report(df, report_path)
    
    print(f"\n{'='*50}")
    print("処理完了! 以下のファイルが作成されました:")
    print(f"✓ 拡張版全体: {enhanced_path}")
    print(f"✓ 正解のみ: {correct_path}")
    print(f"✓ 不正解のみ: {incorrect_path}")
    if args.report:
        print(f"✓ レポート: {report_path}")

if __name__ == '__main__':
    main()