#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import shutil

class ImageCleanupTool:
    """JSONファイルと一致しない画像を削除するツール"""
    
    def __init__(self, image_dir, json_file, backup_dir=None, dry_run=False):
        self.image_dir = Path(image_dir)
        self.json_file = Path(json_file)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.dry_run = dry_run
        
        # サポートする画像拡張子
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        print(f"📁 画像ディレクトリ: {self.image_dir}")
        print(f"📄 JSONファイル: {self.json_file}")
        print(f"🔍 ドライランモード: {'ON' if self.dry_run else 'OFF'}")
        if self.backup_dir:
            print(f"💾 バックアップディレクトリ: {self.backup_dir}")
    
    def load_json_filenames(self):
        """JSONファイルからfilenameリストを読み込み"""
        print(f"\n📋 JSONファイル読み込み: {self.json_file}")
        
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            filenames = set()
            
            if isinstance(json_data, dict):
                # 辞書形式の場合（キーがファイル名）
                filenames = set(json_data.keys())
                print(f"   形式: 辞書 ({len(filenames)}個のキー)")
                
            elif isinstance(json_data, list):
                # リスト形式の場合
                print(f"   形式: リスト ({len(json_data)}個の要素)")
                
                for i, item in enumerate(json_data):
                    if isinstance(item, dict):
                        # 複数のフィールドからファイル名を取得
                        for field in ['filename', 'original_filename', 'copied_filename']:
                            if field in item and item[field]:
                                # パスが含まれている場合はベースネームを取得
                                filename = os.path.basename(item[field])
                                filenames.add(filename)
                                break
                        else:
                            print(f"   警告: アイテム{i}にファイル名フィールドが見つかりません")
                    elif isinstance(item, str):
                        # 文字列の場合（ファイル名のみ）
                        filename = os.path.basename(item)
                        filenames.add(filename)
            
            else:
                raise ValueError(f"未対応のJSONデータ形式: {type(json_data)}")
            
            print(f"✅ 有効なファイル名: {len(filenames)}個")
            
            # サンプル表示
            sample_filenames = list(filenames)[:5]
            print(f"   サンプル: {sample_filenames}")
            
            return filenames
            
        except FileNotFoundError:
            print(f"❌ JSONファイルが見つかりません: {self.json_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析エラー: {e}")
            raise
        except Exception as e:
            print(f"❌ JSONファイル読み込みエラー: {e}")
            raise
    
    def find_image_files(self):
        """画像ディレクトリから画像ファイルを検索"""
        print(f"\n🔍 画像ファイル検索: {self.image_dir}")
        
        if not self.image_dir.exists():
            print(f"❌ 画像ディレクトリが存在しません: {self.image_dir}")
            raise FileNotFoundError(f"画像ディレクトリが見つかりません: {self.image_dir}")
        
        image_files = []
        
        # 再帰的に画像ファイルを検索
        for file_path in self.image_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                image_files.append(file_path)
        
        print(f"✅ 検出した画像ファイル: {len(image_files)}個")
        
        # 拡張子別の統計
        ext_counts = {}
        for file_path in image_files:
            ext = file_path.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        print(f"   拡張子別統計: {dict(ext_counts)}")
        
        return image_files
    
    def identify_orphan_images(self, image_files, valid_filenames):
        """JSONに存在しない孤立した画像を特定"""
        print(f"\n🔍 孤立画像の特定")
        
        orphan_images = []
        matched_images = []
        
        for image_path in image_files:
            filename = image_path.name
            
            if filename in valid_filenames:
                matched_images.append(image_path)
            else:
                orphan_images.append(image_path)
        
        print(f"✅ 一致した画像: {len(matched_images)}個")
        print(f"⚠️  孤立した画像: {len(orphan_images)}個")
        
        # 孤立画像のサンプル表示
        if orphan_images:
            print(f"\n   孤立画像サンプル:")
            for orphan in orphan_images[:10]:
                print(f"     {orphan}")
            if len(orphan_images) > 10:
                print(f"     ... 他 {len(orphan_images) - 10}個")
        
        return orphan_images, matched_images
    
    def create_backup_dir(self):
        """バックアップディレクトリを作成"""
        if self.backup_dir and not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"💾 バックアップディレクトリ作成: {self.backup_dir}")
    
    def delete_orphan_images(self, orphan_images):
        """孤立した画像を削除（またはバックアップ）"""
        if not orphan_images:
            print(f"\n🎉 削除対象の孤立画像はありません")
            return
        
        print(f"\n🗑️  孤立画像の削除処理")
        print(f"   対象ファイル数: {len(orphan_images)}個")
        
        if self.dry_run:
            print(f"   ドライランモード: 実際の削除は行いません")
            
            # ドライランでの削除予定リスト表示
            print(f"\n   削除予定ファイル:")
            for i, orphan in enumerate(orphan_images):
                print(f"     {i+1:3d}. {orphan}")
                if i >= 20:  # 最初の20個まで表示
                    print(f"     ... 他 {len(orphan_images) - 20}個")
                    break
            return
        
        # 実際の削除処理
        deleted_count = 0
        backed_up_count = 0
        failed_count = 0
        
        for i, orphan_path in enumerate(orphan_images):
            try:
                if self.backup_dir:
                    # バックアップ作成
                    relative_path = orphan_path.relative_to(self.image_dir)
                    backup_path = self.backup_dir / relative_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.move(str(orphan_path), str(backup_path))
                    backed_up_count += 1
                    
                    if (i + 1) % 100 == 0:
                        print(f"     進捗: {i + 1}/{len(orphan_images)} 個バックアップ完了")
                else:
                    # 直接削除
                    orphan_path.unlink()
                    deleted_count += 1
                    
                    if (i + 1) % 100 == 0:
                        print(f"     進捗: {i + 1}/{len(orphan_images)} 個削除完了")
                        
            except Exception as e:
                print(f"   ❌ エラー ({orphan_path}): {e}")
                failed_count += 1
        
        # 結果サマリー
        print(f"\n📊 削除処理結果:")
        if self.backup_dir:
            print(f"   バックアップ済み: {backed_up_count}個")
        else:
            print(f"   削除済み: {deleted_count}個")
        
        if failed_count > 0:
            print(f"   失敗: {failed_count}個")
        
        print(f"✅ 処理完了")
    
    def create_report(self, orphan_images, matched_images, output_file="cleanup_report.txt"):
        """処理結果のレポートを作成"""
        report_path = Path(output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("画像クリーンアップレポート\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"処理日時: {Path(__file__).stat().st_mtime}\n")
            f.write(f"画像ディレクトリ: {self.image_dir}\n")
            f.write(f"JSONファイル: {self.json_file}\n")
            f.write(f"ドライランモード: {'ON' if self.dry_run else 'OFF'}\n")
            if self.backup_dir:
                f.write(f"バックアップディレクトリ: {self.backup_dir}\n")
            f.write(f"\n")
            
            f.write(f"処理結果:\n")
            f.write(f"  一致した画像: {len(matched_images)}個\n")
            f.write(f"  孤立した画像: {len(orphan_images)}個\n\n")
            
            if orphan_images:
                f.write("孤立した画像リスト:\n")
                for i, orphan in enumerate(orphan_images):
                    f.write(f"  {i+1:4d}. {orphan}\n")
        
        print(f"📄 レポート作成: {report_path}")
    
    def run(self):
        """メイン処理実行"""
        print(f"{'='*60}")
        print(f"画像クリーンアップツール実行開始")
        print(f"{'='*60}")
        
        try:
            # 1. JSONからファイル名リストを読み込み
            valid_filenames = self.load_json_filenames()
            
            # 2. 画像ファイルを検索
            image_files = self.find_image_files()
            
            # 3. 孤立画像を特定
            orphan_images, matched_images = self.identify_orphan_images(image_files, valid_filenames)
            
            # 4. バックアップディレクトリ作成
            if self.backup_dir:
                self.create_backup_dir()
            
            # 5. 孤立画像を削除
            self.delete_orphan_images(orphan_images)
            
            # 6. レポート作成
            self.create_report(orphan_images, matched_images)
            
            print(f"\n🎉 画像クリーンアップ処理完了!")
            
            return {
                'total_images': len(image_files),
                'matched_images': len(matched_images),
                'orphan_images': len(orphan_images)
            }
            
        except Exception as e:
            print(f"\n❌ 処理中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    parser = argparse.ArgumentParser(
        description="JSONファイルと一致しない孤立した画像を削除",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ドライラン（実際の削除は行わない）
  python image_cleanup.py --image_dir ./dataset_before --json_file ./dataset_before/train_metadata.json --dry-run

  # 削除実行
  python image_cleanup.py --image_dir ./dataset_before --json_file ./dataset_before/train_metadata.json

  # バックアップ付き削除
  python image_cleanup.py --image_dir ./dataset_before --json_file ./dataset_before/train_metadata.json --backup_dir ./backup

  # 複数JSONファイル対応（OR条件）
  python image_cleanup.py --image_dir ./dataset_before --json_file ./dataset_before/train_metadata.json --json_file ./dataset_before/validation_metadata.json
        """
    )
    
    parser.add_argument('--image_dir', '-i', required=True, type=str,
                        help='画像が保存されているディレクトリパス')
    parser.add_argument('--json_file', '-j', action='append', required=True,
                        help='JSONファイルのパス（複数指定可能）')
    parser.add_argument('--backup_dir', '-b', type=str, default=None,
                        help='削除する前にファイルをバックアップするディレクトリ')
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help='ドライランモード（実際の削除は行わない）')
    parser.add_argument('--report', '-r', type=str, default='cleanup_report.txt',
                        help='レポートファイル名（デフォルト: cleanup_report.txt）')
    
    args = parser.parse_args()
    
    # 複数JSONファイルの場合は統合処理
    print("🚀 画像クリーンアップツール")
    print(f"画像ディレクトリ: {args.image_dir}")
    print(f"JSONファイル: {args.json_file}")
    
    if len(args.json_file) == 1:
        # 単一JSONファイルの場合
        cleanup = ImageCleanupTool(
            image_dir=args.image_dir,
            json_file=args.json_file[0],
            backup_dir=args.backup_dir,
            dry_run=args.dry_run
        )
        result = cleanup.run()
        
    else:
        # 複数JSONファイルの場合 - 統合処理
        print(f"📋 複数JSONファイルを統合処理: {len(args.json_file)}個")
        
        all_valid_filenames = set()
        
        # 全JSONファイルからfilename収集
        for json_file in args.json_file:
            print(f"\n📄 処理中: {json_file}")
            temp_cleanup = ImageCleanupTool(
                image_dir=args.image_dir,
                json_file=json_file,
                backup_dir=args.backup_dir,
                dry_run=True  # 統合時は一時的にdry-run
            )
            filenames = temp_cleanup.load_json_filenames()
            all_valid_filenames.update(filenames)
        
        print(f"\n📊 統合結果: {len(all_valid_filenames)}個のユニークファイル名")
        
        # 統合されたファイル名リストで実際の処理
        # 一時的なJSONファイルを作成して処理
        temp_json = Path("temp_merged_filenames.json")
        with open(temp_json, 'w', encoding='utf-8') as f:
            json.dump(list(all_valid_filenames), f, indent=2)
        
        try:
            cleanup = ImageCleanupTool(
                image_dir=args.image_dir,
                json_file=temp_json,
                backup_dir=args.backup_dir,
                dry_run=args.dry_run
            )
            result = cleanup.run()
        finally:
            # 一時ファイル削除
            if temp_json.exists():
                temp_json.unlink()
    
    print(f"\n📊 最終結果:")
    print(f"  総画像数: {result['total_images']}")
    print(f"  一致画像数: {result['matched_images']}")
    print(f"  削除対象画像数: {result['orphan_images']}")

if __name__ == "__main__":
    main()