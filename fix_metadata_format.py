#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import sys

def load_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, p: Path):
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Replace subset metadata entries with entries from all_data_162.json using filename matching.')
    parser.add_argument('--all', required=True, help='path to all_data_162.json')
    parser.add_argument('--inputs', required=True, nargs='+', help='metadata files to convert')
    parser.add_argument('--inplace', action='store_true', help='overwrite input files (default: write to <input>.fixed.json)')
    args = parser.parse_args()

    all_path = Path(args.all)
    if not all_path.exists():
        print(f'Error: all file not found: {all_path}', file=sys.stderr)
        sys.exit(1)

    all_data = load_json(all_path)
    print(f'Loaded all_data: {all_path} ({len(all_data)} entries)')
    
    # all_data_162.jsonから filename をキーとするインデックスを作成
    filename_to_entry = {}
    for entry in all_data:
        filename = entry['filename']
        if filename in filename_to_entry:
            print(f'Warning: duplicate filename found: {filename}')
        filename_to_entry[filename] = entry
    
    print(f'Created filename index with {len(filename_to_entry)} entries')

    for inp in args.inputs:
        inp_path = Path(inp)
        if not inp_path.exists():
            print(f'Warning: input not found, skipping: {inp_path}', file=sys.stderr)
            continue

        subset = load_json(inp_path)
        out = {}
        missing = []
        
        for subset_key in subset.keys():
            # subset_key (例: "4138_2014_4_9_KM2_5637_6.jpg") で検索
            if subset_key in filename_to_entry:
                # all_data_162.json からエントリを取得（filenameフィールドは除外）
                entry = filename_to_entry[subset_key].copy()
                entry.pop('filename', None)  # filenameフィールドを削除
                out[subset_key] = entry
            else:
                missing.append(subset_key)
                out[subset_key] = subset[subset_key]  # fallback: keep original

        if missing:
            print(f'Warning: {len(missing)} entries from {inp_path.name} not found in {all_path.name}. Kept original entries for them.')
            for m in missing[:5]:  # 最初の5個だけ表示
                print('  missing:', m)
            if len(missing) > 5:
                print(f'  ... and {len(missing) - 5} more')

        out_path = inp_path if args.inplace else inp_path.with_suffix(inp_path.suffix + '.fixed.json')
        save_json(out, out_path)
        print(f'Wrote {len(out)} entries to {out_path}')

if __name__ == '__main__':
    main()