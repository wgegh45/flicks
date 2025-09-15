input.txt を読み込み、漢字部分だけをピンイン（声調記号なし）に変換して output.txt に書く（非漢字はそのまま保持）
# to_pinyin.py
import re
from pypinyin import lazy_pinyin, Style

# 漢字ブロックを検出する正規表現
ch_re = re.compile(r'([\u4e00-\u9fff]+)')

def han_to_pinyin(match):
    han = match.group(1)
    # Style.NORMAL -> 声調記号なし
    py = lazy_pinyin(han, style=Style.NORMAL)
    return ' '.join(py)

def convert_file(in_path='input.txt', out_path='output.txt'):
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            new = ch_re.sub(han_to_pinyin, line)
            fout.write(new)

if __name__ == '__main__':
    convert_file('input.txt', 'output.txt')




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
from collections import Counter
from pypinyin import lazy_pinyin, Style

# 対象ディレクトリ
INPUT_DIR = "passwords"
# 出力CSVファイル
OUTPUT_CSV = "summary.csv"
# 調査したい特定のパスワード
TARGET_PASSWORDS = ["password123", "123456"]

# 文字タイプ判定関数
def char_type(c):
    if c.islower():
        return "lower"
    elif c.isupper():
        return "upper"
    elif c.isdigit():
        return "digit"
    else:
        return "symbol"

# パスワードの構造判定
def structure(s):
    result = []
    for c in s:
        t = char_type(c)
        if t == "lower":
            result.append("L")
        elif t == "upper":
            result.append("U")
        elif t == "digit":
            result.append("D")
        else:
            result.append("S")
    return "".join(result)

# 構造圧縮（例: Pa12!! -> U1L1D2S2）
def compress_structure(s):
    if not s:
        return ""
    compressed = []
    last = s[0]
    count = 1
    for c in s[1:]:
        if c == last:
            count += 1
        else:
            compressed.append(f"{last}{count}")
            last = c
            count = 1
    compressed.append(f"{last}{count}")
    return "".join(compressed)

# シャノンエントロピー計算
def shannon_entropy(s):
    if not s:
        return 0.0
    counter = Counter(s)
    length = len(s)
    entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())
    return entropy

# CSV出力ヘッダ
header = [
    "ファイル名","ファイルパス","ファイルサイズ[byte]","総行数","ユニーク数","重複有無",
    "平均長","中央値","最短長","最長長",
    "Top1","Top1件数","Top2","Top2件数","Top3","Top3件数","Top4","Top4件数","Top5","Top5件数",
    "長さTop1","長さTop1件数","長さTop1割合[%]",
    "長さTop2","長さTop2件数","長さTop2割合[%]",
    "長さTop3","長さTop3件数","長さTop3割合[%]",
    "長さTop4","長さTop4件数","長さTop4割合[%]",
    "長さTop5","長さTop5件数","長さTop5割合[%]",
    "構造Top1","構造Top1件数",
    "構造Top2","構造Top2件数",
    "構造Top3","構造Top3件数",
    "構造Top4","構造Top4件数",
    "構造Top5","構造Top5件数",
    "構造詳細Top1","構造詳細Top1件数",
    "構造詳細Top2","構造詳細Top2件数",
    "構造詳細Top3","構造詳細Top3件数",
    "構造詳細Top4","構造詳細Top4件数",
    "構造詳細Top5","構造詳細Top5件数",
    "小文字[%]","大文字[%]","数字[%]","記号[%]","シャノン推定[bit]",
    "対象パスワード含行数"
]

# 出力CSVを追記モードで開く
with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # ファイルが空の場合のみヘッダを書き込む
    if os.stat(OUTPUT_CSV).st_size == 0:
        writer.writerow(header)

    # 再帰的にファイルを処理
    for root, dirs, files in os.walk(INPUT_DIR):
        for fname in files:
            file_path = os.path.join(root, fname)
            rel_path = os.path.relpath(file_path)
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    passwords = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"読み込み失敗: {file_path} ({e})")
                continue

            total_lines = len(passwords)
            unique_passwords = set(passwords)
            duplicate_flag = "あり" if len(unique_passwords) < total_lines else "なし"
            lengths = [len(p) for p in passwords]
            avg_len = sum(lengths)/total_lines if total_lines else 0
            sorted_lengths = sorted(lengths)
            median_len = sorted_lengths[total_lines//2] if total_lines else 0
            min_len = min(lengths) if lengths else 0
            max_len = max(lengths) if lengths else 0

            # Top5パスワード
            pw_counter = Counter(passwords)
            top_pw = pw_counter.most_common(5)
            top_pw_flat = []
            for i in range(5):
                if i < len(top_pw):
                    top_pw_flat.extend([top_pw[i][0], top_pw[i][1]])
                else:
                    top_pw_flat.extend(["",""])

            # 長さTop5
            length_counter = Counter(lengths)
            top_len = length_counter.most_common(5)
            top_len_flat = []
            for i in range(5):
                if i < len(top_len):
                    count = top_len[i][1]
                    perc = round(count/total_lines*100,2)
                    top_len_flat.extend([top_len[i][0], count, perc])
                else:
                    top_len_flat.extend(["","",""])

            # 構造Top5
            struct_counter = Counter([structure(p) for p in passwords])
            top_struct = struct_counter.most_common(5)
            top_struct_flat = []
            for i in range(5):
                if i < len(top_struct):
                    top_struct_flat.extend([top_struct[i][0], top_struct[i][1]])
                else:
                    top_struct_flat.extend(["",""])

            # 構造詳細Top5（圧縮形式）
            struct_detail_counter = Counter([compress_structure(structure(p)) for p in passwords])
            top_struct_detail = struct_detail_counter.most_common(5)
            top_struct_detail_flat = []
            for i in range(5):
                if i < len(top_struct_detail):
                    top_struct_detail_flat.extend([top_struct_detail[i][0], top_struct_detail[i][1]])
                else:
                    top_struct_detail_flat.extend(["",""])

            # 文字種割合
            total_chars = sum(lengths)
            char_counter = Counter(c for p in passwords for c in p)
            lower_perc = round(sum(char_counter[c] for c in char_counter if c.islower())/total_chars*100,2) if total_chars else 0
            upper_perc = round(sum(char_counter[c] for c in char_counter if c.isupper())/total_chars*100,2) if total_chars else 0
            digit_perc = round(sum(char_counter[c] for c in char_counter if c.isdigit())/total_chars*100,2) if total_chars else 0
            symbol_perc = round(total_chars - sum(char_counter[c] for c in char_counter if c.isalnum())/total_chars*100,2) if total_chars else 0

            # Shannonエントロピー
            entropy_bits = round(shannon_entropy("".join(passwords)),2)

            # 対象パスワード含行数
            target_count = sum(1 for p in passwords if p in TARGET_PASSWORDS)

            # ファイルサイズ
            file_size = os.path.getsize(file_path)

            row = [
                fname, rel_path, file_size, total_lines, len(unique_passwords), duplicate_flag,
                round(avg_len,2), median_len, min_len, max_len
            ] + top_pw_flat + top_len_flat + top_struct_flat + top_struct_detail_flat + \
                [lower_perc, upper_perc, digit_perc, symbol_perc, entropy_bits,
