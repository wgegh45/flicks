大規模辞書 → ユニーク数だけ HLL
その他の処理はストリーム集計で対応可能

元コード：
unique_passwords = set()
for pwd in passwords:
    unique_passwords.add(pwd)
row["ユニーク数"] = len(unique_passwords)

変更後：
import hyperloglog
# 大規模辞書のしきい値（例：100万行以上）
LARGE_FILE_THRESHOLD = 1_000_000

if total_lines > LARGE_FILE_THRESHOLD:
    # HyperLogLog を使って近似ユニーク数
    hll = hyperloglog.HyperLogLog(0.01)  # 精度 1%
    for pwd in passwords:
        hll.add(pwd)
    row["ユニーク数"] = len(hll)
else:
    # 通常のセットで正確にカウント
    unique_passwords = set(passwords)
    row["ユニーク数"] = len(unique_passwords)

「シャノン推定 [bit]」という形で出力される値は、シャノンエントロピー（情報量）をパスワード文字列に対して計算したもの
例：パスワード "abc123"
文字数 = 6
文字の出現確率
a: 1/6
b: 1/6
c: 1/6
1: 1/6
2: 1/6
3: 1/6
H=−6×1/6 ​log2 1/​6​=log2​ 6≈2.585 [bit]

例：パスワード "aaaaaa"
文字数 = 6
文字の出現確率
a: 6/6 = 1
H=−1×log2 ​1=0 [bit]

例：パスワード "password"
文字数 = 8
文字の出現確率
p: 1/8
a: 1/8
s: 2/8
w: 1/8
o: 1/8
r: 1/8
d: 1/8
H≈−(1/8 log2 1/8×6+2/8 log2 2/8)≈2.75 [bit]

1文字あたりのエントロピー:H=−∑ i=1 n pi log2 pi
推定総情報量（ビット数）:Htotal=H×L
L：パスワード長（文字数）
単位：ビット [bit]

1. "abc123"（長さ 6）
1文字あたりのエントロピー: H=log2 6≈2.585 [bit]
総情報量: Htotal=2.585×6≈15.5 [bit]

H（エントロピー値） … パスワード1文字の「多様性」を表す
H × L（総ビット数） … パスワード全体がどれくらい「ランダムに近い」かを評価

128bit の推定情報量があれば「AES-128 の鍵空間と同等」
64bit 程度なら「現実的には総当たりされる可能性がある」


元コード（例）：
# シャノンエントロピー計算（1文字あたりの情報量）
entropy = -sum(p * math.log2(p) for p in char_freq.values())
row["シャノン推定[bit]"] = round(entropy, 4)

変更後：
# シャノンエントロピー計算（1文字あたりの情報量）
entropy_per_char = -sum(p * math.log2(p) for p in char_freq.values())
# 平均パスワード長を掛けて総情報量にする
estimated_info = entropy_per_char * avg_length
row["推定総情報量[bit]"] = round(estimated_info, 4)



nomal-gen
修正後
def gen_sample(model, tokenizer, GEN_BATCH_SIZE, device):
    """1バッチのパスワードを生成"""
    inputs = ""
    tokenizer_forgen_result = tokenizer.encode_forgen(inputs)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokenizer_forgen_result.view([1, -1]).to(device),
            pad_token_id=tokenizer.pad_token_id,
            max_length=MAX_LEN,
            do_sample=True,
            num_return_sequences=GEN_BATCH_SIZE,
        )
    return tokenizer.batch_decode(outputs)

修正後
def gen_parallel(vocab_file, batch_size, test_model_path, N, gen_passwords_path, num_threads):    
    print(f'Load tokenizer.')
    tokenizer = CharTokenizer(vocab_file=vocab_file, 
                              bos_token="<BOS>",
                              eos_token="<EOS>",
                              sep_token="<SEP>",
                              unk_token="<UNK>",
                              pad_token="<PAD>"
                              )
    tokenizer.padding_side = "left"

    device = torch.device("cpu")
    model = GPT2LMHeadModel.from_pretrained(test_model_path)
    model.to(device)
    model.eval()

    total_start = time.time()
    total_round = N // batch_size
    print('*' * 30)
    print(f'Generation begin.')
    print('Total generation needs {} batches.'.format(total_round))

    gen_passwords_path = os.path.join(gen_passwords_path, 'Normal-GEN.txt')
    with open(gen_passwords_path, 'w', encoding='utf-8', errors='ignore') as f_gen:
        for i in range(total_round):
            new_passwords = gen_sample(model, tokenizer, batch_size, device)
            for pw in new_passwords:
                f_gen.write(pw + '\n')
            if (i+1) % 10 == 0:
                print(f'[{i+1}/{total_round}] generated {len(new_passwords)}.')

    total_end = time.time()
    print('Generation file saved in: {}'.format(gen_passwords_path))
    print('Generation done.')
    print('*' * 30)
    print('Use time:{}'.format(total_end - total_start))


