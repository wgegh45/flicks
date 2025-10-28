# This file aims to realize D&C-GEN.
# このファイルは、D＆C-Genを実現することを目的としています。

# 構造パターン（例: 文字種の順序）を事前知識として与えて重複を抑えつつヒット率を高める

from typing import Any
import torch
from transformers import GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList, LogitsProcessorList
from tokenizer import CharTokenizer
import time
import threading
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="directory of pagpassgpt", type=str, required=True)
parser.add_argument("--vocabfile_path", help="path of vocab file", type=str, default='./tokenizer/vocab.json')
parser.add_argument("--pattern_path", help="path of pattern rate file", type=str, default='patterns.txt')
parser.add_argument("--output_path", help="directory of output file path", type=str, required=True)
parser.add_argument("--generate_num", help="total guessing number", default=1000000, type=int)
parser.add_argument("--save_num", help="per n passwords generated save once", default=20000000, type=int)
parser.add_argument("--batch_size", help="generate batch size", default=5000, type=int)
parser.add_argument("--gpu_num", help="gpu num", default=1, type=int)
parser.add_argument("--gpu_index", help="Starting GPU index", default=0, type=int)

args = parser.parse_args()
print(args) # Namespace(model_path='./model/last-step/', vocabfile_path='./tokenizer/vocab.json', pattern_path='patterns.txt', output_path='$output_path', generate_num=10000, save_num=20000000, batch_size=5000, gpu_num=1, gpu_index=0)

BRUTE_DICT = {'L':52, 'N':10, 'S':32}   # L has 52 different letters, N has 10 different numbers and S has 32.
                                        # Lには52種類の文字があり、Nには10種類の数字があり、Sには32種類があります。

# the span of three types adhere to vocab.json
# 3つのタイプのスパンは、vocab.jsonに付着します
TYPE_ID_DICT = {'L':(51, 103),  # A:51 B:52 ・・・ 102:z
                'N':(41, 51),   # 0:41 1:42 ・・・ 9:50
                'S':(103, 135), # !:103 \:104 ・・・ 134:~
                }

model_path = args.model_path     # ./model/last-step/
vocab_file = args.vocabfile_path # ./tokenizer/vocab.json
pattern_file = args.pattern_path # patterns.txt
output_path = args.output_path   # ./generate/

n = args.generate_num          # 1000000
save_num = args.save_num       # 20000000
batch_size = args.batch_size   # 5000
gpu_num = args.gpu_num         # 1
gpu_index = args.gpu_index     # 0

# create new folder to store generation passwords
# 生成パスワードを保存する新しいフォルダーを作成します
output_path = output_path + str(n) + '/'  # ./generate/n/
folder = os.path.exists(output_path)      # フォルダがあるとき、True
if not folder:
    os.makedirs(output_path) # not Falseのとき --output_path=./generate/10000/


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids # [4]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


class SplitBigTask2SmallTask():
    def __init__(self, pcfg_pattern, gen_num, device, tokenizer) -> None:
        self.tasks_list = []
        
        self.pcfg_pattern = pcfg_pattern # N5
        # self.gen_num = gen_num
        self.device = device             # cuda:0
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = tokenizer
        print(tokenizer)
        # CharTokenizer(name_or_path='', vocab_size=135, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<BOS>', 'eos_token': '<EOS>', 'unk_token': '<UNK>', 'sep_token': '<SEP>', 'pad_token': '<PAD>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
        #   AddedToken("<BOS>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        #   AddedToken("<SEP>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        #   AddedToken("<EOS>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        #   AddedToken("<UNK>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        #   AddedToken("<PAD>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        # }
        
        # pcfg_pattern(N5)から[0, 12]を作る L8N1から[0, 21, 16]を作る
        init_input_ids = tokenizer.encode_forgen(pcfg_pattern) 
        print(init_input_ids) # tensor([ 0, 12]) 
        # SEPトークン(1)を追加する
        init_input_ids = torch.concat([init_input_ids, torch.tensor([tokenizer.sep_token_id])]).view(1, -1)
        print(init_input_ids) # tensor([[ 0, 12,  1]])
        
        self.patterns_list = pcfg_pattern.split(' ')
        print(self.patterns_list) # ['N5']
        
        self.type_list = []             # パターンタイプ(N5ならN N N N N)を格納
        for pattern in self.patterns_list:
                char_type = pattern[:1] # パターンの1文字目だけ求める
                print(char_type) # N
                length = pattern[1:]    # パターンの2文字目以降だけ求める
                print(length)    # 5
                for i in range(int(length)): # パターンタイプを格納
                    self.type_list.append(char_type) # N5ならN N N N N
        self.prefix_length = len(self.patterns_list) + 2 # 1+2　　len(N5)
        print(self.prefix_length) # 3
        print(self.patterns_list) # ['N5']
        
        # パターンタイプの最大パスワード数を求める（N5なら100000）
        # judge_gen_num_overflow関数へ        
        max_gen_num = self.judge_gen_num_overflow() # 100000
        print(max_gen_num) # 100000
        print(gen_num) # 143
        
        # N5なら100000までしかないので、それ以上作らないようにする
        if max_gen_num < gen_num: # 100000 < 143
            gen_num = max_gen_num
            
        self.tasks_list.append((init_input_ids, gen_num)) # tensor([[ 0, 12,  1]]) 143
        print(self.tasks_list) # [(tensor([[ 0, 12,  1]]), 143)]
        self.gen_passwords = []

        
    def __call__(self):
        more_gen_num = 0
        while(len(self.tasks_list) != 0): #  3=len(tensor([[ 0, 12,  1]]))　　[(tensor([[ 0, 12,  1, 42]]), 111), (tensor([[ 0, 12,  1, 44]]), 7), (tensor([[ 0, 12,  1, 48]]), 4), (tensor([[ 0, 12,  1, 50]]), 4), (tensor([[ 0, 12,  1, 47]]), 3), (tensor([[ 0, 12,  1, 43]]), 3), (tensor([[ 0, 12,  1, 45]]), 2), (tensor([[ 0, 12,  1, 46]]), 2), (tensor([[ 0, 12,  1, 41]]), 1), (tensor([[ 0, 12,  1, 49]]), 1)]
            print(self.tasks_list)      # [(tensor([[ 0, 12,  1]]), 143)]　　　[(tensor([[ 0, 12,  1, 42]]), 111), (tensor([[ 0, 12,  1, 44]]), 7), (tensor([[ 0, 12,  1, 48]]), 4), (tensor([[ 0, 12,  1, 50]]), 4), (tensor([[ 0, 12,  1, 47]]), 3), (tensor([[ 0, 12,  1, 43]]), 3), (tensor([[ 0, 12,  1, 45]]), 2), (tensor([[ 0, 12,  1, 46]]), 2), (tensor([[ 0, 12,  1, 41]]), 1), (tensor([[ 0, 12,  1, 49]]), 1)]
            print(len(self.tasks_list)) # 10 9 8 ・・・ 1  4
            (input_ids, gen_num) = self.tasks_list.pop() # 確率の低い順に取り出す
            print(input_ids) # tensor([[ 0, 12,  1]])  tensor([[ 0, 12,  1, 49]])  tensor([[ 0, 12,  1, 44, 45]])
            print(gen_num)   # 143 1 1 1
            
            print(input_ids[0])        # tensor([ 0, 12,  1])  tensor([ 0, 12,  1, 49])  tensor([[ 0, 12,  1, 41]])   tensor([ 0, 12,  1, 44, 45])
            print(self.type_list)      # ['N', 'N', 'N', 'N', 'N']
            print(len(input_ids[0]))   # 3 4 4         (tensor([[ 0, 12,  1]]), 143)なら3
            print(self.prefix_length)  # 3(=1+2) 3 3 3
            print(len(self.type_list)) # 5 5 5  ['N', 'N', 'N', 'N', 'N']なら5
            if len(input_ids[0]) == self.prefix_length + len(self.type_list): # 3(0,12,1) == 3(len(N5)+2) + 5(N5)
                self.gen_passwords.append(self.tokenizer.decode(input_ids[0]).split(' ')[1]) # 指定したパターンの文字列以上になっていたら、パスワードを格納する
                print(self.gen_passwords.append) # <built-in method append of list object at 0x0000023BA29E9480>
                                                 # tensor([ 0, 12,  1, 42, 42, 42, 42, 42]) tensor([ 0, 12,  1, 42, 43, 42, 43, 43]) tensor([ 0, 12,  1, 42, 43, 42, 43, 44])
                more_gen_num = gen_num - 1
                print(gen_num) # 1 2 5 6 16 
                continue
            
            print(gen_num)      # 143 1 1 1
            print(more_gen_num) # 0   0 0   0
            gen_num = gen_num + more_gen_num # 143+0
            
            if gen_num <= batch_size:   # <= 5000　　# directly_gen関数へ　パスワードを生成する
                new_passwords = directly_gen(self.tokenizer, self.device, input_ids, gen_num)
                new_passwords_num = len(new_passwords)
                print(new_passwords)     # ['8iO454']
                print(new_passwords_num) # 1
                self.gen_passwords.extend(new_passwords)
                more_gen_num = gen_num - new_passwords_num # 1-1
            else: 
                # get_predict_probability_from_model関数へ
                next_ids, next_probs = self.get_predict_probability_from_model(input_ids.to(self.device)) # tensor([[ 0, 12,  1]])
                print(next_ids)   # tensor([[42, 　　　　44, 　　　48,　　　　 50,　　　 47,　　　　 43,　　　 45,　　　　 46, 　　　　41, 　　　49]])
                print(next_probs) # tensor([[0.7793, 0.0493, 0.0301, 0.0294, 0.0279, 0.0279, 0.0176, 0.0169, 0.0112, 0.0105]])
                
                next_gen_num = next_probs * gen_num # *143 *7  
                filtered_gen_num = next_gen_num[next_gen_num>=1].view(-1,1) # next_gen_numが1未満のものは代入されない（例：数字のとき10から3に減る。生成数が7の場合は残り3つで3,2,1個のパスワードを作る）
                print(filtered_gen_num) # tensor([[111.4414], [  7.0462], ・・・　[  1.5030]])  tensor([[2.4967], [1.7899], [1.2796]])
                
                remain_id_num = len(filtered_gen_num) # tensor([[111.4414], [  7.0462], ・・・　[  1.5030]])
                print(remain_id_num) # 10 3
                
                next_ids = next_ids[:,:remain_id_num]
                print(next_ids)   # tensor([[42, 44, 48, 50, 47, 43, 45, 46, 41, 49]])
                                  # vocab.jsonで置き換えると「1,3,7,9,6,2,4,5,0,8」 確率の降順にしているため
                next_probs = next_probs[:,:remain_id_num]
                print(next_probs) # tensor([[0.7793, 0.0493, 0.0301, 0.0294, 0.0279, 0.0279, 0.0176, 0.0169, 0.0112, 0.0105]])  tensor([[0.3567, 0.2557, 0.1828]])
                
                sum_prob = next_probs.sum() # 確率の総和を求める
                print(sum_prob)   # tensor(1.0000)  tensor(0.7952)
                print(next_probs) # tensor([[0.7793, 0.0493, 0.0301, 0.0294, 0.0279, 0.0279, 0.0176, 0.0169, 0.0112, 0.0105]])  tensor([[0.3567, 0.2557, 0.1828]])
                
                next_probs = next_probs/sum_prob    # /1  /0.7952
                next_gen_num = next_probs * gen_num # *143 *7
                print(next_gen_num) # tensor([[111.4413, 7.0462, 4.2981, 4.2030, 3.9912, 3.9866, 2.5119, 2.4134, 1.6052, 1.5030]])  tensor([[3.1398, 2.2509, 1.6092]])
                
                print(range(remain_id_num)) # range(0, 10)
                for i in range(remain_id_num):
                    new_input_ids = torch.cat([input_ids, next_ids[:,i:i+1]], dim=1)
                    print(new_input_ids) # tensor([[ 0, 12,  1, 42]])  tensor([[ 0, 12,  1, 44]])  tensor([[ 0, 12,  1, 48]])  tensor([[ 0, 12,  1, 50]])  tensor([[ 0, 12,  1, 47]])  tensor([[ 0, 12,  1, 43]])　　tensor([[ 0, 12,  1, 44, 43]])
                    
                    new_gen_num = int(next_gen_num[0][i]) # 生成数　多いものから代入される
                    print(new_gen_num)   # (111 7 4 4 3 3 2 2 1 1) (3,2,1)
                    self.tasks_list.append((new_input_ids, new_gen_num)) # tensor([[ 0, 12,  1, 42]]) 111  tensor([[ 0, 12,  1, 44]]) 7   tensor([[ 0, 12,  1, 44, 43]]) 3
                more_gen_num = 0
        
        return self.gen_passwords



    def get_predict_probability_from_model(self, input_ids): # tensor([[ 0, 12,  1]])
        cur_type = self.type_list[len(input_ids[0])-self.prefix_length] # 3(0,12,1)-3(len(N5)+2)
        print(input_ids[0])   # tensor([ 0, 12,  1], device='cuda:0')  tensor([ 0, 12,  1, 44], device='cuda:0')
        print(cur_type)       # N N
        with torch.no_grad(): # 1回しか下の処理をしない
            output = self.model(input_ids=input_ids) ###### 次に来るトークン候補の確率を求める（正確にはすべてのトークンの確率、その後にnext_token_logitsで「直前の入力に続く次トークン候補のスコア」を取り出している） ###########
            print(output) # ・・・ [-0.7068, -0.3433,  0.1185,  ...,  0.0705, -0.0574,  0.0297]]]], device='cuda:0'))), hidden_states=None, attentions=None, cross_attentions=None)
            
            next_token_logits = output.logits[:, -1, :]
            print(next_token_logits) # tensor([[-2.1185e-01, -1.7857e-01,  1.3675e+00, ・・・ -6.3891e-01, -8.3588e-02]], device='cuda:0')
            
            type_id_pair = TYPE_ID_DICT[cur_type] # A:51・・・102:z  0:41・・・9:50  !:103・・・134:~
            print(type_id_pair) # (41, 51)(数字なので)

            selected_logits = next_token_logits[:, type_id_pair[0]:type_id_pair[1]]
            print(selected_logits)  # (数字なので10個の要素) tensor([[-0.2170, 4.0232, 0.6927, 1.2622, 0.2308, 0.1908, 0.6938, 0.7679, -0.2828, 0.7455]], device='cuda:0')
            # softmaxで値の総和が1になるよう調整
            selected_softmax = torch.softmax(selected_logits, dim=-1)
            print(selected_softmax) # tensor([[0.0112, 0.7793, 0.0279, 0.0493, 0.0176, 0.0169, 0.0279, 0.0301, 0.0105, 0.0294]], device='cuda:0')
                                    #          0       1       2       3       4       5       6       7       8       9
            sorted_indices = torch.argsort(selected_softmax, descending=True, dim=-1) # 値を降順にしたときのインデックスを返す
            print(sorted_indices)   # tensor([[1, 3, 7, 9, 6, 2, 4, 5, 0, 8]], device='cuda:0')  tensor([[2, 1, 4, 3, 8, 5, 9, 0, 7, 6]], device='cuda:0')
            # 数字のインデックスの値に合わせる
            sorted_indexes = sorted_indices + type_id_pair[0] # +41
            print(sorted_indexes)   # tensor([[42, 44, 48, 50, 47, 43, 45, 46, 41, 49]], device='cuda:0')  tensor([[43, 42, 45, 44, 49, 46, 50, 41, 48, 47]], device='cuda:0')
            # 値を降順に並び替えて返す
            sorted_softmax = selected_softmax[:, sorted_indices[0]]
            print(sorted_softmax)       # tensor([[0.7793, 0.0493, 0.0301, 0.0294, 0.0279, 0.0279, 0.0176, 0.0169, 0.0112, 0.0105]], device='cuda:0')  tensor([[0.3567, 0.2557, 0.1828, 0.0980, 0.0510, 0.0247, 0.0125, 0.0079, 0.0072, 0.0036]], device='cuda:0')
            print(sorted_softmax.cpu()) # tensor([[0.7793, 0.0493, 0.0301, 0.0294, 0.0279, 0.0279, 0.0176, 0.0169, 0.0112, 0.0105]])                   tensor([[0.3567, 0.2557, 0.1828, 0.0980, 0.0510, 0.0247, 0.0125, 0.0079, 0.0072, 0.0036]])
            
            return sorted_indexes.cpu(), sorted_softmax.cpu() # tensor([[42,     44,     48,     50,     47,     43,     45,     46,     41,     49]]
                                                              # tensor([[0.7793, 0.0493, 0.0301, 0.0294, 0.0279, 0.0279, 0.0176, 0.0169, 0.0112, 0.0105]])
    
    # パターンタイプの最大パスワード数を求める（N5なら100000）
    def judge_gen_num_overflow(self) -> int:
        total = 1
        for _ in self.type_list:
            total = total * BRUTE_DICT[_] # BRUTE_DICT={'L': 52, 'N': 10, 'S': 32}
            print(total) # 10 100 1000 10000 100000
        return total # N5なら100000
    

# パスワードを生成する
def directly_gen(tokenizer, device, input_ids, gen_num): # device=cuda:0 input_ids=tensor([ 0, 12,  1]) gen_num=143 148
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device) # 学習済みGPT-2系モデルを読み込み、指定されたGPU/CPU(device)に載せる
    print(model)
    '''
    GPT2LMHeadModel(
     (transformer): GPT2Model(
       (wte): Embedding(135, 384)
       (wpe): Embedding(32, 384)
       (drop): Dropout(p=0.1, inplace=False)
       (h): ModuleList(
         (0-11): 12 x GPT2Block(
           (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
          )
        )
        (ln_f): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=384, out_features=135, bias=False)
    )
    '''
    passwords = []

    stop_ids = [tokenizer.pad_token_id]  # パディングトークン<PAD>の4を代入
    print(stop_ids) # [4]
    
    # 停止条件を作成　 KeywordsStoppingCriteriaクラスへ  StoppingCriteria を自作して、特定のトークンが出たら生成を止める(トークンID = 4 が出たら生成を停止)
    stop_criteria = KeywordsStoppingCriteria(stop_ids)  
    print(stop_criteria) # <__main__.KeywordsStoppingCriteria object at 0x0000022121441650>  <__main__.KeywordsStoppingCriteria object at 0x0000028D5A937B10>  0x0000028D5A937B10はこのオブジェクトが格納されている メモリアドレス
    print(tokenizer.pad_token_id) # 4
    
    # パスワードの元となるトークンIDを生成する
    outputs = model.generate( 
        input_ids= input_ids.view([1,-1]).to(device), # トークン列を [batch_size=1, sequence_length] の形に変換。-1は自動的に長さを推定を表す。GPU (cuda) に転送
        pad_token_id=tokenizer.pad_token_id,          # 4
        stopping_criteria=StoppingCriteriaList([stop_criteria]), # 生成を打ち切る条件を設定
        max_new_tokens=13,                            # 生成する最大トークン数。
        do_sample=True,                               # サンプリングを有効化(Greedy (確率最大の1語) ではなく、確率分布からランダムに選び、多様性のある生成にする)
        num_return_sequences=gen_num,                 # 一度にgen_num個の生成結果を出力
        )
    print(outputs) # outputsのサイズは[143, 15] [148, 15] 
    # tensor([[ 0, 12,  1,  ...,  8,  2,  4],  <BOS> N12 SEP ... N9 EOS PAD
    #         [ 0, 12,  1,  ..., 47,  2,  4],  <BOS> N12 SEP ... 6  EOS PAD
    #         [ 0, 12,  1,  ..., 81,  2,  4],  <BOS> N12 SEP ... e  EOS PAD
    #         ..., 
    #         [ 0, 12,  1,  ..., 42,  2,  4],
    #         [ 0, 12,  1,  ...,  2,  4,  4],
    #         [ 0, 12,  1,  ...,  4,  4,  4]], device='cuda:0')
    # tensor([[ 0, 21, 16,  ..., 44,  2,  4],
    #         [ 0, 21, 16,  ...,  4,  4,  4],
    #          ・・・
    #         [ 0, 21, 16,  ...,  4,  4,  4]], device='cuda:0')
    
    # char_tokenizer.pyのbatch_decode関数へ　トークンIDを文字列に変換する
    outputs = tokenizer.batch_decode(outputs) 
    print(outputs) # ['N5 s87neN9', 'N5 L11MN8456, ・・・ , 'N5 fron']  ['L8N1 lonesa!23', 'L8N1 lo45665', ・・・ , 'L8N1 ^inN3iS10']
  
    # パスワードだけを1つずつpasswordsに格納
    for output in outputs: 
        passwords.append(output.split(' ')[1])
    passwords = set(passwords) # 重複を排除、順序は保証されない
    
    print(passwords)      # 下と同じ　{'', 's!ettC', 'elomie'　・・・ su12235'}
    print([*passwords,])  # 上と同じ
    return [*passwords,] # return passwordsと同じ  生成したパスワードを返す
        

# パスワードを生成してファイルに書き込む
def single_gpu_task(task_list, gpu_id, tokenizer): # task_list=[('L6', 4388), ('L7', 1942), ('L8', 1366), ('N6', 791), ('L9', 647), ('L5', 287), ('N9', 143), ('L10', 143), ('L8 N1', 143), ('N5', 143)]  gpu_id=0
    gened_passwords = []
    output_count = 1
    finished_task_count = 0
    total_task_num = len(task_list)  # 10  パターンと生成数の組は10個
    print(total_task_num) # 10
    more_gen_num = 0
    
    # 下の処理を行うごとにtask_listは1減る。task_listは後ろのものから使われる
    while(len(task_list) != 0):  # 10 9 8 ・・・ 0 != 0  
        (pcfg_pattern, num) = task_list.pop() # N5 143 　 L8 N1 148
        print((pcfg_pattern, num)) # ('N5', 143)　('L8 N1', 143) ・・・ 
        
        num = num + more_gen_num   # 不足した分をmore_gen_numで作る　143+0  143+5  143+2
        print(num) # 143 148
        
        print(f'[{finished_task_count}/{total_task_num}] cuda:{gpu_id}\tGenerating {pcfg_pattern}: {num}') 
        # [0/10] cuda:0	Generating N5: 143  
        # [1/10] cuda:0	Generating L8 N1: 148
        
        if num <= batch_size: # <= 5000 
            # 通常はこっち
            # char_tokenizer.pyのencode_forgen関数へ  pcfg_pattern(N5)から[ 0, 12]を作る L8N1から[0, 21, 16]を作る
            input_ids = tokenizer.encode_forgen(pcfg_pattern) 
            print(input_ids) # tensor([ 0, 12])  tensor([ 0, 21, 16]) ・・・ 
           
            # 区切りトークン(SEP)の1を追加
            input_ids = torch.concat([input_ids, torch.tensor([tokenizer.sep_token_id])]) # tensor([ 0, 12])  tensor([ 0, 21, 16])  
            print(input_ids) # tensor([ 0, 12,  1])  tensor([ 0, 21, 16,  1]) 
            
            # directly_gen関数へ　パスワードを生成する
            new_passwords = directly_gen(tokenizer, 'cuda:'+str(gpu_id), input_ids, num) # gpu_id=0 input_ids=tensor([ 0, 12,  1]) num=143　  tensor([ 0, 21, 16,  1]) num=148
            print(new_passwords) # ['', 's!ettC', 'elomie'　・・・ su12235']
        else: # SplitBigTask2SmallTaskクラスへ
            split2small = SplitBigTask2SmallTask(pcfg_pattern=pcfg_pattern,  # N5
                                                 gen_num=num,                # 143
                                                 device='cuda:'+str(gpu_id), # + str(0)
                                                 tokenizer=tokenizer)
            print(split2small) # <__main__.SplitBigTask2SmallTask object at 0x0000021A7B30B650>
            new_passwords = split2small() # __call__クラスへ
        
        gened_num = len(new_passwords) # 重複の排除でパスワードの数が減ってる　138 146
        print(gened_num) # 138 146
        
        more_gen_num = num - gened_num # 143-138=5 148-146=2
        print(num) # 143 148
        
        gened_passwords.extend(new_passwords) # パスワードをgened_passwordsに追加する
        print(gened_passwords) # ['', 's!ettC', 'elomie'　・・・ su12235']
        
        finished_task_count += 1 # 0+1 1+1
        print(f'[{finished_task_count}/{total_task_num}] cuda:{gpu_id}\tActually generated {pcfg_pattern}: {gened_num}\t(diff {num-gened_num})')
        # [1/10] cuda:0	Actually generated N5: 138	(diff 5)    
        # [2/10] cuda:0	Actually generated L8 N1: 146	(diff 2)
        # diffは重複排除でヘッダ数 # save_numよりもgened_passwordsが何倍も多い場合はsave_numより小さくなるまで繰り返し実行される
        while len(gened_passwords) > save_num: # > 20000000
            output_passwords = gened_passwords[:save_num] # 1-20000000個のパスワードを格納
            print(output_passwords)
            
            file_path = output_path +'DC-GEN-[cuda:'+ str(gpu_id) + ']-'+str(output_count)+'.txt'
            print(file_path) # --output_path=./generate/10000/DC-GEN-[cuda:0]-1.txt
            
            f = open(file_path, 'w', encoding='utf-8', errors='ignore')
            for password in output_passwords: # パスワードを１つずつファイルに書き込む
                f.write(password+'\n')
                print(password)
            f.close()
            output_count += 1
            print(output_count)
            
            gened_passwords = gened_passwords[save_num:] # 20000001個以降のパスワードを格納
            print(gened_passwords)
            print(f'===> File saved in {file_path}.')

    print(len(gened_passwords)) # 9919 
    
    if len(gened_passwords) != 0:
        # パスワードの出力先のパスを作る
        file_path = output_path + 'DC-GEN-[cuda:'+ str(gpu_id) + ']-last.txt'
        print(file_path) # --output_path=./generate/10000/DC-GEN-[cuda:0]-last.txt
       
        # 生成したパスワードを1つずつファイルに書き込む
        f = open(file_path, 'w', encoding='utf-8', errors='ignore') 
        for password in gened_passwords:
            f.write(password+'\n')
            print(password)
        f.close()
        
        print(f'===> File saved in {file_path}.')



# 各パターンのパスワード生成個数を求める
def prepare_task_list(df, gpu_num): # 1
    threshold = 100
#    threshold = 10

    threshold_rate = threshold/n # 100/10000=0.01 100/1000000  10/100=0.1
    
    # pattern.txtの確率(rate)がthreshold_rate以上のものだけ格納する
    filtered_df = df[df['rate'] >= threshold_rate] # >= 0.1  
    print(filtered_df) # pattern.txtの確率(rate)がthreshold_rate以上のものだけ表示される  
    #                    パスワードの生成数が少ない場合は、より確率の高いパターンだけ使う
    # Empty DataFrame
    # Columns: [pattern, rate]
    # Index: []
#    pattern      rate
# 0       L6  0.426573
# 1       L7  0.188811
# 2       L8  0.132867
# 3       N6  0.076923
# 4       L9  0.062937
# 5       L5  0.027972
# 6       N9  0.013986
# 7      L10  0.013986
# 8    L8 N1  0.013986
# 9       N5  0.013986
   
    # threshold_rate以上の確率のパターンについて、総和を求める
    sum_rate = filtered_df['rate'].sum() 
    print(sum_rate) # 0.9720279720279713
    print(filtered_df['rate'])         # Series([], Name: rate, dtype: object)
    # 0    0.426573　　
    # 1    0.188811
    # ・・・　
    # 9    0.013986
    # Name: rate, dtype: float64
    
    # フィルタされたpattern.txtの確率(rate)/0.9720279720279713をした確率（softmax_rate）を求める（確率の総和が1になるよう調整）
    filtered_df['softmax_rate'] = filtered_df['rate']/sum_rate  
    print(filtered_df['softmax_rate']) # Series([], Name: softmax_rate, dtype: object)  
    # 0    0.438849  
    # 1    0.194245　
    # ・・・ 
    # 9    0.014388　 
    # Name: softmax_rate, dtype: float64
    
    #　初期化
    total_gpu_tasks = []
    for i in range(gpu_num): # 1
        total_gpu_tasks.append([]) # print(total_gpu_tasks) = [[]]
        
    # フィルタされたpattern.txtのパターンと確率を降順に1つずつ読み取り、各パターンの生成数を読み取る
    turn = 0
    for row in filtered_df.itertuples(): 
        print(row) # Pandas(Index=0, pattern='L6', rate=0.4265734265734265, softmax_rate=0.4388489208633096)  Pandas(Index=1, pattern='L7', rate=0.1888111888111888, softmax_rate=0.1942446043165469) ・・・ 
        pcfg_pattern = row[1]
        print(pcfg_pattern) # L6  L7  L8  N6  L9  L5  N9  L10  L8 N1  N5
        # 当該パターンの生成数を求める（softmax_rate×生成数）
        num = int(row[3]*n) # * 10000
        print(num) # 4388 1942 1366 791 647 287 143 143 143 143
  
        total_gpu_tasks[turn].append((pcfg_pattern, num))
        print(total_gpu_tasks[turn]) # [('L6', 4388)]  [('L6', 4388), ('L7', 1942)] ・・・ [('L6', 4388), ('L7', 1942), ('L8', 1366), ('N6', 791), ('L9', 647), ('L5', 287), ('N9', 143), ('L10', 143), ('L8 N1', 143), ('N5', 143)]
  
        turn = (turn + 1) % gpu_num # % 1
        print(turn) # 0 0  ・・・ 0

    print(total_gpu_tasks) # [[('L6', 4388), ('L7', 1942), ('L8', 1366), ('N6', 791), ('L9', 647), ('L5', 287), ('N9', 143), ('L10', 143), ('L8 N1', 143), ('N5', 143)]]
    return total_gpu_tasks


if __name__ == "__main__":
    begin_time = time.time() # 処理の開始時刻を取得 1756552511.2296484

    print(f'Load tokenizer.') # char_tokenizer.pyのCharTokenizerの__init__関数へ
    tokenizer = CharTokenizer(vocab_file=vocab_file,   # ./tokenizer/vocab.json
                                    bos_token="<BOS>",
                                    eos_token="<EOS>",
                                    pad_token="<PAD>",
                                    sep_token="<SEP>",
                                    unk_token="<UNK>"
                                    )
    print(tokenizer)
    # CharTokenizer(name_or_path='', vocab_size=135, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<BOS>', 'eos_token': '<EOS>', 'unk_token': '<UNK>', 'sep_token': '<SEP>', 'pad_token': '<PAD>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
    	#0: AddedToken("<BOS>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	    #1: AddedToken("<SEP>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		#2: AddedToken("<EOS>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		#3: AddedToken("<UNK>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		#4: AddedToken("<PAD>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    #}
    tokenizer.padding_side = "left"
    # CharTokenizer(name_or_path='', vocab_size=135, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<BOS>', 'eos_token': '<EOS>', 'unk_token': '<UNK>', 'sep_token': '<SEP>', 'pad_token': '<PAD>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
		#0: AddedToken("<BOS>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		#1: AddedToken("<SEP>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		#2: AddedToken("<EOS>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		#3: AddedToken("<UNK>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		#4: AddedToken("<PAD>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    #}

    print(f'Load patterns.') 
    # pattern.txtを読み込む
    df = pd.read_csv(pattern_file, sep='\t', header=None, names=['pattern', 'rate']) # pattern.txt
    print(df)
    # Empty DataFrame
    #Columns: [pattern, rate]
    #Index: []
    # pattern      rate
    # 0       L6  0.426573
    # 1       L7  0.188811
    # ・・・ 
    # 13     N10  0.006993　
    
    
    # prepare_task_list関数へ（# 各パターンのパスワード生成個数を求める）
    total_task_list = prepare_task_list(df, gpu_num) # gpu_num=1
    print(total_task_list) # [[('L6', 4388), ('L7', 1942), ('L8', 1366), ('N6', 791), ('L9', 647), ('L5', 287), ('N9', 143), ('L10', 143), ('L8 N1', 143), ('N5', 143)]]

    # multi threading
    threads = []
    print('*'*30) # ******************************
    print(f'Generation begin.')
    for i in range(gpu_num): # 1
    
        ##############################
        # 検証のため追加   
        single_gpu_task(total_task_list[i], i+gpu_index, tokenizer) # [('L6', 4388), ('L7', 1942), ('L8', 1366), ('N6', 791), ('L9', 647), ('L5', 287), ('N9', 143), ('L10', 143), ('L8 N1', 143), ('N5', 143)]  0+0 
        ############################## 
        
        # # パスワードを生成してファイルに書き込む
        thread = threading.Thread(target=single_gpu_task, args=[total_task_list[i], i+gpu_index, tokenizer])
        print(thread) # <Thread(Thread-8 (single_gpu_task), initial)>
        thread.start()
        threads.append(thread)
        print(thread) # <Thread(Thread-8 (single_gpu_task), stopped 18416)>
    
    for t in threads: # 1  threads は threading.Thread オブジェクトを格納したリスト
        t.join() # 全部のスレッドが終わるまでメイン処理を止める
        print(t) # <Thread(Thread-8 (single_gpu_task), stopped 18416)>
    
    end_time = time.time() # 処理の終了時刻を取得
    print('Generation done.')
    print('*'*30) # ******************************
    print(f'Use time: {end_time-begin_time}') # Use time: 954.6592266559601