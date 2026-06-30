import os
import numpy as np
import array
from itertools import islice
from cs336_basics.bpe_tokenizer import BPE

valid_txt = "data/TinyStoriesV2-GPT4-valid.txt"
train_txt = "data/TinyStoriesV2-GPT4-train.txt"
tokenizer = BPE.from_files("data/vocab.json", "data/merges.txt", ["<|endoftext|>"])

batch_size = 1000000
valid_tokens_output_path = "data/valid_tokens.bin"
train_tokens_output_path = "data/train_tokens.bin"

with open(valid_txt, "r", encoding='utf-8') as f:
    valid_token_stream = tokenizer.encode_iterable(f)

    with open(valid_tokens_output_path, "wb") as w:
        while True:
            batch = list(islice(valid_token_stream, batch_size))
            if not batch:
                break
            
            np.array(batch, dtype=np.uint16).tofile(w)

# arr = np.fromfile("data/valid_tokens.bin", dtype=np.uint16)
# with open(valid_txt, "r") as f:
#     original = f.read(200)
#     decoded = tokenizer.decode(arr[:100].tolist())
#     print("原文开头:", original[:100])
#     print("解码开头:", decoded[:100])

with open(train_txt, "r", encoding='utf-8') as f:
    train_token_stream = tokenizer.encode_iterable(f)
    
    with open(train_tokens_output_path, "wb") as w:
        while True:
            batch = list(islice(train_token_stream, batch_size))
            
            if not batch:
                break
            
            np.array(batch, dtype=np.uint16).tofile(w)