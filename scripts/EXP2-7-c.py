import os
import time
from cs336_basics.bpe_tokenizer import BPE


benchmark_text = "data/TinyStoriesV2-GPT4-valid.txt"
tokenizer = BPE.from_files("data/vocab.json", "data/merges.txt", ["<|endoftext|>"])

bytesOfFile = os.path.getsize(benchmark_text)

with open(benchmark_text, "r", encoding='utf-8') as f:
    content = f.read()
    
time1 = 0
for i in range(3):
    startTime1 = time.time()
    # 一次性读取再编码
    token_ids1 = tokenizer.encode(content)
    end_time1 = time.time()
    time1 += end_time1 - startTime1

time2 = 0
for i in range(3):
    startTime2 = time.time()
    with open(benchmark_text, "r", encoding='utf-8') as f:
        list(tokenizer.encode_iterable(f))

    end_time2 = time.time()
    time2 += end_time2 - startTime2

throughput1 = bytesOfFile / (time1/3.0) / 1024**2
throughput2 = bytesOfFile / (time2/3.0) / 1024**2

pile_time = 825 * 1024 / throughput2
print(f"一次性encode花费时间： {time1 / 3.0 :3f} s, throughput: {throughput1} MB/s")
print(f"流式encode花费时间： {time2 / 3.0 :3f} s, throughput: {throughput2} MB/s")
print(f"估算pile花费时间（使用流式编码）：{pile_time} s")