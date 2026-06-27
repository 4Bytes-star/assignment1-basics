import time, tracemalloc
import json
from cs336_basics.bpe_tokenizer import train_bpe

tracemalloc.start()
start = time.time()

vocab, merges = train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    #input_path="data/tinystories_small.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)

elapsed_time = time.time() - start
current, peak = tracemalloc.get_traced_memory()

# vocab: dict[int, bytes] -> json
with open("data/vocab.json", "w") as f:
    json.dump({k:v.decode("latin-1") for k,v in vocab.items()}, f)

# merges: list[tuple[bytes, bytes]] -> human read
with open("data/merges.txt", "w") as f:
    for a, b in merges:
        f.write(f"{a.decode('latin-1')} {b.decode('latin-1')}\n")

longest_id = max(vocab, key=lambda k: len(vocab[k]))
longest_bytes = vocab[longest_id]
longest_str = longest_bytes.decode("utf-8",errors="replace")

print(f"time_used: {elapsed_time:.2f}s")
print(f"peak memory:{peak / 1024 / 1024:.1f} MB")
print(f"the longest token's len:{len(longest_bytes)} ,the longest token'str:" + longest_str)