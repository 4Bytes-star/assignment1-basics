import os
import heapq
from collections import Counter, defaultdict
from multiprocessing import Pool
import regex as re
from .pretokenization_example import find_chunk_boundaries

PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def _pretokenize_chunk(chunk_text:str) -> Counter:
    """
    辅助函数：处理一个chunk，返回预token的字节序列计数
    这里是已经不包含特殊token的chunk了
    """
    # 1.pretokenize
    pre_tokens = []
    for match in re.finditer(PRETOKENIZE_PATTERN,chunk_text):
        text = match.group(0)
        # 转为UTF-8(bytes)
        byte_seq = text.encode('utf-8')
        if byte_seq:
            pre_tokens.append(byte_seq)

    # 2.statistics
    token_counts = Counter()
    for token in pre_tokens:
        # 将 bytes 对象转为 tuple 以便作为 dict key (例如 b'hello' -> (104, 101, ...))
        # 注意：Python 中 bytes 本身是可哈希的，可以直接作为 key，但合并操作时 tuple 更方便索引
        tuple_token = tuple(bytes([b]) for b in token)
        token_counts[tuple_token] += 1
    
    return token_counts

def process_chunk(args):
    """
    Worker 函数：模块顶层定义，用于多进程并行处理。
    """
    start, end, input_path, special_tokens = args
    # 将 bytes 形式的 special tokens 解码为字符串，用于正则切分
    special_tokens_str = [st.decode("utf-8", errors="ignore") for st in special_tokens]
    
    # 1. 读取文件块
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    
    # 2. 解码
    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    
    # 3. 如果没有特殊 token，直接预处理并返回
    if not special_tokens:
        return _pretokenize_chunk(chunk_text)
    # 4. 构造正则 pattern
    # 使用 re.escape 确保特殊 token 中的 <, |, ( 等字符被正确转义
    # 使用 () 包裹每个 token，这样 re.split 会保留匹配到的特殊 token
    escaped_tokens = [f"({re.escape(t)})" for t in special_tokens_str]
    pattern = "|".join(escaped_tokens)
    
    # 5. 切分文本并处理
    # re.split 会返回包含文本和特殊 token 的列表，例如: ["文本1", "特殊tokenA", "文本2"]
    parts = re.split(pattern, chunk_text)
    
    # 6. 合并所有部分的计数
    # 过滤掉空字符串（例如两个特殊 token 相邻时产生的空串）
    parts = [p for p in parts if p]
    
    local_counter = Counter()
    for part in parts:
        # 跳过 special token 本身，只预处理普通文本部分
        if any(part == st for st in special_tokens_str):
            continue
        local_counter.update(_pretokenize_chunk(part))
        
    return local_counter

def _get_token_counts_from_file(input_path: str, special_tokens: list[bytes]) -> dict[tuple[bytes, ...], int]:
    """
    并行处理文件，按特殊 token 切分并统计 token 频率。
    """
    # 1. 找到切分点（基于字节位置，保证多进程读取不重叠）
    # 注意：这里的 boundaries 只是为了把大文件切成几块给不同进程，
    # 具体的特殊 token 切分逻辑已经下沉到 process_chunk 里了
    with open(input_path, "rb") as f:
        num_processes = 4
        # 假设 find_chunk_boundaries 是你之前写的按字节长度粗略切分或按换行符切分的函数
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0])
    
    # 2. 准备任务列表
    # 【关键修改】每个任务元组现在包含 4 个元素，加上 special_tokens
    chunks = [
        (boundaries[i], boundaries[i+1], input_path, special_tokens) 
        for i in range(len(boundaries) - 1)
    ]
    
    # 3. 并行执行
    with Pool(min(num_processes, len(boundaries) - 1)) as pool:
        # pool.map 会自动将 chunks 里的每个元组传给 process_chunk 的 args 参数
        local_counters = pool.map(process_chunk, chunks)
        
    # 4. 合并所有进程的 Counter 结果
    global_token_counts = Counter()
    for counter in local_counters:
        global_token_counts.update(counter)
        
    return dict(global_token_counts)

def train_bpe(input_path: str,vocab_size: int,special_tokens: list[str]) -> tuple:
    """
    train bpe tokenizer
    
    Args:
        input_path: 文本文件路径
        vocab_size: 最终词汇表大小（包含初始 256 字节和特殊 token）
        special_tokens: 特殊 token 列表
        
    Returns:
        vocab: dict[int, bytes] 词汇表
        merges: list[tuple[bytes, bytes]] 合并历史
    """
    # 1. initialize the vocab
    # the initial vocab has all the ascii 
    vocab = {i:bytes([i]) for i in range(256)}
    for st in special_tokens:
        if st not in vocab.values():
            next_id = len(vocab)
            vocab[next_id] = st.encode('utf-8')
    
    merges = []

    # 2.预处理与并行化读取
    encoded_special_tokens = [st.encode('utf-8') for st in special_tokens]
    pre_token_counts = _get_token_counts_from_file(input_path, encoded_special_tokens)
    #pre_token_counts = _get_token_counts_from_file(input_path, special_tokens[0].encode()) 

    # 3.构建初始对（Pair）计数
    pair_counts = Counter()
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    # 遍历所有pre_token以及其频率
    for token_bytes,token_count in pre_token_counts.items():
        # 如果token只有一个字节，没有pair，跳过
        if len(token_bytes) < 2:
            continue
            
        # 遍历token内部每一对相邻字节
        for i in range(len(token_bytes)-1):
            pair = token_bytes[i:i+2]
            pair_counts[pair] += token_count
            
            # 更新pair_to_tokens
            if pair not in pair_to_tokens:
                pair_to_tokens[pair] = set()
            pair_to_tokens[pair].add(token_bytes)
            
    def _get_best_pair(pair_counts: Counter[tuple[bytes, bytes]]):
        # 从Counter类型中找到频率最高的pair，需要性能好
        if not pair_counts:
            return None
        
        # 使用max，key为(Count,pair)
        return max(pair_counts,key = lambda p: (pair_counts[p],p)) 
        
    def merge_token(token:tuple[bytes,...],best_pair:tuple[bytes,bytes],new_bytes:bytes)->tuple[bytes,...]:
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token)-1 and token[i] == best_pair[0] and token[i+1] == best_pair[1]:
                new_token.append(new_bytes)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        return tuple(new_token)

    # 4.合并 
    while len(vocab) < vocab_size:
        # 1.找到频率最高的pair
        best_pair = _get_best_pair(pair_counts)
        
        if best_pair is None:
            break
        
        # 2.合并 并写入vocab中
        b_left, b_right = best_pair
        new_bytes = b_left + b_right
        # 记录合并历史
        merges.append((b_left,b_right))
        next_id = len(vocab)
        vocab[next_id] = new_bytes

        # 3.更新tokens和pair_counts
        # 3.1 获取所有包含 best pair的token
        affected_tokens = pair_to_tokens.pop(best_pair, set()) #移除旧索引，因为 best_pair 不再作为独立 pair 存在
        
        # 临时字典，用于更新pre_token_counts
        new_pre_token_counts_updates = defaultdict(int)
        
        for old_token in affected_tokens:
            count = pre_token_counts.get(old_token,0)
            if count == 0:
                continue
            
            if len(old_token) >= 2:
                for i in range(len(old_token) - 1):
                    p = (old_token[i],old_token[i+1])
                    pair_counts[p] -= count
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                        
                    if p in pair_to_tokens:
                        pair_to_tokens[p].discard(old_token)
                        if not pair_to_tokens[p]:
                            del pair_to_tokens[p]
            new_token = merge_token(old_token, best_pair, new_bytes)

            new_pre_token_counts_updates[new_token] += count
            
        # --- 统一应用 pre_token_counts 的更新 ---
        for old_token in affected_tokens:
            # 只有当它还在字典里时才 pop (防止一个 token 包含多个不同的 affected_pair 导致重复 pop)
            if old_token in pre_token_counts:
                del pre_token_counts[old_token]
                
        for new_token, count in new_pre_token_counts_updates.items():
            pre_token_counts[new_token] = pre_token_counts.get(new_token, 0) + count
            
            # 【关键】为新 token 建立 pair_counts 和 pair_to_tokens 索引
            if len(new_token) >= 2:
                for i in range(len(new_token) - 1):
                    p = (new_token[i], new_token[i+1])
                    pair_counts[p] += count
                    
                    if p not in pair_to_tokens:
                        pair_to_tokens[p] = set()
                    pair_to_tokens[p].add(new_token)
    
    return vocab,merges