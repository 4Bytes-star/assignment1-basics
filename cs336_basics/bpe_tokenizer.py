import os
import heapq
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
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
        num_processes = 8
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

def train_bpe(input_path: str,vocab_size: int,special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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

    # 使用heapq堆来优化到 O(logn)
    class PairObj:
        def __init__(self, count, pair):
            self.count = count
            self.pair = pair
        
        def __lt__(self, other):
            if self.count != other.count:
                return self.count > other.count # count大优先
            else:
                return self.pair > other.pair
    pq = []
    for p,c in pair_counts.items():
        heapq.heappush(pq, PairObj(c,p))

    # 4.合并 
    while len(vocab) < vocab_size:
        # 1.找到频率最高的pair
        best_pair = None
        while pq:
            top = heapq.heappop(pq)
            # 懒删除检查：因为每次取出最大的pair进行合并之后，别的包含这个pair其中元素的pair也需要跟着变
            # 这里我们选择懒更新，每次取出来之后，只有当其等于真实的count时才是有效的，否则删除
            if top.count == pair_counts.get(top.pair,0): # pair_counts里面的元素都是最新状态
                best_pair = top.pair
                break
        
        if best_pair is None or pair_counts[best_pair] < 1:
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
                        # 懒删除，只删除真实计数器，堆内等pop出来比对一下是否有效即可
                    else:
                        # 只要这个pair还有剩余，就得把新的状态压入堆内
                        heapq.heappush(pq, PairObj(pair_counts[p], p))
                        
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
                    # 更新堆内的pair的count数
                    heapq.heappush(pq, PairObj(pair_counts[p], p))

                    if p not in pair_to_tokens:
                        pair_to_tokens[p] = set()
                    pair_to_tokens[p].add(new_token)
    
    return vocab,merges

class BPE:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        vocab: 初始词汇表{index: character}
        merges:BPE合并规则表 [(subword1,subword2),...]
        special_tokens:特殊token
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        从序列化文件构建并返回tokenizer
        """
        import json
        
        # 1.加载vocab.json
        vocab = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
            for k, v in vocab_json.item():
                vocab[int(k)] = v.encode("latin-1")
        
        # 2.加载merges
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.rstrip("\n")
                if not clean_line:
                    continue
                parts = clean_line.rsplit(" ", 1)
                if len(parts) == 2:
                    p1 = parts[0].encode("latin-1")
                    p2 = parts[1].encode("latin-2")
                    merges.append((p1, p2))
        
        # 3.构建并返回BPE实例
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def decode(self, ids: list[int]) -> str:
        """
        将token转为str
        """
        # 建立 id -> token 的反向映射
        bytes_tokens = [self.vocab.get(id) for id in ids]    
        combined_bytes = b"".join(bytes_tokens)
    
        return combined_bytes.decode('utf-8', errors='replace')
    
    def encode(self, text: str) -> list[int]:
        """
        将文本文件转换成token id列表
        1. special_token切分文档
        2. pretokenize
        3. emerge
        """
        bpe_tokens = []
        
        # 构造正则pattern
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [f"({re.escape(t)})" for t in sorted_special_tokens]
            pattern = "|".join(escaped_tokens)

            # 切分文档并处理
            parts = re.split(pattern, text)
        else:
            parts = [text]

        bytes_to_id = {v: k for k,v in self.vocab.items()}
        rank = {pair: i for i, pair in enumerate(self.merges)}
        
        for part in parts:
            if not part:
                continue
            if self.special_tokens and part in self.special_tokens:
                # 找到special token的id
                spec_token_bytes = part.encode('utf-8')
                bpe_tokens.append(bytes_to_id.get(spec_token_bytes))
            else:
                # 普通文本进行pretokenize
                # 1. 用PRETOKENIZE_PATTERN来切分出pretokens
                for match in re.finditer(PRETOKENIZE_PATTERN, part):
                    word_bytes = match.group(0).encode('utf-8')

                    # 2. 变成单字节列表
                    word_list = [bytes([b]) for b in word_bytes]
                    
                    # 3. 循环合并
                    while len(word_list) >= 2:
                        # word并不是很长，先直接遍历所有pair
                        min_pos = len(word_list)
                        min_id = len(self.merges)
                        for i in range(len(word_list) - 1):
                            pair = (word_list[i], word_list[i+1])
                            pair_id = rank.get(pair) # 得到rankd_id
                            if pair_id is not None and pair_id < min_id:
                                min_id = pair_id
                                min_pos = i
                        # 遍历完，如果找到可以合并的就合并，找不到就退出while
                        if min_pos != len(word_list):
                            #  合并第i和第i+1个元素
                            word_list[min_pos] = word_list[min_pos] + word_list[min_pos+1]
                            word_list.pop(min_pos+1)
                        else:
                            break
                    
                    # 把合并完的word_list 里每个bytes查出id并append入结果
                    for token_byte in word_list:
                        bpe_tokens.append(bytes_to_id[token_byte])

        return bpe_tokens

    def encode_iterable(self, iterable: Iterable[str])-> Iterator[int]:
        """
        流式编码， 按块读取节省内存
        """
        for text_chunk in iterable:
            for token_id in self.encode(text_chunk):
                yield token_id