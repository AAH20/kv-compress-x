import time
import threading
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class MatchResult:
    matched_tokens: List[int]
    kv_block_handle: Optional[str]
    is_full_match: bool
    prefix_length: int

class RadixNode:
    def __init__(self, token_seq: List[int], kv_block_handle: Optional[str] = None):
        self.token_seq: List[int] = token_seq
        self.kv_block_handle: Optional[str] = kv_block_handle
        self.children: Dict[int, "RadixNode"] = {} # first_token -> RadixNode
        self.last_accessed: float = time.time()
        self.hit_count: int = 0

class RadixPrefixTrie:
    """
    High-Performance Radix Prefix Trie for Zero-Prefill KV Cache Reuse.
    Indexes token sequences and maps shared prefixes (system prompts, agent cards)
    to pre-computed GPU memory block handles in sub-millisecond lookups.
    """
    def __init__(self):
        self.root = RadixNode(token_seq=[])
        self._lock = threading.Lock()
        self.total_lookups = 0
        self.total_hits = 0

    def insert(self, tokens: List[int], kv_block_handle: str):
        if not tokens:
            return

        with self._lock:
            curr = self.root
            idx = 0
            while idx < len(tokens):
                first_tok = tokens[idx]
                if first_tok not in curr.children:
                    # New branch
                    curr.children[first_tok] = RadixNode(
                        token_seq=tokens[idx:],
                        kv_block_handle=kv_block_handle
                    )
                    return
                else:
                    child = curr.children[first_tok]
                    # Find common prefix length between child.token_seq and tokens[idx:]
                    common_len = 0
                    min_len = min(len(child.token_seq), len(tokens) - idx)
                    while common_len < min_len and child.token_seq[common_len] == tokens[idx + common_len]:
                        common_len += 1

                    if common_len == len(child.token_seq):
                        # Exact match on child sequence, move down
                        curr = child
                        idx += common_len
                    else:
                        # Split child node
                        split_node = RadixNode(
                            token_seq=child.token_seq[:common_len],
                            kv_block_handle=None
                        )
                        child.token_seq = child.token_seq[common_len:]
                        split_node.children[child.token_seq[0]] = child
                        
                        # Add new branch if needed
                        rem_tokens = tokens[idx + common_len:]
                        if rem_tokens:
                            split_node.children[rem_tokens[0]] = RadixNode(
                                token_seq=rem_tokens,
                                kv_block_handle=kv_block_handle
                            )
                        else:
                            split_node.kv_block_handle = kv_block_handle

                        curr.children[first_tok] = split_node
                        return

    def match_prefix(self, tokens: List[int]) -> MatchResult:
        with self._lock:
            self.total_lookups += 1
            curr = self.root
            idx = 0
            best_handle = None
            matched_len = 0

            while idx < len(tokens):
                first_tok = tokens[idx]
                if first_tok not in curr.children:
                    break

                child = curr.children[first_tok]
                common_len = 0
                min_len = min(len(child.token_seq), len(tokens) - idx)
                while common_len < min_len and child.token_seq[common_len] == tokens[idx + common_len]:
                    common_len += 1

                if common_len == len(child.token_seq):
                    curr = child
                    curr.last_accessed = time.time()
                    curr.hit_count += 1
                    matched_len += common_len
                    idx += common_len
                    if curr.kv_block_handle:
                        best_handle = curr.kv_block_handle
                else:
                    matched_len += common_len
                    break

            if matched_len > 0 and best_handle:
                self.total_hits += 1

            return MatchResult(
                matched_tokens=tokens[:matched_len],
                kv_block_handle=best_handle,
                is_full_match=(matched_len == len(tokens)),
                prefix_length=matched_len
            )
