import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers import LlamaTokenizer, LlamaTokenizerFast
from typing import Callable, List, Dict, Any, Optional, Tuple
import logging
import random
import re
from tree_sitter import Parser
from tree_sitter_languages import get_parser

log = logging.getLogger(__name__)

_ALL_LANGUAGES = ['java', 'cpp', 'rust', 'go', 'python']
_WHITELIST_TYPES = ["constructor_declaration", "method_declaration", "class_definition", "function_definition", "if_statement", "enhanced_for_statement", "for_statement"]

def wrapper_get_item(method):
    def f(self, idx):
        item = method(idx)
        item['split'] = self.split
        return item
    return f


class FIMTransformer:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_seq_len: int,
                 fim_prefix_token = "<|fim_prefix|>",
                 fim_middle_token = "<|fim_middle|>",
                 fim_suffix_token = "<|fim_suffix|>",
                 ):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = tokenizer.eos_token_id

        self.fim_prefix_token_id = self.tokenizer.convert_tokens_to_ids(fim_prefix_token)
        self.fim_middle_token_id = self.tokenizer.convert_tokens_to_ids(fim_middle_token)
        self.fim_suffix_token_id = self.tokenizer.convert_tokens_to_ids(fim_suffix_token)

        unk_token_id = getattr(self.tokenizer, 'unk_token_id', None)
        if self.fim_prefix_token_id == unk_token_id:
            log.warning(f"FIM prefix token {fim_prefix_token} not found in tokenizer vocabulary")
        if self.fim_middle_token_id == unk_token_id:
            log.warning(f"FIM middle token {fim_middle_token} not found in tokenizer vocabulary" )
        if self.fim_suffix_token_id == unk_token_id:
            log.warning(f"FIM suffix token {fim_suffix_token} not found in tokenizer vocabulary")

        self.fim_tokens_exist_in_vocab = (self.fim_prefix_token_id != unk_token_id) and (self.fim_middle_token_id != unk_token_id) and (self.fim_suffix_token_id != unk_token_id)

        if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            self.char_level_split_method = self._sp_charlevel_split
        elif isinstance(tokenizer, PreTrainedTokenizerBase):
            self.char_level_split_method = self._bpe_charlevel_split
        else:
            raise RuntimeError(f"Tokenizer type={type(tokenizer)} not supported for FIM!")

    def split_context_into_doc(self, tokens: List[int]):
        parts = np.split(tokens, np.where(tokens==self.eos_token_id)[0])
        return [ part if i == 0 else part[1:] for i, part in enumerate(parts) ]

    def _sp_charlevel_split(self, token_id: int):
        def sp_encode(string):
            return self.tokenizer.encode("\n" + string, add_special_tokens=False)[2:]

        return self._base_charlevel_split(token_id, sp_encode)

    def _bpe_charlevel_split(self, token_id: int):
        def bpe_encode(string):
            return self.tokenizer.encode(string, add_special_tokens=False)

        return self._base_charlevel_split(token_id, bpe_encode)

    def _base_charlevel_split(self,
                              token_id: int,
                              encode_method: Callable):
        """Decode token_id to text, randomly split it on two parts on level of chars and tokenize this parts."""
        token = self.tokenizer.decode([int(token_id)])
        idx = np.random.choice(len(token), 1, replace=False)
        left, right = token[:idx[0]], token[idx[0]:]

        left = encode_method(left) if left else []
        right = encode_method(right) if right else []

        return left, right

    def _error_count(self, node):
        return (node.type == "ERROR") + sum(self._error_count(c) for c in node.children)


    def _get_best_parser(self, code_bytes: bytes):

        best_score = float("inf")
        best_lang  = None
        best_parser: Parser | None = None

        for lang in _ALL_LANGUAGES:
            try:
                parser = get_parser(lang)
                tree   = parser.parse(code_bytes)
                score  = self._error_count(tree.root_node)
            except Exception:
                continue

            if score < best_score:
                best_score, best_lang, best_parser = score, lang, parser
                if score == 0:           # perfect parse – stop early
                    break

        if best_parser is None:
            raise RuntimeError("No suitable Tree-Sitter grammar found.")

        return best_parser, best_lang


    def _all_descendants(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed(n.children))

    def _code_for(self, node, source):
        return source[node.start_byte : node.end_byte]

    def _pick_random_multiline_nonleaf_node(self, tree, source):
        candidates = []
        for n in self._all_descendants(tree.root_node):
            if n.parent is None or not n.children:
                continue
            snippet = self._code_for(n, source)
            if len(snippet) < 10:
                continue
            if sum(1 for ln in snippet.split("\n") if ln.strip()) < 2:
                continue
            if n.type not in _WHITELIST_TYPES:
                continue
            candidates.append(n)
        if not candidates:
            return None
        return random.choice(candidates)

    def _split_three_random(self, text):
        if len(text) < 3:
            return text, "", ""

        a = random.randint(1, len(text) - 2)
        b = random.randint(a + 1, len(text) - 1)

        part1, part2, part3 = text[:a], text[a:b], text[b:]

        newline_at = text.find("\n", b)
        if newline_at != -1:
            part2 = text[a: newline_at + 1]
            part3 = text[newline_at + 1:]
        else:
            part2 = text[a:]
            part3 = ""

        return part1, part2, part3

    def syntax_aware_split(self, tokens: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_code = self.tokenizer.decode(tokens)

        try:
            parser, language = self._get_best_parser(source_code.encode())
            tree = parser.parse(source_code.encode())
            node = self._pick_random_multiline_nonleaf_node(tree, source_code)

            if node is None:
                return self.multiline_fim_split(tokens)

            snippet = self._code_for(node, source_code)
            part1, part2, part3 = self._split_three_random(snippet)

            start_pos = source_code.find(snippet)
            pos1 = start_pos + len(part1)
            pos2 = pos1 + len(part2)

            full_text = source_code
            prefix_text = full_text[:pos1]
            middle_text = full_text[pos1:pos2]
            suffix_text = full_text[pos2:]

            prefix_tokens = np.array(self.tokenizer.encode(prefix_text, add_special_tokens=False))
            middle_tokens = np.array(self.tokenizer.encode(middle_text, add_special_tokens=False))
            suffix_tokens = np.array(self.tokenizer.encode(suffix_text, add_special_tokens=False))

            return prefix_tokens, middle_tokens, suffix_tokens

        except Exception as e:
            log.warning(f"Syntax-aware splitting failed: {e}. Falling back to random splitting.")
            return self.random_split_tokens_legacy(tokens)

    def inline_fim_split(self, tokens: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_code = self.tokenizer.decode(tokens)

        lines = source_code.split('\n')
        if not lines:
            return self.random_split_tokens_legacy(tokens)

        line_indices = [i for i, line in enumerate(lines) if line.strip()]
        if not line_indices:
            return self.random_split_tokens_legacy(tokens)
        line_idx = np.random.choice(line_indices)
        selected_line = lines[line_idx]

        if not selected_line:
            return self.random_split_tokens_legacy(tokens)

        char_pos = random.randint(0, len(selected_line) - 1)

        middle_length = random.randint(0, 120)
        middle_length = min(middle_length, len(selected_line) - char_pos)

        prefix_text = '\n'.join(lines[:line_idx]) + ('\n' if line_idx > 0 else '') + selected_line[:char_pos]
        middle_text = selected_line[char_pos:char_pos + middle_length]
        suffix_text = selected_line[char_pos + middle_length:] + ('\n' if line_idx < len(lines) - 1 else '') + '\n'.join(lines[line_idx + 1:])

        prefix_tokens = np.array(self.tokenizer.encode(prefix_text, add_special_tokens=False))
        middle_tokens = np.array(self.tokenizer.encode(middle_text, add_special_tokens=False))
        suffix_tokens = np.array(self.tokenizer.encode(suffix_text, add_special_tokens=False))

        return prefix_tokens, middle_tokens, suffix_tokens

    def multiline_fim_split(self, tokens: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_code: str = self.tokenizer.decode(tokens)

        if '\n' not in source_code:
            return self.random_split_tokens_legacy(tokens)

        lines = source_code.splitlines(keepends=True)
        line_idx = random.randrange(len(lines))
        p = 0.1
        n = 1
        while random.random() > p and (line_idx + n) < len(lines):
            n += 1

        char_at_line_i = sum(len(l) for l in lines[:line_idx])

        ws_match = re.match(r'\s*', source_code[char_at_line_i:])
        ws_len = ws_match.end()

        prefix_end = (
            char_at_line_i
            if ws_len == 0
            else random.randrange(char_at_line_i, char_at_line_i + ws_len + 1)
        )

        closing_line_re = re.compile(r"\s*[),;:\\]}]*\s*$")
        middle_end_line_idx = min(line_idx + n + 1, len(lines))
        for idx in range(line_idx, min(line_idx + n + 1, len(lines))):
            if closing_line_re.fullmatch(lines[idx].rstrip('\n')):
                middle_end_line_idx = idx
                break

        middle_end_char = sum(len(l) for l in lines[:middle_end_line_idx])

        prefix_text = source_code[:prefix_end]
        middle_text = source_code[prefix_end:middle_end_char]
        suffix_text = source_code[middle_end_char:]

        prefix_tokens = np.array(self.tokenizer.encode(prefix_text, add_special_tokens=False))
        middle_tokens = np.array(self.tokenizer.encode(middle_text, add_special_tokens=False))
        suffix_tokens = np.array(self.tokenizer.encode(suffix_text, add_special_tokens=False))

        return prefix_tokens, middle_tokens, suffix_tokens


    def random_split_tokens_legacy(self, tokens: List[int]):
        idxs = np.random.choice(np.arange(len(tokens)), size=2, replace=False)
        idxs.sort()

        pre_mid_left, pre_mid_right = self.char_level_split_method(tokens[idxs[0]])
        mid_suf_left, mid_suf_right = self.char_level_split_method(tokens[idxs[1]])
        prefix = np.concatenate([tokens[:idxs[0]], pre_mid_left])
        middle = np.concatenate([pre_mid_right, tokens[idxs[0] + 1 : idxs[1]], mid_suf_left])
        suffix = np.concatenate([mid_suf_right, tokens[idxs[1] + 1:]])

        return prefix, middle, suffix

    def random_split_tokens(self, tokens: List[int], fim_variant: str = 'vanilla_fim'):
        if fim_variant == 'vanilla_fim':
            return self.random_split_tokens_legacy(tokens)
        elif fim_variant == 'inline_fim':
            return self.inline_fim_split(tokens)
        elif fim_variant == 'syntax_aware':
            return self.syntax_aware_split(tokens)
        else:
            return self.multiline_fim_split(tokens)

    def concat_psm(self, prefix, middle, suffix):
        return np.concatenate([
                            [self.fim_prefix_token_id], prefix,
                            [self.fim_suffix_token_id], suffix,
                            [self.fim_middle_token_id], middle,
                            [self.eos_token_id]
                            ])

    def concat_spm(self, prefix, middle, suffix):
        return np.concatenate([
                            [self.fim_suffix_token_id], suffix,
                            [self.fim_prefix_token_id], prefix,
                            [self.fim_middle_token_id], middle,
                            [self.eos_token_id]
                            ])

    def concat_casual(self, tokens):
        return np.concatenate([
                            tokens,
                            [self.eos_token_id]
                            ])

    def sample_transform(self, tokens, fim_rates: dict):
        total_fim_rate = sum(fim_rates.values())

        if total_fim_rate <= 0 or len(tokens) < 30 or np.random.sample(1) > total_fim_rate:
            return self.concat_casual(tokens)

        if tokens[-1] == self.eos_token_id:
            tokens = tokens[:-1]

        fim_variants = list(fim_rates.keys())
        fim_probs = [fim_rates[v] / total_fim_rate for v in fim_variants]
        chosen_variant = np.random.choice(fim_variants, p=fim_probs)

        # if some of the parts are too short, the split is considered bad,
        # and we try again. But if we have not achieved a satisfactory split
        # in 10 tries, we give up and just use a bad split anyway.
        for _ in range(10):
            prefix, middle, suffix = self.random_split_tokens(tokens, chosen_variant)
            if len(prefix) >= 5 and len(middle) >= 5 and len(suffix) >= 5:
                break

        if np.random.sample(1) < 0.5:  # 50 / 50 spm vs psm
            return self.concat_psm(prefix, middle, suffix)
        else:
            return self.concat_spm(prefix, middle, suffix)

    def transform(self, tokens, fim_rates: dict):
        if not self.fim_tokens_exist_in_vocab:
            raise RuntimeError(f"Attempthing FIM transformation, but FIM tokens (<|fim_prefix|>, etc) do not exist in vocab.")

        parts = self.split_context_into_doc(tokens)
        parts = [self.sample_transform(part, fim_rates) for part in parts]
        tokens_transformed = np.concatenate(parts).astype(np.int64)[:self.max_seq_len]

        if len(tokens_transformed) < self.max_seq_len:
            pad_len = self.max_seq_len - len(tokens_transformed)
            tokens_transformed = np.concatenate(
                [tokens_transformed,
                 np.full(pad_len, self.eos_token_id, dtype=np.int64)]
            )

        return torch.from_numpy(tokens_transformed)