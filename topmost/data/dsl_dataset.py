"""Soft-label targets from a language model (DSL).

Improving Topic Modeling by Distilling Soft Labels from Language Models. ICML 2026.
Raymond Li, Amirhossein Abaskohi, Chuyuan Li, Gabriel Murray, Giuseppe Carenini.
https://proceedings.mlr.press/v306/li26a.html

A small causal language model reads each document inside a prompt that asks for its
theme. Its next-token distribution over the vocabulary replaces the bag-of-words as the
reconstruction target, and its last hidden state replaces the bag-of-words as the
encoder input. `DSLDataset` prepares both, in the `[target | embedding]` layout that
`CombinedTM` already uses, so models built with `contextual_embed_size` train on them
with `BasicTrainer` unchanged.
"""
from __future__ import annotations


from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

DEFAULT_INSTRUCTION = "Generate a label that captures the overall theme of the document below."
DEFAULT_TEMPLATE = "{instruction}\n\nDocument: {document}\n\nTopic:"


@dataclass
class VocabularyMap:
    """How vocabulary words map onto LM tokens."""

    token_ids: np.ndarray  # (n_vocab,) id of the first token of " word"
    single_token: np.ndarray  # (n_vocab,) bool, the word is exactly one token


def map_vocabulary(tokenizer, vocabulary: Sequence[str]) -> VocabularyMap:
    """Find the token that starts each vocabulary word when it follows a space.

    Words made of one token are scored exactly. Words made of several tokens are
    scored by their first token, which is only an approximation, so `single_token`
    marks the words whose score is exact. `soft_labels` leaves the others out.
    """
    pieces = [tokenizer.encode(f" {word}", add_special_tokens=False) for word in vocabulary]
    if any(len(p) == 0 for p in pieces):
        raise ValueError("A vocabulary word produced no tokens.")
    # SentencePiece tokenizers that emit a stand-alone space token before every word:
    # the distinguishing token is then the second one.
    first = {p[0] for p in pieces}
    offset = 1 if (len(first) == 1 and len(pieces) > 1 and all(len(p) > 1 for p in pieces)) else 0
    token_ids = np.array([p[offset] for p in pieces], dtype=np.int64)
    single = np.array(
        [len(p) == offset + 1 and word.isalpha() for p, word in zip(pieces, vocabulary)], dtype=bool
    )
    return VocabularyMap(token_ids=token_ids, single_token=single)


def build_prompts(
    documents: Iterable[str],
    tokenizer,
    instruction: str = DEFAULT_INSTRUCTION,
    template: str = DEFAULT_TEMPLATE,
    max_length: Optional[int] = None,
) -> list[str]:
    """Render one prompt per document; shorten the document if the prompt is too long.

    Truncating the tokenized prompt would cut off its end ("Topic:"), which is the
    position we read from, so the document text is shortened instead.
    """
    prompts = []
    for document in documents:
        prompt = template.format(instruction=instruction, document=document).rstrip()
        if max_length is not None:
            n_tokens = len(tokenizer.encode(prompt))
            if n_tokens > max_length:
                overhead = len(tokenizer.encode(template.format(instruction=instruction, document="").rstrip()))
                budget = max(max_length - overhead - 2, 1)
                doc_ids = tokenizer.encode(document, add_special_tokens=False)[:budget]
                shortened = tokenizer.decode(doc_ids, skip_special_tokens=True)
                prompt = template.format(instruction=instruction, document=shortened).rstrip()
        prompts.append(prompt)
    return prompts


class SoftLabelTeacher:
    """Runs the language model and returns vocabulary logits and hidden states."""

    def __init__(
        self,
        model_name: str = "baidu/ERNIE-4.5-0.3B-PT",
        instruction: str = DEFAULT_INSTRUCTION,
        template: str = DEFAULT_TEMPLATE,
        batch_size: int = 16,
        max_length: Optional[int] = 512,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.instruction = instruction
        self.template = template
        self.batch_size = batch_size
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding keeps the last real token at position -1 for every row.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        ).eval().to(self.device)

    @torch.no_grad()
    def extract(self, documents: Sequence[str], vocabulary: Sequence[str], progress=None):
        """Return (logits, hidden): arrays of shape (n_docs, n_vocab) and (n_docs, hidden_size)."""
        vocab_map = map_vocabulary(self.tokenizer, vocabulary)
        token_ids = torch.as_tensor(vocab_map.token_ids, device=self.device)
        prompts = build_prompts(documents, self.tokenizer, self.instruction, self.template, self.max_length)
        lengths = [len(self.tokenizer.encode(p)) for p in prompts]
        order = np.argsort(lengths)[::-1]  # longest first: an out-of-memory error shows up at once
        logits = np.empty((len(prompts), len(vocabulary)), dtype=np.float32)
        hidden = None
        for start in range(0, len(order), self.batch_size):
            rows = order[start : start + self.batch_size]
            batch = self.tokenizer([prompts[i] for i in rows], return_tensors="pt", padding=True)
            batch = {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            out = self.model(**batch, use_cache=False, output_hidden_states=True)
            last_logits = out.logits[:, -1, :].index_select(-1, token_ids).float().cpu().numpy()
            last_hidden = out.hidden_states[-1][:, -1, :].float().cpu().numpy()
            if hidden is None:
                hidden = np.empty((len(prompts), last_hidden.shape[-1]), dtype=np.float32)
            logits[rows] = last_logits
            hidden[rows] = last_hidden
            if progress is not None:
                progress(min(start + self.batch_size, len(order)), len(order))
        return logits, hidden, vocab_map


def soft_labels(logits, temperature=3.0, mask=None):
    """softmax(logits / temperature) over the vocabulary, row-wise.

    `mask` (bool, one entry per word) selects the words that take part; the others
    get zero. Pass `VocabularyMap.single_token` to keep exactly scored words only.
    """
    z = np.asarray(logits, dtype=np.float64) / temperature
    if mask is not None and not np.all(mask):
        if not np.any(mask):
            raise ValueError("No vocabulary word is a single alphabetic token for this language model.")
        z = np.where(np.asarray(mask)[None, :], z, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


class DSLDataset:
    """Documents with soft-label targets and language-model embeddings.

    Args:
        docs: the raw documents.
        preprocess: a `topmost.Preprocess`; builds the vocabulary (default: 2,000 words).
        model_name: any causal language model on the Hugging Face Hub.
        temperature: softmax temperature of the soft label. Pass the same value to the
            model (`temperature=dataset.temperature`) so its output is softened alike.
        target_weight: the soft label is multiplied by this before training. The
            reconstruction losses are linear in the target, so this is the weight
            lambda on the distillation term (KL up to a constant).
        teacher: a ready `SoftLabelTeacher`, to reuse one model across datasets.

    Attributes:
        train_data: tensor `[target_weight * soft label (vocab_size) | embedding]`.
        train_bow: the usual bag-of-words counts, kept for evaluation.
        soft_labels, train_contextual_embed, contextual_embed_size, vocab, train_texts.
        single_token: which vocabulary words are part of the soft label.
    """

    def __init__(self,
                 docs,
                 preprocess=None,
                 model_name="baidu/ERNIE-4.5-0.3B-PT",
                 temperature=3.0,
                 target_weight=1000.0,
                 batch_size=200,
                 device='cpu',
                 as_tensor=True,
                 lm_batch_size=16,
                 max_length=512,
                 lm_device=None,
                 teacher=None,
                 verbose=False
                ):
        from scipy.sparse import issparse
        from torch.utils.data import DataLoader
        from topmost.preprocess import Preprocess

        if preprocess is None:
            preprocess = Preprocess(vocab_size=2000, verbose=verbose)

        rst = preprocess.preprocess(docs)
        self.train_bow = rst["train_bow"]
        if issparse(self.train_bow):
            self.train_bow = self.train_bow.toarray()
        self.train_bow = self.train_bow.astype("float32")
        self.train_texts = rst["train_texts"]
        self.vocab = list(rst["vocab"])
        self.vocab_size = len(self.vocab)

        if teacher is None:
            teacher = SoftLabelTeacher(model_name, batch_size=lm_batch_size, max_length=max_length,
                                       device=lm_device or device)
        self.teacher = teacher
        logits, hidden, vocab_map = teacher.extract(list(docs), self.vocab)
        self.single_token = vocab_map.single_token
        self.temperature = temperature
        self.soft_labels = soft_labels(logits, temperature, mask=vocab_map.single_token)
        self.train_contextual_embed = hidden
        self.contextual_embed_size = hidden.shape[1]

        if as_tensor:
            self.train_data = np.concatenate((target_weight * self.soft_labels, hidden), axis=1)
            self.train_data = torch.from_numpy(self.train_data).float().to(device)
            self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
