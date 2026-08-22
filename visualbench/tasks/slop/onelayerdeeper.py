"""One Layer Deeper benchmark from <https://onelayerdeeper.ai/problem>.

Given a modulus ``N`` (product of two secret primes ``p`` and ``q``), a starting
value ``x`` and a step count ``T``, predict the residue after squaring modulo
``N`` exactly ``T`` times:

    x_0 = x mod N
    x_t = x_{t-1} ** 2 mod N
    y   = x_T

Prompts use decimal tokens with field markers, e.g. ``N77X2T4ANS9`` means
``N=77, x=2, T=4`` and the answer is ``9``.

The evaluator knows ``p`` and ``q`` and can reduce the exponent with
``phi(N) = (p-1)(q-1)``: ``e = 2**T mod phi(N)``, ``y = x**e mod N``.
"""
import math
import random
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ...utils import CUDA_IF_AVAILABLE
from .dataset import DatasetBenchmark

# ---------------------------------- vocab ---------------------------------- #
N_TOKEN = 10   # 'N' field marker
X_TOKEN = 11   # 'X' field marker
T_TOKEN = 12   # 'T' field marker
ANS_TOKEN = 13 # 'ANS' field marker
PAD_TOKEN = 14 # padding token

VOCAB_SIZE = 15
"""vocabulary: tokens 0-9 are decimal digits, 10-13 are field markers, 14 is padding"""


def _digits(n: int) -> list[int]:
    return [int(c) for c in str(n)]


def _tokenize_prompt(N: int, x: int, T: int) -> list[int]:
    return [N_TOKEN] + _digits(N) + [X_TOKEN] + _digits(x) + [T_TOKEN] + _digits(T) + [ANS_TOKEN]


def decode_sample(prompt_tokens, answer_tokens) -> tuple[int, int, int, int]:
    """Decodes tokenized prompt and answer back into ``(N, x, T, y)`` integers."""
    toks = [int(t) for t in prompt_tokens if t != PAD_TOKEN]
    x_pos = toks.index(X_TOKEN)
    t_pos = toks.index(T_TOKEN)
    a_pos = toks.index(ANS_TOKEN)
    N = int("".join(str(t) for t in toks[1:x_pos]))
    x = int("".join(str(t) for t in toks[x_pos + 1:t_pos]))
    T = int("".join(str(t) for t in toks[t_pos + 1:a_pos]))
    y = int("".join(str(int(t)) for t in answer_tokens if t != PAD_TOKEN))
    return N, x, T, y


# ------------------------------ label generation ----------------------------- #
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
"""deterministic Miller-Rabin bases, exact for n < ~3.2e23 (about 97 bits)"""


def _is_prime(n: int) -> bool:
    if n < 2: return False
    for p in _MR_BASES:
        if n % p == 0: return n == p

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for a in _MR_BASES:
        if a >= n: continue
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else:
            return False
    return True


def _random_prime(bits: int, rng: random.Random) -> int:
    if bits < 2: raise ValueError(f'bits must be >= 2, got {bits}')
    while True:
        n = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if _is_prime(n): return n


def _brute_square(x: int, T: int, N: int) -> int:
    """Exact label via T serial squarings, works for any x."""
    y = x % N
    for _ in range(T):
        y = y * y % N
    return y


def _fast_label(N: int, x: int, T: int, phi_N: int) -> int:
    """Exact label via Euler's theorem: e = 2**T mod phi(N), y = x**e mod N.

    Valid only when gcd(x, N) == 1, otherwise falls back to serial squaring.
    """
    if math.gcd(x, N) == 1:
        e = pow(2, T, phi_N)
        return pow(x, e, N)
    return _brute_square(x, T, N)


def generate_data(
    bits: int = 32,
    max_T: int = 32,
    num_samples: int = 10_000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates tokenized samples of the one layer deeper problem.

    Returns ``(X, y)`` where:

    - ``X`` is ``(num_samples, prompt_len)`` int64 token ids, each row is
      ``N<digits>X<digits>T<digits>ANS`` padded with ``PAD_TOKEN``.
    - ``y`` is ``(num_samples, answer_len)`` int64 token ids, each row is the
      decimal digits of ``y = x**(2**T) mod N`` padded with ``PAD_TOKEN``.

    Prompt length is ``4 + 2 * digits(N) + digits(T)`` and answer length is
    ``digits(N)`` (the largest possible answer), where ``digits(n)`` is the
    number of decimal digits ``n`` can take. ``bits > ~96`` may produce
    probabilistic primes.
    """
    rng = random.Random(seed)
    p_bits = bits // 2
    q_bits = bits - p_bits

    digits_max = len(str(2 ** bits - 1))
    len_T = len(str(max_T))
    prompt_len = 4 + 2 * digits_max + len_T
    answer_len = digits_max

    X = torch.full((num_samples, prompt_len), PAD_TOKEN, dtype=torch.int64)
    y = torch.full((num_samples, answer_len), PAD_TOKEN, dtype=torch.int64)

    for i in range(num_samples):
        p = _random_prime(p_bits, rng)
        q = _random_prime(q_bits, rng)
        N = p * q
        x = rng.randrange(N)
        T = rng.randrange(1, max_T + 1)
        label = _fast_label(N, x, T, phi_N=(p - 1) * (q - 1))

        prompt = _tokenize_prompt(N, x, T)
        X[i, :len(prompt)] = torch.tensor(prompt, dtype=torch.int64)
        ans = _digits(label)
        y[i, :len(ans)] = torch.tensor(ans, dtype=torch.int64)

    return X, y


def default_criterion(y_hat: torch.Tensor, y: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    return F.cross_entropy(y_hat.transpose(1, 2), y, ignore_index=PAD_TOKEN, reduction=reduction)


# ---------------------------------- model ---------------------------------- #
class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        inv = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos * inv)
        pe[:, 1::2] = torch.cos(pos * inv)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:length].unsqueeze(0)


class OneLayerDeeperTransformer(nn.Module):
    """Small query-based transformer for the one layer deeper problem.

    Input - ``(B, prompt_len)`` token ids.

    output - ``(B, answer_len, VOCAB_SIZE)`` logits.

    A fixed number of learnable query vectors are appended to the prompt and
    attend to it with a bidirectional transformer encoder, so all answer
    digits are predicted in a single forward pass (non-autoregressive).
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        out_len: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_len = out_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos = _SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.queries = nn.Parameter(torch.randn(1, out_len, d_model) * 0.02)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        pad_mask = x == PAD_TOKEN

        h = self.emb(x) + self.pos(x.size(1)).to(x.device)
        queries = self.queries.expand(B, -1, -1)
        src = torch.cat([h, queries], dim=1)
        src_mask = torch.cat(
            [pad_mask, torch.zeros(B, self.out_len, dtype=torch.bool, device=x.device)], dim=1)

        out = self.encoder(src, src_key_padding_mask=src_mask)
        return self.head(out[:, -self.out_len:])


# --------------------------------- benchmark --------------------------------- #
class OneLayerDeeper(DatasetBenchmark):
    """Repeated squaring mod N, from <https://onelayerdeeper.ai/problem>.

    Given a modulus ``N``, a starting value ``x`` and a step count ``T``, the
    task is to predict ``y = x**(2**T) mod N``. ``N`` is the product of two
    secret primes, so the model cannot reduce the exponent and has to carry out
    the serial squarings in order.

    Data is tokenized as decimal digits with field markers, e.g.
    ``N77X2T4ANS9`` means ``N=77, x=2, T=4`` and the answer is ``9``.

    Input - ``(B, prompt_len)`` int64 token ids:
    ``N<digits>X<digits>T<digits>ANS`` padded with ``PAD_TOKEN``.

    output - ``(B, answer_len, VOCAB_SIZE)`` logits: one token per answer digit,
    padded positions are ignored by the default criterion.

    Default model is ``OneLayerDeeperTransformer``. Any model that maps
    ``(B, prompt_len)`` to ``(B, answer_len, VOCAB_SIZE)`` logits will work.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        bits: int = 32,
        max_T: int = 32,
        num_samples: int = 10_000,
        criterion: Callable | None = None,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split: float = 0.8,
        seed: int = 0,
        device=CUDA_IF_AVAILABLE,
    ):
        X, y = generate_data(bits=bits, max_T=max_T, num_samples=num_samples, seed=seed)

        self.bits = bits
        self.max_T = max_T
        self.prompt_len = X.size(1)
        self.answer_len = y.size(1)
        self.vocab_size = VOCAB_SIZE

        if model is None:
            model = OneLayerDeeperTransformer(out_len=self.answer_len)
        if criterion is None:
            criterion = default_criterion

        super().__init__(
            data_train=(X, y),
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            train_split=train_split,
            dtypes=(torch.int64, torch.int64),
            data_device=device,
        )

    def _log_metrics(self, y: torch.Tensor, pred: torch.Tensor, prefix: str = ""):
        mask = y != PAD_TOKEN
        if not mask.any(): return
        correct = (pred == y) & mask
        tok = correct.sum() / mask.sum()
        em = ((pred == y) | ~mask).all(-1).float().mean()
        self.log(prefix + 'token accuracy', tok)
        self.log(prefix + 'exact match accuracy', em)

    def after_get_loss(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
        pred = y_hat.argmax(-1)

        if self.test_start_idx is not None:
            split = self.test_start_idx
            if self.training:
                self._log_metrics(y[:split], pred[:split], prefix='train ')
            self._log_metrics(y[split:], pred[split:], prefix='test ')
        else:
            self._log_metrics(y, pred)