"""generate.py -- sample sentences from any model that has next_word_probs (n-gram or RNN/LSTM)."""
import numpy as np


def sample_sentence(model, vocab, temperature=1.0, rng=None, max_len=40, forbid=("<s>", "<unk>")):
    """Ancestral sampling from <s>. `temperature` rescales the distribution (p ** (1/T), renormalised);
    <s> and <unk> are never sampled. Stops at </s> or after max_len tokens. Returns a list of ids."""
    rng = rng or np.random.default_rng(0)
    banned = [vocab.stoi[t] for t in forbid]
    ctx, out = [vocab.bos_id], []
    for _ in range(max_len):
        p = np.asarray(model.next_word_probs(ctx), dtype=np.float64).copy()
        p[banned] = 0.0
        p = p ** (1.0 / temperature)
        p /= p.sum()
        t = int(rng.choice(len(p), p=p))
        if t == vocab.eos_id:
            break
        out.append(t)
        ctx.append(t)
    return out
