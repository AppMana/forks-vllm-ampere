# SPDX-License-Identifier: Apache-2.0
"""Gate 1 of the INT8 indexer-cache campaign: top-k recall on real tensors.

Loads the capture written by sparse_attn_indexer.py under
APPMANA_DSV4_INDEXER_DUMP (real fp8 q/k, per-token k scales, head weights,
causal ranges from a prefill chunk) and asks: if the indexer cache and q were
stored as per-token symmetric INT8 instead of FP8 e4m3, how much of the
current path's top-k selection is preserved?

The reference is the current FP8 representation evaluated in fp32 (the
deployed selection); the INT8 simulation re-quantizes the dequantized vectors
per token (k) and per token-head (q) at absmax/127. Recall is measured per
query token over its causal range.

    .venv/bin/python tools/ampere/dsv4_indexer_int8_recall.py /tmp/indexer_dump
"""

import sys

import torch


def main() -> None:
    dump = torch.load(f"{sys.argv[1]}/indexer_dump.pt", weights_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    q_fp8 = dump["q_fp8"].to(device)            # [T, H, D] fp8
    k_fp8 = dump["k_fp8"].to(device)            # [S, D] fp8
    k_scale = dump["k_scale"].to(device).reshape(-1)[: k_fp8.shape[0]]  # [S]
    weights = dump["weights"].to(device).float()  # [T, H]
    ks = dump["cu_seqlen_ks"].to(device)
    ke = dump["cu_seqlen_ke"].to(device)
    topk = dump["topk_tokens"]

    T, H, D = q_fp8.shape
    S = k_fp8.shape[0]
    print(f"T={T} H={H} D={D} S={S} topk={topk}")

    q_ref = q_fp8.float()                        # fp8 values as deployed
    k_ref = k_fp8.float()                        # unscaled fp8 values
    k_true = k_ref * k_scale[:, None]            # dequantized K

    def logits_for(q, k_unscaled, k_post_scale):
        # scores[t,h,s] = q[t,h,:] . k[s,:]; relu(score * k_post_scale[s])
        # weighted by weights[t,h], summed over heads.
        scores = torch.einsum("thd,sd->ths", q, k_unscaled)
        weighted = torch.relu(scores * k_post_scale[None, None, :])
        return (weighted * weights[:, :, None]).sum(dim=1)  # [T, S]

    ref_logits = logits_for(q_ref, k_ref, k_scale)

    # INT8 simulation: per-token K, per token-head Q, symmetric absmax/127.
    k_s8 = k_true.abs().amax(dim=1).clamp(min=1e-30) / 127.0
    k_int8 = torch.round(k_true / k_s8[:, None]).clamp(-127, 127)
    q_s8 = q_ref.abs().amax(dim=2).clamp(min=1e-30) / 127.0   # [T, H]
    q_int8 = torch.round(q_ref / q_s8[:, :, None]).clamp(-127, 127)

    scores = torch.einsum("thd,sd->ths", q_int8, k_int8)
    scores = scores * q_s8[:, :, None] * k_s8[None, None, :]
    int8_logits = (torch.relu(scores) * weights[:, :, None]).sum(dim=1)

    # Causal mask and per-token recall.
    arange_s = torch.arange(S, device=device)
    recalls = []
    contexts = []
    for t in range(T):
        lo, hi = int(ks[t]), int(ke[t])
        n_ctx = hi - lo
        if n_ctx < topk:
            continue
        mask = (arange_s >= lo) & (arange_s < hi)
        ref_top = torch.topk(ref_logits[t].masked_fill(~mask, float("-inf")), topk).indices
        new_top = torch.topk(int8_logits[t].masked_fill(~mask, float("-inf")), topk).indices
        inter = len(set(ref_top.tolist()) & set(new_top.tolist()))
        recalls.append(inter / topk)
        contexts.append(n_ctx)

    r = torch.tensor(recalls)
    print(
        f"tokens evaluated: {len(recalls)} (context {min(contexts)}-{max(contexts)})\n"
        f"top-{topk} recall vs deployed fp8 selection: "
        f"mean={r.mean():.4f} p05={r.quantile(0.05):.4f} min={r.min():.4f}"
    )
    snr = 10 * torch.log10(
        ref_logits.pow(2).sum() / (ref_logits - int8_logits).pow(2).sum()
    )
    print(f"logits SNR int8-vs-fp8-path: {snr:.1f} dB")


if __name__ == "__main__":
    main()
