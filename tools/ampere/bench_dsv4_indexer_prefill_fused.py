import os, time
import torch
torch.set_default_device('cuda')
import vllm.models.deepseek_v4.attention  # noqa
from vllm.utils import deep_gemm as dg

d = torch.load('/tmp/idx_off/indexer_dump.pt', weights_only=True)
q = d['q_fp8'].cuda()
k_fp8 = d['k_fp8'].cuda()
k_scale = d['k_scale'].cuda().reshape(-1)[:k_fp8.shape[0]]
w = d['weights'].cuda().float()
ks, ke = d['cu_seqlen_ks'].cuda(), d['cu_seqlen_ke'].cuda()
TOPK = d['topk_tokens']
k_true = k_fp8.float() * k_scale[:, None]
k_s8 = k_true.abs().amax(dim=1).clamp(min=1e-4) / 127.0
k_i8 = torch.round(k_true / k_s8[:, None]).clamp(-127, 127).to(torch.int8)
out_ref = torch.empty(q.shape[0], TOPK, dtype=torch.int32)
out_imma = torch.empty_like(out_ref)
os.environ['APPMANA_DSV4_INDEXER_IMMA'] = '0'
dg._fp8_mqa_logits_topk_torch((q, None), (k_fp8, k_scale), w, ks, ke, TOPK, out=out_ref)
os.environ['APPMANA_DSV4_INDEXER_IMMA'] = '1'
dg._fp8_mqa_logits_topk_torch((q, None), (k_i8, k_s8), w, ks, ke, TOPK, out=out_imma)
torch.cuda.synchronize()
recalls = []
for t in range(0, q.shape[0], 13):
    if int(ke[t]) - int(ks[t]) < TOPK: continue
    a = set(out_ref[t].tolist()); b = set(out_imma[t].tolist())
    a.discard(-1); b.discard(-1)
    if a: recalls.append(len(a & b) / len(a))
r = torch.tensor(recalls)
print(f'recall vs fp8 reference: mean={r.mean():.4f} p05={r.quantile(0.05):.4f} min={r.min():.4f} (n={len(recalls)})')
def bench(fmt, kv):
    os.environ['APPMANA_DSV4_INDEXER_IMMA'] = fmt
    o = torch.empty_like(out_ref)
    for _ in range(2): dg._fp8_mqa_logits_topk_torch((q, None), kv, w, ks, ke, TOPK, out=o)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(8): dg._fp8_mqa_logits_topk_torch((q, None), kv, w, ks, ke, TOPK, out=o)
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/8*1000
t_ref = bench('0', (k_fp8, k_scale))
t_imma = bench('1', (k_i8, k_s8))
print(f'fp32 path: {t_ref:.1f} ms | imma path: {t_imma:.1f} ms | speedup {t_ref/t_imma:.2f}x')
