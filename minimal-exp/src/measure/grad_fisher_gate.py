# src/measure/grad_fisher_gate.py
import os, json, argparse
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data.glue import load_glue_dataset
from src.model.deberta_head_gating import DebertaV2HeadGate, HeadGatingConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--subset_path", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lambda_eps", type=float, default=1e-6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=True)
    ds = load_glue_dataset(args.task, tok, max_len=args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt_dir).to(device)

    with open(args.subset_path, "r") as f:
        idx = json.load(f)
    eval_ds = Subset(ds["eval_raw"], idx)
    dl = DataLoader(eval_ds, batch_size=args.bsz, shuffle=False, collate_fn=ds["collate_fn"])

    cfg = HeadGatingConfig(
        num_layers=len(model.deberta.encoder.layer),
        num_heads=model.config.num_attention_heads,
        hidden_size=model.config.hidden_size
    )
    gatewrap = DebertaV2HeadGate(model, cfg, device=device)

    # Accumulate gradients on gate
    model.train()
    gatewrap.gates.grad = None

    sum_abs = torch.zeros_like(gatewrap.gates, device=device)
    sum_sq = torch.zeros_like(gatewrap.gates, device=device)
    tot = 0

    for batch in dl:
        model.zero_grad(set_to_none=True)
        gatewrap.gates.grad = None
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()

        g = gatewrap.gates.grad.detach()  # [L,H]
        sum_abs += g.abs()
        sum_sq += g.pow(2)
        tot += 1

    G = (sum_abs / max(1, tot)).cpu()
    F = (sum_sq / max(1, tot)).cpu()
    Ppred = (G.pow(2) / (F + args.lambda_eps)).cpu()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for l in range(cfg.num_layers):
            for h in range(cfg.num_heads):
                rec = dict(layer=l, head=h, G=float(G[l,h]), F=float(F[l,h]), Ppred=float(Ppred[l,h]), n_batches=tot)
                f.write(json.dumps(rec) + "\n")

    gatewrap.remove()

if __name__ == "__main__":
    main()
