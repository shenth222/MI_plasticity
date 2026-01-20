# src/measure/importance_ablation.py
import os, json, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data.glue import load_glue_dataset
from src.model.deberta_head_gating import DebertaV2HeadGate, HeadGatingConfig

@torch.no_grad()
def eval_loss(model, dl, device):
    model.eval()
    tot_loss, tot_n = 0.0, 0
    from tqdm import tqdm
    for batch in tqdm(dl, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        n = batch["labels"].shape[0]
        tot_loss += float(loss) * n
        tot_n += n
    return tot_loss / max(1, tot_n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--subset_path", required=True)   # json list of indices
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=True)
    ds = load_glue_dataset(args.task, tok, max_len=args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt_dir).to(device)

    # Fixed subset
    with open(args.subset_path, "r") as f:
        idx = json.load(f)
    eval_ds = Subset(ds["eval_raw"], idx)  # keep a raw dataset that yields tensors
    dl = DataLoader(eval_ds, batch_size=args.bsz, shuffle=False, collate_fn=ds["collate_fn"])

    cfg = HeadGatingConfig(
        num_layers=len(model.deberta.encoder.layer),
        num_heads=model.config.num_attention_heads,
        hidden_size=model.config.hidden_size
    )
    gatewrap = DebertaV2HeadGate(model, cfg, device=device)

    # baseline
    gatewrap.set_all_ones()
    loss_base = eval_loss(model, dl, device)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for l in range(cfg.num_layers):
            for h in range(cfg.num_heads):
                gatewrap.ablate_one(l, h)
                loss_ab = eval_loss(model, dl, device)
                I = loss_ab - loss_base
                rec = dict(layer=l, head=h, I=float(I), loss_base=float(loss_base), loss_ablate=float(loss_ab), n=len(idx))
                f.write(json.dumps(rec) + "\n")

    gatewrap.remove()

if __name__ == "__main__":
    main()
