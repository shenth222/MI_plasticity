# src/measure/importance_ablation_lora.py
import os, json, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
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
    ap.add_argument("--base_model", required=True, help="基础模型路径")
    ap.add_argument("--lora_ckpt", required=True, help="LoRA权重路径")
    ap.add_argument("--subset_path", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--merge_weights", action="store_true", 
                    help="是否合并LoRA权重到基础模型（推荐用于ablation）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    ds = load_glue_dataset(args.task, tok, max_len=args.max_len)

    # 加载LoRA模型
    base_model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.lora_ckpt)
    
    # 合并LoRA权重以便进行head ablation
    if args.merge_weights:
        print("合并LoRA权重到基础模型...")
        model = model.merge_and_unload()
    
    model = model.to(device)

    # 加载固定的评估子集
    with open(args.subset_path, "r") as f:
        idx = json.load(f)
    eval_ds = Subset(ds["eval_raw"], idx)
    dl = DataLoader(eval_ds, batch_size=args.bsz, shuffle=False, collate_fn=ds["collate_fn"])

    # 配置head gating
    # 注意：如果模型被合并了，直接访问.deberta；否则需要访问.base_model.deberta
    if hasattr(model, 'deberta'):
        encoder_layers = model.deberta.encoder.layer
    else:
        encoder_layers = model.base_model.deberta.encoder.layer
    
    cfg = HeadGatingConfig(
        num_layers=len(encoder_layers),
        num_heads=model.config.num_attention_heads if hasattr(model, 'config') else model.base_model.config.num_attention_heads,
        hidden_size=model.config.hidden_size if hasattr(model, 'config') else model.base_model.config.hidden_size
    )
    
    gatewrap = DebertaV2HeadGate(model, cfg, device=device)

    # 计算基线loss
    gatewrap.set_all_ones()
    loss_base = eval_loss(model, dl, device)
    # print(f"基线loss: {loss_base:.4f}")

    # 逐个ablation每个head
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for l in range(cfg.num_layers):
            for h in range(cfg.num_heads):
                gatewrap.ablate_one(l, h)
                loss_ab = eval_loss(model, dl, device)
                I = loss_ab - loss_base
                rec = dict(
                    layer=l, 
                    head=h, 
                    I=float(I), 
                    loss_base=float(loss_base), 
                    loss_ablate=float(loss_ab), 
                    n=len(idx)
                )
                f.write(json.dumps(rec) + "\n")
                # print(f"Layer {l}, Head {h}: I={I:.6f}")

    gatewrap.remove()
    print(f"重要性测量完成，结果保存到: {args.out_jsonl}")

if __name__ == "__main__":
    main()
