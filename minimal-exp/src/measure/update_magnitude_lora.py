# src/measure/update_magnitude_lora.py
import os, json, argparse
import torch
from transformers import AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np

def frob(x):
    """计算Frobenius范数"""
    return float(torch.norm(x, p="fro"))

def get_lora_delta(lora_A, lora_B, scaling):
    """
    计算LoRA的delta权重: delta = B @ A * scaling
    lora_A: [r, in_features]
    lora_B: [out_features, r]
    返回: [out_features, in_features]
    """
    delta = (lora_B @ lora_A) * scaling
    return delta

def slice_out_dim(W, h, head_dim):
    """切分输出维度（用于Q/K/V）"""
    o0 = h * head_dim
    o1 = (h + 1) * head_dim
    return W[o0:o1, :]

def slice_in_dim(W, h, head_dim):
    """切分输入维度（用于O）"""
    i0 = h * head_dim
    i1 = (h + 1) * head_dim
    return W[:, i0:i1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_init", required=True, help="初始基础模型路径")
    ap.add_argument("--ckpt_final", required=True, help="LoRA微调后的模型路径")
    ap.add_argument("--out_jsonl", required=True, help="输出jsonl文件路径")
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    # 加载初始基础模型
    m0 = AutoModelForSequenceClassification.from_pretrained(args.ckpt_init)
    
    # 加载LoRA微调后的模型
    base_model = AutoModelForSequenceClassification.from_pretrained(args.ckpt_init)
    m1 = PeftModel.from_pretrained(base_model, args.ckpt_final)
    
    # 合并LoRA权重到基础模型，以便直接比较
    m1 = m1.merge_and_unload()

    num_layers = len(m0.deberta.encoder.layer)
    num_heads = m0.config.num_attention_heads
    hidden = m0.config.hidden_size
    head_dim = hidden // num_heads

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    
    with open(args.out_jsonl, "w") as f:
        for l in range(num_layers):
            layer0 = m0.deberta.encoder.layer[l]
            layer1 = m1.deberta.encoder.layer[l]

            # 获取初始权重
            Wq0 = layer0.attention.self.query_proj.weight.data
            Wk0 = layer0.attention.self.key_proj.weight.data
            Wv0 = layer0.attention.self.value_proj.weight.data
            Wo0 = layer0.attention.output.dense.weight.data

            # 获取微调后权重（已合并LoRA）
            Wq1 = layer1.attention.self.query_proj.weight.data
            Wk1 = layer1.attention.self.key_proj.weight.data
            Wv1 = layer1.attention.self.value_proj.weight.data
            Wo1 = layer1.attention.output.dense.weight.data

            # 对每个head计算更新量
            for h in range(num_heads):
                # Q/K/V: head在输出维度
                dq = slice_out_dim(Wq1, h, head_dim) - slice_out_dim(Wq0, h, head_dim)
                dk = slice_out_dim(Wk1, h, head_dim) - slice_out_dim(Wk0, h, head_dim)
                dv = slice_out_dim(Wv1, h, head_dim) - slice_out_dim(Wv0, h, head_dim)

                # O: head在输入维度
                do = slice_in_dim(Wo1, h, head_dim) - slice_in_dim(Wo0, h, head_dim)

                # 计算Frobenius范数
                Uq, Uk, Uv, Uo = frob(dq), frob(dk), frob(dv), frob(do)
                U = (Uq**2 + Uk**2 + Uv**2 + Uo**2) ** 0.5

                # 相对更新量（归一化到初始权重范数）
                nq = frob(slice_out_dim(Wq0, h, head_dim)) + args.eps
                nk = frob(slice_out_dim(Wk0, h, head_dim)) + args.eps
                nv = frob(slice_out_dim(Wv0, h, head_dim)) + args.eps
                no = frob(slice_in_dim(Wo0, h, head_dim)) + args.eps
                Urel = U / (nq + nk + nv + no)

                rec = dict(
                    layer=l, 
                    head=h, 
                    U=U, 
                    Urel=Urel, 
                    Uq=Uq, 
                    Uk=Uk, 
                    Uv=Uv, 
                    Uo=Uo
                )
                f.write(json.dumps(rec) + "\n")

    print(f"LoRA更新量已保存到: {args.out_jsonl}")
    print(f"总共测量了 {num_layers * num_heads} 个heads")

if __name__ == "__main__":
    main()
