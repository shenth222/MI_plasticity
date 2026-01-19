# src/measure/update_magnitude.py
import os, json, argparse
import torch
from transformers import AutoModelForSequenceClassification

def slice_out_dim(W, h, head_dim):
    # W: [out, in] if Linear weight in PyTorch is [out_features, in_features]
    o0 = h * head_dim
    o1 = (h + 1) * head_dim
    return W[o0:o1, :]

def slice_in_dim(W, h, head_dim):
    i0 = h * head_dim
    i1 = (h + 1) * head_dim
    return W[:, i0:i1]

def frob(x):
    return float(torch.norm(x, p="fro"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_init", required=True)
    ap.add_argument("--ckpt_final", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    m0 = AutoModelForSequenceClassification.from_pretrained(args.ckpt_init)
    m1 = AutoModelForSequenceClassification.from_pretrained(args.ckpt_final)

    num_layers = len(m0.deberta.encoder.layer)
    num_heads = m0.config.num_attention_heads
    hidden = m0.config.hidden_size
    head_dim = hidden // num_heads

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for l in range(num_layers):
            layer0 = m0.deberta.encoder.layer[l]
            layer1 = m1.deberta.encoder.layer[l]

            Wq0 = layer0.attention.self.query_proj.weight.data
            Wk0 = layer0.attention.self.key_proj.weight.data
            Wv0 = layer0.attention.self.value_proj.weight.data
            Wo0 = layer0.attention.output.dense.weight.data  # [out=hidden, in=hidden]

            Wq1 = layer1.attention.self.query_proj.weight.data
            Wk1 = layer1.attention.self.key_proj.weight.data
            Wv1 = layer1.attention.self.value_proj.weight.data
            Wo1 = layer1.attention.output.dense.weight.data

            # Note: PyTorch Linear weight shape is [out_features, in_features]
            for h in range(num_heads):
                # For Q/K/V: head lives on OUT dimension (each head produces head_dim outputs)
                dq = slice_out_dim(Wq1, h, head_dim) - slice_out_dim(Wq0, h, head_dim)
                dk = slice_out_dim(Wk1, h, head_dim) - slice_out_dim(Wk0, h, head_dim)
                dv = slice_out_dim(Wv1, h, head_dim) - slice_out_dim(Wv0, h, head_dim)

                # For O: head lives on IN dimension (it mixes concatenated heads)
                do = slice_in_dim(Wo1, h, head_dim) - slice_in_dim(Wo0, h, head_dim)

                Uq, Uk, Uv, Uo = frob(dq), frob(dk), frob(dv), frob(do)
                U = (Uq**2 + Uk**2 + Uv**2 + Uo**2) ** 0.5

                # Relative (normalize by initial slice norms)
                nq = frob(slice_out_dim(Wq0, h, head_dim)) + args.eps
                nk = frob(slice_out_dim(Wk0, h, head_dim)) + args.eps
                nv = frob(slice_out_dim(Wv0, h, head_dim)) + args.eps
                no = frob(slice_in_dim(Wo0, h, head_dim)) + args.eps
                Urel = U / (nq + nk + nv + no)

                rec = dict(layer=l, head=h, U=U, Urel=Urel, Uq=Uq, Uk=Uk, Uv=Uv, Uo=Uo)
                f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
