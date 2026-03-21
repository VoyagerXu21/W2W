import os, torch
from safetensors.torch import save_file

src = "./checkpoint/LMTraj-SUP-RLR/pytorch_model.bin"   # ← 改成你的 .bin 路径
dst = os.path.join(os.path.dirname(src), "model.safetensors")

# 1) 加载 state_dict（尽量用 weights_only）
try:
    state = torch.load(src, map_location="cpu", weights_only=True)
except TypeError:
    state = torch.load(src, map_location="cpu")

# 2) 识别共享内存的参数组（按底层 storage 指针聚类）
groups = {}
for k, v in list(state.items()):
    if not torch.is_tensor(v):
        continue
    sid = int(v.untyped_storage().data_ptr())
    groups.setdefault(sid, []).append(k)

# 3) 对每个“共享组”，除第一个键以外都 clone 一份，打断别名
n_fixed = 0
for ks in groups.values():
    if len(ks) > 1:
        master = ks[0]
        for k in ks[1:]:
            if state[k].untyped_storage().data_ptr() == state[master].untyped_storage().data_ptr():
                state[k] = state[k].clone()
                n_fixed += 1
print(f"[INFO] dedup cloned tensors: {n_fixed}")

# 4) 保存为 safetensors
save_file(state, dst)
print("✅ saved:", dst)