import json
import logging
import math
import os
import random
import cv2
import datasets
import evaluate
import torch
from sympy.physics.units import temperature
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoConfig,
    DataCollatorForSeq2Seq, set_seed, CONFIG_MAPPING, logging as transformers_logging
)
from trl import (
    AutoModelForSeq2SeqLMWithValueHead, create_reference_model,
    PPOTrainer, PPOConfig
)
import numpy as np
import torch
from trl import PPOTrainer, PPOConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from utils.converter import text2traj, batch_text2traj
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, \
    get_scheduler

from model.nltoolkit import init_nltk, postprocess_text

logger = get_logger(__name__)

from transformers import T5ForConditionalGeneration
import torch.nn as nn

from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType
from peft import get_peft_model

def guess_lora_targets(policy_model):
    mt = getattr(getattr(policy_model, "config", None), "model_type", "").lower()
    is_encdec = bool(getattr(getattr(policy_model, "config", None), "is_encoder_decoder", False))
    if is_encdec:
        # T5/UL2/Flan-T5 一类
        return ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
    if "llama" in mt or "mistral" in mt or "qwen" in mt or "gemma" in mt:
        # LLaMA/Mistral/Qwen/Gemma
        return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    if "gptj" in mt or "gpt_neox" in mt:
        return ["q_proj","k_proj","v_proj","out_proj","fc_in","fc_out"]
    if "gpt2" in mt:
        return ["c_attn","c_proj","c_fc"]
    # 兜底：返回 None 时，PEFT 会匹配所有线性层；必要时你可以打印模块名再精修
    return None

class PPOCollatorKeepScene:
    """
    只在 PPO 阶段使用：
    - 先 pop 掉字符串的 'scene'，避免 tokenizer.pad 触碰到它
    - 其余字段交给 tokenizer.pad
    - 再把 'scene' 以 Python list[str] 的形式放回 batch
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        scenes = [f.pop("scene", None) for f in features]  # <- 关键：从样本中先移走 scene
        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch["scene"] = scenes  # 以 list[str] 放回
        return batch

class DataCollatorWithScene:
    """
    Wraps a transformers DataCollator (for seq2seq) but preserves a 'scene' field
    as Python list[str] instead of letting the collator/tokenizer try to turn it into a tensor.
    """
    def __init__(self, base_collator: DataCollatorForSeq2Seq):
        self.base = base_collator

    def __call__(self, examples):
        scenes = []
        examples_wo_scene = []
        for ex in examples:
            ex = ex.copy()              # 避免改动原始数据
            scenes.append(ex.pop("scene", None))  # 确保 scene 被移除
            examples_wo_scene.append(ex)

        collated = self.base(examples_wo_scene)  # 这里只会处理数值型字段
        collated["scene"] = scenes               # 再把 scene 加回
        return collated

from types import SimpleNamespace
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class ValueBackboneAdapter(nn.Module):
    def __init__(self, policy_model_name: str):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(policy_model_name)

    def forward(self, *args, **kwargs):
        local_kwargs = dict(kwargs)

        has_decoder_input = any(
            (k in local_kwargs and local_kwargs[k] is not None)
            for k in ("decoder_input_ids", "decoder_inputs_embeds", "labels")
        )

        # Always request hidden states and disable cache for stable hidden retrieval
        common_call_kwargs = {"return_dict": True, "output_hidden_states": True, "use_cache": False}

        if has_decoder_input:
            call_kwargs = dict(common_call_kwargs)
            for k in (
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_inputs_embeds",
                "decoder_attention_mask",
                "labels",
                "encoder_outputs",
            ):
                if k in local_kwargs:
                    call_kwargs[k] = local_kwargs[k]

            out = self.t5(**call_kwargs)  # Seq2SeqLMOutput

            # --- Robust extraction of a single "last-layer" hidden tensor ---
            hidden_states = None

            # prefer decoder hidden states (last layer)
            if getattr(out, "decoder_hidden_states", None) is not None:
                # decoder_hidden_states is tuple(len_layers, B, T_dec, D)
                dhs = tuple(out.decoder_hidden_states)
                hidden_states = (dhs[-1],) if len(dhs) > 0 else None

            # some variants may expose .hidden_states (take last)
            elif getattr(out, "hidden_states", None) is not None:
                h = out.hidden_states
                hidden_states = (h[-1],) if isinstance(h, (list, tuple)) and len(h) > 0 else None

            # fallback to encoder hidden states (last encoder layer) — still useful for value
            elif getattr(out, "encoder_hidden_states", None) is not None:
                ehs = tuple(out.encoder_hidden_states)
                hidden_states = (ehs[-1],) if len(ehs) > 0 else None

            # further fallback to encoder_last_hidden_state / last_hidden_state
            elif getattr(out, "encoder_last_hidden_state", None) is not None:
                hidden_states = (out.encoder_last_hidden_state,)
            elif getattr(out, "last_hidden_state", None) is not None:
                hidden_states = (out.last_hidden_state,)

            if hidden_states is None:
                raise RuntimeError("Adapter: couldn't extract hidden states from full T5 output")

            return SimpleNamespace(hidden_states=hidden_states, raw_output=out)

        else:
            # encoder-only path
            enc_call_kwargs = dict(common_call_kwargs)
            for k in ("input_ids", "attention_mask"):
                if k in local_kwargs:
                    enc_call_kwargs[k] = local_kwargs[k]

            enc_out = self.t5.encoder(**enc_call_kwargs)

            if getattr(enc_out, "hidden_states", None) is not None:
                hs = tuple(enc_out.hidden_states)
                hidden_states = (hs[-1],) if len(hs) > 0 else None
            elif getattr(enc_out, "last_hidden_state", None) is not None:
                hidden_states = (enc_out.last_hidden_state,)
            elif getattr(enc_out, "encoder_last_hidden_state", None) is not None:
                hidden_states = (enc_out.encoder_last_hidden_state,)
            else:
                raise RuntimeError("Adapter: couldn't extract hidden states from encoder output")

            return SimpleNamespace(hidden_states=hidden_states, raw_output=enc_out)

class SharedValueModel(nn.Module):
    base_model_prefix = "shared_value_model"

    def __init__(self, policy_model_name="t5-small", freeze_backbone=True):
        super().__init__()
        self._backbone_adapter = ValueBackboneAdapter(policy_model_name)
        self.shared_value_model = self._backbone_adapter

        d_model = self._backbone_adapter.t5.config.d_model
        self.value_head = nn.Linear(d_model, 1)

        if freeze_backbone:
            for p in self._backbone_adapter.parameters():
                p.requires_grad = False

        for p in self.value_head.parameters():
            p.requires_grad = True

    def forward(self, *args, **kwargs):
        out = self.shared_value_model(*args, **kwargs)
        hidden_states = out.hidden_states
        last_hidden = hidden_states[-1]  # [B, T, D]
        values = self.value_head(last_hidden).squeeze(-1)  # [B, T]
        return values

    def score(self, hidden_states: torch.Tensor):
        return self.value_head(hidden_states).squeeze(-1)

class RewardModel(torch.nn.Module):
    def __init__(self, tokenizer, masks, device, penalty_weight=0.5, l2_weight=1.5, debug=False, id2scene=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.masks = masks            # dict: {scene_name: bool tensor [H, W], True=不可走（示例假设）}
        self.device = device
        self.penalty_weight = penalty_weight
        self.l2_weight = l2_weight
        self.debug = debug
        self.id2scene = id2scene or {}

    def check_penalty(self, traj, scene):
        """
        traj: np.array or torch.Tensor of shape [T, 2]
        scene: str
        return: int (惩罚次数)
        """
        if scene not in self.masks:
            return 0

        mask = self.masks[scene]                  # bool tensor on device
        traj_tensor = torch.as_tensor(traj, device=mask.device)
        coords = traj_tensor.round().long()       # [T, 2]
        xs, ys = coords[:, 0], coords[:, 1]

        valid = (xs >= 0) & (ys >= 0) & (ys < mask.shape[0]) & (xs < mask.shape[1])
        if not torch.any(valid):
            return 0

        penalty_points = mask[ys[valid], xs[valid]]   # True 表示不可走
        return int(penalty_points.sum().item())

    @torch.no_grad()
    def forward(self, queries, responses, batch):
        """
        返回 shape = [B] 的 1D tensor（每个样本一个标量 reward）。
        - 兼容 responses 为 tensor 或 list[tensor]
        - 兼容 labels 为 tensor 或 numpy
        - 使用运行时的 device（以 responses 或 model 参数为准）
        """
        import torch
        import numpy as np

        # --- 决定 batch 大小和 device ---
        if isinstance(responses, torch.Tensor):
            B = responses.size(0)
            in_device = responses.device
        elif isinstance(responses, list):
            B = len(responses)
            # 如果列表里有 tensor，则取第一个 tensor 的 device
            first = next((r for r in responses if isinstance(r, torch.Tensor)), None)
            in_device = first.device if first is not None else (
                next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device(
                    "cpu"))
        else:
            raise RuntimeError("Unsupported responses type in RewardModel.forward")

        # Prefer the runtime device (so we don't rely on self.device arg at init)
        device = in_device

        # --- 准备 labels 和 scene_ids ---
        labels = batch.get("labels", None)
        if labels is None:
            raise RuntimeError("RewardModel requires batch['labels']")
        if isinstance(labels, torch.Tensor):
            labels = torch.where(labels != -100,
                                 labels,
                                 torch.full_like(labels, self.tokenizer.pad_token_id))
            labels_list = labels.detach().cpu().tolist()
        elif isinstance(labels, np.ndarray):
            labels_list = np.where(labels != -100, labels, self.tokenizer.pad_token_id).tolist()
        else:
            labels_list = list(labels)  # assume already list of lists

        scene_ids = batch.get("scene_id", [None] * B)
        if isinstance(scene_ids, torch.Tensor):
            scene_ids = scene_ids.detach().cpu().tolist()
        elif isinstance(scene_ids, np.ndarray):
            scene_ids = scene_ids.tolist()

        # --- 准备 responses -> texts ---
        # 尽量减少 CPU↔GPU 往返；只在需要时 decode
        if isinstance(responses, torch.Tensor):
            responses_list = responses.detach().tolist()
        elif isinstance(responses, list) and len(responses) > 0 and isinstance(responses[0], torch.Tensor):
            responses_list = [r.detach().tolist() for r in responses]
        else:
            responses_list = responses

        gen_texts = self.tokenizer.batch_decode(responses_list, skip_special_tokens=True)
        ref_texts = self.tokenizer.batch_decode(labels_list,   skip_special_tokens=True)

        out_len = min(len(gen_texts), len(ref_texts), B)

        rewards = []
        for idx in range(out_len):
            gen_text = gen_texts[idx].strip() if isinstance(gen_texts[idx], str) else ""
            ref_text = ref_texts[idx].strip() if isinstance(ref_texts[idx], str) else ""
            sid = int(scene_ids[idx]) if idx < len(scene_ids) and scene_ids[idx] is not None else None
            scene = self.id2scene.get(sid, None)

            # empty / parse fail safety
            if (not gen_text):
                rewards.append(-10)
                continue

            gen_traj = text2traj(gen_text)
            ref_traj = text2traj(ref_text)
            if gen_traj is None or ref_traj is None or len(gen_traj) == 0 or len(ref_traj) == 0:
                rewards.append(-1.0)
                continue

            # compute penalty and L2
            penalty_count = self.check_penalty(gen_traj, scene)
            penalty_value = penalty_count * self.penalty_weight

            T = min(len(gen_traj), len(ref_traj))
            if T == 0:
                rewards.append(-1.0)
                continue
            gen_t = np.array(gen_traj[:T])
            ref_t = np.array(ref_traj[:T])
            l2_error = float(np.mean(np.linalg.norm(gen_t - ref_t, axis=1)))
            l2_reward = -l2_error * self.l2_weight

            final_reward = l2_reward - penalty_value
            rewards.append(final_reward)

        # If some weird length mismatch, pad/trim to exactly B
        if len(rewards) < B:
            # pad with very low reward (or zeros) — 这里用 -1.0 保守处理
            rewards.extend([-1.0] * (B - len(rewards)))
        elif len(rewards) > B:
            rewards = rewards[:B]

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        # 末尾形状不符：静默裁剪/填充后返回
        if rewards_tensor.dim() != 1 or rewards_tensor.size(0) != B:
            rewards_tensor = rewards_tensor.reshape(-1)[:B]
            if rewards_tensor.size(0) < B:
                pad = torch.full((B - rewards_tensor.size(0),), -1.0, dtype=torch.float32, device=device)
                rewards_tensor = torch.cat([rewards_tensor, pad], dim=0)

        return rewards_tensor

def trainval(cfg):
    # —— Imports（放函数内，避免全局污染）——
    import os, logging
    import cv2, torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq,
        set_seed
    )
    from trl import (
        AutoModelForSeq2SeqLMWithValueHead, create_reference_model,
        PPOTrainer, PPOConfig
    )

    # ---------- 初始化 ----------
    init_nltk()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    # ---------- 数据加载 ----------
    preprocessed_train_dataset_name = f"{cfg.dataset_name}-train-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}-multimodal.json"
    preprocessed_val_dataset_name = f"{cfg.dataset_name}-val-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json"
    preprocessed_dataset_path = os.path.join(cfg.dataset_path, "preprocessed")

    data_files = {
        "train": os.path.join(preprocessed_dataset_path, preprocessed_train_dataset_name),
        "validation": os.path.join(preprocessed_dataset_path, preprocessed_val_dataset_name)
    }
    for _, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessed dataset not found: {path}")

    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cfg.cache_dir)

    # ---------- 模型与 tokenizer ----------
    config = AutoConfig.from_pretrained(
        cfg.model_config_name or cfg.model_name_or_path,
        trust_remote_code=False, cache_dir=cfg.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name or cfg.model_name_or_path,
        trust_remote_code=False, cache_dir=cfg.cache_dir,
        use_fast=not cfg.use_slow_tokenizer
    )

    # pad 兜底
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        cfg.model_name_or_path, config=config, trust_remote_code=False, cache_dir=cfg.cache_dir
    )
    if hasattr(model, "pretrained_model"):
        model.pretrained_model.config.use_cache = False
        try:
            model.pretrained_model.gradient_checkpointing_enable()
        except Exception:
            pass

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    # 确保 pad ≠ eos
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.pretrained_model.resize_token_embeddings(len(tokenizer))

    ref_model = create_reference_model(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    print("pad_token:", tokenizer.pad_token, "=>", tokenizer.pad_token_id)
    print("eos_token:", tokenizer.eos_token, "=>", tokenizer.eos_token_id)

    # 2. 给 **原始模型** 设置 generation_config
    model.pretrained_model.generation_config.decoder_start_token_id = 0
    ref_model.pretrained_model.generation_config.decoder_start_token_id = 0

    # ---------- 强制对齐 generation_config，确保训练安全 ----------
    tk_pad = getattr(tokenizer, "pad_token_id", None)
    tk_eos = getattr(tokenizer, "eos_token_id", None)
    dec_start = getattr(model.pretrained_model.config, "decoder_start_token_id", None)

    # 确保 policy model 有 generation_config
    if not hasattr(model, "generation_config") and hasattr(model, "pretrained_model"):
        model.generation_config = model.pretrained_model.generation_config
    # 确保 ref model 有 generation_config
    if not hasattr(ref_model, "generation_config") and hasattr(ref_model, "pretrained_model"):
        ref_model.generation_config = ref_model.pretrained_model.generation_config

    # 对齐 tokenizer 和 config
    if tk_pad is not None:
        model.generation_config.pad_token_id = int(tk_pad)
        ref_model.generation_config.pad_token_id = int(tk_pad)
    if tk_eos is not None:
        model.generation_config.eos_token_id = int(tk_eos)
        ref_model.generation_config.eos_token_id = int(tk_eos)
    if dec_start is not None:
        model.generation_config.decoder_start_token_id = int(dec_start)
        ref_model.generation_config.decoder_start_token_id = int(dec_start)



    # 打印检查
    print("== Final generation_config ==")
    print("Policy pad_token_id:", model.generation_config.pad_token_id)
    print("Policy eos_token_id:", model.generation_config.eos_token_id)
    print("Policy decoder_start_token_id:", model.generation_config.decoder_start_token_id)
    print("Ref pad_token_id:", ref_model.generation_config.pad_token_id)
    print("Ref eos_token_id:", ref_model.generation_config.eos_token_id)
    print("Ref decoder_start_token_id:", ref_model.generation_config.decoder_start_token_id)

    # 兼容补丁：部分环境下 ValueHead wrapper 不暴露 generation_config
    if not hasattr(model, "generation_config") and hasattr(model, "pretrained_model"):
        model.generation_config = model.pretrained_model.generation_config
    if not hasattr(ref_model, "generation_config") and hasattr(ref_model, "pretrained_model"):
        ref_model.generation_config = ref_model.pretrained_model.generation_config

    # 调整 tokenizer embedding（仅 policy；ref_model 由 create_reference_model 复制，保持一致即可）
    if len(tokenizer) > model.pretrained_model.get_input_embeddings().weight.shape[0]:
        model.pretrained_model.resize_token_embeddings(len(tokenizer))

    if model.pretrained_model.config.decoder_start_token_id is None:
        raise ValueError("decoder_start_token_id must be defined in config")

    # ---------- 场景 mask ----------
    mask_paths = {
        "biwi_eth": "./datasets/masks/eth_mask.png",
        "biwi_hotel": "./datasets/masks/hotel_mask.png",
        "crowds_zara01": "./datasets/masks/zara1_mask.png",
        "crowds_zara02": "./datasets/masks/zara2_mask.png",
        "students001": "./datasets/masks/univ_mask.png",
        "students003": "./datasets/masks/univ_mask.png",
        "crowds_zara03": "./datasets/masks/zara1_mask.png",
        "uni_examples": "./datasets/masks/univ_mask.png"
    }
    scene2id = {name: i for i, name in enumerate(mask_paths.keys())}
    id2scene = {i: name for name, i in scene2id.items()}

    device = torch.device("cpu")
    masks = {}
    for scene_name, path in mask_paths.items():
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {path}")
        masks[scene_name] = torch.tensor(mask < 128, dtype=torch.bool)  # 先放 CPU

    # ---------- 数据预处理 ----------
    column_names = raw_datasets["train"].column_names

    def preprocess_function(example):
        # 输入 observation
        inp = example[cfg.history_column]
        sc = example["scene"]

        if not isinstance(inp, str) or not inp.strip():
            inp = ""

        model_inputs = tokenizer(
            inp,
            max_length=cfg.max_source_length,
            padding="max_length" if cfg.pad_to_max_length else False,
            truncation=True,
        )

        # scene_id
        sid = scene2id.get(sc, -1)

        # labels 来自 future_column
        future = example[cfg.future_column]  # 例如 "forecast"
        if not isinstance(future, str) or not future.strip():
            future = ""

        labels = tokenizer(
            future,
            max_length=cfg.max_target_length,
            padding=False,  # 交给 DataCollatorForSeq2Seq 统一pad
            truncation=True,
        )["input_ids"]

        # 确保长度 >= 2
        ids = model_inputs["input_ids"]
        am = model_inputs["attention_mask"]
        if len(ids) < 2:
            ids.append(tokenizer.eos_token_id)
            am.append(1)

        if len(labels) < 2:
            labels.append(tokenizer.eos_token_id)

        return {
            "input_ids": ids,
            "attention_mask": am,
            "labels": labels,  # <<< 新增
            "scene_id": sid,
        }

    train_dataset = raw_datasets["train"].map(
        preprocess_function,
        batched=False,  # <<< 改成 False
        remove_columns=column_names,
    )

    val_dataset = raw_datasets["validation"].map(
        preprocess_function,
        batched=False,  # <<< 改成 False
        remove_columns=column_names,
    )

    # ---------- Reward & Value 模型 ----------
    # 这里假设你已在别处定义 RewardModel / SharedValueModel
    reward_model = RewardModel(tokenizer, masks, device, penalty_weight=0.5, l2_weight=1.0, id2scene=id2scene)
    value_model = SharedValueModel(policy_model_name=cfg.model_name_or_path, freeze_backbone=True).to(device)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.pretrained_model if hasattr(model, "pretrained_model") else model,
        label_pad_token_id=-100,  # 训练友好的label padding
        pad_to_multiple_of=8
    )

    # ---------- PPO 配置 ----------
    # ------- 稳健构造 PPOConfig（拷贝替换你原来创建 ppo_config 的地方） -------
    import os, torch
    from trl import PPOConfig

    def get_world_size_fallback():
        # robust fallback for world size detection
        if torch.distributed.is_initialized():
            try:
                return int(torch.distributed.get_world_size())
            except Exception:
                pass
        for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            v = os.environ.get(key)
            if v:
                try:
                    return max(1, int(v))
                except Exception:
                    pass
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cvd is not None:
            if cvd.strip() == "":
                return 1
            return max(1, len(cvd.split(",")))
        return 1

    # 你期望的 per-device batch 从 cfg 里读
    per_device = int(cfg.per_device_train_batch_size)  # e.g., 256
    world_size = get_world_size_fallback()
    total_batch = per_device * max(1, world_size)

    # 推荐的 num_mini_batches：尽量能分成合理的 local_mini_batch_size
    # local_mini_batch_size = (per_device * grad_accum) / num_mini_batches  应 >= 8 if whiten_rewards
    # 这里先给个合理默认（可按需改）
    guess_num_mini = max(1, min(32, total_batch // max(16, per_device)))  # heuristic fallback

    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=int(total_batch),  # global batch (will be recomputed but safe to set)
        mini_batch_size=int(max(1, total_batch // guess_num_mini)),
        num_mini_batches=int(guess_num_mini),
        per_device_train_batch_size=int(per_device),  # **关键：显式设置 per-device**
        gradient_accumulation_steps=int(getattr(cfg, "gradient_accumulation_steps", 1)),
        logging_steps=getattr(cfg, "logging_steps", 10),
        save_steps=getattr(cfg, "save_steps", 500),
        eval_steps=getattr(cfg, "eval_steps", None),
        output_dir=str(checkpoint_path),
        seed=cfg.seed,
        remove_unused_columns=False,
        num_sample_generations=2,
        save_safetensors=False,
        num_train_epochs=8,
        num_ppo_epochs=1,
        cliprange=0.05,
        cliprange_value=0.10,
        whiten_rewards=True,
        max_grad_norm=0.5,
        kl_estimator="k3",             # 更稳的 KL 估计器（若已默认则忽略）
        kl_coef=0.25,                   # 初始 KL 系数偏高一点，牵回参考分布
    )

    # >>> 紧接在 ppo_config 下面、创建 PPOTrainer 之前
    is_encdec = bool(getattr(model.config, "is_encoder_decoder", False))
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=guess_lora_targets(model),
        task_type=(TaskType.SEQ_2_SEQ_LM if is_encdec else TaskType.CAUSAL_LM),
        inference_mode=False
    )

    model.pretrained_model = get_peft_model(model.pretrained_model, lora_cfg)

    # generation kwargs（保留你原来的）
    ppo_config.generation_kwargs = {k: v for k, v in {
        "max_new_tokens": cfg.get("max_new_tokens", 64),
        "min_new_tokens": 1,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.85,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id) if getattr(tokenizer, "eos_token_id", None) is not None else None,
        "decoder_start_token_id": int(model.generation_config.decoder_start_token_id) if getattr(model,
                                                                                                 "generation_config",
                                                                                                 None) and getattr(
            model.generation_config, "decoder_start_token_id", None) is not None else None,
        "return_prompt": False,
        "remove_invalid_values": True,
    }.items() if v is not None}

    # Create trainer

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,  # policy：WithValueHead，底座已加 LoRA
        ref_model=ref_model,  # reference：WithValueHead 拷贝，底座无 LoRA
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=val_dataset,
        # peft_config=None           # 不写这一项，或显式传 None 都行
    )

    print("policy pretrained_model class:", type(model.pretrained_model))
    print("ref pretrained_model class:", type(ref_model.pretrained_model))

    # === 每个 epoch 保存一次：用“每个 epoch 的 step 数”作为 save_steps ===
    import math
    updates_per_epoch = max(1, math.ceil(len(train_dataset) / trainer.args.batch_size))
    trainer.args.save_steps = updates_per_epoch
    # （可选）日志更细一些
    trainer.args.logging_steps = max(1, updates_per_epoch // 2)

    print(f"[SAVE POLICY] updates_per_epoch={updates_per_epoch}, save_steps={trainer.args.save_steps}")

    # 让辅助资源在 trainer 建好后再放到正确设备
    accel_device = trainer.accelerator.device
    for k in list(masks.keys()):
        masks[k] = masks[k].to(accel_device, non_blocking=True)
    reward_model.device = accel_device
    # 立即检查 trainer.args，一定要打印确认！
    print("[CHECK AFTER CREATE] ppo_config.batch_size=", ppo_config.batch_size,
          "per_device_train_batch_size=", getattr(ppo_config, "per_device_train_batch_size", None),
          "num_mini_batches=", getattr(ppo_config, "num_mini_batches", None))
    print("[CHECK AFTER CREATE] trainer.args.batch_size=", trainer.args.batch_size,
          "local_batch_size=", getattr(trainer.args, "local_batch_size", None),
          "local_mini_batch_size=", getattr(trainer.args, "local_mini_batch_size", None),
          "per_device_train_batch_size=", getattr(trainer.args, "per_device_train_batch_size", None))


    # sample = train_dataset[0]
    # input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    # labels = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    #
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(input_ids=input_ids, labels=labels)

    # TRL 的 WithValueHead 返回的是 tuple
    # logits, loss, values = outputs
    # first_sample = logits[0]  # (seq_len, vocab_size)
    # print("logits[0, :5, :10] =")
    # print(first_sample[:5, :10])  # 5×10 的矩阵
    # print("loss:", loss.item())
    # print("values shape:", values.shape)  # (batch, seq_len, 1)

    # from torch.utils.data import DataLoader
    # dl = DataLoader(train_dataset, batch_size=cfg.per_device_train_batch_size,
    #                 collate_fn = data_collator, num_workers = 4, pin_memory = True, persistent_workers = True)
    # from trl.trainer.ppo_trainer import unwrap_model_for_generation
    #
    # for batch in dl:
    #     # batch 中通常含 input_ids 和 attention_mask（由 DataCollatorWithPadding 提供）
    #     input_ids = batch["input_ids"].to(accel_device, non_blocking=True)
    #     attn_mask = batch.get("attention_mask", None)
    #     if attn_mask is not None:
    #         attn_mask = attn_mask.to(accel_device, non_blocking=True)
    #
    #     # Use the unwrapped model that Trainer/Accelerator will generate with
    #     with unwrap_model_for_generation(trainer.model, trainer.accelerator) as gen_model:
    #         # gen_model now is the HF model that supports .generate() on the right device
    #         dec_start = getattr(getattr(gen_model, "generation_config", None), "decoder_start_token_id", None)
    #         if dec_start is None:
    #             # fallback to model config or tokenizer pad
    #             dec_start = getattr(getattr(gen_model, "config", None), "decoder_start_token_id", None)
    #         if dec_start is None:
    #             dec_start = tokenizer.pad_token_id
    #
    #         gen = gen_model.pretrained_model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attn_mask,
    #             min_new_tokens=1,
    #             max_new_tokens=50,
    #             do_sample=True,
    #             pad_token_id=int(tokenizer.pad_token_id),
    #             eos_token_id=(
    #                 int(tokenizer.eos_token_id) if getattr(tokenizer, "eos_token_id", None) is not None else None),
    #             decoder_start_token_id=int(dec_start) if dec_start is not None else None,
    #             # 可按需指定 temperature/top_k/top_p
    #         )

    #     # decode & quick check
    #     response = gen  # gen on same device as model
    #     # decode safely on CPU
    #     decoded = tokenizer.batch_decode(response.detach().cpu().tolist(), skip_special_tokens=True)
    #     print("Sample decode:", decoded[0][:400])
    #     lengths = (response != tokenizer.pad_token_id).sum(-1)
    #     if (lengths == 0).any():
    #         print("发现空 response！input_ids:", input_ids[lengths == 0])
    #         break
    # else:
    #     print("抽检 1 个 batch 无空 response")

    # device = torch.device("cpu")
    # vm = SharedValueModel(policy_model_name=cfg.model_name_or_path).to(device)
    #
    # B, qlen = 2, 12
    # dummy_q = torch.randint(0, 1000, (B, qlen), device=device)
    # dummy_attn = torch.ones_like(dummy_q, device=device)
    # resp_len = 6
    # dummy_dec = torch.randint(0, 1000, (B, resp_len), device=device)
    #
    # orig_kwargs = {"input_ids": dummy_q, "attention_mask": dummy_attn}
    # # call adapter without decoder inputs
    # out_enc = vm.shared_value_model(**orig_kwargs)
    # assert hasattr(out_enc, "hidden_states")
    # # ensure orig_kwargs was not mutated
    # assert "output_hidden_states" not in orig_kwargs
    #
    # # call adapter with decoder inputs
    # orig_kwargs2 = {"input_ids": dummy_q, "attention_mask": dummy_attn, "decoder_input_ids": dummy_dec}
    # out_dec = vm.shared_value_model(**orig_kwargs2)
    # assert hasattr(out_dec, "hidden_states")
    # assert "output_hidden_states" not in orig_kwargs2
    #
    # print("Adapter tests ok: orig kwargs unchanged and hidden_states returned.")

    trainer.train()
    logging.info("  - Trainer finishes!")

    # ---------- 保存 / 推送 ----------
    trainer.save_model(checkpoint_path)
    logging.info(f"Save model to {checkpoint_path}")

    if getattr(cfg, "push_to_hub", False):
        logging.info("  - Push checkpoints to hub")
        trainer.push_to_hub()

if __name__ == "__main__":
    from utils.config import get_exp_config, DotDict

    args = get_exp_config()
    cfg = DotDict(args.__dict__)
    trainval(cfg)

