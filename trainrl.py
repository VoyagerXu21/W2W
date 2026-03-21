import json
import logging
import math
import os
import random
import cv2
import datasets
import evaluate
import numpy as np
import torch
from trl import PPOTrainer, PPOConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.converter import batch_text2traj, text2traj

import transformers
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, \
    get_scheduler

from model.nltoolkit import init_nltk, postprocess_text

logger = get_logger(__name__)


def preprocess_function(examples):
    inputs = examples[history_column]
    targets = examples[future_column]
    model_inputs = tokenizer(inputs, max_length=cfg.max_source_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=targets, max_length=cfg.max_target_length, padding=padding, truncation=True)

    if padding == "max_length":
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in
                               labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


from accelerate import Accelerator
from trl import PPOTrainer, PPOConfig


def main_rl_training(cfg, tokenizer, train_dataset, masks):
    # === 0. 初始化 Accelerator ===
    accelerator = Accelerator()
    print(f"[RL] Detected device: {accelerator.device}")

    # === 1. 加载CE模型 ===
    if not os.path.isdir(cfg.ce_checkpoint_dir):
        raise ValueError(f"CE checkpoint dir not found: {cfg.ce_checkpoint_dir}")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.ce_checkpoint_dir,
        use_safetensors=cfg.use_safetensors
    )
    model.to(accelerator.device)  # 用 accelerate 管理设备

    # === 2. 多卡 DataParallel（不推荐，accelerate 会处理） ===
    if getattr(cfg, "use_dataparallel", False) and torch.cuda.device_count() > 1:
        print(f"[RL] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # === 3. 初始化 PPOConfig ===
    ppo_config = PPOConfig(
        learning_rate=getattr(cfg, "ppo_learning_rate", 1e-5),
        batch_size=getattr(cfg, "ppo_batch_size", 128),
        num_ppo_epochs=getattr(cfg, "ppo_epochs_per_batch", 8),
    )

    # === 4. 创建 DataLoader ===
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100
    )
    rl_dataloader = DataLoader(
        train_dataset,
        batch_size=getattr(cfg, "ppo_batch_size", 128),
        shuffle=True,
        collate_fn=data_collator
    )

    # === 5. 初始化 PPOTrainer（trl 0.18.0 新写法） ===
    ppo_trainer = PPOTrainer(
        ppo_config,
        model=model
    )

    # === 6. 运行 RL 训练 ===
    run_rl_stage(cfg, tokenizer, model, ppo_trainer, rl_dataloader, masks, accelerator.device)

    print("[RL] Training finished.")
    return ppo_trainer, model


def run_rl_stage(cfg, tokenizer, model, ppo_trainer, rl_dataloader, masks, device):
    model.train()

    # RL loop
    global_step = 0
    for epoch in range(getattr(cfg, "ppo_train_epochs", 1)):
        print(f"[RL] Epoch {epoch + 1}/{getattr(cfg, 'ppo_train_epochs', 1)}")
        for batch in rl_dataloader:
            # --- prepare prompts (queries) as strings ---
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            queries_all = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            # move tensors for generation
            input_ids_device = input_ids.to(device)
            attention_mask_device = attention_mask.to(device) if attention_mask is not None else None

            # --- generate with current policy (sampling) ---
            # === CHANGED ===: use model.generate to produce text, then pass strings to ppo_trainer.step()
            gen_ids = model.generate(
                input_ids=input_ids_device,
                attention_mask=attention_mask_device,
                max_length=getattr(cfg, "max_length", 128),
                do_sample=True,
                top_k=getattr(cfg, "gen_top_k", 50),
                top_p=getattr(cfg, "gen_top_p", 0.95),
                temperature=getattr(cfg, "gen_temperature", 1.0),
                num_return_sequences=1,
            )
            pred_texts_all = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # --- parse generated texts into trajectories and keep valid indices ---
            pred_trajs_list = batch_text2traj(pred_texts_all, frame=cfg.pred_len)
            valid_indices = [i for i, t in enumerate(pred_trajs_list) if t is not None]
            if len(valid_indices) == 0:
                continue

            # --- get GT texts and parse ---
            all_gt_texts = tokenizer.batch_decode(
                batch["labels"].masked_fill(batch["labels"] == -100, tokenizer.pad_token_id),
                skip_special_tokens=True
            )
            gt_trajs_list = []
            ok = True
            for i in valid_indices:
                traj = text2traj(all_gt_texts[i], frame=cfg.pred_len)
                if traj is None:
                    ok = False
                    break
                gt_trajs_list.append(torch.from_numpy(traj).float().to(device))
            if not ok:
                continue

            # --- compute rewards for the valid subset ---
            rewards = []
            queries_valid = []
            responses_valid = []
            for idx_i, batch_i in enumerate(valid_indices):
                pred_coords = torch.from_numpy(pred_trajs_list[batch_i]).float().to(device)
                gt_coords = gt_trajs_list[idx_i]
                scene = batch["scene"][batch_i]
                r = 0.0
                if scene in masks:
                    mask = masks[scene].to(device)
                    coords = pred_coords.round().long()
                    h, w = mask.shape
                    coords[..., 0] = coords[..., 0].clamp(0, w - 1)
                    coords[..., 1] = coords[..., 1].clamp(0, h - 1)
                    collision_rate = mask[coords[..., 1], coords[..., 0]].float().mean().item()
                    r -= float(getattr(cfg, "collision_weight", 1.0)) * float(collision_rate)
                l2_err = torch.norm(pred_coords - gt_coords, p=2, dim=-1).mean().item()
                r -= float(getattr(cfg, "l2_weight", 1.0)) * float(l2_err)

                rewards.append(r)
                queries_valid.append(queries_all[batch_i])
                responses_valid.append(pred_texts_all[batch_i])

            # optional normalize rewards
            if len(rewards) > 1 and getattr(cfg, "normalize_rewards", True):
                r_mean = float(np.mean(rewards))
                r_std = float(np.std(rewards)) + 1e-8
                rewards = [(r - r_mean) / r_std for r in rewards]

            # === CHANGED: pass string lists to ppo_trainer.step() ===
            # trl will internally compute logprobs / advantages for these string responses
            ppo_trainer.step(queries_valid, responses_valid, rewards)

            # === ADDED: optional hybrid CE small-step (independent optimizer) ===
            if getattr(cfg, "hybrid_rl", False):
                raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                ce_input_ids = torch.stack([input_ids_device[i] for i in valid_indices]).to(device)
                ce_attention_mask = attention_mask_device.to(device) if attention_mask_device is not None else None
                ce_labels = torch.stack([batch["labels"][i] for i in valid_indices]).to(device)

                ce_outputs = raw_model(input_ids=ce_input_ids, attention_mask=ce_attention_mask, labels=ce_labels)
                ce_loss = ce_outputs.loss * float(getattr(cfg, "ce_weight", 0.1))

                # lazy init CE optimizer attached to raw_model (so we don't interfere with PPOTrainer internals)
                if not hasattr(run_rl_stage, "_ce_optimizer"):
                    run_rl_stage._ce_optimizer = torch.optim.AdamW(raw_model.parameters(),
                                                                   lr=getattr(cfg, "ce_lr", 1e-5))
                ce_optimizer = run_rl_stage._ce_optimizer
                ce_optimizer.zero_grad()
                ce_loss.backward()
                ce_optimizer.step()

            global_step += 1
            # logging & checkpointing
            if global_step % getattr(cfg, "rl_log_steps", 100) == 0:
                avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
                print(f"[RL] step={global_step} avg_reward={avg_reward:.4f} sample_resp={responses_valid[0]}")

            if global_step % getattr(cfg, "rl_save_steps", 1000) == 0:
                save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                out_dir = os.path.join(cfg.rl_checkpoint_dir, f"step_{global_step}")
                os.makedirs(out_dir, exist_ok=True)
                save_model.save_pretrained(out_dir, safe_serialization=cfg.use_safetensors)
                tokenizer.save_pretrained(out_dir)
                print(f"[RL] Saved checkpoint to {out_dir}")

        # end of epoch save
        out_dir = os.path.join(cfg.rl_checkpoint_dir, f"epoch_{epoch + 1}")
        os.makedirs(out_dir, exist_ok=True)
        save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        save_model.save_pretrained(out_dir, safe_serialization=cfg.use_safetensors)
        tokenizer.save_pretrained(out_dir)
        print(f"[RL] End epoch {epoch + 1}, saved to {out_dir}")

    print("[RL] Training finished.")
    return ppo_trainer, model

if __name__ == "__main__":
    import argparse
    import os
    import cv2
    import torch
    import numpy as np
    from datasets import load_dataset
    from utils.config import get_exp_config, DotDict
    # 其他必要导入...

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/config.json", type=str, help="Config file path.")
    parser.add_argument('--dataset', default="eth", type=str, help="Dataset name.",
                        choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument('--tag', default="LMTraj", type=str, help="Personal tag for the model.")
    parser.add_argument('--test', default=False, action='store_true', help="Evaluation mode.")
    args = parser.parse_args()


    def print_arguments(args_dict):
        for k, v in args_dict.items():
            print(f"{k}: {v}")


    print("===== Arguments =====")
    print_arguments(vars(args))

    cfg = get_exp_config(args.cfg)
    print("===== Configs =====")
    print_arguments(cfg)

    # 覆盖配置项
    cfg.dataset_name = args.dataset
    cfg.checkpoint_name = args.tag
    cfg.test_mode = args.test
    cfg = DotDict(cfg)

    # 初始化nltk
    init_nltk()

    preprocessed_train_dataset_name = f"{cfg.dataset_name}-train-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}-multimodal.json"
    preprocessed_val_dataset_name = f"{cfg.dataset_name}-val-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json"
    preprocessed_dataset_path = os.path.join(cfg.dataset_path, "preprocessed")

    data_files = {
        "train": os.path.join(preprocessed_dataset_path, preprocessed_train_dataset_name),
        "validation": os.path.join(preprocessed_dataset_path, preprocessed_val_dataset_name)
    }

    if not os.path.exists(data_files["train"]) or not os.path.exists(data_files["validation"]):
        raise ValueError(
            f"Preprocessed dataset files not found: {data_files['train']} or {data_files['validation']}. Please run `./script/preprocessor.sh` first.")

    extension = "json"  # 直接写死，避免错误
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cfg.cache_dir)

    # 后续模型、tokenizer加载等代码...
    config = AutoConfig.from_pretrained(cfg.ce_checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.ce_checkpoint_dir, config=config)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    if cfg.tokenizer_name is not None:
        model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names

    history_column = cfg.history_column
    if history_column not in column_names:
        raise ValueError(
            f"--history_column' value '{cfg.history_column}' needs to be one of: {', '.join(column_names)}")
    future_column = cfg.future_column
    if future_column not in column_names:
        raise ValueError(f"--future_column' value '{cfg.future_column}' needs to be one of: {', '.join(column_names)}")

    padding = "max_length" if cfg.pad_to_max_length else False
    # 处理masks
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
    masks = {}
    for scene_name, path in mask_paths.items():
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {path}")
        mask = (mask < 128).astype(np.uint8)
        masks[scene_name] = torch.from_numpy(mask).cpu()

    # 预处理数据集
    train_dataset = raw_datasets["train"].map(preprocess_function,
                                              batched=True,
                                              num_proc=cfg.preprocessing_num_workers,
                                              remove_columns=raw_datasets["train"].column_names,
                                              load_from_cache_file=not cfg.overwrite_cache,
                                              desc="Running tokenizer on train dataset")

    val_dataset = raw_datasets["validation"].map(preprocess_function,
                                                 batched=True,
                                                 num_proc=cfg.preprocessing_num_workers,
                                                 remove_columns=raw_datasets["validation"].column_names,
                                                 load_from_cache_file=not cfg.overwrite_cache,
                                                 desc="Running tokenizer on val dataset")

    if cfg.run_rl:
        main_rl_training(cfg, tokenizer, train_dataset, masks)



