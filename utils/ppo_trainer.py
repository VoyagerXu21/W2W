# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch.nn.functional as F
import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from accelerate.utils import DistributedDataParallelKwargs
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available

from trl.core import masked_mean, masked_whiten
from trl.models import create_reference_model
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb


INVALID_LOGPROB = -1e9
# === BEGIN PATCH 1: Global speed/silence toggles ===
import builtins as _bi
import os as _os
import torch as _torch
# 关闭本文件的零碎 print（临时想看日志：PPO_VERBOSE=1 python xxx.py）
if _os.environ.get("PPO_VERBOSE", "0") != "1":
    print = lambda *a, **k: None  # noqa: E731

# TF32 提速（Ampere+）
try:
    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.set_float32_matmul_precision("high")
except Exception:
    pass
# === END PATCH 1 ===
def _safe_logprob(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 在 float32 上做 log_softmax，避免半精度下溢
    lsm = F.log_softmax(logits.float(), dim=-1)
    out = lsm.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # 把非有限与极端值钳住，防止 exp 爆炸
    out = torch.nan_to_num(out, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)
    return out.to(logits.dtype)
# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

        # === 关键：属性转发（proxy）===
        # 让 wrapper“看起来像”policy，本质上只是把属性透传给内层
        self.config = getattr(policy, "config", None)
        self.generation_config = getattr(policy, "generation_config", None)
        self.pretrained_model = getattr(policy, "pretrained_model", None)

    def forward(self, **kwargs):
        FILTER_KEYS = {
            "output_hidden_states","return_dict","output_attentions","use_cache",
            "past_key_values","position_ids","decoder_position_ids",
        }
        # 先走 value 的 backbone（注意：这会单独算一遍 T5，代价大，见文末性能提示）
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])

        policy_kwargs = {k: v for k, v in kwargs.items() if k not in FILTER_KEYS}
        return self.policy(**policy_kwargs), logits

class PPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        args: PPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        ref_model: Optional[nn.Module],
        reward_model: nn.Module,
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int

        # Check that the kl estimator is valid
        if self.args.kl_estimator not in {"k1", "k3"}:
            raise ValueError(
                "kl_estimator must be either 'k1' (straightforward, unbiased) or 'k3' (lower variance, unbiased, "
                "appears to be a strictly better estimator). See "
                "[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) for details."
            )

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.policy_model)

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,  # ★ 关键
            broadcast_buffers=False,  # 建议关闭，减少无效同步
            gradient_as_bucket_view=True,  # 可选
        )

        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=(
                "bf16" if getattr(args, "bf16", False) else ("fp16" if getattr(args, "fp16", False) else "no")),
            device_placement=True,
            kwargs_handlers=[ddp_kwargs],  # ★ 关键
        )
        self.accelerator = accelerator

        args.world_size = accelerator.num_processes
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert args.local_mini_batch_size >= 8, (
                f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
            )
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        # --- ADD: rollout 前向批次与训练批次解耦，控制生成/前向的切片大小 ---
        if not hasattr(args, "local_rollout_forward_batch_size") or args.local_rollout_forward_batch_size is None:
            # 生成与value/logprob前向常常更吃显存/时间，这里建议比 per_device_train_batch_size 更小一点
            args.local_rollout_forward_batch_size = args.per_device_train_batch_size
        # === BEGIN PATCH 2: default chunk sizes & cache ===
        if not hasattr(args, "generation_chunk_size") or args.generation_chunk_size is None:
            args.generation_chunk_size = args.per_device_train_batch_size

        if not hasattr(args, "empty_cache_every"):
            args.empty_cache_every = 0  # 0 表示禁用周期性 empty_cache
        # === END PATCH 2 ===

        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.per_device_train_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
            if module is not None:
                disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)

        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        num_workers = 1  # 先开到 4 以内
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=(4 if num_workers > 0 else 2),
        )

        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        # --- 关闭梯度检查点（用显存换速度）---
        try:
            # policy（ValueHead 外壳可能没有这个方法，优先关底层 pretrained_model）
            pm = getattr(self.policy_model, "pretrained_model", self.policy_model)
            if hasattr(pm, "gradient_checkpointing_disable"):
                pm.gradient_checkpointing_disable()
            # value（你的 SharedValueModel -> _backbone_adapter.t5）
            if self.value_model is not None:
                t5_v = getattr(getattr(self.value_model, "_backbone_adapter", None), "t5", None)
                if t5_v is not None and hasattr(t5_v, "gradient_checkpointing_disable"):
                    t5_v.gradient_checkpointing_disable()
        except Exception as e:
            self.accelerator.print(f"[WARN] disabling gradient checkpointing failed: {e}")

        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again
        # --- ADD: 给 DDP 外壳打补丁，透传关键属性 ---
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)  # 拿到真正的 nn.Module
            base = getattr(unwrapped, "policy", unwrapped)  # 你的 wrapper 里 policy 才有 config
            # 把常用属性挂到 DDP 外层，满足 TRL/你代码里的 self.model.config 用法
            self.model.config = getattr(base, "config", None)
            self.model.generation_config = getattr(base, "generation_config", None)
            self.model.pretrained_model = getattr(base, "pretrained_model", None)
        except Exception as e:
            self.accelerator.print(f"[WARN] failed to proxy attrs to DDP wrapper: {e}")
        self.model.policy = getattr(unwrapped, "policy", None)
        self.model.value_model = getattr(unwrapped, "value_model", None)

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)
        self._guard_skip_streak = 0
        self._max_consec_guard_skips = getattr(self.args, "max_consec_guard_skips", 2)
        self._guard_free_warmup = getattr(self.args, "guard_free_warmup", 10)  # 前 10 个 update 不触发护栏

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model.policy).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        保存 policy 为标准 HF 目录；不在“活体模型”上 merge LoRA，只在 CPU 副本上 merge。
        兼容 ZeRO/FSDP（当我们不做副本时用 accelerator.get_state_dict 聚合）。
        """
        # --- barrier：所有 rank 同步 ---
        try:
            self.accelerator.wait_for_everyone()
        except Exception:
            pass

        import os, copy, torch
        is_main = self.accelerator.is_main_process
        output_dir = output_dir or self.args.output_dir

        # 1) 拿到“活体” policy（去掉 DDP/包装）
        unwrapped = self.accelerator.unwrap_model(self.model)
        live_policy = getattr(unwrapped, "policy", unwrapped)

        # （可选）保存前打印一次活体权重 L1，便于你对比
        try:
            with torch.no_grad():
                wsum = 0.0
                for p in live_policy.parameters():
                    if getattr(p, "requires_grad", False):
                        wsum += p.float().abs().sum().item()
            if is_main:
                self.accelerator.print(f"[DEBUG] (before save) live policy L1: {wsum:.3f}")
        except Exception:
            pass

        # 2) 在 CPU 上做一个“副本快照”，所有破坏性操作只对副本做
        snapshot = None
        try:
            with torch.no_grad():
                snapshot = copy.deepcopy(live_policy)
            try:
                snapshot = snapshot.to("cpu")
            except Exception:
                pass
        except Exception as e:
            snapshot = None
            if is_main:
                self.accelerator.print(f"[WARN] deepcopy(live_policy) 失败：{e}；将跳过 merge，直接保存未合并权重。")

        # 3) 只在“副本”上 merge LoRA（如果需要）
        try:
            merge_lora = bool(getattr(self.args, "merge_lora_on_save", True))
        except Exception:
            merge_lora = True

        if snapshot is not None and merge_lora and is_peft_available():
            try:
                from peft import PeftModel
                # 3a) snapshot 本体是 PeftModel
                if isinstance(snapshot, PeftModel):
                    try:
                        snapshot = snapshot.merge_and_unload()
                    except Exception as e:
                        if is_main:
                            self.accelerator.print(f"[WARN] snapshot.merge_and_unload 失败：{e}；将保存未合并副本。")
                # 3b) 或者 snapshot.pretrained_model 是 PeftModel
                pm = getattr(snapshot, "pretrained_model", None)
                if isinstance(pm, PeftModel):
                    try:
                        merged_pm = pm.merge_and_unload()
                        try:
                            snapshot.pretrained_model = merged_pm
                        except Exception:
                            # 某些外壳没有该属性，直接把 snapshot 指向底座
                            snapshot = merged_pm
                    except Exception as e:
                        if is_main:
                            self.accelerator.print(f"[WARN] snapshot.pretrained_model.merge_and_unload 失败：{e}。")
            except Exception as e:
                if is_main:
                    self.accelerator.print(f"[WARN] PEFT 导入失败或不可用，跳过 merge：{e}")

        # 4) 决定真正要保存的模块（优先底座）
        to_save = snapshot if snapshot is not None else live_policy
        if hasattr(to_save, "pretrained_model"):
            to_save = to_save.pretrained_model

        if not hasattr(to_save, "save_pretrained"):
            raise ValueError(
                "Policy 不是 HuggingFace PreTrainedModel（缺少 save_pretrained）。"
                "请确认 policy 或其 pretrained_model 为 HF 模型。"
            )

        # 5) 拿 state_dict
        if snapshot is not None:
            # 副本不在 accelerate 管理下，直接取本地 state_dict
            state_dict = to_save.state_dict()
        else:
            # 没有副本（deepcopy 失败/不 merge），需要聚合 live 权重以兼容分布式/ZeRO
            try:
                state_dict = self.accelerator.get_state_dict(to_save)
            except Exception:
                # 退路：从训练引擎拿全量，再筛出 policy.*
                try:
                    full_sd = self.accelerator.get_state_dict(self.model)

                    def _strip_module(k: str) -> str:
                        return k[7:] if k.startswith("module.") else k

                    policy_sd = {}
                    for k, v in full_sd.items():
                        k2 = _strip_module(k)
                        if k2.startswith("policy."):
                            policy_sd[k2.split("policy.", 1)[1]] = v
                    state_dict = policy_sd if policy_sd else to_save.state_dict()
                except Exception:
                    state_dict = to_save.state_dict()

        # 6) 主进程落盘
        if is_main:
            os.makedirs(output_dir, exist_ok=True)
            to_save.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)

            # 保存 tokenizer/processor
            proc = getattr(self, "processing_class", None)
            if hasattr(proc, "save_pretrained"):
                try:
                    proc.save_pretrained(output_dir)
                except Exception:
                    pass

            # 保存 generation_config（如有）
            try:
                gen_cfg = getattr(to_save, "generation_config", None)
                if gen_cfg is not None and hasattr(gen_cfg, "save_pretrained"):
                    gen_cfg.save_pretrained(output_dir)
            except Exception:
                pass

        # 7) 同步 & 释放副本
        try:
            self.accelerator.wait_for_everyone()
        except Exception:
            pass
        try:
            del snapshot  # 释放大对象
        except Exception:
            pass

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        # 在 train() 里替换生成配置初始化
        if getattr(args, "generation_kwargs", None):
            generation_config = GenerationConfig(**args.generation_kwargs)
        else:
            generation_config = GenerationConfig(
                max_new_tokens=args.response_length,
                temperature=0.3,
                top_k=50,
                top_p=0.85,
                do_sample=True,
            )

        accelerator.print("===training policy===")
        start_time = time.time()

        model.train()
        # --- 统计标量（在线累计法），放在 GPU/CPU 都行；这里用 Python float 最省显存 ---
        stat_keys = ["approxkl", "pg_clipfrac", "pg_loss", "vf_loss", "vf_clipfrac", "entropy", "ratio"]
        stats_sum = {k: 0.0 for k in stat_keys}
        stats_cnt = 0  # 统计步数（micro-batch 次数）
        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size

            data = next(iter_dataloader)

            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                # ----- Replacement: robust, explicit generation (use pretrained_model.generate + explicit decoder_input_ids) -----
                self.accelerator.wait_for_everyone()

                with unwrap_model_for_generation(
                        self.model, self.accelerator,
                        gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:

                    # Helper: find HF model / prefer .policy if present (same as before)
                    def _get_real_generate_model(obj):
                        cand = getattr(obj, "policy", obj)
                        if hasattr(cand, "generate"):
                            return cand
                        if hasattr(cand, "policy") and hasattr(cand.policy, "generate"):
                            return cand.policy
                        if hasattr(cand, "pretrained_model") and hasattr(cand.pretrained_model, "generate"):
                            return cand.pretrained_model
                        raise RuntimeError(f"Could not locate a model with .generate() on {type(obj)}")

                    real_gen_model = _get_real_generate_model(unwrapped_model)

                    # Choose device for generation
                    try:
                        device_policy = next(real_gen_model.parameters()).device
                    except Exception:
                        device_policy = queries.device

                    # move inputs to generation device
                    queries_dev = queries.to(device_policy, non_blocking=True)
                    attn = data.get("attention_mask", None)
                    attn_dev = attn.to(device_policy, non_blocking=True) if attn is not None else torch.ones_like(
                        queries_dev).to(device_policy)

                    # Build safe gen kwargs (do NOT mutate model.generation_config globally)
                    safe_gen_kwargs = dict(
                        max_new_tokens=getattr(generation_config, "max_new_tokens", args.response_length),
                        do_sample=getattr(generation_config, "do_sample", True),
                        temperature=getattr(generation_config, "temperature", 0.3),
                        top_k=int(getattr(generation_config, "top_k", 0) or 50),
                        top_p=getattr(generation_config, "top_p", 0.985),
                        return_dict_in_generate=True,
                        output_scores=False,
                        min_new_tokens=getattr(generation_config, "min_new_tokens", 1),
                        pad_token_id=getattr(processing_class, "pad_token_id", None),
                        eos_token_id=getattr(processing_class, "eos_token_id", None),
                    )

                    # Determine a safe decoder_start_token_id (prefer model->config -> tokenizer.eos -> tokenizer.pad)
                    dec_start = None
                    # try real_gen_model.generation_config
                    dec_start = getattr(getattr(real_gen_model, "generation_config", None), "decoder_start_token_id",
                                        None)
                    if dec_start is None:
                        dec_start = getattr(getattr(real_gen_model, "config", None), "decoder_start_token_id", None)
                    if dec_start is None:
                        dec_start = getattr(processing_class, "eos_token_id", None)
                    if dec_start is None:
                        dec_start = getattr(processing_class, "pad_token_id", None)
                    # defensive cast
                    if dec_start is None:
                        raise RuntimeError(
                            "Unable to determine a decoder_start_token_id (no config nor tokenizer provided)")

                    # repetition controls to mitigate collapse (can tune/remove later)
                    safe_gen_kwargs.update({
                        # "no_repeat_ngram_size": 3,
                        # "repetition_penalty": 1.2,
                    })

                    # Prefer to call underlying pretrained_model.generate() if available (same code path as your outer quick-test)
                    if hasattr(real_gen_model, "pretrained_model"):
                        real_for_generate = real_gen_model.pretrained_model
                    else:
                        real_for_generate = real_gen_model

                    # Temporarily set eval() and disable grad / inference mode for generation, then restore training state
                    was_training = real_for_generate.training
                    real_for_generate.eval()

                    # === BEGIN: chunked generation (fixed) ===
                    B = queries_dev.size(0)
                    gen_chunk = int(getattr(self.args, "generation_chunk_size", 8))
                    seq_chunks = []
                    logits_chunks = []  # only for decoder-only models

                    # 把不随 chunk 变化的 kwargs 先拷出来；确保里面**没有** decoder_input_ids
                    common_gen_kwargs = dict(safe_gen_kwargs)
                    common_gen_kwargs.pop("decoder_input_ids", None)

                    is_enc_dec = bool(getattr(real_for_generate.config, "is_encoder_decoder", False))
                    max_T = 0
                    with torch.inference_mode():
                        for s in range(0, B, gen_chunk):
                            q_s = queries_dev[s:s + gen_chunk]
                            a_s = attn_dev[s:s + gen_chunk] if attn_dev is not None else None

                            if is_enc_dec:
                                # 每个 chunk 单独构造 decoder_input_ids: [gs, 1]
                                di_s = torch.full(
                                    (q_s.size(0), 1),
                                    int(dec_start),
                                    device=device_policy,
                                    dtype=torch.long,
                                )
                                out_s = real_for_generate.generate(
                                    input_ids=q_s,
                                    attention_mask=a_s,
                                    decoder_input_ids=di_s,
                                    **common_gen_kwargs,
                                )
                            else:
                                out_s = real_for_generate.generate(
                                    input_ids=q_s,
                                    attention_mask=a_s,
                                    **common_gen_kwargs,
                                )

                            t_seq = out_s.sequences  # [gs, T_s]
                            seq_chunks.append(t_seq)
                            # 维护该批内的最大长度
                            max_T = max(max_T, t_seq.size(1))

                            if not is_enc_dec:
                                try:
                                    fw_out_s = real_for_generate(input_ids=t_seq, return_dict=True)
                                except TypeError:
                                    fw_out_s = real_for_generate(t_seq)
                                t_logits = fw_out_s.logits if hasattr(fw_out_s, "logits") else fw_out_s[
                                    0]  # [gs, T_s, V]
                                logits_chunks.append(t_logits)

                    # 选择 pad id（优先 tokenizer.pad_token_id，其次模型 config 的 pad_token_id，最后退化 0）
                    pad_id = int(
                        getattr(processing_class, "pad_token_id",
                                getattr(getattr(real_for_generate, "config", None), "pad_token_id", 0))
                    )

                    # 右侧补齐 sequences 到同一 T=max_T  —— 只 pad 最后一维
                    padded_seq_chunks = []
                    for t in seq_chunks:
                        if t.size(1) < max_T:
                            pad_T = max_T - t.size(1)
                            # 关键：二维 [B, T] 只 pad 最后一维 => (left=0, right=pad_T)
                            t = F.pad(t, (0, pad_T), value=pad_id)
                        padded_seq_chunks.append(t)

                    # 拼接前做个 sanity check
                    for idx, t in enumerate(padded_seq_chunks):
                        assert t.size(1) == max_T, f"seq chunk {idx} after pad -> {tuple(t.shape)} (expect T={max_T})"

                    sequences = torch.cat(padded_seq_chunks, dim=0)  # [B, max_T]

                    logitss = None
                    if not is_enc_dec:
                        # logits 的形状是 [gs, T_s, V]，同样按 T 维右侧补齐到 max_T（pad 值用 0.0）
                        padded_logits_chunks = [
                            (t if t.size(1) == max_T else F.pad(t, (0, 0, 0, max_T - t.size(1), 0, 0), value=0.0))
                            for t in logits_chunks
                        ]
                        logitss = torch.cat(padded_logits_chunks, dim=0).to(queries.device, non_blocking=True)

                    if was_training:
                        real_for_generate.train()

                    query_responses = sequences.to(queries.device, non_blocking=True)
                    # === END: chunked generation (fixed) ===

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    # 切片并送到 device（确保 device 一致）
                    query = queries[i: i + args.local_rollout_forward_batch_size].to(device)
                    query_response = query_responses[i: i + args.local_rollout_forward_batch_size].to(device)

                    is_enc_dec = bool(getattr(self.model.config, "is_encoder_decoder", False))
                    if is_enc_dec:
                        # HF 的 enc-dec generate 返回的 sequences 形如 [decoder_start, y1, y2, ...]
                        # 标签应该是 [y1, y2, ...]；decoder_input_ids 则是 shift-right 后的 [decoder_start, y1, ...]
                        response = query_response[:, 1:]
                    else:
                        response = query_response[:, context_length:]

                    # ---------- 计算 policy logits（encoder-decoder 与 decoder-only 分支） ----------
                    if is_enc_dec:
                        # enc-dec（如 T5）：encoder 输入是 query，decoder_input_ids 用 shift-right 的 response
                        dec_start = int(getattr(self.model.config, "decoder_start_token_id", 0) or 0)

                        # attention mask：优先使用 data，否则根据 pad 推导
                        if "attention_mask" in data:
                            enc_attn_mask = data["attention_mask"][i: i + args.local_rollout_forward_batch_size].to(
                                device)
                        else:
                            enc_attn_mask = (query != processing_class.pad_token_id).long().to(device)

                        # shift-right response -> decoder_input_ids
                        dec_in = torch.nn.functional.pad(response, (1, 0), value=dec_start)[:, :-1].to(device)

                        with torch.no_grad():
                            try:
                                out = self.model.policy(
                                    input_ids=query,
                                    attention_mask=enc_attn_mask,
                                    decoder_input_ids=dec_in,
                                    use_cache=False,
                                    return_dict=True,
                                )
                            except TypeError:
                                out = self.model.policy(
                                    input_ids=query,
                                    attention_mask=enc_attn_mask,
                                    decoder_input_ids=dec_in,
                                    use_cache=False,
                                )

                        logits = out.logits if hasattr(out, "logits") else out[0]
                        logits = logits.to(device)
                        if logits.dim() != 3:
                            raise RuntimeError(f"Expected logits shape [B, T, V], got {tuple(logits.shape)}")
                        if logits.size(1) != response.size(1):
                            if logits.size(1) > response.size(1):
                                logits = logits[:, -response.size(1):, :].contiguous()
                            else:
                                raise RuntimeError(
                                    f"Logits time-dim ({logits.size(1)}) < response time-dim ({response.size(1)})"
                                )
                        del out

                    else:
                        # decoder-only（GPT）：logitss 是 prompt+response 的前向结果
                        logits = logitss[i: i + args.local_rollout_forward_batch_size].to(device)
                        # 裁掉 prompt 部分使 time-dim 对齐 response
                        if logits.size(1) != response.size(1):
                            logits = logits[:, -response.size(1):, :].contiguous()

                    # 现在 logits.shape[:2] == response.shape
                    # logprob = selective_log_softmax(logits, response)
                    logprob = _safe_logprob(logits, response)
                    del logits

                    # ---------- 计算 ref_logprob（同样分支处理） ----------
                    if getattr(self.model.config, "is_encoder_decoder", False):
                        # 选择使用 ref_policy（若存在）或在 null_ref_context 下使用 model.policy
                        if ref_policy is None:
                            ctx_mgr = self.null_ref_context()
                            ref_m = self.model.policy
                        else:
                            ctx_mgr = nullcontext()
                            ref_m = ref_policy

                        with ctx_mgr:
                            try:
                                ref_out = ref_m(
                                    input_ids=query,
                                    attention_mask=enc_attn_mask,
                                    decoder_input_ids=dec_in,
                                    use_cache=False,
                                    return_dict=True,
                                )
                            except TypeError:
                                ref_out = ref_m(
                                    input_ids=query,
                                    attention_mask=enc_attn_mask,
                                    decoder_input_ids=dec_in,
                                    use_cache=False,
                                )

                        if hasattr(ref_out, "logits"):
                            ref_logits = ref_out.logits
                        elif isinstance(ref_out, (tuple, list)):
                            ref_logits = ref_out[0]
                        else:
                            raise RuntimeError(f"Unexpected ref_out type: {type(ref_out)}")

                        if ref_logits.size(1) != response.size(1):
                            if ref_logits.size(1) > response.size(1):
                                ref_logits = ref_logits[:, -response.size(1):, :].contiguous()
                            else:
                                raise RuntimeError(
                                    f"ref_logits time-dim ({ref_logits.size(1)}) < response time-dim ({response.size(1)})"
                                )


                        # ref_logprob = selective_log_softmax(ref_logits, response)
                        ref_logprob = _safe_logprob(ref_logits, response)
                        del ref_out, ref_logits


                    else:
                        # decoder-only 保持原逻辑（forward wrapper）
                        if ref_policy is None:
                            with self.null_ref_context():
                                ref_output = forward(self.model.policy, query_response, processing_class.pad_token_id)
                        else:
                            ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)

                        if hasattr(ref_output, "logits"):
                            ref_logits_full = ref_output.logits
                        elif isinstance(ref_output, (list, tuple)):
                            ref_logits_full = ref_output[0]
                        else:
                            raise RuntimeError(f"Unexpected ref_output type: {type(ref_output)}")

                        if ref_logits_full.size(1) != response.size(1):
                            ref_logits = ref_logits_full[:, -response.size(1):, :].contiguous()
                        else:
                            ref_logits = ref_logits_full


                        # ref_logprob = selective_log_softmax(ref_logits, response)
                        ref_logprob = _safe_logprob(ref_logits, response)
                        del ref_output, ref_logits_full, ref_logits


                    # ---------- Response processing 和 reward/value 计算 ----------
                    postprocessed_response = response
                    if self.stop_token_id is not None:
                        postprocessed_response = truncate_response(self.stop_token_id, processing_class.pad_token_id,
                                                                   response)

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    # safer computation of sequence_length: count non-pad tokens
                    pad_id = processing_class.pad_token_id
                    # 非 pad 的 token 数量（每条样本）
                    nonpad_counts = (postprocessed_response != pad_id).long().sum(dim=1)
                    # last non-pad token index (如果没有非pad，则结果为 -1)
                    sequence_length = nonpad_counts - 1

                    # unwrap value model
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model

                    # 计算 value：enc-dec 做手工前向并稳健解析输出格式
                    if getattr(self.model.config, "is_encoder_decoder", False):
                        with torch.no_grad():
                            try:
                                val_out = unwrapped_value_model(
                                    input_ids=query,
                                    attention_mask=enc_attn_mask,
                                    decoder_input_ids=dec_in,
                                    return_dict=False,
                                )
                            except TypeError:
                                val_out = unwrapped_value_model(
                                    input_ids=query,
                                    attention_mask=enc_attn_mask,
                                    decoder_input_ids=dec_in,
                                )

                        # 兼容 tuple/list/ModelOutput/tensor
                        if isinstance(val_out, torch.Tensor):
                            full_value = val_out
                        elif isinstance(val_out, (tuple, list)):
                            # 找第一个 tensor
                            found = None
                            for it in val_out:
                                if isinstance(it, torch.Tensor):
                                    found = it
                                    break
                            if found is None:
                                raise RuntimeError(f"Unexpected val_out tuple contents: {type(val_out)}")
                            full_value = found
                        elif hasattr(val_out, "value"):
                            full_value = val_out.value
                        else:
                            raise RuntimeError(f"Unexpected val_out type {type(val_out)}")

                        # 标准化 dim -> [B, T]（或 [B,1]）
                        if full_value.dim() == 3:
                            full_value = full_value.squeeze(-1)
                        elif full_value.dim() == 2:
                            pass
                        elif full_value.dim() == 1:
                            full_value = full_value.unsqueeze(1)
                        else:
                            raise RuntimeError(f"Unsupported full_value.dim() = {full_value.dim()}")

                        # 对齐到 response 长度
                        resp_len = response.size(1)
                        if full_value.size(1) >= resp_len:
                            full_value = full_value[:, -resp_len:].contiguous()
                        else:
                            if full_value.size(1) == 1:
                                full_value = full_value.expand(-1, resp_len).contiguous()
                            else:
                                raise RuntimeError(
                                    f"Cannot align full_value shape {full_value.shape} with response shape {response.shape}")

                        value = full_value  # shape [B, resp_len]

                        del val_out, full_value


                    else:
                        # decoder-only：复用原 get_reward 返回的 full_value（注意 get_reward 的行为）
                        full_value, _, _ = get_reward(unwrapped_value_model, query_response,
                                                      processing_class.pad_token_id, context_length)
                        value = full_value[:, context_length - 1: -1].squeeze(-1)

                    # ----- START PATCH: support custom RewardModel (HF-style or custom callable) -----
                    if hasattr(reward_model, "base_model_prefix"):
                        # HF-style reward model (use existing helper)
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                    else:
                        # Custom RewardModel: call it directly.
                        # Build a minimal batch slice expected by your RewardModel (labels/scene_id if present)
                        batch_slice = {}
                        # `data` is the outer-dataloader batch (ensure variable name matches your trainer)
                        # slice indices: i .. i + args.local_rollout_forward_batch_size (same as upstream loop)
                        start_idx = i
                        end_idx = i + args.local_rollout_forward_batch_size
                        for k in ("labels", "scene_id",):  # add any other keys your RewardModel expects
                            if k in data:
                                v = data[k][start_idx:end_idx]
                                # keep on CPU/tensor as original RewardModel expects; move to device inside model if needed
                                batch_slice[k] = v

                        # Call your RewardModel. It should accept (queries, responses, batch) per your implementation.
                        # Here `query` and `postprocessed_response` are tensors on device.
                        with torch.no_grad():
                            # ensure reward_model on same device
                            try:
                                reward_model = reward_model.to(device)
                            except Exception:
                                pass

                            # **Note**: your RewardModel.forward currently defined as forward(queries, responses, batch)
                            # so call accordingly. If your impl uses different arg names, adapt here.
                            raw_score = reward_model(query, postprocessed_response, batch_slice)

                            # normalize into a tensor on device
                            if isinstance(raw_score, torch.Tensor):
                                score = raw_score.to(device)
                            else:
                                score = torch.tensor(raw_score, dtype=torch.float32, device=device)

                    # ----- END PATCH -----

                    # 保存当前小批次的结果
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)

                # === DEBUG: 检查是否被截到空、以及截断前后非 PAD 长度 ===
                try:
                    pad_id = processing_class.pad_token_id
                    eos_id = processing_class.eos_token_id
                except Exception:
                    pad_id, eos_id = 0, None

                with torch.no_grad():
                    # 截断前后（response vs postprocessed_response）的非 PAD token 计数
                    resp_nonpad = (responses != pad_id).sum(dim=1)  # [B]
                    post_nonpad = (postprocessed_responses != pad_id).sum(dim=1)  # [B]

                    empty_rate_before = (resp_nonpad == 0).float().mean().item()
                    empty_rate_after = (post_nonpad == 0).float().mean().item()
                    # “被 truncate 清空”的比例：原本有 token，但截断后变 0
                    trunc_to_zero_rate = ((resp_nonpad > 0) & (post_nonpad == 0)).float().mean().item()

                    # 批级告警
                    if empty_rate_after == 1.0:
                        self.accelerator.print(
                            f"[WARN][update {update}] postprocessed_responses 全是 PAD；"
                            f"截断前空比例={empty_rate_before:.2%}，截断后空比例={empty_rate_after:.2%}，"
                            f"由截断变空的比例={trunc_to_zero_rate:.2%}；"
                            f"stop_token_id={self.stop_token_id}, pad_id={pad_id}, eos_id={eos_id}"
                        )
                    elif empty_rate_after > 0.5 or trunc_to_zero_rate > 0.5:
                        self.accelerator.print(
                            f"[WARN][update {update}] 大量空响应；"
                            f"截断前空={empty_rate_before:.2%}，截断后空={empty_rate_after:.2%}，"
                            f"由截断变空={trunc_to_zero_rate:.2%}"
                        )

                    # 抽一个“被截空”的样本，打印截断前后文本，帮助肉眼确认
                    if (post_nonpad == 0).any():
                        bad_idx = torch.nonzero(post_nonpad == 0, as_tuple=False)[:1].view(-1)
                        try:
                            txt_before = processing_class.decode(
                                responses[bad_idx].detach().cpu().tolist()[0], skip_special_tokens=False
                            )
                            txt_after = processing_class.decode(
                                postprocessed_responses[bad_idx].detach().cpu().tolist()[0], skip_special_tokens=False
                            )
                            self.accelerator.print("[DEBUG] 样本(截断前): ", txt_before.replace("\n", "\\n")[:200])
                            self.accelerator.print("[DEBUG] 样本(截断后): ", txt_after.replace("\n", "\\n")[:200])
                        except Exception:
                            pass

                if 'logitss' in locals() and logitss is not None:
                    del logitss

                # 这些变量在 for 循环里每步都会重新赋值，这里删除只是锦上添花，非必须
                try:
                    del logprob
                except NameError:
                    pass
                try:
                    del ref_logprob
                except NameError:
                    pass
                try:
                    del value
                except NameError:
                    pass
                try:
                    del score
                except NameError:
                    pass
                # full_value 只在某些分支里存在，别无条件 del
                try:
                    del full_value
                except NameError:
                    pass
                # 有些实现中 unwrapped_model 是 with 上下文管理的，可能已出作用域
                try:
                    del unwrapped_model
                except NameError:
                    pass



                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                T = responses.size(1)
                pos = torch.arange(T, device=responses.device)  # [T]
                padding_mask = pos.unsqueeze(0) > sequence_lengths.unsqueeze(1)  # [B,T]
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = pos.unsqueeze(0) > sequence_lengths_p1.unsqueeze(1)
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl_mask = ~padding_mask  # 有效 token
                logr = torch.where(kl_mask, ref_logprobs - logprobs, torch.zeros_like(logprobs))

                # raw KL（用于判断与日志）
                kl_raw = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr
                kl_pos_raw = torch.clamp(kl_raw, min=0.0)

                # === KL 护栏（raw KL, 限次跳过 + 预热，且**全局同步**）===
                mean_kl_token = masked_mean(kl_raw, kl_mask)  # 原始 KL 的带 mask 均值（可保留做日志）
                mean_kl_token_pos_local = masked_mean(torch.clamp(kl_raw, min=0.0), kl_mask)

                # 1) 先全局聚合 -> 得到**一致的**全局正 KL 均值
                mean_kl_token_pos_global = self.accelerator.gather_for_metrics(mean_kl_token_pos_local).mean()

                kl_guard = getattr(self.args, "kl_guard_token", None)  # None => 关护栏
                guard_enabled = (kl_guard is not None) and (update > getattr(self, "_guard_free_warmup", 0))

                # 2) 由 rank0 计算布尔，再广播，让所有 rank 拿到**同一个**决策
                if self.accelerator.is_main_process:
                    skip_update_bool = bool(guard_enabled and (mean_kl_token_pos_global.item() > kl_guard))
                else:
                    skip_update_bool = False
                skip_flag = torch.tensor(int(skip_update_bool), device=self.accelerator.device)
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(skip_flag, src=0)
                skip_update = bool(skip_flag.item())

                if skip_update:
                    self._guard_skip_streak += 1
                    # 越界时温和抬高 KL 系数帮助拉回分布（所有 rank 都做同样的调整）
                    self.args.kl_coef = float(min(0.5, self.args.kl_coef * 1.2))

                    self.accelerator.print(
                        f"[GUARD] mean_kl_token_pos_global={mean_kl_token_pos_global.item():.3f} > {kl_guard} "
                        f"(streak={self._guard_skip_streak}), kl_coef->{self.args.kl_coef:.3g}"
                    )

                    if self._guard_skip_streak < getattr(self, "_max_consec_guard_skips", 2):
                        # **所有 rank**一致地轻跳过：不做 advantage/backward，只做调度与回调，然后进入下一轮
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        self.lr_scheduler.step()
                        self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                        continue
                    else:
                        # 达到最大跳过次数：强制做一次更新（使用裁剪后的 KL），并清零计数
                        self.accelerator.print("[GUARD] hit max skips -> proceed with clipped-KL update.")
                        self._guard_skip_streak = 0
                else:
                    self._guard_skip_streak = 0

                # reward 用裁剪过的 KL，避免极端 batch 爆奖励
                kl_clip_token = getattr(self.args, "kl_clip_token", 0.05)
                kl_pos = torch.clamp(kl_pos_raw, max=kl_clip_token)
                non_score_reward = -args.kl_coef * kl_pos
                rewards = non_score_reward.clone()

                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)

                assert scores.dim() == 1 and scores.size(0) == responses.size(0), \
                    f"Expect scores=[B], got {tuple(scores.shape)} vs B={responses.size(0)}"

                # —— 保留原始分数用于日志；训练用分数可选归一化 ——
                raw_scores = scores.detach()

                # 开关：是否对奖励用的分数做 z-score 归一化（默认 False；如需打开可在 PPOConfig 里加 normalize_scores=True）
                use_norm = getattr(self.args, "normalize_scores", False)
                if use_norm:
                    scores_for_reward = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-6)
                else:
                    scores_for_reward = scores

                # 只把（可选归一化后的）分数加到序列末尾那个位置的 reward 上
                rewards.scatter_add_(1, actual_end.unsqueeze(1), scores_for_reward.unsqueeze(1))

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=True)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values

                valid_mask = ~padding_mask
                if valid_mask.any():
                    advantages = masked_whiten(advantages, valid_mask)  # 只白化一次
                else:
                    accelerator.print("[WARN] valid_mask.sum()==0 -> all tokens are padding for this mini-batch. "
                                      "Skipping whiten for advantages and filling zeros.")
                    advantages = torch.zeros_like(advantages)

                # 把 padding 位置显式置零（保留这句）
                advantages = torch.masked_fill(advantages, padding_mask, 0)


            # ===== INSERT A: 动态按本轮实际 batch 尺寸（B）训练 =====
            B = advantages.size(0)
            assert B == responses.size(0) == returns.size(0) == values.size(0) == logprobs.size(
                0) == query_responses.size(0), \
                f"Batch dim mismatch: {B=}, shapes={tuple(responses.shape), tuple(returns.shape), tuple(values.shape), tuple(logprobs.shape), tuple(query_responses.shape)}"
            assert B > 0, "No samples collected for PPO step."

            # 每轮按 B 均分成 num_mini_batches 份（不整除就丢尾，保持整除可简化索引）
            local_mini = max(1, B // args.num_mini_batches)
            B_eff = local_mini * args.num_mini_batches
            if B_eff < B:
                advantages = advantages[:B_eff]
                returns = returns[:B_eff]
                responses = responses[:B_eff]
                query_responses = query_responses[:B_eff]
                logprobs = logprobs[:B_eff]
                values = values[:B_eff]
                padding_mask = padding_mask[:B_eff]
                padding_mask_p1 = padding_mask_p1[:B_eff]
            B = B_eff  # 之后一律用 B（已整除

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(B)  # 现在 B 是本轮有效 batch（已整除）
                minibatch_idx = 0

                # === 正确的 mini-loop（按 local_mini 切成 num_mini_batches 份）===
                for mini_batch_start in range(0, B, local_mini):
                    mini_batch_end = min(mini_batch_start + local_mini, B)
                    minibatch_inds = b_inds[mini_batch_start:mini_batch_end]

                    gradient_accumulation_idx = 0  # 每个 mini 里重新计数

                    # === MICRO LOOP: drop-in replacement (manual grad accumulation; one step per mini) ===
                    num_micro = math.ceil(len(minibatch_inds) / args.per_device_train_batch_size)

                    for micro_batch_start in range(0, len(minibatch_inds), args.per_device_train_batch_size):
                        micro_batch_end = min(micro_batch_start + args.per_device_train_batch_size, len(minibatch_inds))
                        mb_idx = minibatch_inds[micro_batch_start:micro_batch_end]
                        if mb_idx.size == 0:
                            continue

                        # indices on same device
                        mb_idx_t = torch.as_tensor(mb_idx, device=responses.device, dtype=torch.long)

                        # Gather slices
                        mb_advantage = advantages.index_select(0, mb_idx_t)
                        mb_responses = responses.index_select(0, mb_idx_t)
                        mb_query_responses = query_responses.index_select(0, mb_idx_t)
                        mb_logprobs = logprobs.index_select(0, mb_idx_t)
                        mb_return = returns.index_select(0, mb_idx_t)
                        mb_values = values.index_select(0, mb_idx_t)
                        pad_m = padding_mask.index_select(0, mb_idx_t)
                        pad_m1 = padding_mask_p1.index_select(0, mb_idx_t)

                        # Forward pass (enc-dec vs dec-only)
                        if getattr(self.model.config, "is_encoder_decoder", False):
                            enc_input = queries.index_select(0, mb_idx_t).to(device)
                            if "attention_mask" in data:
                                enc_attn_mask = data["attention_mask"].index_select(0, mb_idx_t).to(device)
                            else:
                                enc_attn_mask = (enc_input != processing_class.pad_token_id).long().to(device)
                            dec_start = int(getattr(self.model.config, "decoder_start_token_id", 0) or 0)
                            dec_in = F.pad(mb_responses, (1, 0), value=dec_start)[:, :-1].to(device)

                            output, vpred_temp = model(
                                input_ids=enc_input,
                                attention_mask=enc_attn_mask,
                                decoder_input_ids=dec_in,
                            )
                            pol_logits = (output.logits if hasattr(output, "logits") else output[0]).to(device)
                            logits = pol_logits.contiguous()
                            vpred = vpred_temp
                        else:
                            dec_input = mb_query_responses.to(device)
                            dec_attn = (dec_input != processing_class.pad_token_id).long().to(device)
                            output, vpred_temp = model(input_ids=dec_input, attention_mask=dec_attn)
                            pol_logits = (output.logits if hasattr(output, "logits") else output[0]).to(device)
                            resp_len = mb_responses.size(1)
                            logits = pol_logits[:, -resp_len:, :].contiguous() if pol_logits.size(
                                1) != resp_len else pol_logits
                            vpred = vpred_temp[:, context_length - 1: -1]

                        # New logprobs (masked)
                        # new_logprobs = selective_log_softmax(logits, mb_responses)
                        new_logprobs = _safe_logprob(logits, mb_responses)
                        new_logprobs = torch.masked_fill(new_logprobs, pad_m, INVALID_LOGPROB)

                        # Value pred shape/mask
                        vpred = vpred.squeeze(-1) if vpred.dim() in (2, 3) else vpred
                        vpred = torch.masked_fill(vpred, pad_m1, 0)

                        # 在进入 micro 循环之前，先在当前 mini 的作用域里加一个标志
                        if minibatch_inds.size == 0:
                            continue
                        mini_has_valid = False  # <<< 新增：本 mini 是否出现过有效 token

                        # All-pad guard
                        if (~pad_m).sum() == 0 and (~pad_m1).sum() == 0:
                            self.accelerator.print(
                                f"[WARN][step {self.state.global_step} | epoch {ppo_epoch_idx} | mini {minibatch_idx}] "
                                f"micro-batch 全 PAD -> 跳过有效反传"
                            )
                            tiny = 0.0
                            for p in self.accelerator.unwrap_model(self.model).parameters():
                                if p.requires_grad:
                                    tiny = tiny + (p.float().sum() * 0.0)
                            accelerator.backward(tiny)
                            # free temps
                            if getattr(self.model.config, "is_encoder_decoder", False):
                                del enc_input
                            else:
                                del dec_input
                            del output, vpred_temp, pol_logits, logits, new_logprobs, vpred
                            continue
                        else:
                            mini_has_valid = True  # <<< 只要有一个 micro 不是全 PAD，就认为本 mini 有效

                        # PPO losses
                        vpredclipped = torch.clamp(vpred, mb_values - args.cliprange_value,
                                                   mb_values + args.cliprange_value)
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), ~pad_m1)
                        vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~pad_m1)

                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), ~pad_m)

                        # Entropy bonus
                        prob_dist = F.softmax(logits.float(), dim=-1)
                        token_entropy = torch.logsumexp(logits.float(), dim=-1) - torch.sum(prob_dist * logits.float(), dim=-1)
                        entropy_for_loss = masked_mean(token_entropy, ~pad_m)
                        ent_coef = getattr(self.args, "ent_coef", 0.01)  # 推荐略大于 1e-3

                        # Final loss (manual accumulation)
                        loss = pg_loss + args.vf_coef * vf_loss - ent_coef * entropy_for_loss
                        loss = loss / num_micro  # 关键：均摊到本 mini 的 micro 次数

                        # Backward with/without sync
                        is_last_micro = (micro_batch_end == len(minibatch_inds))
                        if not is_last_micro:
                            with accelerator.no_sync(model):  # 前面的 micro 不同步
                                accelerator.backward(loss)
                        else:
                            accelerator.backward(loss)  # 最后一个 micro 同步

                        # Metrics accumulation（在线）
                        with torch.no_grad():
                            pg_clipfrac = masked_mean((pg_losses2 > pg_losses).float(), ~pad_m)
                            approxkl = masked_mean(0.5 * (logprobs_diff ** 2), ~pad_m)
                            ratio_valid = torch.masked_select(torch.exp(logprobs_diff), ~pad_m)

                        stats_sum["approxkl"] += float(approxkl.detach().item())
                        stats_sum["pg_clipfrac"] += float(pg_clipfrac.detach().item())
                        stats_sum["pg_loss"] += float(pg_loss.detach().item())
                        stats_sum["vf_loss"] += float(vf_loss.detach().item())
                        stats_sum["vf_clipfrac"] += float(vf_clipfrac.detach().item())
                        stats_sum["entropy"] += float(entropy_for_loss.detach().item())
                        stats_sum["ratio"] += float(ratio_valid.mean().detach().item())
                        stats_cnt += 1

                        # free temps
                        if getattr(self.model.config, "is_encoder_decoder", False):
                            del enc_input
                        else:
                            del dec_input
                        del output, vpred_temp, pol_logits, logits, new_logprobs, vpred, prob_dist, token_entropy
                        del ratio, pg_losses, pg_losses2, logprobs_diff, vf_losses1, vf_losses2

                    # === end micro loop ===
                    # ✅ One optimizer step per MINI batch (不要在 micro 内 step)
                    torch.nn.utils.clip_grad_norm_(
                        self.accelerator.unwrap_model(self.model).parameters(),
                        getattr(self.args, "max_grad_norm", 1.0),
                    )

                    # === end micro loop ===
                    if not mini_has_valid:
                        self.accelerator.print(
                            f"[WARN][step {self.state.global_step} | epoch {ppo_epoch_idx} | mini {minibatch_idx}] "
                            f"本 mini 全部 micro 都是 PAD —— 建议跳过 optimizer.step()"
                        )
                    optimizer.step()
                    optimizer.zero_grad()

            with torch.no_grad():
                # 基本项
                metrics = {}
                # 用 raw KL 记日志
                mean_kl_token = masked_mean(kl_raw, kl_mask)
                mean_kl_token_pos = masked_mean(torch.clamp(kl_raw, min=0.0), kl_mask)

                factor_up, factor_down = 1.5, 1.2
                min_coef, max_coef = 1e-3, 1.0
                target_kl_token = getattr(self.args, "target_kl_token", 0.03)

                if mean_kl_token_pos.item() > target_kl_token * factor_up:
                    self.args.kl_coef = float(min(max_coef, self.args.kl_coef * factor_up))
                elif mean_kl_token_pos.item() < target_kl_token / factor_down:
                    self.args.kl_coef = float(max(min_coef, self.args.kl_coef / factor_down))
                # mean_entropy = (-logprobs).sum(1).mean()
                mean_kl_penalty = non_score_reward.sum(1).mean()  # KL 惩罚（通常为负）
                mean_score = raw_scores.mean()  # 用原始分数记日志，更可读
                rlhf_reward = mean_kl_penalty + mean_score

                # 吞吐：以 response token 数为口径
                # 吞吐
                batch_gen_tokens_local = (sequence_lengths + 1).clamp_min(0).sum()
                batch_gen_tokens_global = self.accelerator.gather_for_metrics(batch_gen_tokens_local).sum().item()
                elapsed = max(1e-6, time.time() - start_time)
                world = self.accelerator.num_processes
                samples_global = responses.size(0) * world

                # 占比
                score_share = torch.abs(mean_score) / (torch.abs(mean_score) + torch.abs(mean_kl_penalty) + 1e-8)

                # —— 分布式“加权”平均（聚合和/聚合计数），防止不同 rank 统计步数略有出入导致偏差 ——
                def _reduce_avg(sum_local: float, cnt_local: int):
                    dev = self.accelerator.device
                    sum_t = torch.tensor(sum_local, device=dev, dtype=torch.float32)
                    cnt_t = torch.tensor(cnt_local, device=dev, dtype=torch.float32)
                    sum_all = self.accelerator.gather_for_metrics(sum_t).sum()
                    cnt_all = self.accelerator.gather_for_metrics(cnt_t).sum().clamp_min(1)
                    return (sum_all / cnt_all).item()

                # 先写速度/奖励/目标函数这批（之前你算了但没写进去）
                metrics["speed/tok_per_s"] = batch_gen_tokens_global / elapsed
                metrics["speed/samples_per_s"] = samples_global / elapsed
                metrics["reward/score"] = self.accelerator.gather_for_metrics(mean_score).mean().item()
                metrics["reward/kl_penalty"] = self.accelerator.gather_for_metrics(mean_kl_penalty).mean().item()
                metrics["reward/score_share"] = self.accelerator.gather_for_metrics(score_share).mean().item()
                metrics["reward/rlhf_total"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/kl_token_mean"] = self.accelerator.gather_for_metrics(mean_kl_token).mean().item()

                # 下面这些来自“在线累计”的训练期统计（用 _reduce_avg 聚合）
                metrics["policy/approxkl_avg"] = _reduce_avg(stats_sum["approxkl"], stats_cnt)
                metrics["policy/clipfrac_avg"] = _reduce_avg(stats_sum["pg_clipfrac"], stats_cnt)
                metrics["loss/policy_avg"] = _reduce_avg(stats_sum["pg_loss"], stats_cnt)
                metrics["loss/value_avg"] = _reduce_avg(stats_sum["vf_loss"], stats_cnt)
                metrics["val/clipfrac_avg"] = _reduce_avg(stats_sum["vf_clipfrac"], stats_cnt)
                metrics["policy/entropy_avg"] = _reduce_avg(stats_sum["entropy"], stats_cnt)
                metrics["val/ratio"] = _reduce_avg(stats_sum["ratio"], stats_cnt)
                metrics["gen/avg_len"] = self.accelerator.gather_for_metrics(
                    (sequence_lengths + 1).float().mean()).mean().item()
                metrics["gen/eos_rate"] = self.accelerator.gather_for_metrics(
                    contain_eos_token.float().mean()).mean().item()
                metrics["objective/kl_coef"] = float(self.args.kl_coef)

                # 重置累计器，进入下一 update
                for k in stats_sum:
                    stats_sum[k] = 0.0
                stats_cnt = 0

                # EOS 计数（全局）
                num_eos_local = (responses == processing_class.eos_token_id).sum()
                num_eos_global = self.accelerator.gather_for_metrics(num_eos_local).sum().item()
                metrics["val/num_eos_tokens"] = num_eos_global

                # 训练进度
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len
                self.state.global_step += 1

                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            if self.control.should_save:
                with torch.no_grad():
                    pol = self.accelerator.unwrap_model(self.model).policy
                    wsum = 0.0
                    for p in pol.parameters():
                        if p.requires_grad:
                            wsum += p.float().abs().sum().item()
                self.accelerator.print(f"[DEBUG][step {self.state.global_step}] weight L1 sum before save: {wsum:.3f}")

                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            for _v in ("kl_raw", "mean_kl_token", "mean_kl_penalty", "scores",
                       "raw_scores", "scores_for_reward",
                       "metrics", "non_score_reward", "rlhf_reward"):
                if _v in locals():
                    del locals()[_v]

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )


        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        """
        Rank0-only eval generation, strictly no collectives inside.
        """
        # 仅在主进程执行；其它 rank 直接返回，且本函数内部不做任何 gather/broadcast。
        if self.accelerator.num_processes > 1 and not self.accelerator.is_main_process:
            return

        try:
            self.reward_model = self.reward_model.to(self.accelerator.device)
        except Exception as e:
            print(f"[WARN] reward_model move failed: {e}")

        args = self.args
        tokenizer = self.processing_class

        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=0.3,
            top_k=50,
            top_p=0.85,
            do_sample=True,
        )

        table = defaultdict(list)

        with unwrap_model_for_generation(
                self.model, self.accelerator, gather_deepspeed3_params=args.ds3_gather_for_generation
        ) as unwrapped_model:

            def _get_real_generate_model(obj):
                cand = getattr(obj, "policy", obj)
                if hasattr(cand, "generate"):
                    return cand
                if hasattr(cand, "policy") and hasattr(cand.policy, "generate"):
                    return cand.policy
                if hasattr(cand, "pretrained_model") and hasattr(cand.pretrained_model, "generate"):
                    return cand.pretrained_model
                raise RuntimeError(f"Could not locate a model with .generate() on {type(obj)}")

            real_gen_model = _get_real_generate_model(unwrapped_model)
            real_for_generate = getattr(real_gen_model, "pretrained_model", real_gen_model)

            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                B = query.size(0)
                device = query.device

                with torch.no_grad():
                    # 选择生成所用 device
                    try:
                        gen_device = next(real_for_generate.parameters()).device
                    except Exception:
                        gen_device = query.device

                    # attention mask（若缺省则按 pad 推导）
                    attn = batch.get("attention_mask", None)
                    if attn is None:
                        pad_id = getattr(tokenizer, "pad_token_id", None)
                        if pad_id is None:
                            pad_id = getattr(getattr(real_for_generate, "generation_config", None), "pad_token_id",
                                             None)
                        if pad_id is None and hasattr(real_for_generate, "config"):
                            pad_id = getattr(real_for_generate.config, "pad_token_id", None)
                        attn = torch.ones_like(query) if pad_id is None else (query != pad_id).long()

                    query_dev = query.to(gen_device, non_blocking=True)
                    attn_dev = attn.to(gen_device, non_blocking=True)

                    # 逐次调用时的 kwargs（不修改 model.generation_config）
                    gen_kwargs = dict(
                        max_new_tokens=getattr(generation_config, "max_new_tokens", args.response_length),
                        do_sample=getattr(generation_config, "do_sample", True),
                        temperature=getattr(generation_config, "temperature", 1.0),
                        top_k=int(getattr(generation_config, "top_k", 50) or 0),
                        top_p=getattr(generation_config, "top_p", 1.0),
                        return_dict_in_generate=True,
                        output_scores=False,
                    )

                    pad_id = (getattr(tokenizer, "pad_token_id", None)
                              or getattr(getattr(real_for_generate, "generation_config", None), "pad_token_id", None)
                              or getattr(getattr(real_for_generate, "config", None), "pad_token_id", None))
                    eos_id = (getattr(tokenizer, "eos_token_id", None)
                              or getattr(getattr(real_for_generate, "generation_config", None), "eos_token_id", None)
                              or getattr(getattr(real_for_generate, "config", None), "eos_token_id", None))
                    if pad_id is not None:
                        gen_kwargs["pad_token_id"] = int(pad_id)
                    if eos_id is not None:
                        gen_kwargs["eos_token_id"] = int(eos_id)

                    dec_conf = (getattr(getattr(real_for_generate, "generation_config", None), "decoder_start_token_id",
                                        None)
                                or getattr(getattr(real_for_generate, "config", None), "decoder_start_token_id", None))
                    if dec_conf is not None:
                        chosen_decoder_start = int(dec_conf)
                    elif pad_id is not None:
                        chosen_decoder_start = int(pad_id)
                    elif eos_id is not None:
                        chosen_decoder_start = int(eos_id)
                    else:
                        raise RuntimeError("Cannot determine decoder_start_token_id")

                    gen_kwargs["decoder_start_token_id"] = chosen_decoder_start

                    is_enc_dec = bool(getattr(real_for_generate.config, "is_encoder_decoder", False))
                    if is_enc_dec:
                        gen_kwargs["decoder_input_ids"] = torch.full(
                            (B, 1), chosen_decoder_start, dtype=torch.long, device=gen_device
                        )

                    gen_kwargs.setdefault("no_repeat_ngram_size", 3)
                    gen_kwargs.setdefault("repetition_penalty", 1.2)
                    gen_kwargs.setdefault("min_new_tokens", 1)

                    was_training = real_for_generate.training
                    real_for_generate.eval()
                    gen_out = real_for_generate.generate(input_ids=query_dev, attention_mask=attn_dev, **gen_kwargs)
                    if was_training:
                        real_for_generate.train()

                    sequences = gen_out.sequences.to(device, non_blocking=True)

                    if is_enc_dec:
                        response = sequences[:, 1:]  # 去掉 decoder_start_token
                    else:
                        context_length = query.shape[1]
                        response = sequences[:, context_length:]

                    postprocessed_response = response
                    if self.stop_token_id is not None:
                        postprocessed_response = truncate_response(self.stop_token_id, tokenizer.pad_token_id, response)

                    postprocessed_query_response = torch.cat((query, postprocessed_response), dim=1)

                    # 记录（仅本地）
                    table["query"].extend(tokenizer.batch_decode(query.detach().cpu().tolist(),
                                                                 skip_special_tokens=True))
                    table["model response"].extend(tokenizer.batch_decode(
                        postprocessed_response.detach().cpu().tolist(), skip_special_tokens=True))

                    # compute reward（仅本地）
                    queries_slice = postprocessed_query_response[:, : query.shape[1]]
                    responses_slice = postprocessed_query_response[:, query.shape[1]:]

                    if hasattr(self.reward_model, "base_model_prefix"):
                        _, score, _ = get_reward(self.reward_model, postprocessed_query_response,
                                                 tokenizer.pad_token_id, query.shape[1])
                    else:
                        batch_slice = {}
                        if "labels" in batch:
                            batch_slice["labels"] = batch["labels"].to(device)
                        else:
                            batch_slice["labels"] = torch.full((B, responses_slice.size(1)), tokenizer.pad_token_id,
                                                               dtype=torch.long, device=device)
                        if "scene_id" in batch:
                            batch_slice["scene_id"] = batch["scene_id"].to(device)
                        else:
                            batch_slice["scene_id"] = torch.full((B,), -1, dtype=torch.long, device=device)
                        score = self.reward_model(queries_slice, responses_slice, batch_slice)

                    s = (score if score.dim() == 1 else score.squeeze(-1)).detach().float().cpu().tolist()
                    table["score"].extend(s)

                if sampling:
                    break

        # 仅 rank0 汇总日志
        df = pd.DataFrame(table)
        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0:5])
            if "wandb" in args.report_to:
                import wandb
                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})
            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(name="completions.csv", table=df)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
