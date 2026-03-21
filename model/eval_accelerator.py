import os
import torch
from tqdm import tqdm
import numpy as np

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed

from torch.utils.data import DataLoader
from model.nltoolkit import init_nltk, postprocess_text
from utils.converter import batch_text2traj

from utils.dataloader import get_dataloader
from utils.homography import generate_homography, image2world, world2image
from utils.postprocessor import postprocess_trajectory

import warnings

warnings.filterwarnings('ignore')

from accelerate.logging import get_logger

logger = get_logger(__name__)
from accelerate import Accelerator
# ==== 新增 import ====
from PIL import Image, ImageDraw
import os


# ==== 新增：可视化工具 ====
def _clip_xy_to_img(points_xy, w, h):
    pts = points_xy.copy().astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def _to_fullres_pixels_from_world(world_traj, H, scale):
    """world -> (下采样像素) -> 还原到原图像素"""
    from utils.homography import world2image
    px_down = world2image(world_traj, H)  # 下采样后像素系
    px_full = px_down / scale  # 还原到原图大小（例如 0.25 -> ×4）
    return px_full


def _to_fullres_pixels_from_down_px(down_px_traj, scale):
    """（下采样像素系） -> 还原到原图像素"""
    return down_px_traj / scale


def visualize_trajectories_on_map(
        preds,  # 形状 [N, S, T, 2] ，S=样本数（deterministic 时一般为1；stochastic经过postprocess后通常为best_of_n）
        obs_traj,  # [N, To, 2] world 系
        gts_world,  # [N, Tp, 2] world 系（可为 None）
        seq_start_end,  # 序列起止 index 列表
        scene_id,  # 每个行人所属 scene 的 id
        homography,  # scene->H（已含下采样缩放）
        cfg,  # 配置，需含 image_scale_down、map_image_path、vis_output_dir、deterministic、best_of_n(如为采样)
):
    # 读取背景图
    map_image_path = getattr(cfg, "map_image_path", "./datasets/image/crowds_zara02_bg_bg.png")
    bg = Image.open(map_image_path).convert("RGB")
    W, H = bg.size  # 期望 640x480

    # 输出目录
    out_dir = getattr(cfg, "vis_output_dir", os.path.join(os.getcwd(), "vis_out"))
    os.makedirs(out_dir, exist_ok=True)

    # 每个序列导出一张图
    for seq_idx, (s, e) in enumerate(seq_start_end):
        canvas = bg.copy()
        draw = ImageDraw.Draw(canvas, "RGBA")

        for ped_id in range(s, e):
            H_ped = homography[scene_id[ped_id]]

            # 1) 观测轨迹（world -> fullres 像素）
            obs_px_full = _to_fullres_pixels_from_world(obs_traj[ped_id], H_ped, cfg.image_scale_down)
            obs_px_full = _clip_xy_to_img(obs_px_full, W, H)
            if len(obs_px_full) > 1:
                draw.line([tuple(p) for p in obs_px_full], fill=(0, 0, 255, 255), width=2)
                # 起点小圆点
                x0, y0 = obs_px_full[-1]
                r = 2
                draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=(0, 0, 255, 255))

            # 2) 预测轨迹（downsample 像素 -> fullres 像素）
            ped_preds = preds[ped_id]  # [S, T, 2] 或 [T, 2]
            if ped_preds.ndim == 2:
                ped_preds = ped_preds[None, ...]
            # deterministic 时只画1条；非确定性时画到 best_of_n（postprocess 后通常就是 best_of_n 条）
            draw_count = 1 if getattr(cfg, "deterministic", False) else ped_preds.shape[0]
            if hasattr(cfg, "best_of_n"):
                draw_count = min(draw_count, cfg.best_of_n)

            for sidx in range(draw_count):
                pred_px_full = _to_fullres_pixels_from_down_px(ped_preds[sidx], cfg.image_scale_down)
                pred_px_full = _clip_xy_to_img(pred_px_full, W, H)
                if len(pred_px_full) > 1:
                    draw.line([tuple(p) for p in pred_px_full], fill=(255, 0, 0, 200), width=2)
                    x1, y1 = pred_px_full[-1]
                    r = 2
                    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=(255, 0, 0, 200))

            # 3) （可选）GT 未来轨迹（world -> fullres 像素）
            if gts_world is not None:
                gt_px_full = _to_fullres_pixels_from_world(gts_world[ped_id], H_ped, cfg.image_scale_down)
                gt_px_full = _clip_xy_to_img(gt_px_full, W, H)
                if len(gt_px_full) > 1:
                    draw.line([tuple(p) for p in gt_px_full], fill=(0, 255, 0, 255), width=2)

        canvas.save(os.path.join(out_dir, f"seq_{seq_idx:05d}.png"))


@torch.no_grad()
def test(cfg):
    # ==== 新增：配置可视化输出目录和背景图路径（可按需改）====
    cfg.vis_output_dir = os.path.join(cfg.checkpoint_path, "viszara2-rl0.7T")  # 模型目录下建 vis
    cfg.map_image_path = "./datasets/image/crowds_zara02_bg.png"  # 你的 640x480 背景图
    # Initialize the Natural language toolkit
    init_nltk()

    # Initialize the accelerator.
    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)
    accelerator_log_kwargs = {}
    if cfg.use_logger:
        accelerator_log_kwargs["log_with"] = cfg.logger_type
        accelerator_log_kwargs["project_dir"] = checkpoint_path

    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Reproducibility settings
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Get the datasets
    dataloader = get_dataloader(os.path.join(cfg.dataset_path, cfg.dataset_name), 'test', cfg.obs_len, cfg.pred_len,
                                batch_size=1e8)
    obs_traj = dataloader.dataset.obs_traj.numpy()
    pred_traj = dataloader.dataset.pred_traj.numpy()
    non_linear_ped = dataloader.dataset.non_linear_ped.numpy()
    homography = dataloader.dataset.homography
    scene_id = dataloader.dataset.scene_id
    scene_img = dataloader.dataset.scene_img
    scene_map = dataloader.dataset.scene_map
    seq_start_end = dataloader.dataset.seq_start_end

    # batch_size_per_gpu = obs_traj.shape[0] // accelerator.state.num_processes + 1
    # if batch_size_per_gpu < cfg.per_device_inference_batch_size:
    #     print(f"per_device_inference_batch_size is automatically reduced from {cfg.per_device_inference_batch_size} to {batch_size_per_gpu}.")
    #     cfg.per_device_inference_batch_size = batch_size_per_gpu
    print("per_device_inference_batch_size", cfg.per_device_inference_batch_size)

    # Scale down the scene
    for k, v in homography.items():
        cfg.image_scale_down = 0.25
        homography[k] = v.copy() @ generate_homography(scale=cfg.image_scale_down)

    preprocessed_test_dataset_name = f"{cfg.dataset_name}-test-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json"
    preprocessed_dataset_path = os.path.join(cfg.dataset_path, "preprocessed")

    data_files = {}
    data_files["test"] = os.path.join(preprocessed_dataset_path, preprocessed_test_dataset_name)

    if not os.path.exists(data_files["test"]):
        raise ValueError(
            f"Preprocessed dataset files not found: {data_files['train']} or {data_files['validation']}. Please run `./script/preprocessor.sh` first.")

    extension = data_files["test"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cfg.cache_dir)

    # Load the model
    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=False, cache_dir=cfg.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=False, cache_dir=cfg.cache_dir,
                                              use_fast=not cfg.use_slow_tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, config=config, trust_remote_code=False,
                                                  cache_dir=cfg.cache_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if accelerator.is_local_main_process:
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Number of parameters: {count_parameters(model)}")

    # Preprocessing the datasets.
    column_names = raw_datasets["test"].column_names

    history_column = cfg.history_column
    if history_column not in column_names:
        raise ValueError(
            f"--history_column' value '{cfg.history_column}' needs to be one of: {', '.join(column_names)}")
    future_column = cfg.future_column
    if future_column not in column_names:
        raise ValueError(f"--future_column' value '{cfg.future_column}' needs to be one of: {', '.join(column_names)}")

    padding = "max_length" if cfg.pad_to_max_length else False

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

    test_dataset = raw_datasets["test"].map(preprocess_function,
                                            batched=True,
                                            num_proc=cfg.preprocessing_num_workers,
                                            remove_columns=column_names,
                                            load_from_cache_file=not cfg.overwrite_cache,
                                            desc="Running tokenizer on test dataset")

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id, )
    eval_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=cfg.per_device_inference_batch_size)

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    progress_bar = tqdm(range(len(obs_traj)), desc="Generating", disable=not accelerator.is_local_main_process)
    progress_step = cfg.per_device_inference_batch_size * accelerator.state.num_processes

    all_obs = np.array(raw_datasets['test']['obs_traj']).astype(np.float32)
    all_gts = np.array(raw_datasets['test']['pred_traj']).astype(np.float32)
    all_preds = []
    error_ids = []

    for step, batch in enumerate(eval_dataloader):
        if cfg.deterministic:
            # Most-likely prediction
            generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"].to(device),
                                                                        attention_mask=batch["attention_mask"].to(
                                                                            device),
                                                                        max_length=cfg.max_target_length,
                                                                        num_beams=cfg.num_beams)
        else:
            # Probabilistic sampling
            generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"].to(device),
                                                                        attention_mask=batch["attention_mask"].to(
                                                                            device),
                                                                        max_length=cfg.max_target_length,
                                                                        do_sample=True,
                                                                        num_return_sequences=cfg.num_samples,
                                                                        temperature=cfg.temperature,
                                                                        top_k=cfg.top_k)

        generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
        generated_tokens = accelerator.gather_for_metrics(
            (generated_tokens.view(-1, cfg.num_samples, generated_tokens.size(-1))))
        generated_tokens = generated_tokens.view(-1, generated_tokens.size(-1)).cpu().numpy()
        generated_tokens = generated_tokens[0] if isinstance(generated_tokens, tuple) else generated_tokens

        if not cfg.use_slow_tokenizer:
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        else:
            # make sure that special tokens are not decoded using sentencepiece model
            filtered_tokens = np.where(generated_tokens >= tokenizer.sp_model.get_piece_size(), 0, generated_tokens)
            decoded_preds = tokenizer.sp_model.decode(filtered_tokens.tolist())
        decoded_preds = [pred.strip() for pred in decoded_preds]
        traj_data = batch_text2traj(decoded_preds, frame=cfg.pred_len, dim=2)

        for pid in range(len(traj_data)):
            if traj_data[pid] is None:
                ped_id = cfg.per_device_inference_batch_size * accelerator.state.num_processes * step + pid // cfg.num_samples
                error_ids.append(ped_id)
                # Assume the pedestrian is not moving
                traj_data[pid] = np.tile(all_obs[ped_id, -1], (cfg.pred_len, 1))

        traj_data = np.stack(traj_data, axis=0).reshape(-1, cfg.num_samples, cfg.pred_len, 2)
        all_preds.append(traj_data)
        progress_bar.update(progress_step)

    all_preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    progress_bar.n = len(obs_traj)
    progress_bar.close()

    # ==== 重要：先做 postprocess（让轨迹合法/去异常），再可视化 ====
    if accelerator.is_local_main_process:
        all_preds = postprocess_trajectory(all_preds, obs_traj, seq_start_end, scene_id, homography, scene_map, cfg)

        # # ==== 新增：可视化（使用 postprocess 后的预测），all_obs/all_gts 是 world 系 ====
        # try:
        #     visualize_trajectories_on_map(
        #         preds=all_preds.copy(),  # 复制一份，避免后续评估里被改写坐标系
        #         obs_traj=obs_traj,
        #         gts_world=pred_traj,  # 如果不想画 GT，可传 None
        #         seq_start_end=seq_start_end,
        #         scene_id=scene_id,
        #         homography=homography,
        #         cfg=cfg
        #     )
        #     print(f"Visualization saved to: {cfg.vis_output_dir}")
        # except Exception as e:
        #     print(f"[WARN] Visualization failed: {e}")

        ADE = []
        FDE = []
        for ped_id in range(all_preds.shape[0]):

            # Homography warping
            if cfg.metric == "pixel":
                H = homography[scene_id[ped_id]]
                all_preds[ped_id] = image2world(all_preds[ped_id], H)
                all_gts[ped_id] = pred_traj[ped_id]

            error = np.linalg.norm(all_preds[ped_id] - all_gts[ped_id], ord=2, axis=-1)
            ADE.append(np.mean(error, axis=-1).min())
            FDE.append(error[:, -1].min())

        print(f"Test dataset: {cfg.dataset_name}")
        print(f"Total pedestrian number: {all_preds.shape[0]}")
        print(f"ADE: {np.mean(ADE)}")
        print(f"FDE: {np.mean(FDE)}")


if __name__ == "__main__":
    from utils.config import get_exp_config, DotDict

    args = get_exp_config()
    cfg = DotDict(args.__dict__)
    test(cfg)

