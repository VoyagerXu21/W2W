from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from utils.homography import world2image


# =========================
# 可靠打印：不被 tqdm 吞 + 强制 flush
# =========================
def _pp_print(msg: str):
    try:
        tqdm.write(str(msg))
    except Exception:
        print(str(msg), flush=True)
    try:
        import sys
        sys.stdout.flush()
    except Exception:
        pass


def postprocess_trajectory_new(traj, obs_traj, seq_start_end, scene_id, homography, scene_map, cfg):
    # ======== 统计（不影响逻辑） ========
    checked_end = 0
    oob_end = 0
    end_free = 0
    end_wall = 0
    fix_applied = 0

    skipped_start_in_wall = 0  # new 版本里其实没有“起点在墙内跳过”的逻辑，但保留统计位
    abnormal_masked = 0
    clustered_peds = 0  # new 版本里 clustering 仍有

    # 强制打印：证明跑到了这个函数
    _pp_print(f"[postprocess_new] ENTER | traj.shape={getattr(traj,'shape',None)} | deterministic={cfg.deterministic}")

    def _bounds(a, x, y):
        # 注意：这里 x,y 是你函数里 check_nonzero(a, x, y) 的语义
        # a[x,y] => x 对应行(H), y 对应列(W)
        return (0 <= x < a.shape[0]) and (0 <= y < a.shape[1])

    # postprocess the trajectory
    def check_nonzero(a, x, y):
        try:
            if 0 <= x < a.shape[0] and 0 <= y < a.shape[1]:
                return a[x, y] == 1
            return False
        except IndexError:
            return False

    def nearest_nonzero_idx(a, x, y):
        try:
            if 0 <= x < a.shape[0] and 0 <= y < a.shape[1]:
                if a[x, y] != 0:
                    return x, y
        except IndexError:
            pass

        r, c = np.nonzero(a)
        min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
        return r[min_idx], c[min_idx]

    if cfg.deterministic:
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess", disable=False)):
            map_temp = scene_map[scene_id[s]]
            # map_temp = np.ones_like(map_temp)  # Uncomment it if you don't want to use image map
            for ped_id in range(s, e):
                sample = 0
                endpoint = (traj[ped_id, sample, -1] / cfg.image_scale_down).astype(np.int32)

                # ===== 统计：越界/可走/墙 =====
                checked_end += 1
                ex, ey = int(endpoint[0]), int(endpoint[1])
                if not _bounds(map_temp, ey, ex):  # 注意：a[x,y] 里 x=endpoint[1]
                    oob_end += 1
                if check_nonzero(map_temp, ey, ex):
                    end_free += 1
                else:
                    end_wall += 1

                if not check_nonzero(map_temp, endpoint[1], endpoint[0]):
                    # Pedestrian is in the wall,
                    obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                    startpoint = obs_traj_temp[-1].copy()
                    endpoint_new = np.array(nearest_nonzero_idx(map_temp, endpoint[1], endpoint[0]))[::-1]
                    scale = np.clip((endpoint_new - startpoint) / (endpoint - startpoint), a_min=0.01, a_max=1.0)
                    traj_temp = (traj[ped_id, sample].copy() - startpoint) * scale + startpoint
                    traj[ped_id, sample] = traj_temp
                    fix_applied += 1  # new 版本这里视为“做了修正”

    else:
        new_traj = np.zeros([traj.shape[0], cfg.best_of_n, cfg.pred_len, 2])
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess", disable=False)):
            map_temp = scene_map[scene_id[s]]
            # map_temp = np.ones_like(map_temp)  # Uncomment it if you don't want to use image map
            for ped_id in range(s, e):
                obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                startpoint = obs_traj_temp[-1].copy()

                # Sample removal if there are abnormal movements
                THRESHOLD = 100
                mask = np.diff(traj[ped_id, :, :, :], n=1, axis=1)
                mask = np.linalg.norm(mask, ord=2, axis=-1)
                mask = np.any(np.greater(mask, THRESHOLD), axis=1)

                abnormal_masked += int(mask.sum())

                # traj_filtered = traj[ped_id, ~mask]
                traj_filtered = traj[ped_id].copy()
                traj_filtered[mask, :, 0] = startpoint[0]
                traj_filtered[mask, :, 1] = startpoint[1]
                max_samples_filtered = traj_filtered.shape[0]

                obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                startpoint = obs_traj_temp[-1].copy()

                for sample in range(max_samples_filtered):
                    # ⚠️ 注意：你原代码这里用的是 traj[ped_id, sample, -1]（不是 traj_filtered）
                    # 我保持不改逻辑
                    endpoint = (traj[ped_id, sample, -1] / cfg.image_scale_down).astype(np.int32)

                    # ===== 统计：越界/可走/墙 =====
                    checked_end += 1
                    ex, ey = int(endpoint[0]), int(endpoint[1])
                    if not _bounds(map_temp, ey, ex):
                        oob_end += 1
                    if check_nonzero(map_temp, ey, ex):
                        end_free += 1
                    else:
                        end_wall += 1

                    if not check_nonzero(map_temp, endpoint[1], endpoint[0]):
                        # Pedestrian is in the wall,
                        obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                        startpoint = obs_traj_temp[-1].copy()
                        endpoint_new = np.array(nearest_nonzero_idx(map_temp, endpoint[1], endpoint[0]))[::-1]
                        scale = np.clip((endpoint_new - startpoint) / (endpoint - startpoint), a_min=0.01, a_max=1.0)
                        traj_temp = (traj[ped_id, sample].copy() - startpoint) * scale + startpoint
                        traj_filtered[sample] = traj_temp
                        fix_applied += 1

                # Clustering
                if max_samples_filtered > cfg.best_of_n:
                    clustered_peds += 1
                    temp = traj_filtered.reshape(max_samples_filtered, -1)
                    centroids = KMeans(
                        n_clusters=cfg.best_of_n,
                        random_state=0,
                        init="k-means++",
                        n_init=1
                    ).fit(temp).cluster_centers_
                    traj_filtered = centroids.reshape(cfg.best_of_n, cfg.pred_len, -1)

                new_traj[ped_id, :traj_filtered.shape[0]] = traj_filtered

        traj = new_traj

    # ======== 汇总打印（保证可见） ========
    total = max(1, checked_end)
    _pp_print(
        f"[postprocess_new] checked_end={checked_end} | "
        f"oob_end={oob_end} ({100.0*oob_end/total:.2f}%) | "
        f"end_free={end_free} ({100.0*end_free/total:.2f}%) | "
        f"end_wall={end_wall} ({100.0*end_wall/total:.2f}%) | "
        f"fix_applied={fix_applied}"
    )
    if not cfg.deterministic:
        _pp_print(f"[postprocess_new] abnormal_masked(samples)={abnormal_masked} | clustered_peds={clustered_peds}")

    return traj


def postprocess_trajectory(traj, obs_traj, seq_start_end, scene_id, homography, scene_map, cfg):
    # postprocess the trajectory
    def get_value(S, i, j):
        try:
            return S[i, j]
        except IndexError:
            return 0

    # ======== 统计（不影响逻辑） ========
    checked_end = 0
    oob_end = 0
    end_free = 0
    end_wall = 0
    fix_applied = 0
    fix_failed = 0  # 扫完 100 个 scale 仍未找到 1（但原逻辑仍会赋值 traj_temps[i]）

    skipped_start_in_wall = 0
    oob_start = 0

    abnormal_masked = 0
    clustered_peds = 0

    # 强制打印：证明跑到了这个函数
    _pp_print(f"[postprocess] ENTER | traj.shape={getattr(traj,'shape',None)} | deterministic={cfg.deterministic}")

    # 注意：统计用“严格边界”，不改变你原来的 get_value 行为（逻辑不变）
    def _in_bounds(M, y, x):
        return (0 <= y < M.shape[0]) and (0 <= x < M.shape[1])

    if cfg.deterministic:
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess", disable=False)):
            map_temp = scene_map[scene_id[s]]
            # map_temp = np.ones_like(map_temp)  # Uncomment it if you don't want to use image map
            for ped_id in range(s, e):
                sample = 0
                endpoint = (traj[ped_id, sample, -1] / cfg.image_scale_down).astype(np.int32)

                # ===== 统计：endpoint 越界/可走/墙 =====
                checked_end += 1
                ex, ey = int(endpoint[0]), int(endpoint[1])
                if not _in_bounds(map_temp, ey, ex):
                    oob_end += 1
                v_end = get_value(map_temp, ey, ex)
                if v_end == 1:
                    end_free += 1
                else:
                    end_wall += 1

                if get_value(map_temp, endpoint[1], endpoint[0]) != 1:
                    # Pedestrian is in the wall, scale down.
                    obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                    # startpoint = traj[ped_id, sample, 0].copy()
                    startpoint = obs_traj_temp[-1].copy()

                    if cfg.dataset_name not in ["eth", "hotel", "univ", "zara1", 'zara2']:
                        # ===== 统计：start 越界 =====
                        sx, sy = int(startpoint[0]), int(startpoint[1])
                        if not _in_bounds(map_temp, sy, sx):
                            oob_start += 1

                        if get_value(map_temp, startpoint[1], startpoint[0]) == 0:
                            # Pedestrian is already in the wall, skip.
                            skipped_start_in_wall += 1
                            continue

                    scale = np.linspace(1.0, 0.01, 100)
                    traj_temp = traj[ped_id, sample].copy() - startpoint
                    traj_temps = np.tile(traj_temp[None, :, :], [len(scale), 1, 1])
                    traj_temps *= np.tile(scale[:, None, None], [1, *traj[ped_id, sample].shape])
                    traj_temps += startpoint
                    endpoints = (traj_temps[:, -1] / cfg.image_scale_down).astype(np.int32)

                    # Find the first endpoint that is not in the wall
                    found = False
                    for i in range(len(endpoints)):
                        if get_value(map_temp, endpoints[i, 1], endpoints[i, 0]) == 1:
                            found = True
                            break

                    traj[ped_id, sample] = traj_temps[i]  # 原逻辑不变

                    if found:
                        fix_applied += 1
                    else:
                        fix_failed += 1

    else:
        new_traj = np.zeros([traj.shape[0], cfg.best_of_n, cfg.pred_len, 2])
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess", disable=False)):
            map_temp = scene_map[scene_id[s]]
            # map_temp = np.ones_like(map_temp)  # Uncomment it if you don't want to use image map
            for ped_id in range(s, e):
                obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                startpoint = obs_traj_temp[-1].copy()

                # Sample removal if there are abnormal movements
                THRESHOLD = 100
                mask = np.diff(traj[ped_id, :, :, :], n=1, axis=1)
                mask = np.linalg.norm(mask, ord=2, axis=-1)
                mask = np.any(np.greater(mask, THRESHOLD), axis=1)

                abnormal_masked += int(mask.sum())

                # traj_filtered = traj[ped_id, ~mask]
                traj_filtered = traj[ped_id].copy()
                traj_filtered[mask, :, 0] = startpoint[0]
                traj_filtered[mask, :, 1] = startpoint[1]
                max_samples_filtered = traj_filtered.shape[0]

                obs_traj_temp = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])
                startpoint = obs_traj_temp[-1].copy()

                # ===== 统计：start 越界（仅统计，不改逻辑） =====
                sx, sy = int(startpoint[0]), int(startpoint[1])
                if not _in_bounds(map_temp, sy, sx):
                    oob_start += 1

                for sample in range(max_samples_filtered):
                    endpoint = (traj_filtered[sample, -1] / cfg.image_scale_down).astype(np.int32)

                    # ===== 统计：endpoint 越界/可走/墙 =====
                    checked_end += 1
                    ex, ey = int(endpoint[0]), int(endpoint[1])
                    if not _in_bounds(map_temp, ey, ex):
                        oob_end += 1
                    v_end = get_value(map_temp, ey, ex)
                    if v_end == 1:
                        end_free += 1
                    else:
                        end_wall += 1

                    if get_value(map_temp, endpoint[1], endpoint[0]) != 1:
                        # Pedestrian is in the wall, scale down.

                        if cfg.dataset_name not in ["eth", "hotel", "univ", "zara1", 'zara2']:
                            if get_value(map_temp, startpoint[1], startpoint[0]) == 0:
                                # Pedestrian is already in the wall, skip.
                                skipped_start_in_wall += 1
                                continue

                        scale = np.linspace(1.0, 0.01, 100)
                        traj_temp = traj_filtered[sample].copy() - startpoint
                        traj_temps = np.tile(traj_temp[None, :, :], [len(scale), 1, 1])
                        traj_temps *= np.tile(scale[:, None, None], [1, cfg.pred_len, 2])
                        traj_temps += startpoint
                        endpoints = (traj_temps[:, -1] / cfg.image_scale_down).astype(np.int32)

                        # Find the first endpoint that is not in the wall
                        found = False
                        for i in range(len(endpoints)):
                            if get_value(map_temp, endpoints[i, 1], endpoints[i, 0]) == 1:
                                found = True
                                break

                        traj_filtered[sample] = traj_temps[i]  # 原逻辑不变

                        if found:
                            fix_applied += 1
                        else:
                            fix_failed += 1

                # Clustering
                if max_samples_filtered > cfg.best_of_n:
                    clustered_peds += 1
                    temp = traj_filtered.reshape(max_samples_filtered, -1)
                    centroids = KMeans(
                        n_clusters=cfg.best_of_n,
                        random_state=0,
                        init='k-means++',
                        n_init=1
                    ).fit(temp).cluster_centers_
                    traj_filtered = centroids.reshape(cfg.best_of_n, cfg.pred_len, -1)

                new_traj[ped_id, :traj_filtered.shape[0]] = traj_filtered

        traj = new_traj

    # ======== 汇总打印（保证可见） ========
    total = max(1, checked_end)
    _pp_print(
        f"[postprocess] checked_end={checked_end} | "
        f"oob_end={oob_end} ({100.0*oob_end/total:.2f}%) | "
        f"end_free={end_free} ({100.0*end_free/total:.2f}%) | "
        f"end_wall={end_wall} ({100.0*end_wall/total:.2f}%)"
    )
    _pp_print(
        f"[postprocess] fix_applied={fix_applied} | fix_failed={fix_failed} | "
        f"fix_success_over_wall={100.0*fix_applied/max(1,end_wall):.2f}% | "
        f"fix_applied_over_checked={100.0*fix_applied/total:.2f}%"
    )
    if not cfg.deterministic:
        _pp_print(f"[postprocess] abnormal_masked(samples)={abnormal_masked} | clustered_peds={clustered_peds}")
    _pp_print(f"[postprocess] skipped_start_in_wall={skipped_start_in_wall} | oob_start={oob_start}")

    return traj


def postprocess_trajectory_simple(traj, obs_traj, seq_start_end, scene_id, homography, cfg):
    # 原样不改（你没要求这里统计）
    if cfg.deterministic:
        pass
    else:
        new_traj = np.zeros([traj.shape[0], cfg.best_of_n, cfg.pred_len, 2])
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess")):
            for ped_id in range(s, e):
                if cfg.metric == "pixel":
                    startpoint = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])[-1].copy()
                else:
                    startpoint = obs_traj[ped_id][-1].copy()

                # Sample removal if there are abnormal movements
                THRESHOLD = 100 if cfg.metric == "pixel" else 5
                mask = np.diff(traj[ped_id, :, :, :], n=1, axis=1)
                mask = np.linalg.norm(mask, ord=2, axis=-1)
                mask = np.any(np.greater(mask, THRESHOLD), axis=1)
                # traj_filtered = traj[ped_id, ~mask]
                traj_filtered = traj[ped_id].copy()
                traj_filtered[mask, :, 0] = startpoint[0]
                traj_filtered[mask, :, 1] = startpoint[1]
                max_samples_filtered = traj_filtered.shape[0]

                # Clustering
                if max_samples_filtered > cfg.best_of_n:
                    temp = traj_filtered.reshape(max_samples_filtered, -1)
                    centroids = KMeans(n_clusters=cfg.best_of_n, random_state=0, init='k-means++', n_init=1).fit(temp).cluster_centers_
                    traj_filtered = centroids.reshape(cfg.best_of_n, cfg.pred_len, -1)

                new_traj[ped_id, :traj_filtered.shape[0]] = traj_filtered

        traj = new_traj
    return traj

