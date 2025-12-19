#!/usr/bin/env python3
"""
creat_session.py
用于生成 LIMIT/FSCIL 实验所需的会话索引文件。

功能修改说明：
1. 类别划分 (Class Division)：
   - Base Session (Session 0): classes [0, 1, 4, 7] -> 生成 session_1.txt
   - Inc Session 1 (Session 1): classes [2, 3]    -> 生成 session_2.txt
   - Inc Session 2 (Session 2): classes [5, 6]    -> 生成 session_3.txt
   - Inc Session 3 (Session 3): classes [8, 9]    -> 生成 session_4.txt

2. 样本数量控制 (Sample Counts):
   - 基础任务: 可通过 --base-shots 指定每类样本数 (0表示全部使用)。
   - 增量任务: 可通过 --new-shots 指定每类样本数 (例如 5-shot)。

Usage:
  # 5-shot 设置 (基础任务用全量数据，增量任务每类5个)
  python creat_session.py --base-shots 0 --new-shots 5
"""
import os
import json
import argparse
import numpy as np
from collections import defaultdict

DEFAULT_PROCESSED_DIR = "/data1/zhangyong/lq/mbhm_dataset/processed_fscil" # 请确保指向包含 .npy 的文件夹
OUT_DIR = "./data/index_list/mbhm"

# ==========================================
# 1. 任务类别划分配置 (Class Configuration)
# ==========================================
# 基础任务包含的类别 (对应 session_1.txt)
BASE_CLASSES = [0, 1, 4, 7]

# # 增量任务序列 (对应 session_2.txt, session_3.txt, ...)
INC_SESSIONS = [
    [2, 3],   # Incremental Session 1
    [5, 6],   # Incremental Session 2
    [8, 9],   # Incremental Session 3
]

# BASE_CLASSES = [0, 2, 5, 8]

# # 增量任务序列 (对应 session_2.txt, session_3.txt, ...)
# INC_SESSIONS = [
#     [1, 3],   # Incremental Session 1
#     [4, 6],   # Incremental Session 2
#     [7, 9],   # Incremental Session 3
# ]


def parse_args():
    p = argparse.ArgumentParser(description="Create session index files for LIMIT.")
    
    p.add_argument("--processed-dir", type=str, default=DEFAULT_PROCESSED_DIR,
                   help="Directory containing train_labels_mbhm.npy.")
    p.add_argument("--out-dir", type=str, default=OUT_DIR,
                   help="Output directory to write session index files.")
    
    # ==========================================
    # 2. 样本数量参数 (Sample Count Arguments)
    # ==========================================
    p.add_argument("--base-shots", type=int, default=0, 
                   help="基础任务每类样本数。0 表示使用该类别的所有可用样本 (Default: 0).")
    
    p.add_argument("--new-shots", type=int, default=5, 
                   help="增量任务每类样本数 (Few-Shot setting). (Default: 5).")
    
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--allow-replacement", action="store_true",
                   help="Allow sampling with replacement if not enough unique candidates.")
    p.add_argument("--manifest", type=str, default="selected_manifest.json",
                   help="File name for manifest.")
    return p.parse_args()

def select_indices(class_to_indices, classes, shots, seed_rng, allow_replacement=False):
    """
    辅助函数：从指定类别中采样指定数量的样本索引
    """
    selected_indices = []
    
    for c in classes:
        # 获取该类别的所有可用索引
        candidates = class_to_indices.get(int(c), [])
        candidates = np.array(candidates, dtype=int)
        
        if len(candidates) == 0:
            print(f"Warning: Class {c} has no samples!")
            continue
            
        # 确定采样数量
        # 如果 shots=0，则取全部；否则取 min(现有数量, 目标shots)
        if shots == 0:
            count = len(candidates)
            chosen = candidates
        else:
            count = shots
            if len(candidates) < count:
                if allow_replacement:
                    print(f"Warning: Class {c} has {len(candidates)} samples, requested {count}. Sampling with replacement.")
                    chosen = seed_rng.choice(candidates, size=count, replace=True)
                else:
                    print(f"Warning: Class {c} has {len(candidates)} samples, requested {count}. Using all available.")
                    chosen = candidates
            else:
                chosen = seed_rng.choice(candidates, size=count, replace=False)
        
        selected_indices.extend(chosen)
    
    return np.array(selected_indices, dtype=int)

def main():
    args = parse_args()
    
    # 设置随机种子
    rng = np.random.default_rng(args.seed)
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")

    # 加载标签文件
    labels_path = os.path.join(args.processed_dir, "train_label_mbhm.npy") # 注意文件名可能需要根据 mbhm_data.py 的输出调整
    if not os.path.exists(labels_path):
        # 尝试兼容旧文件名
        labels_path_alt = os.path.join(args.processed_dir, "train_labels_mbhm.npy")
        if os.path.exists(labels_path_alt):
            labels_path = labels_path_alt
        else:
            raise FileNotFoundError(f"Label file not found in {args.processed_dir}")

    print(f"Loading labels from {labels_path}...")
    labels = np.load(labels_path)
    print(f"Total samples loaded: {len(labels)}")

    # 构建 类别 -> 索引 的映射
    class_to_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        class_to_indices[int(lbl)].append(int(idx))

    manifest = {
        "args": vars(args),
        "sessions": {}
    }
    
    # 全局去重集合 (如果需要确保增量任务样本不与基础任务重复，虽然这里类别通常不重叠)
    # 但如果未来改为类别重叠的设置，这很重要。
    # global_selected = set() 

    # ==========================================
    # 生成 Session 1 (Base Task)
    # ==========================================
    # Trainer 读取的是 session_1.txt
    print(f"\nGenerating Session 1 (Base) for classes {BASE_CLASSES}...")
    base_indices = select_indices(class_to_indices, BASE_CLASSES, args.base_shots, rng, args.allow_replacement)
    
    base_file = os.path.join(args.out_dir, "session_1.txt")
    np.savetxt(base_file, base_indices, fmt="%d")
    manifest["sessions"]["1"] = [int(i) for i in base_indices]
    
    print(f" -> Saved {len(base_indices)} indices to {base_file}")
    
    # ==========================================
    # 生成 Session 2+ (Incremental Tasks)
    # ==========================================
    for i, inc_classes in enumerate(INC_SESSIONS):
        session_id = i + 2 # Session 1 是 Base，所以从 2 开始
        print(f"Generating Session {session_id} (Inc) for classes {inc_classes}...")
        
        inc_indices = select_indices(class_to_indices, inc_classes, args.new_shots, rng, args.allow_replacement)
        
        inc_file = os.path.join(args.out_dir, f"session_{session_id}.txt")
        np.savetxt(inc_file, inc_indices, fmt="%d")
        manifest["sessions"][str(session_id)] = [int(x) for x in inc_indices]
        
        print(f" -> Saved {len(inc_indices)} indices to {inc_file}")

    # 保存 manifest
    manifest_path = os.path.join(args.out_dir, args.manifest)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

if __name__ == "__main__":
    main()