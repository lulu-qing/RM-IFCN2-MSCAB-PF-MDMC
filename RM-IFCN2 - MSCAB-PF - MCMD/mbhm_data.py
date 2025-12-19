import h5py
import sqlite3
import numpy as np
import os
import random
from collections import defaultdict
from scipy.fft import dct
from sklearn.model_selection import train_test_split

# ================= 配置路径 =================
# 请修改为您实际的 MBHM 数据集路径
dataset_path = '/data1/zhangyong/lq/mbhm_dataset' 
data_path = os.path.join(dataset_path, 'data.hdf5')
meta_path = os.path.join(dataset_path, 'metadata.sqlite')
output_dir = os.path.join(dataset_path, 'processed_fscil')
os.makedirs(output_dir, exist_ok=True)

SIGNAL_LENGTH = 24000

# ================= DCN 预处理函数 =================
def dcn(data, length=24000):
    """
    Discrete Cosine Normalization (BearLLM Paper)
    """
    # 1. DCT 变换
    data = dct(data) # Type 2 DCT
    
    # 2. 截断或补零
    if len(data) < length:
        data = np.pad(data, (0, length - len(data)))
    else:
        data = data[:length]
        
    # 3. 归一化
    power = np.sum(data ** 2)
    if power > 0:
        data = data * np.sqrt(len(data) / power) * 0.01
    return data.astype(np.float32)

# ================= 处理函数 =================
def process_subset(rows, rows_by_condition, vib_file, subset_name):
    """
    处理一个子集（训练集或测试集）
    """
    data_list = []
    label_list = []
    valid_count = 0
    skipped_count = 0
    
    print(f"Processing {subset_name} set...")
    
    for file_id, label, cond_id in rows:
        # 1. 寻找同工况下的无故障参考信号
        candidates = rows_by_condition.get(cond_id, [])
        # 排除自身
        refs = [fid for fid, lbl in candidates if lbl == 0 and fid != file_id]
        
        if not refs:
            # BearLLM 核心要求：必须有同工况参考信号
            skipped_count += 1
            continue
            
        # 随机选择一个参考信号
        ref_id = random.choice(refs)
        
        # 2. 读取原始数据
        # 修正：直接使用整数索引访问 HDF5 Dataset，移除 try-except 字符串访问
        try:
            raw_query = vib_file[int(file_id)]
            raw_ref = vib_file[int(ref_id)]
        except Exception as e:
            print(f"Error reading file_id {file_id}: {e}")
            skipped_count += 1
            continue
        
        # 3. DCN 预处理
        feat_query = dcn(raw_query, SIGNAL_LENGTH)
        feat_ref = dcn(raw_ref, SIGNAL_LENGTH)
        
        # 4. 堆叠: [Query, Ref] -> Shape (2, 24000)
        sample = np.stack([feat_query, feat_ref], axis=0)
        
        data_list.append(sample)
        label_list.append(label)
        valid_count += 1
        
        if valid_count % 2000 == 0:
            print(f"  Processed {valid_count} samples in {subset_name}...")

    print(f"{subset_name} Done. Valid: {valid_count}, Skipped: {skipped_count}")
    
    if len(data_list) == 0:
        return None, None
        
    return np.stack(data_list, axis=0).astype(np.float32), np.array(label_list, dtype=np.int64)

def process_and_save():
    print(f"Reading metadata from {meta_path}...")
    
    # 1. 读取元数据
    conn = sqlite3.connect(meta_path)
    cursor = conn.cursor()
    # 尝试检查是否有 condition_id 列
    try:
        cursor.execute('SELECT file_id, label, condition_id FROM file_info')
        rows = cursor.fetchall()
    except Exception:
        print("Error: metadata.sqlite structure might be different. Please check column names.")
        return
    conn.close()
    
    # 2. 建立索引以便查找参考信号
    rows_by_condition = defaultdict(list)
    for file_id, label, cond_id in rows:
        rows_by_condition[cond_id].append((file_id, label))
        
    # 3. 划分训练集和测试集 (按样本划分)
    # 使用 stratify 保证每一类故障在训练集和测试集中分布均匀
    labels = [r[1] for r in rows]
    train_rows, test_rows = train_test_split(rows, test_size=0.2, random_state=42, stratify=labels)
    
    print(f"Total samples: {len(rows)}")
    print(f"Train samples: {len(train_rows)}, Test samples: {len(test_rows)}")
    
    # 4. 处理并保存
    with h5py.File(data_path, 'r') as f:
        # 注意：这里获取的是 Dataset 对象
        vib_dataset = f['vibration']
        
        # 处理训练集
        train_data, train_labels = process_subset(train_rows, rows_by_condition, vib_dataset, 'train')
        if train_data is not None:
            np.save(os.path.join(output_dir, 'train_data_mbhm.npy'), train_data)
            np.save(os.path.join(output_dir, 'train_label_mbhm.npy'), train_labels)
            print("Saved train data.")
            
        # 处理测试集
        test_data, test_labels = process_subset(test_rows, rows_by_condition, vib_dataset, 'test')
        if test_data is not None:
            np.save(os.path.join(output_dir, 'test_data_mbhm.npy'), test_data)
            np.save(os.path.join(output_dir, 'test_label_mbhm.npy'), test_labels)
            print("Saved test data.")

if __name__ == '__main__':
    process_and_save()