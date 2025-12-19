from torch.utils.data import Dataset
import torch
import numpy as np
import os

class MBHM_dataset(Dataset):
    def __init__(self, root, train=True, index=None, base_sess=None):
        """
        Args:
            root: 数据存放目录 (processed_fscil)
            train: True 加载 train_data_mbhm.npy, False 加载 test_data_mbhm.npy
            index: 类别ID列表 或 样本索引列表
            base_sess: 是否为基础会话
        """
        self.root = root
        self.train = train
        
        # 1. 根据 train 标志选择加载对应的文件
        if self.train:
            data_name = 'train_data_mbhm.npy'
            label_name = 'train_label_mbhm.npy'
        else:
            data_name = 'test_data_mbhm.npy'
            label_name = 'test_label_mbhm.npy'
            
        data_path = os.path.join(root, data_name)
        label_path = os.path.join(root, label_name)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}. Please run mbhm_data.py first.")
            
        # 加载数据 [N, 2, 24000]
        # 注意：这里的数据已经是频域数据 (由 mbhm_data.py 处理过)
        self.data = np.load(data_path)
        self.targets = np.load(label_path)
        
        # 转换为 Tensor
        self.data = torch.from_numpy(self.data).float()
        self.targets = torch.from_numpy(self.targets).long()

        # 2. 根据 FSCIL 阶段筛选数据
        if base_sess:
            # Base Session: index 是类别列表，按类别筛选
            self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)
        else:
            # Incremental Session
            if train:
                # 增量训练: index 是样本索引列表
                self.data, self.targets = self.NewClassSelector(self.data, self.targets, index)
            else:
                # 增量测试: index 是类别列表
                self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)

    def __getitem__(self, idx):
        # ==========================================
        # 修复: 直接返回数据，不做重复的 DCT/DCN
        # ==========================================
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data)

    def SelectfromDefault(self, data, targets, index):
        """ 根据类别 ID 列表筛选数据 """
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for class_id in index:
            mask |= (targets == int(class_id))
        return data[mask], targets[mask]

    def NewClassSelector(self, data, targets, index):
        """ 根据样本索引列表筛选数据 """
        indices = torch.tensor([int(i) for i in index], dtype=torch.long)
        return data[indices], targets[indices]