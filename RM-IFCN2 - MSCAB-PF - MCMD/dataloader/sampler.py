import torch
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader

class CategoriesSampler():
    """
    CategoriesSampler: 
    从数据集中按类别进行分层采样。
    
    关键修复：
    不再使用 range(max(label)+1) 遍历所有可能的整数标签，
    而是使用 np.unique(label) 仅遍历数据集中实际存在的类别。
    这解决了非连续类别（如 [0, 1, 4, 7]）导致采样到空集从而报错的问题。
    """
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # 迭代次数
        self.n_cls = n_cls      # 每次迭代选取的类别数 (way)
        self.n_per = n_per      # 每个类别选取的样本数 (shot)

        label = np.array(label)  # all data label
        self.m_ind = []  # 存储每个类别的样本索引列表
        
        # 修复逻辑：只获取存在的唯一类别
        self.unique_classes = np.unique(label)
        
        # 将每个类别的样本索引存入 m_ind
        for i in self.unique_classes:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            
            # 1. 随机选择 n_cls 个类别
            # 注意：如果实际类别数 < n_cls，则选取所有可用类别
            num_available_classes = len(self.m_ind)
            if self.n_cls > num_available_classes:
                # 如果要求的 way 数大于实际拥有的类别数 (例如 args.episode_way=15 但 base只有4类)
                # 则只取所有可用类别 (4类)
                # 这种情况下 batch size 会变小 (4 * 50 而不是 15 * 50)，但这通常比报错好
                classes = torch.randperm(num_available_classes)
            else:
                classes = torch.randperm(num_available_classes)[:self.n_cls]
            
            # 2. 对选中的每个类别进行样本采样
            for c in classes:
                l = self.m_ind[c]  # 获取该类别的所有样本索引
                
                # 如果该类样本数不足 n_per，则允许重复采样 (replace=True)
                if len(l) < self.n_per:
                    pos = torch.randint(0, len(l), (self.n_per,))
                else:
                    pos = torch.randperm(len(l))[:self.n_per]
                
                batch.append(l[pos])

            # 3. 堆叠并展平
            # 此时 batch 中的每个 tensor 长度都保证是 self.n_per，不会报错
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class BasePreserverCategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        
        # 同样应用修复逻辑
        self.unique_classes = np.unique(label)
        
        for i in self.unique_classes:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # 按顺序遍历所有类别
            classes = torch.arange(len(self.m_ind))
            
            for c in classes:
                l = self.m_ind[c]
                if len(l) < self.n_per:
                    pos = torch.randint(0, len(l), (self.n_per,))
                else:
                    pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class NewCategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        
        # 同样应用修复逻辑
        self.unique_classes = np.unique(label)
        
        for i in self.unique_classes:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
    
        # classlist 只需要是 0 到 len(m_ind)-1 的索引
        self.classlist = np.arange(len(self.m_ind))

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for c in self.classlist:
                l = self.m_ind[c]
                if len(l) < self.n_per:
                    pos = torch.randint(0, len(l), (self.n_per,))
                else:
                    pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

if __name__ == '__main__':
    # 测试代码
    labels = torch.tensor([0, 0, 1, 1, 4, 4, 7, 7]) # 模拟非连续标签
    sampler = CategoriesSampler(labels, n_batch=1, n_cls=2, n_per=2)
    print("Sampling from [0, 1, 4, 7]:")
    for idx in sampler:
        print(idx)