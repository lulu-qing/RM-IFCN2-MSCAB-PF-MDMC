from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from dataloader.data_utils import get_session_classes 

def replace_base_fc(trainset, model, args):
    # 切换到评估模式，确保 BN 层统计量不更新
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=0, pin_memory=True, shuffle=False)

    embedding_list = []
    label_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            # 移至 CPU 防止显存溢出
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
            
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # 获取基础会话的类别列表
    if args.dataset == 'mbhm':
        base_classes = get_session_classes(args, 0)
    else:
        base_classes = range(args.base_class)

    for class_index in base_classes:
        idx = int(class_index)
        data_index = (label_list == idx).nonzero()
        
        if data_index.numel() > 0:
            embedding_this = embedding_list[data_index.squeeze(-1)]
            
            # 1. 计算原型 (均值)
            proto = embedding_this.mean(0)
            
            # 2. [新增] 归一化原型
            # 这一步对于 Cosine Classifier 和 Attention 机制非常重要
            # 确保所有类的权重都在同一个单位超球面上
            proto = F.normalize(proto, p=2, dim=0)
            
            # 3. 赋值 (使用更健壮的 device 写法)
            model.module.fc.weight.data[idx] = proto.to(model.module.fc.weight.device)
            
        else:
            print(f"Warning: No samples found for base class {idx} during replace_base_fc")

    return model