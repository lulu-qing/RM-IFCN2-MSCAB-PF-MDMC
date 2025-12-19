from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np

def _map_labels(labels, class_list):
    """
    辅助函数：将全局标签 (如 0,1,4,7) 映射为局部索引 (0,1,2,3)
    """
    if not isinstance(class_list, list):
        class_list = class_list.tolist() if hasattr(class_list, 'tolist') else list(class_list)
        
    target_map = {int(c): i for i, c in enumerate(class_list)}
    mapped_labels = []
    for x in labels:
        item = int(x.item())
        if item in target_map:
            mapped_labels.append(target_map[item])
        else:
            mapped_labels.append(0) # 异常保护
            
    return torch.tensor(mapped_labels, dtype=torch.long, device=labels.device)

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    
    # 获取基础类别列表 (例如 [0, 1, 4, 7])
    # 注意：这里假设 args.base_class 是数量，我们需要具体的类别ID
    # 如果 data_utils.py 里没有把 class_index 传进来，我们需要手动处理
    # 临时方案：对于 mbhm，硬编码或者从 dataloader 获取
    # 更稳妥的方式是假设 dataloader.dataset.index 存储了类别列表
    if hasattr(trainloader.dataset, 'index'):
        base_classes = trainloader.dataset.index
    else:
        # Fallback for mbhm
        base_classes = [0, 1, 4, 7]

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        # 维度兼容
        if args.dataset != 'mbhm' and data.dim() == 4:
            data = data.squeeze(1)

        logits = model(data)
        # 取前 N 个输出 (N=4)
        logits = logits[:, :args.base_class]
        
        # === 关键修改：映射标签 (7 -> 3) ===
        mapped_label = _map_labels(train_label, base_classes)
        
        loss = F.cross_entropy(logits, mapped_label)
        acc = count_acc(logits, mapped_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, model, args):
    # Base 模式预训练通常不需要 replace_base_fc，
    # 因为它是端到端训练 FC 层的。
    # 这里保留原样即可，或者直接返回 model
    return model

def test(model, testloader, epoch, args, session, validation=True):
    # Session 0 测试
    test_class = args.base_class
    model = model.eval()
    vl = Averager()
    va = Averager()
    
    if hasattr(testloader.dataset, 'index'):
        base_classes = testloader.dataset.index
    else:
        base_classes = [0, 1, 4, 7]
    
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            if args.dataset != 'mbhm' and data.dim() == 4:
                data = data.squeeze(1)
                
            logits = model(data)
            logits = logits[:, :test_class]
            
            # === 关键修改：映射标签 ===
            mapped_label = _map_labels(test_label, base_classes)
            
            loss = F.cross_entropy(logits, mapped_label)
            acc = count_acc(logits, mapped_label)

            vl.add(loss.item())
            va.add(acc)

        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl.item(), va.item()))
    
    return vl.item(), va.item(), [], []