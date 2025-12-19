import numpy as np
import torch
from dataloader.sampler import CategoriesSampler
from torch.utils.data import DataLoader

def set_up_datasets(args):
    if args.dataset == 'cwru':
        import dataloader.cwru as Dataset
        args.base_class = 4
        args.num_classes = 10
        args.way = 1
        args.shot = 5
        args.sessions = 7
    elif args.dataset == 'mbhm':
        import dataloader.mbhm as Dataset
        args.base_class = 4   # 基础任务类别数量 (0,1,4,7)
        args.num_classes = 10 # 总类别数量
        args.way = 2          # 每次增量包含的类别数 ([2,3], [5,6]...)
        args.shot = 50         # shot数
        args.sessions = 4     # 总session数 (1 Base + 3 Inc)
        args.signal_length = 24000

    args.Dataset = Dataset
    return args

def get_session_classes(args, session):
    """
    返回直到当前 session 为止的所有可见类别 (全局ID)。
    针对 creat_session.py 的非连续划分进行特殊处理。
    """
    if args.dataset == 'mbhm':
        ##严格对应 creat_session.py 的划分
        # base_classes = [0, 2, 5, 8]
        # inc_sessions = [
        #     [1, 3],   # Session 1
        #     [4, 6],   # Session 2
        #     [7, 9]    # Session 3
        # ]
        
        base_classes = [0, 1, 4, 7]
        inc_sessions = [
            [2, 3],   # Session 1
            [5, 6],   # Session 2
            [8, 9]    # Session 3
        ]

        current_classes = list(base_classes)
        # 累加增量任务的类别
        for i in range(session):
            if i < len(inc_sessions):
                current_classes.extend(inc_sessions[i])
        
        return np.array(current_classes, dtype=int)
    else:
        # CWRU 或其他默认连续情况
        return np.arange(args.base_class + session * args.way)

def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    class_index = np.arange(args.base_class)
    if args.dataset == 'mbhm':
        # 对于 MBHM，我们使用 get_session_classes(0) 来获取基础类别 [0, 1, 4, 7]
        class_index = get_session_classes(args, 0)

    if args.dataset == 'cwru':
        trainset = args.Dataset.CWRU_dataset(root=args.dataroot, train=True, index=class_index, base_sess=True)
        testset = args.Dataset.CWRU_dataset(root=args.dataroot, train=False, index=class_index, base_sess=True)
    elif args.dataset == 'mbhm':
        trainset = args.Dataset.MBHM_dataset(root=args.dataroot, train=True, index=class_index, base_sess=True)
        testset = args.Dataset.MBHM_dataset(root=args.dataroot, train=False, index=class_index, base_sess=True)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_base_dataloader_meta(args):
    class_index = np.arange(args.base_class)
    if args.dataset == 'mbhm':
        class_index = get_session_classes(args, 0)

    if args.dataset == 'cwru':
        trainset = args.Dataset.CWRU_dataset(root=args.dataroot, train=True, index=class_index, base_sess=True)
        testset = args.Dataset.CWRU_dataset(root=args.dataroot, train=False, index=class_index, base_sess=True)
    elif args.dataset == 'mbhm':
        trainset = args.Dataset.MBHM_dataset(root=args.dataroot, train=True, index=class_index, base_sess=True)
        testset = args.Dataset.MBHM_dataset(root=args.dataroot, train=False, index=class_index, base_sess=True)

    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)
    
    # Generic loader for pretraining/stats
    train_gfsl_loader = DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True, 
                                   num_workers=args.num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, train_gfsl_loader, testloader

def get_new_dataloader(args, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    
    class_index = open(txt_path).read().splitlines()
    
    if args.dataset == 'cwru':
        trainset = args.Dataset.CWRU_dataset(root=args.dataroot, train=True, index=class_index, base_sess=False)
    elif args.dataset == 'mbhm':
        trainset = args.Dataset.MBHM_dataset(root=args.dataroot, train=True, index=class_index, base_sess=False)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cwru':
        testset = args.Dataset.CWRU_dataset(root=args.dataroot, train=False, index=class_new, base_sess=False)
    elif args.dataset == 'mbhm':
        testset = args.Dataset.MBHM_dataset(root=args.dataroot, train=False, index=class_new, base_sess=False)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    
    # FSL Sampler for incremental training (optional, used in LIMIT trainer)
    train_fsl_loader = None
    try:
        from dataloader.sampler import NewCategoriesSampler
        # MBHM specific: ensure we use the correct shot info if available, otherwise default to 5
        shot = args.shot if hasattr(args, 'shot') else 5
        test_sampler = NewCategoriesSampler(trainset.targets, 1, len(class_index), shot)
        train_fsl_loader = DataLoader(dataset=trainset, batch_sampler=test_sampler, 
                                      num_workers=args.num_workers, pin_memory=True)
    except:
        train_fsl_loader = trainloader

    return trainset, trainloader, testloader, train_fsl_loader