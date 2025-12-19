#!/usr/bin/env python3
#
import argparse
import importlib
import random
import torch
import os
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import subprocess
import sys

def setup_chinese_font_safely():
    """安全地设置中文字体"""
    try:
        # 方法1: 直接设置文泉驿微米黑
        plt.rcParams['font.family'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 测试字体是否生效
        test_fig, test_ax = plt.subplots(figsize=(1, 1))
        test_ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
        plt.close(test_fig)
        print("✓ 文泉驿微米黑字体设置成功")
        
    except Exception as e:
        print("✗ 文泉驿微米黑字体设置失败，尝试备用方案...")
        
        # 方法2: 查找系统中可用的中文字体
        backup_fonts = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
        available_fonts = set([f.name for f in fm.fontManager.ttflist])
        
        for font in backup_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                print(f"✓ 使用备用字体: {font}")
                break
        else:
            # 方法3: 使用字体文件路径
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Ubuntu
                '/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc',  # CentOS
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font_prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = font_prop.get_name()
                    print(f"✓ 使用字体文件: {font_path}")
                    break
            else:
                print("⚠️ 未找到合适的中文字体，图表可能显示方框")
        
        plt.rcParams['axes.unicode_minus'] = False

# 在导入其他库之前调用
setup_chinese_font_safely()
import numpy as np
import pprint as pprint

from sklearn.metrics import confusion_matrix
import matplotlib

import matplotlib.pyplot as plt 
#plt.rcParams['font.family'] = 'DejaVu Serif'
# _utils_pp = pprint.PrettyPrinter()
plt.rcParams["font.serif"] = ["Liberation Serif", "DejaVu Serif"]  # 优先使用这些字体
plt.rcParams["axes.unicode_minus"] = False  # 修复负号显示问题
import torch

import matplotlib.pyplot as plt
# 设置使用文泉驿微米黑字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 指定字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Ensure a pretty-printer object exists for pprint()
try:
    _utils_pp = pprint.PrettyPrinter()
except Exception:
    _utils_pp = None

def pprint(x):
    if _utils_pp is not None:
        _utils_pp.pprint(x)
    else:
        print(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)

# 求平均值
class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

# 计算准确率
def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

# 计算前k类的准确率
def count_acc_topk(x,y,k=5):
    _,maxk = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1) 
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == maxk).sum().item()
    return float(topk/total)

def count_acc_taskIL(logits, label,args):
    basenum=args.base_class
    incrementnum=(args.num_classes-args.base_class)/args.way
    for i in range(len(label)):
        currentlabel=label[i]
        if currentlabel<basenum:
            logits[i,basenum:]=-1e9
        else:
            space=int((currentlabel-basenum)/args.way)
            low=basenum+space*args.way
            high=low+args.way
            logits[i,:low]=-1e9
            logits[i,high:]=-1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def confmatrix(logits, label, filename):
    pred = torch.argmax(logits, dim=1)
    cm = confusion_matrix(label, pred, normalize='true')

    clss = len(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)

    # 设置适用于10类的刻度
    if clss == 10:
        plt.yticks(range(clss), range(clss), fontsize=16)
        plt.xticks(range(clss), range(clss), fontsize=16)
    else:
        # 如果类数不同，则根据类数动态设置
        step = clss // 5  # 分为5个区间
        plt.yticks(range(0, clss, step), range(0, clss, step), fontsize=16)
        plt.xticks(range(0, clss, step), range(0, clss, step), fontsize=16)

    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    cbar = plt.colorbar(cax)  # This line includes the color bar
    cbar.ax.tick_params(labelsize=16)

    # 设置适用于10类的刻度
    if clss == 10:
        plt.yticks(range(clss), range(clss), fontsize=16)
        plt.xticks(range(clss), range(clss), fontsize=16)
    else:
        step = clss // 5  # 分为5个区间
        plt.yticks(range(0, clss, step), range(0, clss, step), fontsize=16)
        plt.xticks(range(0, clss, step), range(0, clss, step), fontsize=16)

    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + '_cbar.pdf', bbox_inches='tight')
    plt.close()

    return cm



def dummy_matrix(mat,filename):
    # font={'family':'DejaVu Serif','size':18}
    # matplotlib.rc('font',**font)
    # matplotlib.rcParams.update({'font.family':'DejaVu Serif','font.size':18})
    # plt.rcParams["font.family"]="DejaVu Serif"

   
    cm=mat
    

    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm,cmap=plt.cm.jet) 
    cbar = plt.colorbar(cax) 
    cbar.ax.tick_params(labelsize=16)
    plt.yticks([0,19,39,59],[0,20,40,60],fontsize=16)
    plt.xticks([0,19,39],[0,20,40],fontsize=16)

    plt.xlabel('Virtual Label',fontsize=20)
    plt.ylabel('True Label',fontsize=20)
    plt.tight_layout()
    plt.savefig(filename+'.pdf',bbox_inches='tight')
    plt.close()

    print('transpose')
    cm=np.transpose(mat)
    

    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm,cmap=plt.cm.jet) 
    cbar = plt.colorbar(cax,shrink=0.7) 
    cbar.ax.tick_params(labelsize=16)
    plt.xticks([0,19,39,59],[0,20,40,60],fontsize=16)
    plt.yticks([0,19,39],[0,20,40],fontsize=16)

    plt.ylabel('Virtual Label',fontsize=20)
    plt.xlabel('True Label',fontsize=20)
    plt.tight_layout()
    plt.savefig(filename+'_2.pdf',bbox_inches='tight')
    return cm


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


if __name__=='__main__':

    # font={'family':'DejaVu Serif','size':18}
    # matplotlib.rc('font',**font)
    # matplotlib.rcParams.update({'font.family':'DejaVu Serif','font.size':18}')
    # plt.rcParams["font.family"]="Times New Roman"

    cm=np.random.rand(10, 10)  # 修改为10x10的随机矩阵
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm, cmap=plt.cm.jet)

    # 设置适用于10类的刻度
    plt.yticks(range(10), range(10), fontsize=16)  # y轴从0到9
    plt.xticks(range(10), range(10), fontsize=16)  # x轴从0到9
    
    cbar = plt.colorbar(cax)  # 颜色条
    cbar.ax.tick_params(labelsize=16)

    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.tight_layout()

    # 显示图像
    plt.show()

    # 保存为pdf文件
    plt.savefig('2.pdf', bbox_inches='tight')
    plt.close()

# %%
MODEL_DIR=None
DATA_DIR = '/data1/zhangyong/lq/mbhm_dataset/processed'
PROJECT='limit'
# PROJECT='base'

# %%
def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='mbhm',
                        choices=['cwru', 'cub200', 'cifar100','mbhm'])
    
    parser.add_argument('-dataroot', type=str, default='/data1/zhangyong/lq/mbhm_dataset/processed_fscil')   
    

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=5)
    parser.add_argument('-epochs_new', type=int, default=25)
    parser.add_argument('-lr_base', type=float, default=0.001)
    parser.add_argument('-lr_new', type=float, default=0.001)
    parser.add_argument('-schedule', type=str, default='Cosine',
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=64, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=30)
    parser.add_argument('-episode_way', type=int, default=2)
    parser.add_argument('-episode_query', type=int, default=20)

    #for castle
    # parser.add_argument('-meta_class_way', type=int, default=10, help='total classes(including know and unknown) to sample in training process')
    # parser.add_argument('-meta_new_class', type=int, default=4)
    parser.add_argument('-num_tasks', type=int, default=256)
    parser.add_argument('-sample_class', type=int, default=1)
    parser.add_argument('-sample_shot', type=int, default=1)

    # for pretrain
    parser.add_argument('-balance', type=float, default=1.0)
    parser.add_argument('-balance_for_reg', type=float, default=1.0)
    parser.add_argument('-loss_iter', type=int, default=200)
    parser.add_argument('-alpha', type=float, default=2.0)

    parser.add_argument('-fuse', type=float, default=0.04)
    parser.add_argument('-topk', type=int, default=2)
    parser.add_argument('-prototype_momentum', type=float, default=0.99)
    parser.add_argument('-eta', type=float, default=0.5)

    
    parser.add_argument('-lrg', type=float, default=0.001) #lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)
    # for ablation
    parser.add_argument('-shot_num', type=int, default=100)
    parser.add_argument('-start_session', type=int, default=0)
    # 在参数解析代码中添加session参数（通常在main.py或train.py中）
   
    parser.add_argument('--session', type=int, default=0, help='current training session')  ##########################
                   
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')
    # for training
    parser.add_argument('-gpu', default='5')
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-autoaug', type=int, default=1)

    parser.add_argument('--crop_len', type=int, default=4096,
                        help='window length for cropping / sliding-window inference (default: 4096). Set to None for full-length')
    parser.add_argument('--crop_stride', type=int, default=2048,
                        help='stride for sliding-window inference (default: window_size//2)')

    return parser

# %%
parser = get_command_line_parser()
# NOTE: changed to parse real CLI args (was previously parse_args([]) in the minimal-edits run)
args = parser.parse_args()
set_seed(args.seed)
print(vars(args))
args.num_gpu = set_gpu(args)
print(args.num_gpu)

# %%
trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)

# %%
true, pre = trainer.train()

# %%
