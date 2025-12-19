from models.base.fscil_trainer import FSCILTrainer as Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import time

from .helper import *
from utils import *
from dataloader.data_utils import *
from dataloader.sampler import CategoriesSampler, BasePreserverCategoriesSampler, NewCategoriesSampler
from .Network import MYNET

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        
        pass

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
    
        if self.args.model_dir != None:  
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('*********WARNING: NO INIT MODEL**********')
            pass

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_dataloader(self, session):
        if session == 0:
            trainset, train_fsl_loader, train_gfsl_loader, testloader = get_base_dataloader_meta(self.args)
            return trainset, train_fsl_loader, train_gfsl_loader, testloader
        else:
            trainset, trainloader, testloader, train_fsl_loader = get_new_dataloader(self.args, session)
            return trainset, trainloader, testloader, train_fsl_loader

    def get_session_classes(self, session):
        return get_session_classes(self.args, session)

    def get_optimizer_pretrain(self):
        params = [v for k, v in self.model.named_parameters() if ('encoder' in k or 'fc' in k)]
        optimizer = torch.optim.SGD(params, lr=self.args.lr_base, momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)
        return optimizer, scheduler

    def get_optimizer_meta(self):
        params = [v for k, v in self.model.named_parameters() if ('encoder' not in k and 'fc' not in k)]
        optimizer = torch.optim.SGD(params, lr=self.args.lrg, momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)
        return optimizer, scheduler

    def train(self):
        args = self.args
        t_start_time = time.time()
        self.result_list = [args]

        for session in range(args.start_session, args.sessions):
            if session == 0:
                train_set, train_fsl_loader, train_gfsl_loader, testloader = self.get_dataloader(session)
            else:
                train_set, trainloader, testloader, train_fsl_loader = self.get_dataloader(session)
            
            self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # Base Session
                print('new classes for this session:\n', np.unique(train_set.targets))
                
                if self.args.model_dir is None:
                    print("\n=== Stage 1: Pre-training Backbone (MDMC Bypassed) ===")
                    optimizer, scheduler = self.get_optimizer_pretrain()
                    
                    for epoch in range(args.epochs_base):
                        self.model.eval() 
                        tl, ta = self.base_train(self.model, train_fsl_loader, train_gfsl_loader, optimizer, scheduler, epoch, args, session='pretrain')
                        
                        self.model.module.mode = 'avg_cos'
                        if args.set_no_val:
                            pass
                        else:
                            vl, va, true, pre = self.validation()
                            if (va * 100) >= self.trlog['max_acc'][session]:
                                self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                                self.trlog['max_acc_epoch'] = epoch
                                self.best_model_dict = deepcopy(self.model.state_dict())
                                print('Better model found in Pre-train!')
                            
                            lrc = scheduler.get_last_lr()[0]
                            log_str = 'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                                epoch, lrc, tl, ta, vl, va)
                            print(log_str)
                            self.result_list.append(log_str)
                        
                        scheduler.step()

                    print("Stage 1 Finished. Loading best backbone...")
                    self.model.load_state_dict(self.best_model_dict)
                else:
                    print(f"\n>>> Stage 1 Skipped: Loaded Pre-trained Backbone from {self.args.model_dir} <<<")

                print("\n=== Stage 2: Meta-training MDMC (Backbone Frozen) ===")
                optimizer, scheduler = self.get_optimizer_meta() 
                self.trlog['max_acc'][session] = 0.0 
                
                for epoch in range(args.epochs_base):
                    self.model.eval()
                    tl, ta = self.base_train(self.model, train_fsl_loader, train_gfsl_loader, optimizer, scheduler, epoch, args, session=0)
                    
                    self.model.module.mode = 'avg_cos'
                    if args.set_no_val:
                        pass
                    else:
                        vl, va, true, pre = self.validation()
                        if (va * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            print('Better model found in Meta-train!')
                        
                        lrc = scheduler.get_last_lr()[0]
                        log_str = 'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va)
                        print(log_str)
                        self.result_list.append(log_str)
                    
                    scheduler.step()

                print("Session 0 Finished. Finalizing...")
                self.model.load_state_dict(self.best_model_dict)
                
                if hasattr(self.model, 'module'):
                    self.model.module.anchor_initialized = False
                else:
                    self.model.anchor_initialized = False
                
                self.best_model_dict = deepcopy(self.model.state_dict())
                best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), best_model_dir)
         
                self.model.module.mode = 'avg_cos'
                tsl, tsa, true, pre = self.test(self.model, testloader, None, args, session)
                
                final_log = 'Session {}, Test Best Epoch {}, acc {:.4f}'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session])
                print(final_log)
                self.result_list.append(final_log)

            else:  # Incremental Sessions
                print("training session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)
                
                if hasattr(self.model, 'module'):
                    self.model.module.anchor_initialized = True
                else:
                    self.model.anchor_initialized = True
                print(f"Session {session}: MDMC Active.")

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                
                tsl, tsa, true, pre = self.test(self.model, testloader, train_fsl_loader, args, session, validation=False)

                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                
                inc_log = 'Session {}, test Acc {:.3f}'.format(session, self.trlog['max_acc'][session])
                print(inc_log)
                self.result_list.append(inc_log)

        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), self.result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Total time used %.2f mins' % total_time)

        return true, pre

    def validation(self):
        with torch.no_grad():
            model = self.model
            session = 0 
            trainset, trainloader, testloader, train_fsl_loader = self.get_dataloader(session)
            model.module.mode = 'avg_cos'
            model.eval()
            
            vl, va, true, pre = self.test(model, testloader, train_fsl_loader, self.args, session)
            
        return vl, va, true, pre

    def _map_labels(self, labels, class_list):
        target_map = {int(c): i for i, c in enumerate(class_list)}
        mapped_labels = []
        for x in labels:
            item = int(x.item())
            if item in target_map:
                mapped_labels.append(target_map[item])
            else:
                mapped_labels.append(0) 
        return torch.tensor(mapped_labels, dtype=torch.long, device=labels.device)

    def base_train(self, model, train_fsl_loader, train_gfsl_loader, optimizer, scheduler, epoch, args, session=0):
        tl = Averager()
        ta = Averager()
        
        active_classes = self.get_session_classes(0)
        active_indices = torch.from_numpy(active_classes).long().cuda()

        for _, batch in enumerate(zip(train_fsl_loader, train_gfsl_loader)):
            support_data, support_label = batch[0][0].cuda(), batch[0][1].cuda()
            query_data, query_label = batch[1][0].cuda(), batch[1][1].cuda()
            model.module.mode = 'classifier'
          
            logits = model(support_data, query_data, support_label, session=session)
            logits = logits[:, active_indices] 
            
            flat_labels = query_label.view(-1, 1).repeat(1, args.num_tasks).view(-1)
            mapped_flat_labels = self._map_labels(flat_labels, active_classes)
            
            total_loss = F.cross_entropy(logits, mapped_flat_labels)
            acc = count_acc(logits, mapped_flat_labels)

            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return tl.item(), ta.item()

    def test(self, model, testloader, train_fsl_loader, args, session, validation=True):
        true = []
        pre = []
        
        active_classes = self.get_session_classes(session)
        active_indices = torch.from_numpy(active_classes).long().cuda()
        
        model = model.eval()
        vl = Averager()
        va = Averager()
        
        lgt = torch.tensor([])
        lbs = torch.tensor([])

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                query = model(data)
                
                full_logits = model.module.forward_many(query, session=session)
                
                logits = full_logits[:, active_indices]
                mapped_label = self._map_labels(test_label, active_classes)

                true.append(mapped_label)
                pre.append(logits)

                loss = F.cross_entropy(logits, mapped_label)
                acc = count_acc(logits, mapped_label)
                vl.add(loss.item())
                va.add(acc)

                lgt = torch.cat([lgt, logits.cpu()])
                
                # [关键] 这里 lbs 我们要存原始 label，而不是 mapped_label，方便后续判断 Seen/Unseen
                lbs = torch.cat([lbs, test_label.cpu()])
            
            vl = vl.item()
            va = va.item()
        
            # 注意：lgt 是局部 Logits (列数 = 当前Session类别数)
            # lbs 是全局 Label (如 0, 7, 2...)
            lgt = lgt.view(-1, len(active_classes))
            lbs = lbs.view(-1)

            if validation is not True:
                # [核心修复] 正确计算 Seen/Unseen Accuracy
                # 1. 获取模型预测的 全局类别索引
                # lgt 的第 j 列对应 active_classes[j]
                preds_loc = lgt.argmax(dim=1).cpu().numpy()
                preds_glob = active_classes[preds_loc] # 映射回全局 ID
                
                lbs_numpy = lbs.numpy().astype(int)
                
                # 2. 定义 Base 类集合 (MBHM: 0, 1, 4, 7)
                if args.dataset == 'mbhm':
                    base_set = {0, 1, 4, 7}
                else:
                    base_set = set(range(args.base_class))
                
                # 3. 统计 Seen Acc
                seen_mask = np.array([l in base_set for l in lbs_numpy])
                if seen_mask.sum() > 0:
                    seen_correct = (preds_glob[seen_mask] == lbs_numpy[seen_mask]).sum()
                    seenac = seen_correct / seen_mask.sum()
                else:
                    seenac = 0.0
                
                # 4. 统计 Unseen Acc
                unseen_mask = ~seen_mask
                if unseen_mask.sum() > 0:
                    unseen_correct = (preds_glob[unseen_mask] == lbs_numpy[unseen_mask]).sum()
                    unseenac = unseen_correct / unseen_mask.sum()
                else:
                    unseenac = 0.0
                    
                log_seen = 'Seen Acc:%.5f, Unseen ACC:%.5f' % (seenac, unseenac)
                print(log_seen)
                self.result_list.append(log_seen)
                
        return vl, va, true, pre

    def set_save_path(self):
        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        self.args.save_path = self.args.save_path + '%dSC-%dEpo-%.2fT-%dSshot' % (
            self.args.sample_class, self.args.epochs_base, self.args.temperature, self.args.sample_shot)
        self.args.save_path = self.args.save_path + '%.5fDec-%.2fMom-%dQ_' % (
            self.args.decay, self.args.momentum, self.args.batch_size_base,)
        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None