import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
from lib import *
from config import *
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
import math
from visdom import Visdom
from glob import glob
import os

class Trainer:
    def __init__(self, model, lr, epoch, save_fn, validation_interval, save_interval):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        self.validation_interval = validation_interval
        self.save_interval = save_interval
        
    def Tester(self, loader, b_size,we):
        all_pred = np.zeros((b_size, NUM_LABELS, int(LENGTH)))
        all_tar = np.zeros((b_size, NUM_LABELS, int(LENGTH)))
        p_pred = np.zeros((b_size, MAX_MIDI-MIN_MIDI+1, int(LENGTH)))
        pitch_tar = np.zeros((b_size, MAX_MIDI-MIN_MIDI+1, int(LENGTH)))
        loss_IPT = 0.0
        loss_pitch = 0.0
        loss_onset = 0.0

        self.model.eval()
        ds = 0
        for idx,_input in enumerate(loader):
            data, target, target_p, target_o = Variable(_input[0].to(device)),Variable(_input[1].to(device)), Variable(_input[2].to(device)), Variable(_input[3].to(device))
            IPT_pred, pitch_pred, onset_pred= self.model(data) #torch.Size([1, 7, 375])

            loss = sp_loss(IPT_pred, target, we)
            loss_p = pitch_loss(pitch_pred, target_p)
            loss_o = onset_loss(onset_pred, target_o)
            loss_IPT += loss.data
            loss_pitch +=loss_p.data
            loss_onset +=loss_o.data

            all_tar[ds: ds + len(target)] = target.data.cpu().numpy()
            all_pred[ds: ds + len(target)] = F.sigmoid(IPT_pred).data.cpu().numpy()
            pitch_tar[ds: ds + len(target)] = target_p.data.cpu().numpy()
            p_pred[ds: ds + len(target)] = F.sigmoid(pitch_pred).data.cpu().numpy()
            ds += len(target)
        threshold = 0.5
        pred_inst = np.transpose(all_pred, (1, 0, 2)).reshape((NUM_LABELS, -1))  # shape = [10, 8424] , 8424 = 27*312
        tar_inst = np.transpose(all_tar, (1, 0, 2)).reshape((NUM_LABELS, -1))  # shape = [10, 8424] , 8424 = 27*312
        pred_pitch = np.transpose(p_pred, (1, 0, 2)).reshape((MAX_MIDI-MIN_MIDI+1, -1))  # shape = [10, 8424] , 8424 = 27*312
        tar_pitch = np.transpose(pitch_tar, (1, 0, 2)).reshape((MAX_MIDI-MIN_MIDI+1, -1))  # shape = [10, 8424] , 8424 = 27*312
        pred_inst[pred_inst > threshold] = 1
        pred_inst[pred_inst <= threshold] = 0
        pred_pitch[pred_pitch > threshold] = 1
        pred_pitch[pred_pitch <= threshold] = 0

        metrics = compute_metrics(pred_inst, tar_inst)
        metrics_pitch = compute_pitch_metrics(pred_pitch, tar_pitch)
        return loss_IPT / b_size, metrics['metric/IPT_frame/precision'][0],metrics['metric/IPT_frame/recall'][0],metrics['metric/IPT_frame/f1'][0], loss_pitch/b_size, metrics_pitch['metric/pitch_frame/precision'][0], metrics_pitch['metric/pitch_frame/recall'][0], metrics_pitch['metric/pitch_frame/f1'][0], loss_onset/b_size
     
    def fit(self, tr_loader, va_loader, we):
        st = time.time()

        lr = self.lr

        viz = Visdom()
        viz.line([[0., 0.]], [0], win="IPT_loss_" + saveName, opts=dict(title="IPT_loss_" + saveName, legend=['train_loss', 'valid_loss']))
        viz.line([[0.]], [0], win="IPT_precision_" + saveName, opts=dict(title="IPT_precision_" + saveName, legend=['valid_IPT_precision']))
        viz.line([[0.]], [0], win="IPT_recall_" + saveName, opts=dict(title="IPT_recall_" + saveName, legend=['valid_IPT_recall']))
        viz.line([[0.]], [0], win="IPT_F1_" + saveName, opts=dict(title="IPT_F1_" + saveName, legend=['valid_IPT_F1']))
        viz.line([[0., 0.]], [0], win="pitch_loss_" + saveName, opts=dict(title="pitch_loss_" + saveName, legend=['train_loss', 'valid_loss']))
        viz.line([[0.]], [0], win="pitch_precision_" + saveName, opts=dict(title="pitch_precision_" + saveName, legend=['valid_pitch_precision']))
        viz.line([[0.]], [0], win="pitch_recall_" + saveName, opts=dict(title="pitch_recall_" + saveName, legend=['valid_pitch_recall']))
        viz.line([[0.]], [0], win="pitch_F1_" + saveName, opts=dict(title="pitch_F1_" + saveName, legend=['valid_pitch_F1']))
        viz.line([[0., 0.]], [0], win="onset_loss_" + saveName, opts=dict(title="onset_loss_" + saveName, legend=['train_loss', 'valid_loss']))
        best_acc = 0
        last_best_epoch = 1 #for early stopping

        for e in range(1, self.epoch+1):
            # Wheter there is a two-step fine-tuning process
            if TWO_STEP and (e > LIN_EPOCH) and FREEZE_ALL:
                for p in self.model.frontend.parameters():
                    p.requires_grad = True
                    self.model.frontend.model.feature_extractor._freeze_parameters()

            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=1e-4)
            lrf = 0.01
            epochs = 100
            lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

            loss_total_p = 0
            loss_total_i = 0
            loss_total_o = 0
            print ('\n==> Training Epoch #%d lr=%4f'%(e, lr))

            # Training
            for batch_idx, _input in enumerate(tr_loader):
                self.model.train()
                data, target, target_p, target_o = Variable(_input[0].to(device)), Variable(_input[1].to(device)), Variable(_input[2].to(device)), Variable(_input[3].to(device))
            
                #start feed in                
                IPT_pred,pitch_pred,onset_pred = self.model(data) #[batch, time, IPT, pitch]

                #counting loss
                loss = sp_loss(IPT_pred, target, we)
                loss_p = pitch_loss(pitch_pred, target_p)
                loss_o = onset_loss(onset_pred,target_o)
                loss_all = loss + 0.5*loss_p + 0.5*loss_o #0.2
                loss_total_i += loss.data
                loss_total_p += loss_p.data
                loss_total_o += loss_o.data

                optimizer.zero_grad()
                loss_all.backward()

                clip_grad_norm_(self.model.parameters(), 3)

                optimizer.step()
                scheduler.step()
                
                #frush the board
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d]\tLoss %4f\tTime %d'
                        %(e, self.epoch, batch_idx+1, len(tr_loader),
                            loss.data, time.time() - st))
                sys.stdout.flush()
 
            print ('\n')
            print (loss_total_i/len(tr_loader))
            print(loss_total_p/len(tr_loader))
            print(loss_total_o/len(tr_loader))

            if e % self.validation_interval == 1:
                print (self.save_fn)
                eva_result = self.Tester(va_loader, len(va_loader.dataset), we)
                self.model.train()

                viz.line([[float(loss_total_i/len(tr_loader.dataset)), float(eva_result[0])]], [e - 1], win="IPT_loss_" + saveName, update='append')
                viz.line([[float(eva_result[1])]], [e - 1], win="IPT_precision_" + saveName, update='append')
                viz.line([[float(eva_result[2])]], [e - 1], win="IPT_recall_" + saveName, update='append')
                viz.line([[float(eva_result[3])]], [e - 1], win="IPT_F1_" + saveName, update='append')
                viz.line([[float(loss_total_p / len(tr_loader.dataset)), float(eva_result[4])]], [e - 1], win="pitch_loss_" + saveName, update='append')
                viz.line([[float(eva_result[5])]], [e - 1], win="pitch_precision_" + saveName, update='append')
                viz.line([[float(eva_result[6])]], [e - 1], win="pitch_recall_" + saveName, update='append')
                viz.line([[float(eva_result[7])]], [e - 1], win="pitch_F1_" + saveName, update='append')
                viz.line([[float(loss_total_o / len(tr_loader.dataset)), float(eva_result[8])]], [e - 1], win="onset_loss_" + saveName, update='append')
                print("IPT_F1:", eva_result[3])
                print("pitch_F1:", eva_result[7])
                if eva_result[3] > best_acc:
                    best_acc = eva_result[3]
                    last_best_epoch = e

                    rm_lst = glob(self.save_fn + 'best_*')
                    for p in rm_lst:
                        os.remove(p)
                    torch.save(self.model.state_dict(), self.save_fn + 'best'+'_e_%d'%(e-1))
                else:
                    if e-last_best_epoch >= EARLY_STOPPING:
                        print('Early stopping at epoch {}...'.format(e + 1))
                        break

            # if e % self.save_interval == 1:
            #     torch.save(self.model.state_dict(), self.save_fn+'_e_%d'%(e-1))