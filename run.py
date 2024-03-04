import datetime
date = datetime.datetime.now()
import sys
sys.path.append('./function')
from function.fit import *
from function.model import *
from function.load_data import *
from function.config import *
from transformers import Wav2Vec2FeatureExtractor
import torch
import numpy as np
import os
from opendelta import Visualization
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # change

def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
get_random_seed(42)

#Obtain the weight of the loss function to alleviate class imbalance
def get_weight(Ytr):#(1508, 375, 7)
	mp = Ytr[:].sum(0).sum(0) #(7,)
	mmp = mp.astype(np.float32) / mp.sum()
	cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
	inverse_feq = torch.from_numpy(cc)
	return inverse_feq

out_model_fn = './data/model/%d%d%d%d:%d:%d/%s/'%(date.year,date.month,date.day,date.hour,date.minute,date.second,saveName)
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

# load data
wav_dir = DATASET + '/data'
csv_dir = DATASET + '/labels'

groups = ['train']
vali_groups = ['validation']

Xtr,Ytr,Ytr_p,Ytr_o,avg,std = load(wav_dir,csv_dir,groups)
Xva,Yva,Yva_p,Yva_o,va_avg,va_std = load(wav_dir,csv_dir,vali_groups,avg,std)
print ('finishing data loading...')

processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)
Xtrs = processor(Xtr, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
Xvas = processor(Xva, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")

# Build Dataloader
t_kwargs = {'batch_size': BATCH_SIZE, 'num_workers': 2, 'pin_memory': True, 'drop_last': True}
v_kwargs = {'batch_size': 1, 'num_workers': 2, 'pin_memory': True}
tr_loader = torch.utils.data.DataLoader(Data2Torch2([Xtrs["input_values"], Ytr, Ytr_p, Ytr_o]), shuffle=True, **t_kwargs)
va_loader = torch.utils.data.DataLoader(Data2Torch2([Xvas["input_values"], Yva, Yva_p, Yva_o]), **v_kwargs)
print ('finishing data building...')

model = SSLNet(url=URL, class_num=NUM_LABELS*(MAX_MIDI-MIN_MIDI+1),weight_sum=1,freeze_all=FREEZE_ALL).to(device)
print(type(model))
print("before modify:")
print(Visualization(model).structure_graph())

#Obtain the weight of weighted loss
inverse_feq = get_weight(Ytr.transpose(0,2,1))

#Start training
Trer = Trainer(model, 1e-3, 10000, out_model_fn, validation_interval=5, save_interval=100) #0.01
Trer.fit(tr_loader, va_loader,inverse_feq)