import torch.nn.functional as F
import sys
sys.path.append('../fun')
from transformers import AutoModel
from torch import nn
from config import *

class SSLNet(nn.Module):
    def __init__(self,
                 url,
                 class_num,
                 weight_sum=False,
                 freeze_all=False
                 ):
        super().__init__()
        self.num_classes = class_num
        self.url = url
        encode_size = 24 if "330M" in self.url else 12
        self.frontend = HuggingfaceFrontend(url=self.url, use_last=(1-weight_sum), encoder_size=encode_size, freeze_all=freeze_all)
        self.backend = Backend(class_num, encoder_size=encode_size)
        self.backend_onset = Backend(1, encoder_size=encode_size)
        self.backend_attnet_IPT =Backend_Attnet(NUM_LABELS,NUM_LABELS+1)
        self.backend_attnet_pitch = Backend_Attnet(MAX_MIDI-MIN_MIDI+1,MAX_MIDI-MIN_MIDI+1+1)

    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.frontend(x)
        out, feature = self.backend(x) #[batch, time, class_num]
        sizes = out.size()
        out = out.view(sizes[0], sizes[1], NUM_LABELS, MAX_MIDI - MIN_MIDI + 1)
        IPT_pred =  torch.sum(out, dim=3)  # [batch, time, IPT]
        pitch_pred = torch.sum(out, dim=2) # [batch, time, pitch]
        onset_pred, _ = self.backend_onset(x) #[batch, time, class_num]
        onset_pred_deta = onset_pred.detach()

        IPT_onset_cat = torch.cat((IPT_pred, onset_pred_deta), 2)
        pitch_onset_cat = torch.cat((pitch_pred, onset_pred_deta), 2)

        IPT_pred_out = self.backend_attnet_IPT(IPT_onset_cat).transpose(-1,-2)
        pitch_pred_out = self.backend_attnet_pitch(pitch_onset_cat).transpose(-1,-2)
        onset_pred_out = onset_pred.transpose(-1,-2)

        return IPT_pred_out,pitch_pred_out,onset_pred_out


class HuggingfaceFrontend(nn.Module):
    def __init__(self, url, use_last=False, encoder_size=12, freeze_all=False):
        super().__init__()
        print("url is：",url)
        self.model = AutoModel.from_pretrained(URL, trust_remote_code=True)
        if freeze_all:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.feature_extractor._freeze_parameters()

        self.use_last = use_last
        if encoder_size == 12:
            self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(13), requires_grad=True)
        elif encoder_size == 24:
            self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(25), requires_grad=True)

    def forward(self,x):
        x = self.model(x, output_hidden_states=True)
        if self.use_last:
            h = x["last_hidden_state"]
            pad_width = (0, 0, 0, 1)
            h = F.pad(h, pad_width, mode='reflect')
        else:
            h = x["hidden_states"]
            h = torch.stack(h, dim=3)
            pad_width = (0, 0, 0, 0, 0, 1)
            h = F.pad(h, pad_width, mode='reflect')
        if not self.use_last:
            weights = torch.softmax(self.layer_weights,dim=0)
            h = torch.matmul(h, weights)
        return h

class Backend(nn.Module):
    def __init__(self, class_size, encoder_size=12, frame=True) -> None:
        super().__init__()
        assert encoder_size == 12 or encoder_size == 24
        if encoder_size == 12:
            self.feature_dim = 768
        elif encoder_size == 24:
            self.feature_dim = 1024
        else:
            raise NotImplementedError
        self.hidden_dim = 512
        self.proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.hidden_dim, class_size)
        self.frame = frame

    def forward(self, x):
        x = self.proj(x)
        if not self.frame:
            x = x.mean(1, False)
        feature = x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x, feature

class self_attn(nn.Module):
    def __init__(self, embeded_dim, num_heads):
        super(self_attn, self).__init__()
        self.att = nn.MultiheadAttention(embeded_dim, num_heads, batch_first=True)

    def forward(self, x):
        x1 = x #[batch, T/9, FRE*3]
        res_branch, attn_wei = self.att(x1, x1, x1)
        res = torch.add(res_branch, x)
        return res

class Backend_Attnet(nn.Module):
    def __init__(self, class_size, feature_dim):
        super().__init__()
        self.Attn = self_attn(feature_dim, 1)
        self.proj = nn.Linear(feature_dim, class_size)

    def forward(self, x):
        x = self.Attn(x)
        x = self.proj(x)
        return x