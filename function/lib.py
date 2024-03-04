import sys
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from config import *
from collections import defaultdict
from scipy.stats import hmean
from mir_eval.util import midi_to_hz
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes

# Dataset
class Data2Torch2(Dataset):
    def __init__(self, data):

        self.X = data[0]
        self.Y = data[1]
        self.Y_p = data[2]
        self.Y_o = data[3]

    def __getitem__(self, index):

        mX = self.X[index].float()
        mY = torch.from_numpy(self.Y[index]).float()
        mY_p = torch.from_numpy(self.Y_p[index]).float()
        mY_o = torch.from_numpy(self.Y_o[index]).float()
        return mX, mY, mY_p, mY_o
    
    def __len__(self):
        return len(self.X)


def sp_loss(fla_pred, target, gwe): #[batch, IPT, time]

    we = gwe.to(device) # we: tensor([0.9210, 0.4433, 1.3906, 2.0617, 2.6110, 2.0190, 2.0366]
    wwe = 1
    we *= wwe
    
    loss = 0

    for idx, (out, fl_target) in enumerate(zip(fla_pred,target)):
        twe = we.view(-1,1).repeat(1, fl_target.size(1)).type(torch.cuda.FloatTensor) #[IPT_NUM, time_frame]
        ttwe = twe * fl_target.data + (1 - fl_target.data) * wwe
        loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)   
        loss += loss_fn(torch.squeeze(out), fl_target)
 
    return loss

def pitch_loss(pit_pred, pit_tar):
    loss_p = 0
    for idx, (out, fl_target) in enumerate(zip(pit_pred,pit_tar)):
        loss_fn = nn.BCEWithLogitsLoss(size_average=True)
        loss_p += loss_fn(out, fl_target)
    return loss_p

def onset_loss(on_pred, on_tar):
    loss_o = 0
    for idx, (out, fl_target) in enumerate(zip(on_pred,on_tar)):
        ttwe = 10 * fl_target.data + (1 - fl_target.data) * 1 #9 * fl_target.data + 1，这个加权损失只是在加大正样本的权重
        loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)
        loss_o += loss_fn(out, fl_target)
    return loss_o

def num_params(model):
    params = 0
    for i in model.parameters():
        params += i.view(-1).size()[0]
    print ('#params:%d'%(params))

def notes_to_frames_no_inference(roll):

    time = np.arange(roll.shape[-1])
    freqs = [roll[:, t].nonzero()[0] for t in time]
    return time, freqs

def extract_notes(onsets, frames, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, 1]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    """
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8) #形状[时间轴帧数,1]
    frames = (frames > frame_threshold).cpu().to(torch.uint8) #形状[时间轴帧数,88]
    # 因为按照论文里说的，大部分的onset会跨2个frames，所以后一个frame减去前一个frame的onset值=1表示这是一个新的onset。有onset的frame值为1，没有onset的frame值为0
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1 #形状[时间轴帧数,1]

    pitches = []
    intervals = []

    for nonzero in onset_diff.nonzero():
        #例：nonzero：[312, 0]表示第312帧有onset
        frame = nonzero[0].item()
        pitch = frames[frame].nonzero() #形状[n,1] #n表示非0帧的个数

        onset = frame

        # inference的过程
        for i in range(len(pitch)):
            offset = frame
            while onsets[offset, 0].item() or frames[offset, pitch[i, 0].item()].item(): #当frames[offset, pitch].item()为0时停止循环，此时的offset是该音符的offset
                offset += 1
                if offset == onsets.shape[0]: #当offset走到了最后一个时间帧以外则停止循环
                    break

            if offset > onset:
                pitches.append(pitch[i,0].item())
                intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs, roll

def extract_notes_original(onsets, frames, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    """
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8) #形状[时间轴帧数,88]
    frames = (frames > frame_threshold).cpu().to(torch.uint8) #形状[时间轴帧数,88]
    # 因为按照论文里说的，大部分的onset会跨2个frames，所以后一个frame减去前一个frame的onset值=1表示这是一个新的onset。有onset的frame值为1，没有onset的frame值为0
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1 #形状[时间轴帧数,88]

    pitches = []
    intervals = []

    for nonzero in onset_diff.nonzero():
        #例：nonzero：[312, 68]表示第312帧中有68这个音的onset
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame

        # inference的过程
        while onsets[offset, pitch].item() or frames[offset, pitch].item(): #当frames[offset, pitch].item()为0时停止循环，此时的offset是该音符的offset
            offset += 1
            if offset == onsets.shape[0]: #当offset走到了最后一个时间帧以外则停止循环
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals)

#计算进行了post-processing之后的note-level和frame-level指标
def compute_metrics_with_note(pred_IPT, tar_IPT, pred_onset, tar_onset):
    metrics = defaultdict(list)
    eps = sys.float_info.epsilon
    p_ref, i_ref= extract_notes_original(torch.from_numpy(tar_onset).transpose(-1, -2), torch.from_numpy(tar_IPT).transpose(-1, -2))
    p_est, i_est= extract_notes(torch.from_numpy(pred_onset).transpose(-1, -2), torch.from_numpy(pred_IPT).transpose(-1, -2))

    t_ref, f_ref, tt = notes_to_frames(p_ref, i_ref, tar_IPT.transpose(-1,-2).shape)
    t_est, f_est, ee = notes_to_frames(p_est, i_est, pred_IPT.transpose(-1,-2).shape)

    t_ref = t_ref.astype(np.float64) / FEATURE_RATE
    f_ref = [np.array([midi_to_hz(21 + midi) for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) / FEATURE_RATE
    f_est = [np.array([midi_to_hz(21 + midi) for midi in freqs]) for freqs in f_est]

    IPT_frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/IPT_frame/f1'].append(
        hmean([IPT_frame_metrics['Precision'] + eps, IPT_frame_metrics['Recall'] + eps]) - eps)

    for key, loss in IPT_frame_metrics.items():
        metrics['metric/IPT_frame/' + key.lower().replace(' ', '_')].append(loss)

    i_ref = (i_ref /FEATURE_RATE).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(midi) for midi in p_ref])
    i_est = (i_est /FEATURE_RATE).reshape(-1, 2) #scaling = 0.05 一帧的时间 = 1/FEATURE_RATE
    p_est = np.array([midi_to_hz(midi) for midi in p_est])
    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=0.05, pitch_tolerance=0)
    metrics['metric/note/precision'].append(p)
    metrics['metric/note/recall'].append(r)
    metrics['metric/note/f1'].append(f)
    metrics['metric/note/overlap'].append(o)

    return metrics, ee.transpose(-1,-2)

#计算没有用onset进行post-processing时候的note-level指标
def compute_metrics_with_note_no_infer(pred_IPT, tar_IPT, pred_onset, tar_onset):
    metrics = defaultdict(list)
    eps = sys.float_info.epsilon
    p_ref, i_ref = extract_notes_original(torch.from_numpy(tar_onset).transpose(-1, -2), torch.from_numpy(tar_IPT).transpose(-1, -2))
    p_est, i_est= extract_notes_original(torch.from_numpy(pred_IPT).transpose(-1, -2), torch.from_numpy(pred_IPT).transpose(-1, -2))

    t_ref, f_ref, tt = notes_to_frames(p_ref, i_ref, tar_IPT.transpose(-1,-2).shape)
    t_est, f_est, ee = notes_to_frames(p_est, i_est, pred_IPT.transpose(-1,-2).shape)

    t_ref = t_ref.astype(np.float64) / FEATURE_RATE
    f_ref = [np.array([midi_to_hz(21 + midi) for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) / FEATURE_RATE
    f_est = [np.array([midi_to_hz(21 + midi) for midi in freqs]) for freqs in f_est]

    IPT_frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/IPT_frame/f1'].append(
        hmean([IPT_frame_metrics['Precision'] + eps, IPT_frame_metrics['Recall'] + eps]) - eps)

    for key, loss in IPT_frame_metrics.items():
        metrics['metric/IPT_frame/' + key.lower().replace(' ', '_')].append(loss)

    i_ref = (i_ref /FEATURE_RATE).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(midi) for midi in p_ref])
    i_est = (i_est /FEATURE_RATE).reshape(-1, 2) #scaling = 0.05 一帧的时间 = 1/FEATURE_RATE
    p_est = np.array([midi_to_hz(midi) for midi in p_est])
    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=0.05, pitch_tolerance=0)
    metrics['metric/note/precision'].append(p)
    metrics['metric/note/recall'].append(r)
    metrics['metric/note/f1'].append(f)
    metrics['metric/note/overlap'].append(o)

    return metrics, ee.transpose(-1,-2)

#在validation的时候用，只计算帧级别
def compute_metrics(pred_inst, Yte):
    metrics = defaultdict(list)
    eps = sys.float_info.epsilon
    t_ref, f_ref = notes_to_frames_no_inference(Yte)
    t_est, f_est = notes_to_frames_no_inference(pred_inst)

    t_ref = t_ref.astype(np.float64) / FEATURE_RATE
    f_ref = [np.array([midi_to_hz(21 + midi) for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) / FEATURE_RATE
    f_est = [np.array([midi_to_hz(21 + midi) for midi in freqs]) for freqs in f_est]

    IPT_frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/IPT_frame/f1'].append(
        hmean([IPT_frame_metrics['Precision'] + eps, IPT_frame_metrics['Recall'] + eps]) - eps)

    for key, loss in IPT_frame_metrics.items():
        metrics['metric/IPT_frame/' + key.lower().replace(' ', '_')].append(loss)

    return metrics

#在validation的时候用来计算pitch帧级别的指标
def compute_pitch_metrics(pred_inst, Yte):
    metrics = defaultdict(list)
    eps = sys.float_info.epsilon
    t_ref, f_ref = notes_to_frames_no_inference(Yte)
    t_est, f_est = notes_to_frames_no_inference(pred_inst)

    t_ref = t_ref.astype(np.float64) / FEATURE_RATE
    f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) / FEATURE_RATE
    f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

    pitch_frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/pitch_frame/f1'].append(
        hmean([pitch_frame_metrics['Precision'] + eps, pitch_frame_metrics['Recall'] + eps]) - eps)

    for key, loss in pitch_frame_metrics.items():
        metrics['metric/pitch_frame/' + key.lower().replace(' ', '_')].append(loss)

    return metrics
