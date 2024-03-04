import numpy as np
from glob import glob
import os
import csv
import librosa
from tqdm import tqdm
from config import *
import torch
import torchaudio.transforms as T

def load(wav_dir, csv_dir, groups, avg=None, std=None):
    #Return a list of [(audio location, corresponding CSV file location), (,),...]
    if std is None:
        std = np.array([None])
    if avg is None:
        avg = np.array([None])

    def files(wav_dir, csv_dir, group):
        flacs = sorted(glob(os.path.join(wav_dir, group, '*.flac')))
        if len(flacs) == 0:
            flacs = sorted(glob(os.path.join(wav_dir, group, '*.wav')))

        csvs = sorted(glob(os.path.join(csv_dir, group, '*.csv')))
        files = list(zip(flacs, csvs))
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')
        result = []
        for audio_path, csv_path in files:
            result.append((audio_path, csv_path))
        return result

    def get_wav(file):
        sr = SAMPLE_RATE
        y, sr = librosa.load(file, sr=sr)
        sampling_rate = sr
        resample_rate = MERT_SAMPLE_RATE
        if resample_rate != sampling_rate:
            print(f'setting rate from {sampling_rate} to {resample_rate}')
            resampler = T.Resample(sampling_rate, resample_rate)
        else:
            resampler = None

        # audio file is decoded on the fly
        if resampler is None:
            input_audio = y
        else:
            input_audio = resampler(torch.from_numpy(y))
        return input_audio

    def chunk_data(f):
        s = int(FEATURE_RATE*TIME_LENGTH)
        xdata = np.transpose(f) #[time,feat]
        x = []
        length = int(np.ceil(len(xdata) / s) * s)
        app = np.zeros((length - xdata.shape[0], xdata.shape[1]))
        xdata = np.concatenate((xdata, app), 0)
        for i in range(int(length / s)):
            data = xdata[int(i * s):int(i * s + s)]
            x.append(np.transpose(data[:s, :]))

        return np.array(x)

    def chunk_data_test(f):
        s = int(FEATURE_RATE * TIME_LENGTH)
        xdata = np.transpose(f)  # [time,feat]
        x = []
        for i in range(int(np.ceil(len(xdata) / s))):
            data = xdata[int(i * s):min(int(i * s + s), len(xdata))]
            x.append(np.transpose(data[:, :]))
        return x

    #Trim each audio input to a size of 3 seconds
    def chunk_wav(f):
        s = int(MERT_SAMPLE_RATE*TIME_LENGTH)
        xdata = f
        x = []
        length = int(np.ceil(len(xdata) / s) * s)
        app = np.zeros((length - xdata.shape[0]))
        xdata = np.concatenate((xdata, app), 0)
        for i in range(int(length / s)):
            data = xdata[int(i * s):int(i * s + s)]
            x.append(data)
        return np.array(x)

    def chunk_wav_test(f):
        s = int(MERT_SAMPLE_RATE * TIME_LENGTH)
        xdata = f
        x = []
        length = int(np.ceil(len(xdata) / s) * s)
        app = np.zeros((length - xdata.shape[0]))
        xdata = np.concatenate((xdata, app), 0)
        for i in range(int(length / s)):
            data = xdata[int(i * s):int(i * s + s)]
            x.append(data)
        return x

    
    def load_all(audio_path, csv_path):

        saved_data_path = audio_path.replace('.flac', '_multi9898.pt').replace('.wav', '_multi9898.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        #Load audio features
        feature = get_wav(audio_path) #feature's shape(The number of Time frame)

        #Load the ground truth label
        n_steps = int(FEATURE_RATE*len(feature)//MERT_SAMPLE_RATE)
        n_IPTs = NUM_LABELS
        n_keys = MAX_MIDI - MIN_MIDI + 1

        technique = {'chanyin': 0, 'dianyin': 6, 'shanghua': 2, 'xiahua': 3, 'huazhi':4, 'guazou': 4, 'lianmo': 4, 'liantuo': 4, 'yaozhi': 5, 'boxian': 1}

        IPT_label = np.zeros([n_IPTs, n_steps], dtype=int)
        pitch_label = np.zeros([n_keys, n_steps], dtype=int)
        onset_label = np.zeros([1, n_steps], dtype=int)

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:  #each note
                onset = float(label['onset_time'])
                offset = float(label['offset_time'])
                IPT = int(technique[label['IPT']])
                note = int(label['note'])
                left = int(round(onset * FEATURE_RATE))
                onset_right = min(n_steps, left + HOPS_IN_ONSET)
                frame_right = int(round(offset * FEATURE_RATE))
                frame_right = min(n_steps, frame_right)
                fn = int(note) - MIN_MIDI
                IPT_label[IPT, left:frame_right] = 1
                pitch_label[fn, left:frame_right] = 1
                onset_label[0, left:onset_right] = 1

        data = dict(audiuo_path=audio_path, csv_path=csv_path, feature=feature, IPT_label=IPT_label, pitch_label=pitch_label, onset_label=onset_label)
        torch.save(data, saved_data_path)
        return data

    data = []
    print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} ")
    for group in groups:
        for input_files in tqdm(files(wav_dir, csv_dir, group), desc='Loading group %s' % group):
            data.append(load_all(*input_files))

    print("Start chunking data~~~")
    if "test" in groups:
        Xte = []
        Yte = []
        Yte_p = []
        Yte_o = []
        for dic in data:
            x = dic['feature']
            x = chunk_wav_test(x)  # [Batch,time]
            y_i = dic['IPT_label']
            y_i = chunk_data_test(y_i)  # [27, 156, 8]
            y_p = dic['pitch_label']
            y_p = chunk_data_test(y_p)
            y_o = dic['onset_label']
            y_o = chunk_data_test(y_o)
            Xte += x
            Yte += y_i
            Yte_p += y_p
            Yte_o += y_o
        print("len(Xte):", len(Xte))  # (2482, 13, 768, 225)
        print("len(Yte):", len(Yte))  # (2482, 7, 225)
        print("len(Yte_p):", len(Yte_p))
        print("len(Yte_o):", len(Yte_o))
        return Xte, Yte, Yte_p, Yte_o, avg, std
    else:
        i = 0
        for dic in data:
            x = dic['feature']
            x = chunk_wav(x)
            y_i = dic['IPT_label']
            y_i = chunk_data(y_i)
            y_p = dic['pitch_label']
            y_p = chunk_data(y_p)
            y_o = dic['onset_label']
            y_o = chunk_data(y_o)

            if i == 0:
                Xtr = x
                Ytr_i = y_i
                Ytr_p = y_p
                Ytr_o = y_o
                i += 1
            else:
                Xtr = np.concatenate([Xtr,x],axis=0)
                Ytr_i = np.concatenate([Ytr_i,y_i],axis=0)
                Ytr_p = np.concatenate([Ytr_p, y_p], axis=0)
                Ytr_o = np.concatenate([Ytr_o, y_o], axis=0)

        print("Xtr.shape", Xtr.shape) #(2482, 13, 768, 225)
        print("Ytr_i.shape", Ytr_i.shape) #(2482, 7, 225)
        print("Ytr_p.shape", Ytr_p.shape)
        print("Ytr_o.shape", Ytr_o.shape)
        return Xtr, Ytr_i, Ytr_p, Ytr_o, avg, std
