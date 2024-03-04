#config
import torch

URL = "m-a-p/MERT-v1-95M" #model URL
TIME_LENGTH = 5 #5 seconds
LENGTH = 375 #number of frame in 3 seconds,225
NUM_LABELS = 7 #number of IPTs
BATCH_SIZE = 10
SAMPLE_RATE = 44100 #Raw audio sampling rate
MERT_SAMPLE_RATE = 24000 if "MERT" in URL else 16000 #input audio sampling rate of MERT
FEATURE_RATE = 75 # FEATURE_RATE = 1000//ZHEN_LENGTH，Sampling rate of feature extracted from MERT
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
TWO_STEP = False # Whether two-step finetuning
LIN_EPOCH = 5 #If fine-tuning is done in two steps, which epochs should we start fine-tuning the pre-trained model
FREEZE_ALL = False # Whether to freeze all parameters of the self-supervised pre-training model
EARLY_STOPPING = 1000 #early_stopping
saveName = "mul_onset7_pitch_IPT_share_weight_weighted_loss-" + URL.split("/")[-1] #name of the model to save and load
DATASET = "Guzheng_Tech99"

MIN_MIDI = 36 #音域内最低音的midi值 C2 36
MAX_MIDI = 87 #音域内最高音的midi值 Eb6 87
HOPS_IN_ONSET = 1 #onset跨越几帧