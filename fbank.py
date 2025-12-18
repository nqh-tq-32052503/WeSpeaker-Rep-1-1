import torchaudio.compliance.kaldi as kaldi
import torch
from lhotse.features import Fbank, FbankConfig

EXTRACTOR = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80, device="cuda"))

def extract_fbank_using_kaldi(waveform, normalize=True):
    waveform = waveform.to("cuda")
    feat = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, sample_frequency=16000, window_type='hamming')
    if normalize:
        feat = feat - torch.mean(feat, 0)
    return feat

def extract_fbank_using_lhotse(waveform):
    fbank_np = EXTRACTOR.extract(samples=waveform, sampling_rate=16000)  # (T, F) np.float32
    fbank_t = torch.from_numpy(fbank_np)
    return fbank_t

def extract_fbank(waveform, name="kaldi"):
    if name == "kaldi":
        feat = extract_fbank_using_kaldi(waveform)
    else:
        feat = extract_fbank_using_lhotse(waveform)
    return feat