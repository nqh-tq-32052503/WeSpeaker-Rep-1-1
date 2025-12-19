import torchaudio.compliance.kaldi as kaldi
import torch
from tqdm import tqdm
from lhotse.features import Fbank, FbankConfig
from torch.func import vmap

STANDARD_SEGMENT_DURATION = 5
SAMPLE_RATE = 16000
STANDARD_SEGMENT_LENGTH = int(STANDARD_SEGMENT_DURATION * SAMPLE_RATE)
VMAP_FBANK = vmap(lambda w: kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, frame_length=25, frame_shift=10, sample_frequency=16000, window_type='hamming'))
EXTRACTOR = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80, device="cuda"))
WINDOW_OFFSET = torch.arange(STANDARD_SEGMENT_LENGTH).unsqueeze(0).to("cuda")

def handle_short_segment(short_segment: torch.Tensor):
    num_duplicates = int(STANDARD_SEGMENT_LENGTH / short_segment.shape[0]) + 1
    duplicated_segment = torch.cat([short_segment] * num_duplicates, dim=0)
    duplicated_segment = duplicated_segment[:STANDARD_SEGMENT_LENGTH].unsqueeze(0).to("cuda")
    embedding = VMAP_FBANK(duplicated_segment)
    return embedding

def handle_long_segment(long_segment: torch.Tensor, stride=int(1 * SAMPLE_RATE)):
    length = long_segment.size(0)
    window_size = STANDARD_SEGMENT_LENGTH
    max_start = length - window_size
    stride_forward = window_size - stride
    start_indices_forward = torch.arange(0, max_start + 1, stride_forward).to("cuda")
    N1 = len(start_indices_forward)
    # Bắt đầu indices: (N1 x 1) + (1 x W) = (N1 x W)
    indices_N1 = start_indices_forward.unsqueeze(1) + WINDOW_OFFSET
    # Trích xuất dữ liệu (N1 x W)
    chunks_N1 = torch.index_select(long_segment, 0, indices_N1.flatten()).view(N1, window_size)
    embedding_chunks = VMAP_FBANK(chunks_N1)
    return embedding_chunks

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