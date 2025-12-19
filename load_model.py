from models.resnet import ResNet34, ResNet221
from models.campplus import CAMPPlus
from models.eres2net import ERes2Net34_aug
from models.samresnet import SimAM_ResNet34_ASP
from utils import load_checkpoint
from fbank import extract_fbank, handle_long_segment, handle_short_segment, STANDARD_SEGMENT_LENGTH
import torch
from tqdm import tqdm
import yaml
import os

SUPPORTED_MODELS = ["campplus", "chinese", "english", "eres2net", "vblinkf", "vblinkp"]
MAX_CHUNK_SIZE = 200
print("SUPPORTED_MODELS: ", SUPPORTED_MODELS)

class Model(object):
    def __init__(self, model_index=1, device="cuda"):
        self.model_name = SUPPORTED_MODELS[model_index]
        self.device = device
        print("Model Name: ", self.model_name)
        checkpoint_path = "./weights/" + self.model_name + ".pt"
        assert os.path.exists(checkpoint_path), "Checkpoint is not found"
        with open("./configs/" + self.model_name + ".yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if model_index == 0:
            model = CAMPPlus(**config["model_args"])
        elif model_index == 1:
            model = ResNet34(**config["model_args"])
        elif model_index == 2:
            model = ResNet221(**config["model_args"])
        elif model_index == 3:
            model = ERes2Net34_aug(**config["model_args"])
        else:
            model = SimAM_ResNet34_ASP(**config["model_args"])
        
        load_checkpoint(model, checkpoint_path)
        self.model = model 
        self.model.eval().to(self.device)
        print("Init model sucessfully")
    
    def __call__(self, list_segments):
        input_segments = [(segment * (2 ** 15)).to(torch.float32).to(self.device) for segment in list_segments] # Mapping to 16-bit range instead of 0-1 normalization
        short_segments = []
        short_indices = []
        long_segments = []
        long_indices = []
        for segment_index, segment in enumerate(input_segments):
            if segment.size(0) >= STANDARD_SEGMENT_LENGTH:
                long_segments.append(segment)
                long_indices.append(segment_index)
            else:
                short_segments.append(segment)
                short_indices.append(segment_index)
        
        long_fbanks = [handle_long_segment(segment) for segment in tqdm(long_segments, desc="Extracting FBank for long segments...")]
        long_embeddings = [self.extract_embedding(long_fbank) for long_fbank in tqdm(long_fbanks, desc="Extracting Embeddings for long segments...")]
        long_embeddings = [embedding.mean(dim=0) for embedding in long_embeddings]

        short_fbanks = [handle_short_segment(segment) for segment in tqdm(short_segments, desc="Extracting FBank for short segments...")]
        merged_short_fbanks = torch.cat(short_fbanks, dim=0)
        N_short = merged_short_fbanks.size(0)
        short_embeddings_list = []
        for i in tqdm(range(0, N_short, MAX_CHUNK_SIZE), desc="Extracting Embeddings for short segments..."):
            input_fbanks = merged_short_fbanks[i : i + MAX_CHUNK_SIZE]
            output_embeddings = self.extract_embedding(input_fbanks)
            short_embeddings_list.append(output_embeddings)
        short_embeddings = torch.cat(short_embeddings_list, dim=0)

        list_embeddings = [None] * len(list_segments)
        for i, long_index in enumerate(long_indices):
            list_embeddings[long_index] = long_embeddings[i]
        for i, short_index in enumerate(short_indices):
            list_embeddings[short_index] = short_embeddings[i]
        
        torch.cuda.empty_cache()
        
        return list_embeddings

    def extract_embedding(self, input_fbanks):
        # waveform = waveform.to(torch.float32).to(self.device)
        # feature = extract_fbank(waveform)
        # feature = feature.unsqueeze(0).to(self.device)
        # print("Input shape: ", feature.shape)
        with torch.no_grad():
            outputs = self.model(input_fbanks)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            return outputs
    
    def cosine_similarity(self, e1, e2):
        cosine_score = torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2))
        cosine_score = cosine_score.item()
        return (cosine_score + 1.0) / 2  # normalize: [-1, 1] => [0, 1]