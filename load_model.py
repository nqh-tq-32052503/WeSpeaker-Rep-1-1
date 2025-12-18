from models.resnet import ResNet34, ResNet221
from models.campplus import CAMPPlus
from models.eres2net import ERes2Net34_aug
from models.samresnet import SimAM_ResNet34_ASP
from utils import load_checkpoint
from fbank import extract_fbank
import torch
import yaml
import os

SUPPORTED_MODELS = ["campplus", "chinese", "english", "eres2net", "vblinkf", "vblinkp"]

class Model(object):
    def __init__(self, model_index, checkpoint_path, device="cuda"):
        self.model_name = SUPPORTED_MODELS[model_index]
        self.device = device
        print("Model Name: ", self.model_name)
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
    
    def extract_embedding(self, waveform):
        waveform = waveform.to(torch.float32).to(self.device)
        feature = extract_fbank(waveform)
        feature = feature.unsqueeze(0).to(self.device)
        print("Input shape: ", feature.shape)
        with torch.no_grad():
            outputs = self.model(feature)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            embedding = outputs[0]
            return embedding
    
    def cosine_similarity(self, e1, e2):
        cosine_score = torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2))
        cosine_score = cosine_score.item()
        return (cosine_score + 1.0) / 2  # normalize: [-1, 1] => [0, 1]