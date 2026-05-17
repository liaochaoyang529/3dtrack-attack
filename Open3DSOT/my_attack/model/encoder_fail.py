import torch
import torch.nn as nn
from models import get_model

class FrozenM2TrackBackbone(nn.Module):
    def __init__(self, m2track_model):
        super().__init__()
        self.m2track = m2track_model

        self.m2track.eval()
        for p in self.m2track.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        self.m2track.eval()
        return self

    def forward(self, input_dict, detach=True):
        if detach:
            with torch.no_grad():
                out = self.m2track(input_dict, return_point_feature=True)
        else:
            out = self.m2track(input_dict, return_point_feature=True)
        return out["point_feature"]  # [B, 256]
class Encoder(nn.Module):  
    def __init__(self, input_size=256, hidden_size=128, output_size=32):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
class FailureAwareModel(nn.Module):
    def __init__(self, m2track_model, embed_dim=32):
        super().__init__()
        self.backbone = FrozenM2TrackBackbone(m2track_model = m2track_model)
        self.encoder = Encoder(input_size=256, hidden_size=128, output_size=embed_dim)
        self.cls_head = nn.Linear(embed_dim,1)
    def forward(self,input_dict,detach_backbone=True):
        point_feature = self.backbone(input_dict, detach=detach_backbone)
        embeddings = self.encoder(point_feature)
        logit = self.cls_head(embeddings).squeeze(-1)
        return {
            "point_feature": point_feature,
            "embedding": embeddings,
            "failure_logit": logit,
        }

def build_failure_model(cfg, ckpt_path, device):
    model_cls = get_model(cfg.net_model)
    m2track = model_cls.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=cfg,
    )
    m2track = m2track.to(device)
    m2track.eval()

    failure_model = FailureAwareModel(m2track).to(device)
    return failure_model

