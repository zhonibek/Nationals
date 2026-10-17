import torch

from .corpus_warm import FullCorpusAnchor, anchor_definition
from .imitation import POSE_BUDGET_SCALES


ANCHOR_SOURCE = "full-corpus-upper-pace-v1"


def definition(corpus_definition):
    return {**anchor_definition("full-corpus-equal-world-v1", corpus_definition), "method": ANCHOR_SOURCE,
            "paceLoss": "Square positive signed pace error only; slower-than-teacher pace has zero native anchor penalty",
            "poseLoss": "Unchanged symmetric precision-scaled errors on both position references and heading",
            "reduction": "Mean of all four axes times original absolute N/(W*L) world weight, without batch renormalization",
            "warmStart": "Unchanged original symmetric full-data imitation",
            "scope": "Native soft retention only; all four heads remain learned, no inference speed cap/projection, changed reward, physics or gate"}


class DirectionalPaceAnchor(FullCorpusAnchor):
    def loss(self, model, profile="pose-budget-v2"):
        if profile != "pose-budget-v2":
            raise ValueError("Directional native anchor requires the declared precision loss")
        indices = torch.randperm(len(self.features), device="cpu")[:32]
        distribution, _ = model(self.features[indices], encoded=True)
        predicted = distribution.mean * 2 - 1
        if predicted.shape != (len(indices), 4) or not torch.isfinite(predicted).all():
            raise ValueError("Finite matching four-axis native anchor predictions required")
        scales = torch.tensor(POSE_BUDGET_SCALES, dtype=torch.float32, device=predicted.device)
        difference = (predicted - self.targets[indices].to(predicted.device)) * scales
        error = torch.cat((difference[:, :1].clamp_min(0).square(), difference[:, 1:].square()), dim=1)
        return (error.mean(dim=1) * self.row_weights[indices].to(predicted.device)).mean()
