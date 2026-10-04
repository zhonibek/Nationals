"""Compact multi-task neural predictor and a linear comparison baseline."""

import torch
from torch import nn


class FailurePredictor(nn.Module):
    def __init__(self, input_size, label_count, metric_count, width=128, depth=3):
        super().__init__()
        if not 16 <= width <= 512 or depth not in (3, 4):
            raise ValueError("Expected 3-4 hidden layers, width 16-512")
        layers = []
        for layer in range(depth):
            layers.extend([nn.Linear(input_size if layer == 0 else width, width), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.classification = nn.Linear(width, label_count)
        self.regression = nn.Linear(width, metric_count)

    def forward(self, inputs):
        hidden = self.trunk(inputs)
        return self.classification(hidden), self.regression(hidden)


def loss_for(logits, predictions, labels, regression, mask):
    classification_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    observed_error = ((predictions - regression).square() * mask).sum() / mask.sum().clamp_min(1)
    return classification_loss + 0.25 * observed_error


def classification_metrics(probabilities, labels, names):
    output = {}
    for column, name in enumerate(names):
        scores = probabilities[:, column].detach().cpu().tolist()
        observed = labels[:, column].detach().cpu().tolist()
        positive = sum(observed)
        negatives = len(observed) - positive
        true_positive = sum(value >= 0.5 and truth == 1 for value, truth in zip(scores, observed))
        false_negative = sum(value < 0.5 and truth == 1 for value, truth in zip(scores, observed))
        false_positive = sum(value >= 0.5 and truth == 0 for value, truth in zip(scores, observed))
        accumulated_positive = 0
        average_precision = 0.0
        ordering = sorted(range(len(scores)), key=lambda index: -scores[index])
        cursor = 0
        while cursor < len(ordering):
            end = cursor + 1
            while end < len(ordering) and scores[ordering[end]] == scores[ordering[cursor]]:
                end += 1
            found = sum(observed[index] for index in ordering[cursor:end])
            accumulated_positive += found
            average_precision += found * accumulated_positive / end
            cursor = end
        calibration_error = 0.0
        for bucket in range(10):
            members = [index for index, score in enumerate(scores) if min(9, int(score * 10)) == bucket]
            if members:
                predicted = sum(scores[index] for index in members) / len(members)
                actual = sum(observed[index] for index in members) / len(members)
                calibration_error += len(members) / len(scores) * abs(predicted - actual)
        output[name] = dict(count=len(scores), positives=int(positive), negatives=int(negatives),
                            recall=true_positive / positive if positive else None,
                            false_negatives=false_negative, false_positives=false_positive,
                            precision=true_positive / (true_positive + false_positive) if true_positive + false_positive else None,
                            average_precision=average_precision / positive if positive and negatives else None,
                            brier=sum((score - truth) ** 2 for score, truth in zip(scores, observed)) / len(scores),
                            expected_calibration_error_10_bins=calibration_error)
    return output
