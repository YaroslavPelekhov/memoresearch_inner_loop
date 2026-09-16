import hashlib
from typing import Any, Dict, List
from collections import defaultdict

import torch
from torchmetrics import Metric


class PredictionAccuracy(Metric):
    correct: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("correct", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, outputs: Dict, labels: List[str]):
        output_list = outputs["generated_output"]
        assert len(output_list) == len(labels)

        self.correct += sum(out == label for out, label in zip(output_list, labels))
        self.total += len(labels)

    def compute(self):
        return self.correct / self.total


class MMLUPredictionAccuracy(Metric):
    correct: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("correct", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, outputs: Dict, labels: List[str]):
        output_list = outputs["generated_output"]
        assert len(output_list) == len(labels)

        # Extract first symbol of prediction and target, corresponding to answer letter
        self.correct += sum(out[0] == label[0] for out, label in zip(output_list, labels))
        self.total += len(labels)

    def compute(self):
        return self.correct / self.total


class BaseMatchAgggregator(Metric):
    prediction_mapping: List[torch.Tensor]

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("prediction_mapping", default=[], dist_reduce_fx="cat")

    def update(self, outputs: Dict, labels: List[str]):
        output_list = outputs["generated_output"]
        assert len(output_list) == len(labels)

        state_update = []
        for out, label in zip(output_list, labels):
            label_hash = int(hashlib.sha1(label.encode("utf-8")).hexdigest(), 16) % (10 ** 16)
            prediction_match = int(out == label)
            t = torch.tensor([label_hash, prediction_match], dtype=torch.long, device=self.device)
            state_update.append(t)

        self.prediction_mapping += state_update


class MulticlassPredictionAccuracy(BaseMatchAgggregator):
    def compute(self):
        prediction_mapping = self.prediction_mapping
        assert prediction_mapping.shape[0] % 2 == 0

        label_hash_cnt, prediction_match_cnt = defaultdict(int), defaultdict(int)

        for i in range(prediction_mapping.shape[0] // 2):
            label_hash = prediction_mapping[2 * i].item()
            prediction_match = prediction_mapping[2 * i + 1].item()

            label_hash_cnt[label_hash] += 1
            prediction_match_cnt[label_hash] += prediction_match

        return {k: prediction_match_cnt[k] / label_hash_cnt[k] for k in label_hash_cnt}


class AverageRecall(BaseMatchAgggregator):
    def compute(self):
        prediction_mapping = self.prediction_mapping
        assert prediction_mapping.shape[0] % 2 == 0

        label_hash_cnt, prediction_match_cnt = defaultdict(int), defaultdict(int)

        for i in range(prediction_mapping.shape[0] // 2):
            label_hash = prediction_mapping[2 * i].item()
            prediction_match = prediction_mapping[2 * i + 1].item()

            label_hash_cnt[label_hash] += 1
            prediction_match_cnt[label_hash] += prediction_match

        label_hash_cnt, prediction_match_cnt = dict(label_hash_cnt), dict(prediction_match_cnt)
        assert len(label_hash_cnt) == len(prediction_match_cnt)

        num_labels = len(label_hash_cnt)
        recall_list = [prediction_match_cnt[k] / label_hash_cnt[k] for k in label_hash_cnt]

        return sum(recall_list) / num_labels
