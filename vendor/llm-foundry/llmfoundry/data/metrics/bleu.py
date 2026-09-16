import re
from typing import Dict, List

from torchmetrics.text import BLEUScore


class BLEUComposer(BLEUScore):
    def update(self, outputs: Dict, labels: List[str]):
        output_list_copy = outputs["generated_output"].copy()
        labels_copy = labels.copy()
        assert len(output_list_copy) == len(labels_copy)

        for i in range(len(output_list_copy)):
            output_list_copy[i] = re.sub(r'[^\w\s]', '', output_list_copy[i]).lower()
            labels_copy[i] = [re.sub(r'[^\w\s]', '', labels_copy[i]).lower()]

        super().update(output_list_copy, labels_copy)
