from typing import Mapping
from torchmetrics import Metric
import torch
from torch import nn
from torch.nn import functional as F

from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.teacher_student import ComposerTeacherStudentModel
from llmfoundry.models.layers import (
    NORM_CLASS_REGISTRY,
)

from composer.loss import KLLogitsLoss
from composer.metrics.nlp import (
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningQAAccuracy,
)


class ComposerDistilTeacherStudentModel(ComposerTeacherStudentModel):
    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.kl_alpha = om_model_config.kl_alpha
        assert 0 <= self.kl_alpha <= 1  
        self.temperature = om_model_config.get("temperature", 1)

        train_metrics = []
        eval_metrics = [
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]

        super().__init__(
            om_model_config=om_model_config,
            tokenizer=tokenizer,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics
        )

        self.distill_loss_name = om_model_config.get("distill_loss", "kl")
        if self.distill_loss_name == "kl":
            self.loss_fn = KLLogitsLoss(self.temperature)
        elif self.distill_loss_name == "hinton":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"unsupported distillation loss: {self.distill_loss_name}")


    def loss(self, outputs: Mapping, batch: Mapping):
        student_out = outputs
        
        student_loss, student_logits, teacher_out = (
            student_out["loss"], 
            student_out["logits"], 
            student_out["teacher_out"]
        )
    
        if self.teacher_config.use_return_dict:
            teacher_loss, teacher_logits = teacher_out["loss"], teacher_out["logits"]
        else:
            teacher_loss, teacher_logits = teacher_out[0], teacher_out[1]
        
        distill_loss = self.loss_fn(student_logits, teacher_logits)
        total_loss = self.kl_alpha * distill_loss + (1 - self.kl_alpha) * student_loss
        
        output_losses = {"total" : total_loss,
                         f"{self.distill_loss_name}_loss" : distill_loss,
                         "lm_loss" : student_loss,
                         "teacher_lm_loss" : teacher_loss}

        if "z_loss" in student_out:
            output_losses["z_loss"] = student_out["z_loss"]
                
        return output_losses

