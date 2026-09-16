import inspect
import functools
import typing as tp
import warnings
from collections import UserDict

import torch
from omegaconf import DictConfig

from composer.models import HuggingFaceModel
from composer.utils import dist
from transformers.generation import GenerationConfig
from transformers.utils.generic import ModelOutput
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.layers.norm import resolve_norm_class
from llmfoundry.models.gigaspeech.utils import prepare_speech_model_for_fsdp
from llmfoundry.utils.config_utils import to_container
from llmfoundry.utils.console_logger import setup_logger
from llmfoundry.models.gigaspeech.configuration_gigaspeech import GigaSpeechConfig
from llmfoundry.models.gigaspeech.modeling_gigaspeech import GigaSpeechForCausalLM
from llmfoundry.data.metrics import (
    WordErrorRateComposer,
    WordErrorRateE2EComposer,
    PredictionAccuracy,
    MMLUPredictionAccuracy,
    MulticlassPredictionAccuracy,
    AverageRecall,
    BLEUComposer,
)

logger = setup_logger(__name__)


class ComposerGigaSpeechCausalLM(HuggingFaceModel):

    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        om_generation_config: DictConfig,
    ) -> None:
        """ """
        config = GigaSpeechConfig(
            encoder_config=om_model_config.encoder_config,
            projector_config=om_model_config.projector_config,
            subsampler_config=om_model_config.subsampler_config,
            decoder_config=om_model_config.decoder_config,
            vocab_size=tokenizer.vocab_size,  # type: ignore
            **to_container(om_model_config.kwargs),
        )
        generation_config = GenerationConfig.from_dict(
            to_container(om_generation_config)
        )

        # TODO (fedorovgv): добавить сюда мердж config_overrides в основной конфиг
        with dist.run_local_rank_zero_first():
            model = GigaSpeechForCausalLM(config, generation_config=generation_config)

        train_metrics = []
        eval_metrics = [
            WordErrorRateComposer(),
            WordErrorRateE2EComposer(),
            PredictionAccuracy(),
            MMLUPredictionAccuracy(),
            MulticlassPredictionAccuracy(),
            BLEUComposer(),
            AverageRecall(),
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
        )

        if om_model_config.get("pretrained_encoder_path", None):
            state_dict = torch.load(
                om_model_config.pretrained_encoder_path, map_location="cpu"
            )
            missing_keys, unexpected_keys = model.encoder.load_state_dict(
                state_dict, strict=False
            )
            logger.info("Loading modality encoder..")
            logger.warning(f"missing_keys: {missing_keys}\nunexpected_keys:{unexpected_keys}")

        if om_model_config.get("pretrained_adapter_path", False):
            state_dict = torch.load(
                om_model_config.pretrained_adapter_path, map_location="cpu"
            )
            missing_keys, unexpected_keys = model.modality_adapter.load_state_dict(
                state_dict, strict=False
            )
            logger.info("Loading modality projector..")
            logger.warning(f"missing_keys: {missing_keys}\nunexpected_keys:{unexpected_keys}")

        vocab_size = tokenizer.vocab_size
        self.special_tokens_set = set(range(vocab_size, vocab_size + 256)) | set([0, 1, 2])
        warnings.warn('We assume that the last 256 tokens in tokenizer are special.')

        self.model_forward_args: tp.List[str] = inspect.getfullargspec(self.model.forward).args

        prepare_speech_model_for_fsdp(self.model)

    def loss(self, outputs: ModelOutput, batch: tp.Mapping) -> torch.Tensor:
        if self.config.return_dict:
            loss, _ = outputs["loss"], outputs["logits"]
        else:
            loss, *_ = outputs
        return loss

    def forward(self, batch: tp.Mapping) -> tp.Dict:
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                "Unexpected batch type. Expected a dictionary with keys corresponding to the inputs " +
                "to the forward function of the Huggingface model"
            )
        return output

    def eval_forward(self, batch: tp.Dict, **kwargs) -> tp.Union[tp.Dict, torch.Tensor]:
        """Overrided method for composer model for generation.

        Args:
            batch (tp.Dict):
                mode (str): `generate`, `generate_with_logits`
        """
        mode = batch.get("mode", None)

        if mode is not None and isinstance(mode, str) and mode.startswith("generate"):
            assert self.tokenizer is not None
            batch.pop("labels")

            input_ids_split, labels_split, pad_sizes = self.model.split_inputs_for_generation(
                batch["input_ids"], batch["pad_sizes"], self.model.generation_config
            )
            self.labels = labels_split

            return_logits = (mode == "generate_with_logits")

            generated_input_ids, logits = self.model.generate_continuation(
                input_ids=input_ids_split,
                spectrograms=batch.get("spectrograms", None),
                spectrogram_lengths=batch.get("spectrogram_lengths", None),
                pad_sizes=pad_sizes,
                generation_config=self.model.generation_config,
                return_logits=return_logits
            )
            assert len(generated_input_ids.shape) == 2
            assert len(labels_split.shape) == 2

            def remove_special_tokens(inp: tp.List, special_tokens_set: tp.List):
                return [token for token in inp if token not in special_tokens_set]

            detokenized_result, detokenized_labels = [], []
            for predicted_tokens, target_tokens in zip(generated_input_ids, labels_split):
                detokenized_result.append(self.tokenizer.decode(
                    remove_special_tokens(predicted_tokens.tolist(), self.special_tokens_set),
                    clean_up_tokenization_spaces=True
                ))
                detokenized_labels.append(self.tokenizer.decode(
                    remove_special_tokens(target_tokens.tolist(), self.special_tokens_set),
                    clean_up_tokenization_spaces=True
                ))

            self.labels = detokenized_labels

            assert len(detokenized_result) == len(detokenized_labels)
            for predicted, target in zip(detokenized_result, detokenized_labels):
                print(f"Predicted: {[predicted]}, Target: {[target]}")

            if batch.get("wav_filepaths", False):
                samples = enumerate(zip(detokenized_labels, detokenized_result, batch["wav_filepaths"]))
                result = {
                    row : {'target': target, 'prediction': prediction, 'wav_filepath': wav_filepath}
                    for row, (target, prediction, wav_filepath) in samples
                }
            else:
                samples = enumerate(zip(detokenized_labels, detokenized_result))
                result = {
                    row : {'target': target, 'prediction': prediction}
                    for row, (target, prediction) in samples
                }

            output = {"generated_output": detokenized_result, "result": result, "logits": logits}

        else:
            raise ValueError("Wrong batch mode, allowed only `generate`, `generate_test`,"
                             + f"but found {mode} !")

        return output

    def gigafsdp_model_setup(self, fsdp_config: tp.Dict) -> None:
        fsdp_config["gigafsdp_wrappers"]["model"] = self.gigafsdp_language_model_setup(fsdp_config)

    def gigafsdp_language_model_setup(self, fsdp_config: tp.Dict) -> tp.Dict[str, tp.Any]:
        """Setup gigafsdp.
        """
        auto_wrap_policy = lambda model: functools.partial(
            torch.distributed.fsdp.wrap.lambda_auto_wrap_policy,
            lambda_fn=lambda m: (
                m in model.encoder
                or m in model.modality_adapter
                or m is model.decoder.model.embed_tokens
                or m in model.decoder.model.layers
                or m is model.decoder.lm_head
            )
        )

        activation_checkpointing_auto_wrap_policy = lambda model: functools.partial(
            torch.distributed.fsdp.wrap.lambda_auto_wrap_policy,
            lambda_fn=lambda m: (
                m in (
                    next(layer.children()) for layer in model.decoder.model.layers[:(
                        len(model.decoder.model.layers)
                        if fsdp_config['num_layers_to_checkpoint'] is None
                        else fsdp_config['num_layers_to_checkpoint']
                    )]
                ) or m in (
                    layer for layer in model.encoder.layers
                )
            )
        )

        giga_fsdp_modules_to_wrap_with_names = lambda model: (
            [(model.encoder, "model.encoder")]
            + [(model.modality_adapter, "model.modality_adapter")]
            + [(model.decoder.model.embed_tokens, "model.decoder.model.embed_tokens")]
            + [
                (m, f"model.decoder.model.layers.{i}") for i, m in enumerate(model.decoder.model.layers)
            ]
            + [(model.decoder.lm_head, "model.decoder.lm_head")]
        )

        giga_fsdp_rogue_layer_norm_modules_with_names = lambda model: {
            model.decoder.model.norm: "model.decoder.model.norm",
        }
        speech_model = self.model
        decoder_norm_type = getattr(
            getattr(speech_model.decoder.model, "config", None),
            "norm_type",
            "LlamaRMSNorm",
        )

        return {
            "auto_wrap_policy": auto_wrap_policy(self.model),
            "activation_checkpointing_auto_wrap_policy": activation_checkpointing_auto_wrap_policy(self.model),
            "giga_fsdp_modules_to_wrap_with_names": giga_fsdp_modules_to_wrap_with_names(self.model),
            "giga_fsdp_rogue_layer_norm_modules_with_names": giga_fsdp_rogue_layer_norm_modules_with_names(self.model),
            "giga_fsdp_layer_norm_module_cls": resolve_norm_class(decoder_norm_type)
        }
