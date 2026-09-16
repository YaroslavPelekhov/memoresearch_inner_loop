import os
import json
import typing as tp
from pathlib import Path

from transformers import PretrainedConfig
from omegaconf import DictConfig

from composer.utils import dist

from .registry import DECODER_CONFIG_CLASS_RESOLVER
from .modules.modality_adapter import ModalityAdapterConfig
from .modules import build_subsampler_config, build_projector_config, build_encoder_config
from llmfoundry.models.utils.configuration_utils import get_sp_split_type


def build_decoder_config(decoder_config: DictConfig):
    assert decoder_config.get("model_type", "default") in DECODER_CONFIG_CLASS_RESOLVER

    pretrained_model_name_or_path = decoder_config.pop("pretrained_model_name_or_path")
    trust_remote_code = decoder_config.pop("trust_remote_code", True)
    use_auth_token = decoder_config.pop("use_auth_token", False)

    cfg_cls = DECODER_CONFIG_CLASS_RESOLVER[decoder_config.model_type].from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_auth_token=use_auth_token,
        **decoder_config,
    )
    cfg_cls.sp_split_type = get_sp_split_type(split_type=decoder_config.get("sp_split_type", None), attention_type=cfg_cls.attention_type)

    if hasattr(cfg_cls, "tp_size") and cfg_cls.tp_size > 1:
        tp_group_size = dist.get_tp_group_size()
        assert tp_group_size is None or cfg_cls.tp_size == tp_group_size, (
            f"Wrong tensor parallel group size. {tp_group_size} instead of {cfg_cls.tp_size}"
        )

    return cfg_cls


class GigaSpeechConfig(PretrainedConfig):
    model_type = "giga_speech"

    def __init__(
            self,
            encoder_config: tp.Optional[DictConfig] = None,
            projector_config: tp.Optional[DictConfig] = None,
            subsampler_config: tp.Optional[DictConfig] = None,
            decoder_config: tp.Optional[DictConfig] = None,
            role_sep_id: tp.Optional[int] = None,
            message_sep_id: tp.Optional[int] = None,
            audio_token_id: tp.Optional[int] = None,
            chunk_input_audio_size: tp.Optional[int] = None,
            freeze_encoder: tp.Optional[bool] = None,
            varlen_input: bool = False,
            return_dict: bool = False,
            **kwargs: tp.Dict,
        ):
        """Main config for acoustic modality.

        Args:
            chunk_input_audio_size (tp.Optional[int], optional): Chunk size for chunked inference.
        """
        super().__init__(**kwargs)

        self.encoder_config = build_encoder_config(encoder_config) if encoder_config is not None else None

        projector_config_cls = build_projector_config(projector_config) if projector_config is not None else None
        subsampler_config_cls = build_subsampler_config(subsampler_config) if subsampler_config is not None else None

        if subsampler_config_cls is not None and projector_config_cls is not None:
            self.modality_adapter_config = ModalityAdapterConfig(
                projector_cfg=projector_config_cls,
                subsampler_cfg=subsampler_config_cls
            )
        else:
            self.modality_adapter_config = None

        self.decoder_config = build_decoder_config(decoder_config) if decoder_config is not None else None

        self.role_sep_id = role_sep_id
        self.message_sep_id = message_sep_id
        self.audio_token_id = audio_token_id

        self.chunk_input_audio_size = chunk_input_audio_size
        self.freeze_encoder = freeze_encoder

        self.varlen_input = varlen_input
        if self.decoder_config is not None:
            setattr(self.decoder_config, "varlen_input", varlen_input)

        self.vocab_size = self.decoder_config.vocab_size if self.decoder_config else None
        self.return_dict = return_dict

    def _save_part_config(
            self,
            config: PretrainedConfig,
            save_cfg_filepath: Path,
        ) -> None:
        save_cfg_filepath.parent.mkdir(parents=True, exist_ok=True)
        config_dict = config.to_dict()
        config_dict["_name_or_path"] = ""
        config_dict["pretrain_path"] = None
        with open(save_cfg_filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

    def save_pretrained(
            self,
            save_directory: tp.Union[str, os.PathLike],
            push_to_hub: bool = False,
            **kwargs: tp.Dict,
        ) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        encoder_cfg_filepath = save_directory / "encoder" / "config.json"
        self._save_part_config(self.encoder_config, encoder_cfg_filepath)

        modality_adapter_cfg_filepath = save_directory / "modality_adapter" / "config.json"
        self._save_part_config(self.modality_adapter_config, modality_adapter_cfg_filepath)

        decoder_cfg_filepath = save_directory / "config.json"
        self._save_part_config(self.decoder_config, decoder_cfg_filepath)

        return super().save_pretrained(save_directory, push_to_hub, **kwargs)
