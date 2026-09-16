import cv2
import torch
import inspect
import numpy as np

from pathlib import Path
from PIL import Image, UnidentifiedImageError
from typing import List, Tuple, Optional, Union, Callable
from transformers import (
    ProcessorMixin,
    BaseImageProcessor,
    AutoProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)
from tokenizers import processors as tk_processors
import warnings
from .mm_utils import (
    expand2square,
    process_anyres_image,
    process_anyres_image_onevision,
    get_crop_size,
)


class GigaVisionProcessor(ProcessorMixin):
    attributes = []
    valid_kwargs = [
        "image_processor",
        "image_token_id",
        "num_image_tokens_per_crop",
        "image_aspect_ratio",
        "image_grid_pinpoints",
        "replace_unreadable_images",
    ]
    modality = "image"

    def __init__(
        self,
        image_processor: Optional[Union[str, BaseImageProcessor]] = None,
        image_token_id: int = 42042,
        num_image_tokens_per_crop: int = 256,
        image_aspect_ratio: str = "anyres",
        image_grid_pinpoints: Union[
            str, List[List[int]]
        ] = "[[448, 896], [896, 448], [896, 896], [1344, 448], [448, 1344]]",
        replace_unreadable_images: bool = True,
        tokenizer_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_processor = image_processor
        if isinstance(image_processor, str):
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.image_processor
            )
        # Tokenizer needed only in sglang inference and should be ignored in other cases
        if tokenizer_path is not None:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self._tokenizer = None

        self._prepare_tokenizer()
        self.image_token_id = image_token_id
        self.num_img_tokens = num_image_tokens_per_crop
        self.image_aspect_ratio = image_aspect_ratio

        self.image_grid_pinpoints = image_grid_pinpoints

        self.replace_unreadable_images = replace_unreadable_images

        self.auto_map = {
            "AutoProcessor": "gigavision_processor.GigaVisionProcessor",
        }

        self._processor_fallback_count = 0

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: object):
        raise AttributeError(
            "tokenizer is read-only; initialize it via tokenizer_path or from_pretrained()."
        )

    def _prepare_tokenizer(self):
        # for sglang inference
        # Some models (e.g. GigaVision, GigaOmni) include BOS/EOS in their prompt templates.
        # The tokenizer's post_processor adds them again, causing double BOS.
        if self._tokenizer is not None:
            self._tokenizer._tokenizer.post_processor = tk_processors.Sequence([])

    def preprocess(
        self,
        image_paths: Optional[List[Path]] = None,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pil_images: Optional[List[Image.Image]] = None,
        return_dict: Optional[bool] = False,
        mask_fill_value: int = 1,
    ) -> Tuple[torch.Tensor, ...]:
        assert input_ids is not None, "input_ids must be provided"
        assert int(image_paths is None) + int(pil_images is None) == 1, (
            "Only one of image_paths or pil_images can be provided"
        )

        self._processor_fallback_count = 0

        if pil_images is None:
            pil_images = [self._read_image(image_path) for image_path in image_paths]

        tensor_images, num_crops_for_image = self._preprocess_images(pil_images)

        if len(num_crops_for_image) > 0:
            input_ids, labels, attention_mask = self._format_image_tokens(
                input_ids,
                num_crops_for_image,
                labels,
                attention_mask,
                mask_fill_value=mask_fill_value,
            )

        if return_dict:
            dict_output = {
                "input_ids": input_ids,
                "labels": labels,
                "images": tensor_images,
                "processor_fallback_count": self._processor_fallback_count,
            }
            if attention_mask is not None:
                dict_output["attention_mask"] = attention_mask

            return dict_output

        if attention_mask is not None:
            return input_ids, labels, attention_mask, tensor_images

        return input_ids, labels, tensor_images

    def save_pretrained(self, save_directory: str, **kwargs):
        if self.image_processor is not None:
            self.image_processor.save_pretrained(
                save_directory,
                **kwargs,
            )

        output = super().save_pretrained(save_directory, **kwargs)
        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        try:
            # Tokenizer should be in the same directory as the preprocessor
            processor._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path
            )
            processor._prepare_tokenizer()
        except Exception as e:
            processor._tokenizer = None
            warnings.warn(
                f"Tokenizer not found, using None instead. It will be caused error in sglang inference. Error: {e}"
            )
        processor.image_processor = image_processor
        return processor

    @staticmethod
    def _extract_kwargs(func: Callable, **kwargs) -> dict:
        """
        Extract the kwargs that are valid for the given function.
        """
        return {
            k: v for k, v in kwargs.items() if k in inspect.signature(func).parameters
        }

    def _read_image(self, image_path: Path) -> Image.Image:
        try:
            try:
                with Image.open(image_path) as image:
                    img = image.convert("RGB")
            except (OSError, UnidentifiedImageError):
                print(
                    f"Image {image_path} can't be loaded with Image.open(), using opencv"
                )
                numpy_image = cv2.imread(str(image_path))
                numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(np.uint8(numpy_image))
            return img
        except Exception as exc:
            if not self.replace_unreadable_images:
                raise exc

            warnings.warn(f"Unreadable sample {image_path} black image fallback")

            self._processor_fallback_count += 1
            crop_size = get_crop_size(self.image_processor)

            return Image.new("RGB", (crop_size, crop_size))

    def _get_empty_image(self):
        crop_size = get_crop_size(self.image_processor)

        # return Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
        return torch.zeros((1, 3, crop_size, crop_size))

    def _preprocess_images(
        self, images: List[Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_tensors = []
        num_crops_for_image = []
        for image in images:
            if self.image_aspect_ratio == "pad":
                image = expand2square(
                    image, tuple(int(x * 255) for x in self.image_processor.image_mean)
                )
                image_tensor = self.image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"]
            elif self.image_aspect_ratio == "anyres":
                image_tensor = process_anyres_image(
                    image, self.image_processor, self.image_grid_pinpoints
                )
            elif self.image_aspect_ratio == "anyres_onevision":
                image_tensor = process_anyres_image_onevision(
                    image, self.image_processor, self.image_grid_pinpoints
                )
            else:
                raise ValueError(
                    f"image_aspect_ratio={self.image_aspect_ratio} is not supported!"
                )

            image_tensors.append(image_tensor)
            num_crops_for_image.append(image_tensor.shape[0])

        # assert len(image_tensors), "Image tensors is empty, it shouldn't be " #TODO: раскоментить после мерджа и старта использования omni dataloader, обязательная проверка на пустую картинку.
        # После перехода на использование omni dataset - удалить условие, так как при заходе в процессор мы гарантируем, что картинка будет и проверять ее наличие не нужно.
        if image_tensors:
            image_tensors = torch.vstack(image_tensors)
        else:
            image_tensors = self._get_empty_image()

        return image_tensors, num_crops_for_image

    def _format_image_tokens(
        self,
        input_ids: torch.Tensor,
        num_crops_for_image: List[int],
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_fill_value: int = 1,
    ) -> torch.Tensor:
        image_token_positions = (input_ids == self.image_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(image_token_positions) != len(num_crops_for_image):
            raise ValueError(
                f"Number of image tokens in input_ids is not the same as the number of images provided: "
                f"{len(image_token_positions)} vs {len(num_crops_for_image)}!"
            )

        new_token_sample = torch.tensor(
            [], dtype=input_ids.dtype, device=input_ids.device
        )

        if labels is not None:
            new_labels_sample = torch.tensor(
                [], dtype=labels.dtype, device=labels.device
            )

        if attention_mask is not None:
            new_attention_mask = torch.tensor(
                [], dtype=attention_mask.dtype, device=attention_mask.device
            )

        prev_position = 0
        for pos, parts_count in zip(image_token_positions, num_crops_for_image):
            num_new_tokens = parts_count * self.num_img_tokens
            new_tokens = torch.full(
                (num_new_tokens,),
                self.image_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            new_token_sample = torch.cat(
                (new_token_sample, input_ids[prev_position:pos], new_tokens)
            )

            if labels is not None:
                new_labels = torch.full(
                    (num_new_tokens,), -100, dtype=labels.dtype, device=labels.device
                )
                new_labels_sample = torch.cat(
                    (new_labels_sample, labels[prev_position:pos], new_labels)
                )

            if attention_mask is not None:
                new_mask = torch.full(
                    (num_new_tokens,),
                    mask_fill_value,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                new_attention_mask = torch.cat(
                    (new_attention_mask, attention_mask[prev_position:pos], new_mask)
                )
            prev_position = pos + 1

        new_token_sample = torch.cat((new_token_sample, input_ids[prev_position:]))
        if labels is not None:
            labels = torch.cat((new_labels_sample, labels[prev_position:]))

        if attention_mask is not None:
            attention_mask = torch.cat(
                (new_attention_mask, attention_mask[prev_position:])
            )

        return new_token_sample, labels, attention_mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(image_processor={self.image_processor}, image_token_id={self.image_token_id}, "
            f"num_img_tokens={self.num_img_tokens}, image_aspect_ratio={self.image_aspect_ratio})"
        )


GigaVisionProcessor.register_for_auto_class(AutoProcessor)
