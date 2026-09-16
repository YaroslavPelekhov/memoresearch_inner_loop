import os
import av
import cv2
import copy
import torch
import random
import logging
import inspect
import torchvision
import numpy as np

from PIL import Image
from decord import VideoReader, cpu
from omegaconf import OmegaConf, ListConfig
from typing import List, Tuple, Optional, Union, Callable
from transformers import (
    ProcessorMixin,
    BaseImageProcessor,
    AutoProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)

from llmfoundry.data.processors.mm_utils import (
    expand2square,
    process_anyres_image,
    process_anyres_image_onevision,
    get_crop_size,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GigaVideoProcessor(ProcessorMixin):
    attributes = []
    valid_kwargs = [
        "video_processor",
        "video_token_id",
        "num_frame_tokens_per_crop",
        "frame_aspect_ratio",
        "frame_grid_pinpoints",
        "timestamp_placeholder",
        "tokenizer",
        "fps_strategy",
        "fps_schedule",
    ]
    modality = "video"
    fps_strategy_options = ["constant", "dynamic", "random"]
    fps_supported_backends = ["decord", "pyav"]
    _FPS_FALLBACK = 1.0

    def __init__(
        self,
        video_processor: Optional[Union[str, BaseImageProcessor]] = None,
        video_token_id: int = 128006,
        num_frame_tokens_per_crop: int = 256,
        frame_aspect_ratio: str = "pad",
        frame_grid_pinpoints: Union[str, List[List[int]]] = "[[448, 448]]",
        num_frames: int = 8,
        video_decode_backend: str = "decord",
        timestamp_placeholder: str = None,
        tokenizer=None,
        fps_strategy: Optional[str] = None,
        fps_schedule: Optional[List[dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.video_processor = video_processor
        if isinstance(video_processor, str):
            self.video_processor = AutoImageProcessor.from_pretrained(
                self.video_processor
            )

        self.video_token_id = video_token_id
        self.video_decode_backend = video_decode_backend
        self.num_frame_tokens = num_frame_tokens_per_crop
        self.frame_aspect_ratio = frame_aspect_ratio
        self.frame_grid_pinpoints = frame_grid_pinpoints

        self.num_frames = num_frames
        self.fps_strategy = fps_strategy
        self.fps_schedule = fps_schedule or []

        self.timestamp_placeholder = timestamp_placeholder
        self.tokenizer = tokenizer
        self.auto_map = {
            "AutoProcessor": "gigavideo_processor.GigaVideoProcessor",
        }
        self._processor_fallback_count = 0

        if self.fps_strategy:
            self._validate_fps_config()

    def _validate_fps_config(self):
        assert self.fps_strategy in self.fps_strategy_options, (
            f"Unsupported fps_strategy: {self.fps_strategy}, "
            f"expected one of {self.fps_strategy_options}"
        )

        assert self.video_decode_backend in self.fps_supported_backends, (
            f"Unsupported video_decode_backend: {self.video_decode_backend}, "
            f"expected one of {self.fps_supported_backends}"
        )

        if isinstance(self.fps_schedule, ListConfig):
            self.fps_schedule = OmegaConf.to_container(self.fps_schedule, resolve=True)

        if self.fps_schedule:
            self.fps_schedule.sort(key=lambda x: x.get("max_duration", float("inf")))
        else:
            logger.warning(
                f"{self.fps_strategy=} is set, but fps_schedule is empty, "
                f"fallback to {self._FPS_FALLBACK} fps",
            )

        for i, rule in enumerate(self.fps_schedule):
            assert "fps" in rule, (
                f"fps_schedule[{i}] missing required 'fps' key: {rule}"
            )

            fps_val = rule["fps"]
            if self.fps_strategy == "random":
                assert isinstance(fps_val, (list, tuple)), (
                    f"fps_schedule[{i}]: fps must be a non-empty list "
                    f"for 'random' strategy, got {fps_val}"
                )
            else:
                assert isinstance(fps_val, (int, float)), (
                    f"fps_schedule[{i}]: fps must be a number "
                    f"for '{self.fps_strategy}' strategy, got {fps_val}"
                )

    def _get_target_fps(self, duration: float) -> float:
        """
        Select target fps based on strategy and video duration
        Strategies:
            constant: fixed fps regardless of duration (first rule)
            dynamic: fps from first rule whose max_duration covers the video
            random: same as dynamic, but fps is randomly chosen from a list

        Fallback to _FPS_FALLBACK if no schedule or no matching rule
        """
        if not self.fps_schedule:
            return self._FPS_FALLBACK

        if self.fps_strategy == "constant":
            return float(self.fps_schedule[0]["fps"])

        for rule in self.fps_schedule:
            max_duration = rule.get("max_duration", float("inf"))
            if duration > max_duration:
                continue

            if self.fps_strategy == "random":
                return float(random.choice(rule["fps"]))
            else:
                return float(rule["fps"])

        return self._FPS_FALLBACK

    def _get_target_frames(self, total_frames: int, native_fps: float) -> int:
        """
        Calculate number of frames to sample based on target fps
        Result is clamped to [1, total_frames]
        """
        duration = total_frames / native_fps
        target_fps = self._get_target_fps(duration)
        target_frames = duration * target_fps

        return max(min(int(target_frames), total_frames), 1)

    def _get_sample_indices(self, total_frames: int, native_fps: float) -> np.ndarray:
        """
        Build array of frame indices to sample from the video

        With fps_strategy: number of frames depends on duration and target fps
        Without: uniformly samples num_frames

        In both cases, if result exceeds num_frames, it is uniformly subsampled
        """
        if self.fps_strategy:
            target_frames = self._get_target_frames(total_frames, native_fps)
        else:
            target_frames = self.num_frames

        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        if len(frame_indices) > self.num_frames:
            selection_indices = np.linspace(
                0, len(frame_indices) - 1, self.num_frames, dtype=int
            )
            frame_indices = frame_indices[selection_indices]
        return frame_indices

    def _load_and_transform_video(self, video_paths):
        if not video_paths:
            return [], []

        assert len(video_paths) == 1, (
            f"Only one video per sample is supported, but got {len(video_paths)}: {video_paths}"
        )
        video_path = str(video_paths[0])

        if self.video_decode_backend == "decord":
            return self._load_and_transform_video_decord(video_path)
        elif self.video_decode_backend == "opencv":
            return self._load_and_transform_video_opencv(video_path)
        elif self.video_decode_backend == "pyav":
            return self._load_and_transform_video_pyav(video_path)
        elif self.video_decode_backend == "torchvision":
            return self._load_and_transform_video_torchvision(video_path)
        else:
            raise ValueError(
                "`video_decode_backend` must be in (decord, opencv, pyav, torchvision)"
            )

    def _load_and_transform_video_decord(self, video_path):
        try:
            decord_vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(decord_vr)
            native_fps = decord_vr.get_avg_fps() or 30.0

            frame_indices = self._get_sample_indices(total_frames, native_fps)

            list_frames = decord_vr.get_batch(frame_indices).asnumpy()
            list_frame_images = [Image.fromarray(frame) for frame in list_frames]

            timestamps = [idx / native_fps for idx in frame_indices]

        except Exception as exc:
            logger.error(
                f"Got exception when loading video from `{video_path}` with decord. Will return empty frame.\nException: {exc}"
            )
            list_frame_images = [
                self._get_empty_frame() for _ in range(self.num_frames)
            ]
            timestamps = [float(i) for i in range(self.num_frames)]
            self._processor_fallback_count += 1

        return list_frame_images, timestamps

    def _load_and_transform_video_pyav(self, video_path):
        # Constant for converting seconds into microseconds
        # PyAV uses microseconds for global duration and seek operations
        AV_TIME_BASE = 1_000_000
        try:
            with av.open(video_path) as container:
                video_stream = container.streams.video[0]
                native_fps = float(
                    video_stream.average_rate or video_stream.guessed_rate or 30,
                )

                total_frames = video_stream.frames or 0
                if total_frames <= 0:
                    if (
                        video_stream.duration is not None
                        and video_stream.time_base is not None
                    ):
                        duration = float(video_stream.duration * video_stream.time_base)
                        total_frames = int(duration * native_fps)
                    elif container.duration is not None:
                        # microsec -> sec
                        duration = container.duration / AV_TIME_BASE
                        total_frames = int(duration * native_fps)
                    else:
                        raise RuntimeError(
                            f"Cannot determine video duration for {video_path}",
                        )

                frame_indices = self._get_sample_indices(total_frames, native_fps)

                list_frames = []
                for idx in frame_indices:
                    # sec -> microsec
                    target_microsec = int(idx / native_fps * AV_TIME_BASE)
                    container.seek(target_microsec)
                    for frame in container.decode(video=0):
                        list_frames.append(
                            Image.fromarray(
                                frame.to_ndarray(format="rgb24"),
                            ),
                        )
                        break

                assert len(list_frames) > 0, (
                    f"Number of frames decoded from `{video_path}` with PyAV is equal to 0"
                )

                target_len = len(frame_indices)
                if len(list_frames) < target_len:
                    logger.warning(
                        f"PyAV decoded {len(list_frames)}/{target_len} frames from `{video_path}`, padding last frame"
                    )
                    list_frames += [list_frames[-1]] * (target_len - len(list_frames))

                timestamps = [idx / native_fps for idx in frame_indices]

        except Exception as exc:
            logger.error(
                f"Got exception when loading video from `{video_path}` with PyAV. Will return empty frame.\nException: {exc}"
            )
            list_frames = [self._get_empty_frame() for _ in range(self.num_frames)]
            timestamps = [float(i) for i in range(self.num_frames)]
            self._processor_fallback_count += 1

        return list_frames, timestamps

    def _load_and_transform_video_torchvision(self, video_path):
        try:
            tensor_frames, _, info = torchvision.io.read_video(
                video_path, pts_unit="sec"
            )
            fps = info.get("video_fps", 30.0)
            frame_indices = np.linspace(
                0, len(tensor_frames) - 1, self.num_frames, dtype=int
            )
            list_frames = [tensor_frames[idx].numpy() for idx in frame_indices]
            list_frame_images = [Image.fromarray(item) for item in list_frames]

            timestamps = [idx / fps for idx in frame_indices]

            if 0 < len(list_frame_images) < self.num_frames:
                logger.warning(
                    f"Got less frames from video `{video_path}`: {len(list_frame_images)} instead of {self.num_frames}. Will repeat the last frame."
                )
                while len(list_frame_images) < self.num_frames:
                    list_frame_images.append(copy.deepcopy(list_frame_images[-1]))
                    timestamps.append(timestamps[-1])

        except Exception as exc:
            logger.error(
                f"Got exception when loading video from `{video_path}` with torchvision. Will return empty frame.\nException: {exc}"
            )
            list_frame_images = [
                self._get_empty_frame() for _ in range(self.num_frames)
            ]
            timestamps = [float(i) for i in range(self.num_frames)]

            self._processor_fallback_count += 1

        return list_frame_images, timestamps

    def _load_and_transform_video_opencv(self, video_path):
        try:
            cv2_vr = cv2.VideoCapture(video_path)
            duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cv2_vr.get(cv2.CAP_PROP_FPS) or 30.0
            frame_id_list = np.linspace(0, duration - 1, self.num_frames, dtype=int)

            list_frames = []
            timestamps = []
            for frame_idx in frame_id_list:
                cv2_vr.set(1, frame_idx)
                success, frame = cv2_vr.read()
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    list_frames.append(Image.fromarray(frame))
                else:
                    if 0 < len(list_frames) < self.num_frames:
                        frame = list_frames[-1]
                        list_frames.append(Image.fromarray(frame))
                    else:
                        continue
                timestamps.append(frame_idx / fps)

            if len(list_frames) == 0:
                logger.error(
                    f"Can't read any frames from `{video_path}` with opencv. Will return empty frame."
                )
                list_frames = [self._get_empty_frame()]
                self._processor_fallback_count += 1
            elif len(list_frames) < self.num_frames:
                logger.error(
                    f"Can't read {self.num_frames} frames from `{video_path}` with opencv. The last frame from video will "
                    f"be repeated {self.num_frames - len(list_frames)} times."
                )
                list_frames = list_frames + [list_frames[-1]] * (
                    self.num_frames - len(list_frames)
                )

            cv2_vr.release()
        except Exception as exc:
            logger.error(
                f"Got exception when loading video from `{video_path}` with opencv. Will return empty frame.\nException: {exc}"
            )
            list_frames = [self._get_empty_frame()]
            timestamps = [float(i) for i in range(self.num_frames)]

            self._processor_fallback_count += 1

        return list_frames, timestamps

    def _get_empty_frame(self):
        crop_size = get_crop_size(self.video_processor)
        return Image.new("RGB", (crop_size, crop_size), (255, 255, 255))

    def _get_empty_frame_tensor(self):
        crop_size = get_crop_size(self.video_processor)
        return torch.zeros((1, 3, crop_size, crop_size))

    def preprocess(
        self,
        video_paths: List[str],
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        self._processor_fallback_count = 0

        frames, timestamps = self._load_and_transform_video(video_paths)
        tensor_frames, num_crops_for_frame = self._preprocess_frames(frames)

        if len(num_crops_for_frame) > 0:
            input_ids, labels, attention_mask = self._format_video_tokens(
                input_ids, num_crops_for_frame, labels, attention_mask, timestamps
            )

        if return_dict:
            dict_output = {
                "input_ids": input_ids,
                "labels": labels,
                "tensor_frames": tensor_frames,
                "processor_fallback_count": self._processor_fallback_count,
            }
            if attention_mask is not None:
                dict_output["attention_mask"] = attention_mask

            return dict_output

        if attention_mask is not None:
            return input_ids, labels, attention_mask, tensor_frames

        return input_ids, labels, tensor_frames

    def save_pretrained(self, save_directory: str, **kwargs):
        if self.video_processor is not None:
            self.video_processor.save_pretrained(
                save_directory,
                **kwargs,
            )

        if self.tokenizer is not None:
            tok_dir = os.path.join(save_directory, "tokenizer")
            self.tokenizer.save_pretrained(tok_dir)

        output = super().save_pretrained(save_directory, **kwargs)
        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        video_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        processor.video_processor = video_processor

        tok_dir = os.path.join(pretrained_model_name_or_path, "tokenizer")
        if os.path.isdir(tok_dir):
            processor.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        else:
            processor.tokenizer = None
        return processor

    @staticmethod
    def _extract_kwargs(func: Callable, **kwargs) -> dict:
        """
        Extract the kwargs that are valid for the given function.
        """
        return {
            k: v for k, v in kwargs.items() if k in inspect.signature(func).parameters
        }

    def _preprocess_frames(
        self, frames: List[Image.Image]
    ) -> Tuple[torch.Tensor, List[int]]:
        frame_tensors = []
        num_crops_for_frame = []
        for frame in frames:
            if self.frame_aspect_ratio == "pad":
                # self.frame_grid_pinpoints не используется в PAD
                frame = expand2square(
                    frame, tuple(int(x * 255) for x in self.video_processor.image_mean)
                )
                frame_tensor = self.video_processor.preprocess(
                    frame, return_tensors="pt"
                )["pixel_values"]
            elif self.frame_aspect_ratio == "anyres":
                frame_tensor = process_anyres_image(
                    frame, self.video_processor, self.frame_grid_pinpoints
                )
            elif self.frame_aspect_ratio == "anyres_onevision":
                frame_tensor = process_anyres_image_onevision(
                    frame, self.video_processor, self.frame_grid_pinpoints
                )
            else:
                raise ValueError(
                    f"frame_aspect_ratio={self.frame_aspect_ratio} is not supported!"
                )

            frame_tensors.append(frame_tensor)
            num_crops_for_frame.append(frame_tensor.shape[0])

        if num_crops_for_frame:
            num_crops_for_frame = [num_crops_for_frame[0] * len(frames)]

        if frame_tensors:
            frame_tensors = torch.vstack(frame_tensors)
        else:
            frame_tensors = self._get_empty_frame_tensor()

        return frame_tensors, num_crops_for_frame

    def _inject_ts_video(
        self, video_token_ids: torch.Tensor, timestamps: List
    ) -> torch.Tensor:
        """
        Insert timestamps in Video tokens format:
        'Time n. secs/Frame n:' + <VIDEO_TOKEN> * num_frames
        """

        assert self.tokenizer is not None, "Tokenizer is not set in GigaVideoProcessor"

        num_frames_actual = len(timestamps)
        total_tokens = video_token_ids.shape[0]
        tokens_per_frame = total_tokens // num_frames_actual

        out: list[int] = []
        for idx, t in enumerate(timestamps):
            start = idx * tokens_per_frame
            end = (idx + 1) * tokens_per_frame
            video_tokens_slice = video_token_ids[start:end].tolist()

            if self.timestamp_placeholder.lower() == "frame":
                prefix = f"Frame {idx + 1}:"
            elif self.timestamp_placeholder.lower() == "time":
                prefix = f"Time {t:.1f}s:"
            else:
                raise ValueError(
                    f"Unsupported timestamp_placeholder: {self.timestamp_placeholder}"
                )

            prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            out.extend(prefix_ids + video_tokens_slice)

        return torch.tensor(
            out, device=video_token_ids.device, dtype=video_token_ids.dtype
        )

    def _format_video_tokens(
        self,
        input_ids: torch.Tensor,
        num_crops_for_frame: List[int],
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        timestamps: Optional[List[int]] = None,
    ) -> torch.Tensor:
        video_token_positions = (input_ids == self.video_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(video_token_positions) != len(num_crops_for_frame):
            raise ValueError(
                f"Number of video tokens in input_ids is not the same as the number of frames provided: "
                f"{len(video_token_positions)} vs {len(num_crops_for_frame)}!"
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
        for pos, parts_count in zip(video_token_positions, num_crops_for_frame):
            num_new_tokens = parts_count * self.num_frame_tokens
            new_tokens = torch.full(
                (num_new_tokens,),
                self.video_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            if self.timestamp_placeholder is not None:
                assert timestamps is not None, (
                    "timestamp_placeholder is not None, but timestamps are None!"
                )
                new_tokens = self._inject_ts_video(new_tokens, timestamps)
                num_new_tokens = new_tokens.shape[0]

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
                    1,
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
            f"{self.__class__.__name__}(video_processor={self.video_processor}, video_token_id={self.video_token_id}, "
            f"num_frame_tokens={self.num_frame_tokens}, frame_aspect_ratio={self.frame_aspect_ratio})"
        )


GigaVideoProcessor.register_for_auto_class(AutoProcessor)
