from typing import Dict, List
import torch


class VideoCollator:
    def __call__(self, video_examples: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        pixel_values = [ex["pixel_values"] for ex in video_examples]

        processor_fallback_count = sum(
            ex.get("processor_fallback_count", 0) for ex in video_examples
        )
        padded_frames = []
        max_num_frames = max(pv.size(0) for pv in pixel_values)
        for frame_tensor in pixel_values:
            if frame_tensor.shape[0] < max_num_frames:
                padding = torch.zeros(
                    (
                        max_num_frames - frame_tensor.size(0),
                        frame_tensor.size(1),
                        frame_tensor.size(2),
                        frame_tensor.size(3),
                    )
                )
                frame_tensor = torch.cat((frame_tensor, padding), dim=0)
            padded_frames.append(frame_tensor)

        out_dict = {
            "videos": torch.stack(padded_frames),
            "video_processor_fallback_count": processor_fallback_count,
        }

        return out_dict
