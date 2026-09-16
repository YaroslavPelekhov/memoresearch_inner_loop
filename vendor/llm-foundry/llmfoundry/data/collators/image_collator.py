from typing import Dict, List
import torch


class ImageCollator:
    def __call__(self, image_examples: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        pixel_values = [ex["pixel_values"] for ex in image_examples]
        processor_fallback_count = sum(
            ex.get("processor_fallback_count", 0) for ex in image_examples
        )

        padded_images = []
        max_num_images = max(pv.size(0) for pv in pixel_values)
        for image_tensor in pixel_values:
            if image_tensor.shape[0] < max_num_images:
                padding = torch.zeros(
                    (
                        max_num_images - image_tensor.size(0),
                        image_tensor.size(1),
                        image_tensor.size(2),
                        image_tensor.size(3),
                    )
                )
                image_tensor = torch.cat((image_tensor, padding), dim=0)
            padded_images.append(image_tensor)

        out_dict = {
            "images": torch.stack(padded_images),
            "vision_processor_fallback_count": processor_fallback_count,
        }

        return out_dict
