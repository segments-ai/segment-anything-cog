# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cog
import segment_anything
import sys
import cv2
import numpy as np
import base64
import torch

sys.path.append("..")


class Predictor(cog.BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        base_sam_checkpoint = "sam_vit_b_01ec64.pth"  # 375 MB
        large_sam_checkpoint = "sam_vit_l_0b3195.pth"  # 1.25 GB
        huge_sam_checkpoint = "sam_vit_h_4b8939.pth"  # 2.56 GB

        base_sam = segment_anything.sam_model_registry["vit_b"](
            checkpoint=base_sam_checkpoint
        )
        large_sam = segment_anything.sam_model_registry["vit_l"](
            checkpoint=large_sam_checkpoint
        )
        huge_sam = segment_anything.sam_model_registry["vit_h"](
            checkpoint=huge_sam_checkpoint
        )

        base_sam.to(device=device)
        large_sam.to(device=device)
        huge_sam.to(device=device)

        self.base_predictor = segment_anything.SamPredictor(base_sam)
        self.large_predictor = segment_anything.SamPredictor(large_sam)
        self.huge_predictor = segment_anything.SamPredictor(huge_sam)

    def predict(
        self,
        image_path: cog.Path = cog.Input(description="Input image"),
        model_size: str = cog.Input(
            default="base",
            description="Model size",
            choices=["base", "large", "huge"],
        ),
    ) -> str:
        """Generate an image embedding."""

        image = cv2.imread(str(image_path))

        # Select model size
        if model_size == "base":
            self.predictor = self.base_predictor
        elif model_size == "large":
            self.predictor = self.large_predictor
        elif model_size == "huge":
            self.predictor = self.huge_predictor

        # Run model
        self.predictor.set_image(image)
        # Output shape is (1, 256, 64, 64)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()

        # Flatten the array to a 1D array
        flat_arr = image_embedding.flatten()
        # Convert the 1D array to bytes
        bytes_arr = flat_arr.astype(np.float32).tobytes()
        # Encode the bytes to base64
        base64_str = base64.b64encode(bytes_arr).decode("utf-8")

        return base64_str
