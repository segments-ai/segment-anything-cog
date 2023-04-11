# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cog
import segment_anything
import sys
import cv2
import imutils
import json

sys.path.append("..")


class Predictor(cog.BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        device = "cpu"  # or "cuda"

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
            description="The model size.",
            choices=["base", "large", "huge"],
        ),
        resize_width: int = cog.Input(
            default=1024,
            description="The width to resize the image to before running inference.",
        ),
    ) -> str:
        """Generate an image embedding."""
        # Resize image to requested width.
        image = cv2.imread(str(image_path))
        image = imutils.resize(image, width=resize_width)

        # Select model size.
        if model_size == "base":
            self.predictor = self.base_predictor
        elif model_size == "large":
            self.predictor = self.large_predictor
        elif model_size == "huge":
            self.predictor = self.huge_predictor

        # Run model.
        self.predictor.set_image(image)
        # Output shape is (1, 256, 64, 64).
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        output = json.dumps(image_embedding.tolist())

        return output
