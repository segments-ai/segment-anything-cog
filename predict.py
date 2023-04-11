# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from segment_anything import sam_model_registry, SamPredictor
import sys
import json

sys.path.append("..")
import cv2
import imutils
import numpy as np


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        device = "cuda"
        model_type = "default"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        resize_width: int = Input(
            default=1024,
            description="The width to resize the image to before running inference.",
        ),
    ) -> Path:
        """Generate an image embedding."""

        # Resize image to requested width.
        image = cv2.imread(str(image))
        image = imutils.resize(image, width=resize_width)

        # Run model.
        self.predictor.set_image(image)
        # Output shape is (1, 256, 64, 64).
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()

        # Save output to disk as json.
        output = Path("image_embedding.json")
        with open("image_embedding.json", "w") as f:
            json.dump(image_embedding.tolist(), f)

        return output
