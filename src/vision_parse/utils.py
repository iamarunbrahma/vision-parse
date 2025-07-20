import fitz
import numpy as np
import cv2
import base64
from typing import List, Tuple, Literal, ClassVar
from dataclasses import dataclass
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class ImageExtractionError(BaseException):
    """Custom exception for handling Image Extraction errors."""

    pass


@dataclass
class ImageData:
    image_url: str  # URL path for extracted images
    base64_encoded: str | None  # Base64 string if image_mode is base64, None otherwise
    summary: str | None = None  # Summary description of the image
    _lock: ClassVar[Lock] = Lock()  # Lock for thread safety

    @staticmethod
    def _prepare_image_for_detection(image: np.ndarray) -> np.ndarray:
        """Process image to highlight potential image regions."""
        try:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(grayscale, (5, 5), 0)
            binary = cv2.adaptiveThreshold(
                smooth,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        except Exception as e:
            raise ImageExtractionError(f"Image detection failed: {str(e)}")

    @staticmethod
    def _check_region_validity(
        region: np.ndarray, contour: np.ndarray, region_dims: tuple
    ) -> bool:
        """Determine if region contains a valid image based on statistical properties."""
        try:
            width, height = region_dims
            region_area = cv2.contourArea(contour) / (width * height)
            region_variance = cv2.Laplacian(
                cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()

            return (
                np.std(region) > 25
                and 20 < np.mean(region) < 235
                and region_area > 0.4
                and region_variance < 500
            )
        except Exception as e:
            raise ImageExtractionError(
                f"Image validity check for region failed: {str(e)}"
            )

    @classmethod
    def extract_images(
        cls,
        pix: fitz.Pixmap,
        image_mode: Literal["url", "base64", None],
        page_number: int,
        min_dimensions: tuple = (100, 100),
    ) -> List["ImageData"]:
        """Extract images with their coordinates and hash versions based on image_mode."""
        with cls._lock:
            try:
                min_width, min_height = min_dimensions
                page_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                page_image = cv2.cvtColor(
                    page_array, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR
                )
                processed_image = cls._prepare_image_for_detection(page_image)

                contours, _ = cv2.findContours(
                    processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                idx = 1
                extracted_images = []
                img_height, img_width = page_image.shape[:2]

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    if (
                        w < min_width
                        or h < min_height
                        or w > img_width * 0.95
                        or h > img_height * 0.95
                    ):
                        continue

                    # Get the exact region from the original page_image
                    region = page_image[y : y + h, x : x + w].copy()

                    if cls._check_region_validity(region, contour, (w, h)):
                        # Encode image based on mode
                        # Always encode as base64 for LLM processing regardless of image_mode
                        image_url = f"image_{page_number+1}_{idx}.png"
                        img_bytes = cv2.imencode(".png", region)[1].tobytes()
                        if not img_bytes:
                            continue
                            
                        # Create base64 encoding for the image
                        base64_str = base64.b64encode(img_bytes).decode('utf-8', errors='ignore')
                        
                        if image_mode == "url":
                            # Save image to file system
                            if not cv2.imwrite(image_url, region):
                                continue
                                
                            # Store only URL reference, not base64 data
                            extracted_images.append(
                                ImageData(
                                    image_url=image_url,
                                    base64_encoded=None,  # Don't store in final output
                                    # We'll add summary later
                                )
                            )
                        else:  # base64 mode or None
                            # For base64 mode, store the base64 data
                            base64_encoded = f"data:image/png;base64,{base64_str}"
                            
                            extracted_images.append(
                                ImageData(
                                    image_url=image_url,
                                    base64_encoded=base64_encoded,
                                    # We'll add summary later
                                )
                            )
                            
                        # Store the raw base64 string for summary generation
                        # (will be used outside this function)
                        extracted_images[-1]._raw_base64 = base64_str
                        
                        idx += 1

                return extracted_images
            except Exception as e:
                raise ImageExtractionError(f"Image processing failed: {str(e)}")


def get_device_config() -> Tuple[Literal["cuda", "mps", "cpu"], int]:
    """Get optimal number of worker processes based on device."""
    import platform
    import subprocess
    import os

    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if nvidia_smi.returncode == 0:
            return "cuda", min(len(nvidia_smi.stdout.strip().split("\n")) * 2, 8)
    except Exception:
        pass

    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "mps", 4
    return "cpu", max(2, (os.cpu_count() // 2))
