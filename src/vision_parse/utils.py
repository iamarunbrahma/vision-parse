import pypdfium2 as pdfium
import numpy as np
import cv2
import base64
from typing import List, Tuple, Literal, ClassVar, Optional
from dataclasses import dataclass
from threading import Lock
import logging

_logger = logging.getLogger(__name__)


class ImageExtractionError(Exception):
    pass


@dataclass
class ImageData:
    image_url: str
    base64_encoded: Optional[str] = None
    
    _lock: ClassVar[Lock] = Lock()

    @staticmethod
    def _prepare_image_for_detection(image: np.ndarray) -> np.ndarray:
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
        region: np.ndarray, contour: np.ndarray, region_dims: Tuple[int, int]
    ) -> bool:
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
            raise ImageExtractionError(f"Region validity check failed: {str(e)}")

    @classmethod
    def extract_images(
        cls,
        bitmap: pdfium.PdfBitmap,
        image_mode: Literal["url", "base64", None],
        page_number: int,
        min_dimensions: Tuple[int, int] = (100, 100),
    ) -> List["ImageData"]:
        with cls._lock:
            try:
                min_width, min_height = min_dimensions
                # Convert pypdfium2 bitmap to numpy array
                page_array = bitmap.to_numpy()
                
                # pypdfium2 uses BGR by default, but we need to handle different formats
                if page_array.shape[2] == 4:  # BGRA format
                    page_image = cv2.cvtColor(page_array, cv2.COLOR_BGRA2BGR)
                elif page_array.shape[2] == 3:  # BGR format (default for pypdfium2)
                    page_image = page_array
                else:
                    raise ImageExtractionError(f"Unsupported image format with {page_array.shape[2]} channels")
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
                        if image_mode == "url":
                            image_url = f"image_{page_number+1}_{idx}.png"

                            if not cv2.imwrite(image_url, region):
                                continue

                            idx += 1

                            extracted_images.append(
                                ImageData(
                                    image_url=image_url,
                                    base64_encoded=None,
                                )
                            )
                        else:  # base64 mode
                            image_url = f"image_{page_number+1}_{idx}.png"

                            img_bytes = cv2.imencode(".png", region)[1].tobytes()
                            if not img_bytes:
                                continue
                            base64_encoded = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8', errors='ignore')}"
                            idx += 1

                            extracted_images.append(
                                ImageData(
                                    image_url=image_url,
                                    base64_encoded=base64_encoded,
                                )
                            )

                return extracted_images
            except Exception as e:
                raise ImageExtractionError(f"Image processing failed: {str(e)}")


def get_device_config() -> Tuple[Literal["cuda", "mps", "cpu"], int]:
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
    return "cpu", max(2, (os.cpu_count() or 1) // 2)
