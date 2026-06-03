import importlib
import subprocess
import sys

_REQUIRED = [
    ("fal_client", "fal-client>=0.4.1"),
    ("PIL", "pillow>=10.0.0"),
    ("requests", "requests>=2.32.0"),
    ("numpy", "numpy>=1.24.0"),
    ("scipy", "scipy>=1.10.0"),
]

for _module, _package in _REQUIRED:
    if importlib.util.find_spec(_module) is None:
        print(f"[NB2] Installing missing dependency: {_package}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", _package])
        except Exception as _e:
            print(f"[NB2] WARNING: could not auto-install {_package}: {_e}")
            print(f"[NB2] Install manually: python -m pip install {_package}")

from .nodes import (
    InpaintCropNB2,
    InpaintStitchNB2,
    NB2AddAlpha,
    NB2Florence2RegionSelector,
    NB2NanoBanana2Edit,
    NB2OpenAIImageEdit,
    NB2SAM3ImageSegmenter,
    NB2SAM3SmartRegionSelector,
    NB2Seedream45Edit,
    NB2SmartRegionMask,
    NanoBanana2MaskGen,
    SmartMaskMultiStitch,
    SmartMaskCrop,
    SmartMaskStitch,
    SmartObjectIsolateCrop,
)

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "NanoBanana2MaskGen": NanoBanana2MaskGen,
    "NB2SmartRegionMask": NB2SmartRegionMask,
    "SmartMaskCrop": SmartMaskCrop,
    "SmartObjectIsolateCrop": SmartObjectIsolateCrop,
    "SmartMaskStitch": SmartMaskStitch,
    "SmartMaskMultiStitch": SmartMaskMultiStitch,
    "InpaintCropNB2": InpaintCropNB2,
    "InpaintStitchNB2": InpaintStitchNB2,
    "NB2AddAlpha": NB2AddAlpha,
    "NB2Florence2RegionSelector": NB2Florence2RegionSelector,
    "NB2SAM3ImageSegmenter": NB2SAM3ImageSegmenter,
    "NB2SAM3SmartRegionSelector": NB2SAM3SmartRegionSelector,
    "NB2NanoBanana2Edit": NB2NanoBanana2Edit,
    "NB2OpenAIImageEdit": NB2OpenAIImageEdit,
    "NB2Seedream45Edit": NB2Seedream45Edit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana2MaskGen": "NB2 Mask Generator",
    "NB2SmartRegionMask": "NB2 Smart Region",
    "SmartMaskCrop": "Smart Mask Crop",
    "SmartObjectIsolateCrop": "Smart Object Isolate Crop",
    "SmartMaskStitch": "Smart Mask Stitch",
    "SmartMaskMultiStitch": "Smart Mask Multi Stitch",
    "InpaintCropNB2": "NB2 Crop",
    "InpaintStitchNB2": "NB2 Stitch",
    "NB2AddAlpha": "NB2 Add Alpha",
    "NB2Florence2RegionSelector": "Florence-2 Smart Region Selector (FAL API)",
    "NB2SAM3ImageSegmenter": "SAM 3 Image Segmenter (FAL API)",
    "NB2SAM3SmartRegionSelector": "SAM 3 Smart Region Selector (FAL API)",
    "NB2NanoBanana2Edit": "Nano Banana 2 Edit (FAL API)",
    "NB2OpenAIImageEdit": "OpenAI GPT Image Edit",
    "NB2Seedream45Edit": "Seedream 4.5 Edit (FAL API)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
