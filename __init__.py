from .nodes import (
    InpaintCropNB2,
    InpaintStitchNB2,
    NB2AddAlpha,
    NB2Florence2RegionSelector,
    NB2SmartRegionMask,
    NanoBanana2MaskGen,
    SmartMaskCrop,
    SmartMaskStitch,
)

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "NanoBanana2MaskGen": NanoBanana2MaskGen,
    "NB2SmartRegionMask": NB2SmartRegionMask,
    "SmartMaskCrop": SmartMaskCrop,
    "SmartMaskStitch": SmartMaskStitch,
    "InpaintCropNB2": InpaintCropNB2,
    "InpaintStitchNB2": InpaintStitchNB2,
    "NB2AddAlpha": NB2AddAlpha,
    "NB2Florence2RegionSelector": NB2Florence2RegionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana2MaskGen": "NB2 Mask Generator",
    "NB2SmartRegionMask": "NB2 Smart Region",
    "SmartMaskCrop": "Smart Mask Crop",
    "SmartMaskStitch": "Smart Mask Stitch",
    "InpaintCropNB2": "NB2 Crop",
    "InpaintStitchNB2": "NB2 Stitch",
    "NB2AddAlpha": "NB2 Add Alpha",
    "NB2Florence2RegionSelector": "Florence-2 Smart Region Selector (FAL API)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
