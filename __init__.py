from .nodes import NanoBanana2MaskGen, InpaintCropNB2, InpaintStitchNB2, NB2AddAlpha

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "NanoBanana2MaskGen": NanoBanana2MaskGen,
    "InpaintCropNB2":     InpaintCropNB2,
    "InpaintStitchNB2":   InpaintStitchNB2,
    "NB2AddAlpha":        NB2AddAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana2MaskGen": "🎯 NB2 Mask Generator",
    "InpaintCropNB2":     "✂️ NB2 Crop",
    "InpaintStitchNB2":   "✂️ NB2 Stitch",
    "NB2AddAlpha":        "🔲 NB2 Add Alpha",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
