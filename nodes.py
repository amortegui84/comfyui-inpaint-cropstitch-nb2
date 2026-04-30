"""
ComfyUI-Inpaint-CropStitch-NB2
================================
Fork and adaptation of ComfyUI-Inpaint-CropAndStitch by lquesada
  Original: https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch
  License : Apache 2.0 (see LICENSE)

Adaptations by amortegui84 (https://github.com/amortegui84):
  - NanoBanana2MaskGen: helper node that generates a positioned rectangular
    mask whose aspect ratio matches the exact output resolutions of the
    Nano Banana 2 generation model (1K / 2K / 4K in 16:9, 9:16 and 1:1).
  - InpaintCropNB2: simplified crop node that enforces a Nano Banana 2
    target resolution, removing parameters that are unnecessary for that
    workflow (preresize, outpainting extension, debug outputs).
  - InpaintStitchNB2: stitch node extended with:
      * Percentage-based edge feathering so composited borders are smooth.
      * Alpha-channel support: if the inpainted / generated image has four
        channels (RGBA) the alpha is used as an additional blend mask.

Intended workflow
-----------------
  [original image]
       |
  NanoBanana2MaskGen  <-- pick aspect_ratio, resolution, center X/Y
       | mask
  InpaintCropNB2      <-- set same aspect_ratio + resolution; crops and
       | stitcher  cropped_image    scales to exact NB2 resolution
  [Nano Banana 2]     <-- generation (no mask needed, RGB output)
       | generated_image
  InpaintStitchNB2    <-- feathered composite back onto original
       | result image
"""

import comfy.utils
import comfy.model_management
import io
import json
import logging
import math
import nodes
import numpy as np
import os
import re
import requests
import time
import uuid
import torch
import torch.nn.functional as TF
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, grey_dilation, binary_closing, binary_fill_holes
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import folder_paths as _folder_paths
    _HAS_FOLDER_PATHS = True
except ImportError:
    _folder_paths = None
    _HAS_FOLDER_PATHS = False


# ---------------------------------------------------------------------------
# Nano Banana 2 resolution table
# ---------------------------------------------------------------------------

NB2_RESOLUTIONS = {
    "16:9": {
        "1K": (1376, 768),
        "2K": (2752, 1536),
        "4K": (5504, 3072),
    },
    "9:16": {
        "1K": (768, 1376),
        "2K": (1536, 2752),
        "4K": (3072, 5504),
    },
    "1:1": {
        "1K": (1024, 1024),
        "2K": (2048, 2048),
        "4K": (4096, 4096),
    },
}

REGION_ASPECT_RATIO_HINTS = {
    "glasses": "16:9",
    "face": "1:1",
    "upper_body": "1:1",
    "lower_body": "1:1",
    "full_body": "9:16",
}

REGION_EDIT_HINTS = {
    "glasses": {
        "aspect_ratio": "16:9",
        "edit_size": "2752x1536",
        "mask_expand_percent": 18.0,
        "mask_feather_percent": 10.0,
        "context_expand": 1.18,
    },
    "face": {
        "aspect_ratio": "1:1",
        "edit_size": "2048x2048",
        "mask_expand_percent": 10.0,
        "mask_feather_percent": 6.0,
        "context_expand": 1.12,
    },
    "upper_body": {
        "aspect_ratio": "1:1",
        "edit_size": "2048x2048",
        "mask_expand_percent": 8.0,
        "mask_feather_percent": 5.0,
        "context_expand": 1.10,
    },
    "lower_body": {
        "aspect_ratio": "1:1",
        "edit_size": "2048x2048",
        "mask_expand_percent": 8.0,
        "mask_feather_percent": 5.0,
        "context_expand": 1.10,
    },
    "full_body": {
        "aspect_ratio": "9:16",
        "edit_size": "1536x2752",
        "mask_expand_percent": 6.0,
        "mask_feather_percent": 4.0,
        "context_expand": 1.08,
    },
}

EDIT_SIZE_BY_ASPECT_RATIO = {
    "16:9": "2752x1536",
    "9:16": "1536x2752",
    "1:1": "2048x2048",
}


def _coerce_text_value(value):
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _summarize_remote_error(error):
    text = str(error).strip()
    lower = text.lower()
    if "<html" in lower and "internal server error" in lower:
        return (
            "Remote server returned 500 Internal Server Error. "
            "This is a FAL/OpenAI-side failure, often transient or caused by a "
            "request the gateway could not process."
        )
    return text


def _safe_json_loads(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    value = value.strip()
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _aspect_ratio_from_bbox_dims(width, height):
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    ratio = width / height
    if ratio >= 1.2:
        return "16:9"
    if ratio <= (1.0 / 1.2):
        return "9:16"
    return "1:1"


def _recommend_aspect_ratio_for_region(region_type, bbox=None):
    region_type = _coerce_text_value(region_type)
    if region_type in REGION_ASPECT_RATIO_HINTS:
        return REGION_ASPECT_RATIO_HINTS[region_type]
    if bbox:
        x1, y1, x2, y2 = bbox
        return _aspect_ratio_from_bbox_dims(max(1, x2 - x1), max(1, y2 - y1))
    return "1:1"


def _recommend_edit_size_for_aspect_ratio(aspect_ratio):
    return EDIT_SIZE_BY_ASPECT_RATIO.get(aspect_ratio, "1024x1024")


def _get_region_edit_hints(region_type):
    region_type = _coerce_text_value(region_type)
    hints = REGION_EDIT_HINTS.get(region_type)
    if hints:
        return dict(hints)
    aspect_ratio = _recommend_aspect_ratio_for_region(region_type)
    return {
        "aspect_ratio": aspect_ratio,
        "edit_size": _recommend_edit_size_for_aspect_ratio(aspect_ratio),
        "mask_expand_percent": 8.0,
        "mask_feather_percent": 5.0,
        "context_expand": 1.10,
    }


def _extract_region_context(region_info):
    data = _safe_json_loads(region_info)
    bbox_obj = data.get("bbox") if isinstance(data, dict) else None
    bbox = None
    if isinstance(bbox_obj, dict):
        try:
            bbox = (
                int(bbox_obj["x1"]),
                int(bbox_obj["y1"]),
                int(bbox_obj["x2"]),
                int(bbox_obj["y2"]),
            )
        except Exception:
            bbox = None
    elif all(key in data for key in ("x1", "y1", "x2", "y2")):
        try:
            bbox = (int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"]))
        except Exception:
            bbox = None

    region_type = _coerce_text_value(data.get("region_type")) if isinstance(data, dict) else ""
    recommended_aspect_ratio = _coerce_text_value(data.get("recommended_aspect_ratio")) if isinstance(data, dict) else ""
    recommended_edit_size = _coerce_text_value(data.get("recommended_edit_size")) if isinstance(data, dict) else ""

    return {
        "region_type": region_type,
        "bbox": bbox,
        "recommended_aspect_ratio": recommended_aspect_ratio,
        "recommended_edit_size": recommended_edit_size,
        "recommended_mask_expand_percent": float(data.get("recommended_mask_expand_percent", 0.0) or 0.0) if isinstance(data, dict) else 0.0,
        "recommended_mask_feather_percent": float(data.get("recommended_mask_feather_percent", 0.0) or 0.0) if isinstance(data, dict) else 0.0,
        "recommended_context_expand": float(data.get("recommended_context_expand", 0.0) or 0.0) if isinstance(data, dict) else 0.0,
    }



# ---------------------------------------------------------------------------
# Edge-feather helper
# ---------------------------------------------------------------------------

def make_smoothstep_feather(h: int, w: int, feather_h_px: int, feather_w_px: int,
                            device: torch.device) -> torch.Tensor:
    """Return a [H, W] float32 mask: 0 at each edge, 1 in the centre.

    Uses a smoothstep curve so the transition looks natural.
    feather_h_px / feather_w_px control the ramp width in pixels.
    """
    ramp_y = torch.ones(h, device=device, dtype=torch.float32)
    if feather_h_px > 0:
        t = torch.linspace(0.0, 1.0, feather_h_px, device=device)
        smooth = t * t * (3.0 - 2.0 * t)          # smoothstep
        ramp_y[:feather_h_px] = smooth
        ramp_y[h - feather_h_px:] = smooth.flip(0)

    ramp_x = torch.ones(w, device=device, dtype=torch.float32)
    if feather_w_px > 0:
        t = torch.linspace(0.0, 1.0, feather_w_px, device=device)
        smooth = t * t * (3.0 - 2.0 * t)
        ramp_x[:feather_w_px] = smooth
        ramp_x[w - feather_w_px:] = smooth.flip(0)

    return ramp_y.unsqueeze(1) * ramp_x.unsqueeze(0)   # [H, W]


# ---------------------------------------------------------------------------
# Processor base class  (copied verbatim from lquesada's original)
# ---------------------------------------------------------------------------

class ProcessorLogic(ABC):
    @abstractmethod
    def rescale_i(self, samples, width, height, algorithm: str):
        pass

    @abstractmethod
    def rescale_m(self, samples, width, height, algorithm: str):
        pass

    @abstractmethod
    def fillholes_iterative_hipass_fill_m(self, samples):
        pass

    @abstractmethod
    def hipassfilter_m(self, samples, threshold):
        pass

    @abstractmethod
    def expand_m(self, samples, pixels):
        pass

    @abstractmethod
    def invert_m(self, samples):
        pass

    @abstractmethod
    def blur_m(self, samples, pixels):
        pass

    @abstractmethod
    def debug_context_location_in_image(self, image, x, y, w, h):
        pass

    @abstractmethod
    def pad_to_multiple(self, value, multiple):
        pass

    @abstractmethod
    def preresize_imm(self, image, mask, optional_context_mask,
                      downscale_algorithm, upscale_algorithm,
                      preresize_mode, preresize_min_width, preresize_min_height,
                      preresize_max_width, preresize_max_height):
        pass

    @abstractmethod
    def extend_imm(self, image, mask, optional_context_mask,
                   extend_up_factor, extend_down_factor,
                   extend_left_factor, extend_right_factor):
        pass

    @abstractmethod
    def batched_findcontextarea_m(self, mask):
        pass

    def findcontextarea_m(self, mask):
        _, x, y, w, h = self.batched_findcontextarea_m(mask)
        context = mask[:, y[0]:y[0]+h[0], x[0]:x[0]+w[0]]
        return context, x[0].item(), y[0].item(), w[0].item(), h[0].item()

    @abstractmethod
    def batched_growcontextarea_m(self, mask, x, y, w, h, extend_factor):
        pass

    def growcontextarea_m(self, context, mask, x, y, w, h, extend_factor):
        _, nx, ny, nw, nh = self.batched_growcontextarea_m(
            mask,
            torch.tensor([x], device=mask.device),
            torch.tensor([y], device=mask.device),
            torch.tensor([w], device=mask.device),
            torch.tensor([h], device=mask.device),
            extend_factor)
        nx, ny, nw, nh = nx[0].item(), ny[0].item(), nw[0].item(), nh[0].item()
        ctx = mask[:, ny:ny+nh, nx:nx+nw]
        return ctx, nx, ny, nw, nh

    @abstractmethod
    def batched_combinecontextmask_m(self, mask, x, y, w, h, optional_context_mask):
        pass

    def combinecontextmask_m(self, context, mask, x, y, w, h, optional_context_mask):
        _, nx, ny, nw, nh = self.batched_combinecontextmask_m(
            mask,
            torch.tensor([x], device=mask.device),
            torch.tensor([y], device=mask.device),
            torch.tensor([w], device=mask.device),
            torch.tensor([h], device=mask.device),
            optional_context_mask)
        nx, ny, nw, nh = nx[0].item(), ny[0].item(), nw[0].item(), nh[0].item()
        ctx = mask[:, ny:ny+nh, nx:nx+nw]
        return ctx, nx, ny, nw, nh

    @abstractmethod
    def crop_magic_im(self, image, mask, x, y, w, h,
                      target_w, target_h, padding,
                      downscale_algorithm, upscale_algorithm,
                      resize_output=True):
        pass

    @abstractmethod
    def stitch_magic_im(self, canvas_image, inpainted_image, mask,
                        ctc_x, ctc_y, ctc_w, ctc_h,
                        cto_x, cto_y, cto_w, cto_h,
                        downscale_algorithm, upscale_algorithm):
        pass


# ---------------------------------------------------------------------------
# CPU implementation  (copied verbatim from lquesada's original)
# ---------------------------------------------------------------------------

class CPUProcessorLogic(ProcessorLogic):
    def rescale_i(self, samples, width, height, algorithm: str):
        samples = samples.movedim(-1, 1)
        algorithm_enum = getattr(Image, algorithm.upper())
        results = []
        for i in range(samples.shape[0]):
            pil = F.to_pil_image(samples[i].cpu()).resize((width, height), algorithm_enum)
            results.append(F.to_tensor(pil))
        samples = torch.stack(results, dim=0).movedim(1, -1)
        return samples

    def rescale_m(self, samples, width, height, algorithm: str):
        algorithm_enum = getattr(Image, algorithm.upper())
        results = []
        for i in range(samples.shape[0]):
            pil = F.to_pil_image(samples[i].cpu()).resize((width, height), algorithm_enum)
            results.append(F.to_tensor(pil).squeeze(0))
        return torch.stack(results, dim=0)

    def fillholes_iterative_hipass_fill_m(self, samples):
        thresholds = [1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7,
                      0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        results = []
        for i in range(samples.shape[0]):
            mask_np = samples[i].cpu().numpy()
            for threshold in thresholds:
                thresholded_mask = mask_np >= threshold
                closed_mask = binary_closing(thresholded_mask,
                                             structure=np.ones((3, 3)),
                                             border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                mask_np = np.maximum(mask_np,
                                     np.where(filled_mask != 0, threshold, 0))
            results.append(torch.from_numpy(mask_np.astype(np.float32)))
        return torch.stack(results, dim=0)

    def hipassfilter_m(self, samples, threshold):
        filtered = samples.clone()
        filtered[filtered < threshold] = 0
        return filtered

    def expand_m(self, mask, pixels):
        sigma = pixels / 4
        kernel_size = math.ceil(sigma * 1.5 + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        results = []
        for i in range(mask.shape[0]):
            mask_np = mask[i].cpu().numpy()
            dilated = grey_dilation(mask_np, footprint=kernel, mode='reflect')
            results.append(torch.from_numpy(dilated.astype(np.float32)).clamp(0.0, 1.0))
        return torch.stack(results, dim=0)

    def invert_m(self, samples):
        return 1.0 - samples.clone()

    def blur_m(self, samples, pixels):
        sigma = pixels / 4
        results = []
        for i in range(samples.shape[0]):
            mask_np = samples[i].cpu().numpy()
            blurred = gaussian_filter(mask_np, sigma=sigma, mode='reflect')
            results.append(torch.from_numpy(blurred).float().clamp(0.0, 1.0))
        return torch.stack(results, dim=0)

    def debug_context_location_in_image(self, image, x, y, w, h):
        debug = image.clone()
        debug[:, y:y+h, x:x+w, :] = 1.0 - debug[:, y:y+h, x:x+w, :]
        return debug

    def pad_to_multiple(self, value, multiple):
        return int(math.ceil(value / multiple) * multiple)

    def preresize_imm(self, image, mask, optional_context_mask,
                      downscale_algorithm, upscale_algorithm,
                      preresize_mode, preresize_min_width, preresize_min_height,
                      preresize_max_width, preresize_max_height):
        current_width, current_height = image.shape[2], image.shape[1]
        if preresize_mode == "ensure minimum resolution":
            if current_width >= preresize_min_width and current_height >= preresize_min_height:
                return image, mask, optional_context_mask
            sf = max(preresize_min_width / current_width,
                     preresize_min_height / current_height)
            tw = math.ceil(current_width * sf)
            th = math.ceil(current_height * sf)
            image = self.rescale_i(image, tw, th, upscale_algorithm)
            mask = self.rescale_m(mask, tw, th, 'bilinear')
            optional_context_mask = self.rescale_m(optional_context_mask, tw, th, 'bilinear')
        elif preresize_mode == "ensure minimum and maximum resolution":
            if (preresize_min_width <= current_width <= preresize_max_width and
                    preresize_min_height <= current_height <= preresize_max_height):
                return image, mask, optional_context_mask
            sf_min = max(preresize_min_width / current_width,
                         preresize_min_height / current_height)
            sf_max = min(preresize_max_width / current_width,
                         preresize_max_height / current_height)
            assert not (sf_min > 1 and sf_max < 1), \
                "Cannot meet both min and max resolution with aspect-ratio preservation."
            if sf_min > 1:
                sf, algo = sf_min, upscale_algorithm
                tw = math.ceil(current_width * sf)
                th = math.ceil(current_height * sf)
            else:
                sf, algo = sf_max, downscale_algorithm
                tw = int(current_width * sf)
                th = int(current_height * sf)
            image = self.rescale_i(image, tw, th, algo)
            mask = self.rescale_m(mask, tw, th, 'nearest')
            optional_context_mask = self.rescale_m(optional_context_mask, tw, th, 'nearest')
        elif preresize_mode == "ensure maximum resolution":
            if current_width <= preresize_max_width and current_height <= preresize_max_height:
                return image, mask, optional_context_mask
            sf = min(preresize_max_width / current_width,
                     preresize_max_height / current_height)
            tw = int(current_width * sf)
            th = int(current_height * sf)
            image = self.rescale_i(image, tw, th, downscale_algorithm)
            mask = self.rescale_m(mask, tw, th, 'nearest')
            optional_context_mask = self.rescale_m(optional_context_mask, tw, th, 'nearest')
        return image, mask, optional_context_mask

    def extend_imm(self, image, mask, optional_context_mask,
                   extend_up_factor, extend_down_factor,
                   extend_left_factor, extend_right_factor):
        B, H, W, C = image.shape
        new_H = int(H * (extend_up_factor + extend_down_factor - 1.0))
        new_W = int(W * (extend_left_factor + extend_right_factor - 1.0))
        assert new_H >= 0 and new_W >= 0, "Extend factors are too small."
        expanded_image = torch.zeros(B, new_H, new_W, C, device=image.device)
        expanded_mask = torch.ones(B, new_H, new_W, device=mask.device)
        expanded_opt = torch.zeros(B, new_H, new_W, device=optional_context_mask.device)
        up_pad = int(H * (extend_up_factor - 1.0))
        dn_pad = new_H - H - up_pad
        lf_pad = int(W * (extend_left_factor - 1.0))
        rt_pad = new_W - W - lf_pad
        st_u = max(0, up_pad); st_d = min(new_H, up_pad + H)
        st_l = max(0, lf_pad); st_r = min(new_W, lf_pad + W)
        ss_u = max(0, -up_pad); ss_d = min(H, new_H - up_pad)
        ss_l = max(0, -lf_pad); ss_r = min(W, new_W - lf_pad)
        image = image.permute(0, 3, 1, 2)
        expanded_image = expanded_image.permute(0, 3, 1, 2)
        expanded_image[:, :, st_u:st_d, st_l:st_r] = image[:, :, ss_u:ss_d, ss_l:ss_r]
        if up_pad > 0:
            expanded_image[:, :, :up_pad, st_l:st_r] = \
                image[:, :, 0:1, ss_l:ss_r].repeat(1, 1, up_pad, 1)
        if dn_pad > 0:
            expanded_image[:, :, -dn_pad:, st_l:st_r] = \
                image[:, :, -1:, ss_l:ss_r].repeat(1, 1, dn_pad, 1)
        if lf_pad > 0:
            expanded_image[:, :, st_u:st_d, :lf_pad] = \
                expanded_image[:, :, st_u:st_d, lf_pad:lf_pad+1].repeat(1, 1, 1, lf_pad)
        if rt_pad > 0:
            expanded_image[:, :, st_u:st_d, -rt_pad:] = \
                expanded_image[:, :, st_u:st_d, -rt_pad-1:-rt_pad].repeat(1, 1, 1, rt_pad)
        expanded_mask[:, st_u:st_d, st_l:st_r] = mask[:, ss_u:ss_d, ss_l:ss_r]
        expanded_opt[:, st_u:st_d, st_l:st_r] = optional_context_mask[:, ss_u:ss_d, ss_l:ss_r]
        expanded_image = expanded_image.permute(0, 2, 3, 1)
        return expanded_image, expanded_mask, expanded_opt

    def batched_findcontextarea_m(self, mask):
        B, H, W = mask.shape
        device = mask.device
        x_list, y_list, w_list, h_list = [], [], [], []
        for i in range(B):
            nz = torch.nonzero(mask[i])
            if nz.numel() == 0:
                x_list.append(-1); y_list.append(-1)
                w_list.append(-1); h_list.append(-1)
            else:
                by = torch.min(nz[:, 0]).item()
                bx = torch.min(nz[:, 1]).item()
                by_max = torch.max(nz[:, 0]).item()
                bx_max = torch.max(nz[:, 1]).item()
                x_list.append(bx); y_list.append(by)
                w_list.append(bx_max - bx + 1)
                h_list.append(by_max - by + 1)
        return (None,
                torch.tensor(x_list, device=device),
                torch.tensor(y_list, device=device),
                torch.tensor(w_list, device=device),
                torch.tensor(h_list, device=device))

    def batched_growcontextarea_m(self, mask, x, y, w, h, extend_factor):
        img_h, img_w = mask.shape[1], mask.shape[2]
        device = mask.device
        grow_x = (w.float() * (extend_factor - 1.0) / 2.0).round().long()
        grow_y = (h.float() * (extend_factor - 1.0) / 2.0).round().long()
        new_x = torch.clamp(x - grow_x, min=0)
        new_y = torch.clamp(y - grow_y, min=0)
        new_x2 = torch.clamp(x + w + grow_x, max=img_w)
        new_y2 = torch.clamp(y + h + grow_y, max=img_h)
        new_w = new_x2 - new_x
        new_h = new_y2 - new_y
        empty = (w == -1)
        new_x[empty] = 0; new_y[empty] = 0
        new_w[empty] = img_w; new_h[empty] = img_h
        return None, new_x, new_y, new_w, new_h

    def batched_combinecontextmask_m(self, mask, x, y, w, h, optional_context_mask):
        _, ox, oy, ow, oh = self.batched_findcontextarea_m(optional_context_mask)
        neg1 = (x == -1)
        x1 = torch.where(neg1, ox, x); y1 = torch.where(neg1, oy, y)
        w1 = torch.where(neg1, ow, w); h1 = torch.where(neg1, oh, h)
        oneg1 = (ox == -1)
        ox2 = torch.where(oneg1, x1, ox); oy2 = torch.where(oneg1, y1, oy)
        ow2 = torch.where(oneg1, w1, ow); oh2 = torch.where(oneg1, h1, oh)
        new_x = torch.min(x1, ox2); new_y = torch.min(y1, oy2)
        new_xmax = torch.max(x1 + w1, ox2 + ow2)
        new_ymax = torch.max(y1 + h1, oy2 + oh2)
        new_w = new_xmax - new_x; new_h = new_ymax - new_y
        both_empty = (x1 == -1)
        new_x[both_empty] = -1; new_y[both_empty] = -1
        new_w[both_empty] = -1; new_h[both_empty] = -1
        return None, new_x, new_y, new_w, new_h

    def crop_magic_im(self, image, mask, x, y, w, h,
                      target_w, target_h, padding,
                      downscale_algorithm, upscale_algorithm,
                      resize_output=True):
        image = image.clone(); mask = mask.clone()
        if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
            return (image, 0, 0, image.shape[2], image.shape[1],
                    image, mask, 0, 0, image.shape[2], image.shape[1])
        if padding != 0:
            target_w = self.pad_to_multiple(target_w, padding)
            target_h = self.pad_to_multiple(target_h, padding)
        target_ar = target_w / target_h
        B, image_h, image_w, C = image.shape
        ctx_ar = w / h
        if ctx_ar < target_ar:
            new_w = int(h * target_ar); new_h = h
            new_x = x - (new_w - w) // 2; new_y = y
            if new_x < 0:
                shift = -new_x
                new_x = new_x + shift if new_x + new_w + shift <= image_w \
                    else -(new_w - image_w) // 2
            elif new_x + new_w > image_w:
                ov = new_x + new_w - image_w
                new_x = new_x - ov if new_x - ov >= 0 \
                    else -(new_w - image_w) // 2
        else:
            new_w = w; new_h = int(w / target_ar)
            new_x = x; new_y = y - (new_h - h) // 2
            if new_y < 0:
                shift = -new_y
                new_y = new_y + shift if new_y + new_h + shift <= image_h \
                    else -(new_h - image_h) // 2
            elif new_y + new_h > image_h:
                ov = new_y + new_h - image_h
                new_y = new_y - ov if new_y - ov >= 0 \
                    else -(new_h - image_h) // 2
        if not resize_output:
            if new_w < target_w:
                gw = target_w - new_w; new_x -= gw // 2; new_w = target_w
                if new_x < 0:
                    s = -new_x
                    new_x = new_x + s if new_x + new_w + s <= image_w \
                        else -((new_w - image_w) // 2)
                elif new_x + new_w > image_w:
                    ov = new_x + new_w - image_w
                    new_x = new_x - ov if new_x - ov >= 0 \
                        else -((new_w - image_w) // 2)
            if new_h < target_h:
                gh = target_h - new_h; new_y -= gh // 2; new_h = target_h
                if new_y < 0:
                    s = -new_y
                    new_y = new_y + s if new_y + new_h + s <= image_h \
                        else -((new_h - image_h) // 2)
                elif new_y + new_h > image_h:
                    ov = new_y + new_h - image_h
                    new_y = new_y - ov if new_y - ov >= 0 \
                        else -((new_h - image_h) // 2)
        up_pad = dn_pad = lf_pad = rt_pad = 0
        exp_w = image_w; exp_h = image_h
        if new_x < 0:     lf_pad = -new_x;              exp_w += lf_pad
        if new_x + new_w > image_w: rt_pad = new_x + new_w - image_w; exp_w += rt_pad
        if new_y < 0:     up_pad = -new_y;              exp_h += up_pad
        if new_y + new_h > image_h: dn_pad = new_y + new_h - image_h; exp_h += dn_pad
        expanded_image = torch.zeros((B, exp_h, exp_w, C), device=image.device)
        expanded_mask  = torch.ones( (B, exp_h, exp_w),    device=mask.device)
        image = image.permute(0, 3, 1, 2)
        expanded_image = expanded_image.permute(0, 3, 1, 2)
        expanded_image[:, :, up_pad:up_pad+image_h, lf_pad:lf_pad+image_w] = image
        if up_pad > 0:
            expanded_image[:, :, :up_pad, lf_pad:lf_pad+image_w] = \
                expanded_image[:, :, up_pad:up_pad+1, lf_pad:lf_pad+image_w].repeat(1,1,up_pad,1)
        if dn_pad > 0:
            expanded_image[:, :, -dn_pad:, lf_pad:lf_pad+image_w] = \
                expanded_image[:, :, up_pad+image_h-1:up_pad+image_h, lf_pad:lf_pad+image_w].repeat(1,1,dn_pad,1)
        if lf_pad > 0:
            expanded_image[:, :, up_pad:up_pad+image_h, :lf_pad] = \
                expanded_image[:, :, up_pad:up_pad+image_h, lf_pad:lf_pad+1].repeat(1,1,1,lf_pad)
        if rt_pad > 0:
            expanded_image[:, :, up_pad:up_pad+image_h, -rt_pad:] = \
                expanded_image[:, :, up_pad:up_pad+image_h, -rt_pad-1:-rt_pad].repeat(1,1,1,rt_pad)
        expanded_image = expanded_image.permute(0, 2, 3, 1)
        image = image.permute(0, 2, 3, 1)
        expanded_mask[:, up_pad:up_pad+image_h, lf_pad:lf_pad+image_w] = mask
        cto_x, cto_y, cto_w, cto_h = lf_pad, up_pad, image_w, image_h
        canvas_image = expanded_image; canvas_mask = expanded_mask
        ctc_x = new_x + lf_pad; ctc_y = new_y + up_pad
        ctc_w = new_w;            ctc_h = new_h
        cropped_image = canvas_image[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w]
        cropped_mask  = canvas_mask[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w]
        if resize_output:
            if target_w > ctc_w or target_h > ctc_h:
                cropped_image = self.rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
                cropped_mask  = self.rescale_m(cropped_mask,  target_w, target_h, upscale_algorithm)
            else:
                cropped_image = self.rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
                cropped_mask  = self.rescale_m(cropped_mask,  target_w, target_h, downscale_algorithm)
        return (canvas_image, cto_x, cto_y, cto_w, cto_h,
                cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h)

    def stitch_magic_im(self, canvas_image, inpainted_image, mask,
                        ctc_x, ctc_y, ctc_w, ctc_h,
                        cto_x, cto_y, cto_w, cto_h,
                        downscale_algorithm, upscale_algorithm):
        canvas_image = canvas_image.clone()
        inpainted_image = inpainted_image.clone()
        mask = mask.clone()
        B, h, w, _ = inpainted_image.shape
        if ctc_w > w or ctc_h > h:
            resized_image = self.rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
            resized_mask  = self.rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
        else:
            resized_image = self.rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
            resized_mask  = self.rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)
        resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)
        canvas_crop = canvas_image[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w]
        blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop
        canvas_image[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w] = blended
        return canvas_image[:, cto_y:cto_y+cto_h, cto_x:cto_x+cto_w]


# ---------------------------------------------------------------------------
# GPU implementation  (copied verbatim from lquesada's original)
# ---------------------------------------------------------------------------

class GPUProcessorLogic(ProcessorLogic):
    def rescale_i(self, samples, width, height, algorithm: str):
        original_device = samples.device
        samples = samples.movedim(-1, 1)
        algorithm_enum = getattr(Image, algorithm.upper())
        results = []
        for i in range(samples.shape[0]):
            pil = F.to_pil_image(samples[i].float().cpu()).resize((width, height), algorithm_enum)
            results.append(F.to_tensor(pil))
        return torch.stack(results, dim=0).to(original_device).movedim(1, -1)

    def rescale_m(self, samples, width, height, algorithm: str):
        original_device = samples.device
        algorithm_enum = getattr(Image, algorithm.upper())
        results = []
        for i in range(samples.shape[0]):
            pil = F.to_pil_image(samples[i].float().cpu()).resize((width, height), algorithm_enum)
            results.append(F.to_tensor(pil).squeeze(0))
        return torch.stack(results, dim=0).to(original_device)

    def fillholes_iterative_hipass_fill_m(self, samples):
        thresholds = [1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7,
                      0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        results = []
        original_device = samples.device
        for i in range(samples.shape[0]):
            mask_np = samples[i].cpu().numpy()
            for threshold in thresholds:
                thresholded = mask_np >= threshold
                closed = binary_closing(thresholded, structure=np.ones((3, 3)), border_value=1)
                filled = binary_fill_holes(closed)
                mask_np = np.maximum(mask_np, np.where(filled, threshold, 0))
            results.append(torch.from_numpy(mask_np.astype(np.float32)))
        return torch.stack(results, dim=0).to(original_device)

    def hipassfilter_m(self, samples, threshold):
        filtered = samples.clone()
        filtered[filtered < threshold] = 0
        return filtered

    def expand_m(self, mask, pixels):
        sigma = pixels / 4
        kernel_size = math.ceil(sigma * 1.5 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        mask_in = mask.unsqueeze(1)
        mask_padded = TF.pad(mask_in, (padding, padding, padding, padding), mode='reflect')
        dilated = TF.max_pool2d(mask_padded, kernel_size=kernel_size, stride=1, padding=0)
        return dilated.squeeze(1)

    def invert_m(self, samples):
        return 1.0 - samples.clone()

    def blur_m(self, samples, pixels):
        sigma = pixels / 4
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        x = torch.arange(kernel_size, device=samples.device, dtype=samples.dtype) \
            - (kernel_size - 1) / 2
        k1d = torch.exp(-0.5 * (x / sigma).pow(2))
        k1d = k1d / k1d.sum()
        k2d = (k1d.unsqueeze(1) * k1d.unsqueeze(0)).expand(1, 1, kernel_size, kernel_size)
        mask_in = samples.unsqueeze(1)
        pad = kernel_size // 2
        mask_padded = TF.pad(mask_in, (pad, pad, pad, pad), mode='reflect')
        blurred = TF.conv2d(mask_padded, k2d, padding=0, groups=1)
        return blurred.squeeze(1).clamp(0.0, 1.0)

    def debug_context_location_in_image(self, image, x, y, w, h):
        debug = image.clone()
        debug[:, y:y+h, x:x+w, :] = 1.0 - debug[:, y:y+h, x:x+w, :]
        return debug

    def pad_to_multiple(self, value, multiple):
        return int(math.ceil(value / multiple) * multiple)

    def preresize_imm(self, image, mask, optional_context_mask,
                      downscale_algorithm, upscale_algorithm,
                      preresize_mode, preresize_min_width, preresize_min_height,
                      preresize_max_width, preresize_max_height):
        # Delegate to CPU logic (same behaviour, GPU gives no advantage here)
        cpu = CPUProcessorLogic()
        return cpu.preresize_imm(image, mask, optional_context_mask,
                                 downscale_algorithm, upscale_algorithm,
                                 preresize_mode, preresize_min_width, preresize_min_height,
                                 preresize_max_width, preresize_max_height)

    def extend_imm(self, image, mask, optional_context_mask,
                   extend_up_factor, extend_down_factor,
                   extend_left_factor, extend_right_factor):
        cpu = CPUProcessorLogic()
        return cpu.extend_imm(image, mask, optional_context_mask,
                              extend_up_factor, extend_down_factor,
                              extend_left_factor, extend_right_factor)

    def batched_findcontextarea_m(self, mask):
        B, H, W = mask.shape
        device = mask.device
        any_y = mask.max(dim=2).values > 0.0
        any_x = mask.max(dim=1).values > 0.0
        def get_min_max(any_dim, size):
            indices = torch.arange(size, device=device).unsqueeze(0).expand(B, -1)
            min_idx = torch.where(any_dim, indices, torch.tensor(size, device=device))
            max_idx = torch.where(any_dim, indices, torch.tensor(-1, device=device))
            b_min = torch.min(min_idx, dim=1).values
            b_max = torch.max(max_idx, dim=1).values
            empty = ~any_dim.any(dim=1)
            b_min[empty] = -1; b_max[empty] = -1
            return b_min, b_max
        y_min, y_max = get_min_max(any_y, H)
        x_min, x_max = get_min_max(any_x, W)
        w = torch.where(x_min >= 0, x_max - x_min + 1, torch.tensor(-1, device=device))
        h = torch.where(y_min >= 0, y_max - y_min + 1, torch.tensor(-1, device=device))
        return None, x_min, y_min, w, h

    def batched_growcontextarea_m(self, mask, x, y, w, h, extend_factor):
        img_h, img_w = mask.shape[1], mask.shape[2]
        device = mask.device
        grow_x = (w.float() * (extend_factor - 1.0) / 2.0).round().long()
        grow_y = (h.float() * (extend_factor - 1.0) / 2.0).round().long()
        new_x = torch.clamp(x - grow_x, min=0)
        new_y = torch.clamp(y - grow_y, min=0)
        new_x2 = torch.clamp(x + w + grow_x, max=img_w)
        new_y2 = torch.clamp(y + h + grow_y, max=img_h)
        new_w = new_x2 - new_x; new_h = new_y2 - new_y
        empty = (w == -1)
        new_x[empty] = 0; new_y[empty] = 0
        new_w[empty] = img_w; new_h[empty] = img_h
        return None, new_x, new_y, new_w, new_h

    def batched_combinecontextmask_m(self, mask, x, y, w, h, optional_context_mask):
        _, ox, oy, ow, oh = self.batched_findcontextarea_m(optional_context_mask)
        neg1 = (x == -1)
        x1 = torch.where(neg1, ox, x); y1 = torch.where(neg1, oy, y)
        w1 = torch.where(neg1, ow, w); h1 = torch.where(neg1, oh, h)
        oneg1 = (ox == -1)
        ox2 = torch.where(oneg1, x1, ox); oy2 = torch.where(oneg1, y1, oy)
        ow2 = torch.where(oneg1, w1, ow); oh2 = torch.where(oneg1, h1, oh)
        new_x = torch.min(x1, ox2); new_y = torch.min(y1, oy2)
        nxmax = torch.max(x1 + w1, ox2 + ow2); nymax = torch.max(y1 + h1, oy2 + oh2)
        new_w = nxmax - new_x; new_h = nymax - new_y
        both_empty = (x1 == -1)
        new_x[both_empty] = -1; new_y[both_empty] = -1
        new_w[both_empty] = -1; new_h[both_empty] = -1
        return None, new_x, new_y, new_w, new_h

    def crop_magic_im(self, image, mask, x, y, w, h,
                      target_w, target_h, padding,
                      downscale_algorithm, upscale_algorithm,
                      resize_output=True):
        # Run on CPU to keep PIL rescaling simple; restore device on outputs.
        device = image.device
        cpu = CPUProcessorLogic()
        (canvas, cto_x, cto_y, cto_w, cto_h,
         cropped, cmask,
         ctc_x, ctc_y, ctc_w, ctc_h) = cpu.crop_magic_im(
            image.cpu(), mask.cpu(), x, y, w, h,
            target_w, target_h, padding,
            downscale_algorithm, upscale_algorithm,
            resize_output)
        return (canvas.to(device), cto_x, cto_y, cto_w, cto_h,
                cropped.to(device), cmask.to(device),
                ctc_x, ctc_y, ctc_w, ctc_h)

    def stitch_magic_im(self, canvas_image, inpainted_image, mask,
                        ctc_x, ctc_y, ctc_w, ctc_h,
                        cto_x, cto_y, cto_w, cto_h,
                        downscale_algorithm, upscale_algorithm):
        device = canvas_image.device
        cpu = CPUProcessorLogic()
        result = cpu.stitch_magic_im(
            canvas_image.cpu(), inpainted_image.cpu(), mask.cpu(),
            ctc_x, ctc_y, ctc_w, ctc_h,
            cto_x, cto_y, cto_w, cto_h,
            downscale_algorithm, upscale_algorithm)
        return result.to(device)


# ---------------------------------------------------------------------------
# Preview helper — draws the crop rectangle and returns tensor + temp file
# ---------------------------------------------------------------------------

def _make_nb2_preview(img_tensor: torch.Tensor, y1: int, x1: int,
                      ch: int, cw: int):
    """Draw the NB2 crop rectangle on the image.

    Returns
    -------
    preview_tensor : torch.Tensor  [1, H, W, 3]  float32 in [0, 1]
    temp_info      : dict or None   ComfyUI image-info for ui.nb2_preview
    """
    # Always work in RGB
    img_np = (img_tensor.cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
    rgb_np = img_np[:, :, :3]  # strip alpha if present

    base = Image.fromarray(rgb_np, "RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    draw.rectangle([x1, y1, x1 + cw - 1, y1 + ch - 1], fill=(0, 140, 255, 45))
    for t in range(3):
        draw.rectangle([x1 + t, y1 + t, x1 + cw - 1 - t, y1 + ch - 1 - t],
                       outline=(0, 200, 255, 220))
    cx, cy = x1 + cw // 2, y1 + ch // 2
    cs = max(10, min(cw, ch) // 25)
    draw.line([cx - cs, cy, cx + cs, cy], fill=(255, 255, 255, 210), width=2)
    draw.line([cx, cy - cs, cx, cy + cs], fill=(255, 255, 255, 210), width=2)

    base.alpha_composite(overlay)
    rgb_out = base.convert("RGB")

    # tensor [1, H, W, 3]
    preview_tensor = torch.from_numpy(
        np.array(rgb_out).astype(np.float32) / 255.0
    ).unsqueeze(0)

    # save to temp for JS widget
    temp_info = None
    if _HAS_FOLDER_PATHS:
        try:
            temp_dir = _folder_paths.get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            fname = f"nb2_prev_{uuid.uuid4().hex[:10]}.png"
            rgb_out.save(os.path.join(temp_dir, fname))
            temp_info = {"filename": fname, "subfolder": "", "type": "temp"}
        except Exception as exc:
            print(f"[NB2] Preview save failed: {exc}")

    return preview_tensor, temp_info


def _fit_nb2_rect_to_mask(mask_2d: torch.Tensor, image_w: int, image_h: int,
                          target_ar: float, padding_percent: float,
                          crop_scale: float):
    """Fit a target-aspect rectangle around a semantic mask."""
    nz = torch.nonzero(mask_2d > 0.0)
    if nz.numel() == 0:
        raise ValueError("region_mask is empty; cannot derive an NB2 rectangle.")

    y1 = int(torch.min(nz[:, 0]).item())
    x1 = int(torch.min(nz[:, 1]).item())
    y2 = int(torch.max(nz[:, 0]).item()) + 1
    x2 = int(torch.max(nz[:, 1]).item()) + 1

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_x = int(round(box_w * (padding_percent / 100.0)))
    pad_y = int(round(box_h * (padding_percent / 100.0)))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image_w, x2 + pad_x)
    y2 = min(image_h, y2 + pad_y)

    padded_w = max(1, x2 - x1)
    padded_h = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    if (padded_w / padded_h) < target_ar:
        rect_h = float(padded_h)
        rect_w = rect_h * target_ar
    else:
        rect_w = float(padded_w)
        rect_h = rect_w / target_ar

    rect_w *= max(1.0, crop_scale)
    rect_h *= max(1.0, crop_scale)

    max_rect_w = float(image_w)
    max_rect_h = max_rect_w / target_ar
    if max_rect_h > image_h:
        max_rect_h = float(image_h)
        max_rect_w = max_rect_h * target_ar

    if rect_w > max_rect_w or rect_h > max_rect_h:
        rect_w = max_rect_w
        rect_h = max_rect_h

    cw = max(1, int(round(rect_w)))
    ch = max(1, int(round(rect_h)))

    ch = max(1, min(image_h, int(round(cw / target_ar))))
    cw = max(1, min(image_w, int(round(ch * target_ar))))

    if cw > image_w or ch > image_h:
        fit_w = image_w
        fit_h = int(round(fit_w / target_ar))
        if fit_h > image_h:
            fit_h = image_h
            fit_w = int(round(fit_h * target_ar))
        cw = max(1, min(image_w, fit_w))
        ch = max(1, min(image_h, fit_h))

    rect_x1 = int(round(center_x - cw / 2.0))
    rect_y1 = int(round(center_y - ch / 2.0))

    rect_x1 = max(0, min(rect_x1, image_w - cw))
    rect_y1 = max(0, min(rect_y1, image_h - ch))

    return rect_x1, rect_y1, cw, ch


def _normalize_mask_to_image(mask: torch.Tensor, image: torch.Tensor,
                             processor: ProcessorLogic,
                             node_name: str,
                             depad_florence: bool = True) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Make a MASK tensor match an IMAGE tensor in rank, batch, and spatial size.

    Florence / segmentation nodes may emit a valid MASK tensor whose width and
    height do not match the source IMAGE. For crop logic that derives a bbox
    from the mask, that mismatch shifts the selected region.

    depad_florence: when True (default), detects the square letterbox padding
    that Florence2 adds internally and crops it out before resizing, so the
    bbox lands on the correct region in non-square images.
    """
    note_parts = []

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        note_parts.append("mask [H,W] -> [1,H,W]")
    elif mask.ndim != 3:
        raise ValueError(
            f"{node_name} expected MASK [H,W] or [B,H,W], got {tuple(mask.shape)}."
        )

    if image.ndim != 4:
        raise ValueError(
            f"{node_name} expected IMAGE [B,H,W,C], got {tuple(image.shape)}."
        )

    if mask.shape[0] > 1 and image.shape[0] == 1:
        image = image.expand(mask.shape[0], -1, -1, -1).clone()
        note_parts.append(f"image batch expanded 1->{mask.shape[0]}")
    if image.shape[0] > 1 and mask.shape[0] == 1:
        mask = mask.expand(image.shape[0], -1, -1).clone()
        note_parts.append(f"mask batch expanded 1->{image.shape[0]}")

    if image.shape[0] != mask.shape[0]:
        raise ValueError(f"{node_name}: image and mask batch sizes are incompatible.")

    target_h, target_w = image.shape[1], image.shape[2]
    target_ar = target_w / target_h

    # Fix 2: kijai's Florence2Run outputs the mask at the original image size
    # but places detections in Florence's 1024-pixel coordinate space, so all
    # active pixels land in the top-left region (columns < 1024 and/or rows <
    # 1024), making the mask appear tiny and stuck to the upper-left corner.
    # Detect this by checking that at least one image dimension exceeds 1024 and
    # that no active pixels exist beyond the 1024-px boundary in that dimension.
    # Then extract the true content area (removing Florence's letterbox padding)
    # and rescale it to the full image size.
    _FLORENCE_SIZE = 1024
    _wide = target_w > _FLORENCE_SIZE
    _tall = target_h > _FLORENCE_SIZE
    if (depad_florence
            and mask.shape[1] == target_h
            and mask.shape[2] == target_w
            and (_wide or _tall)
            and mask.any()
            and (not _tall or not mask[:, _FLORENCE_SIZE:, :].any())
            and (not _wide or not mask[:, :, _FLORENCE_SIZE:].any())):
        # Compute Florence's letterbox offsets for this image's aspect ratio,
        # then extract only the content pixels (without the padding rows/cols).
        if target_w >= target_h:  # landscape or square
            content_h = max(1, round(_FLORENCE_SIZE * target_h / target_w))
            pad_y     = (_FLORENCE_SIZE - content_h) // 2
            row_start = min(pad_y, target_h - 1)
            row_end   = min(pad_y + content_h, target_h)
            col_end   = min(target_w, _FLORENCE_SIZE)
            content   = mask[:, row_start:row_end, :col_end]
            fix2_note = (f"Florence coord fix: landscape {target_w}x{target_h}"
                         f" depad y[{row_start}:{row_end}/{_FLORENCE_SIZE}]")
        else:  # portrait
            content_w = max(1, round(_FLORENCE_SIZE * target_w / target_h))
            pad_x     = (_FLORENCE_SIZE - content_w) // 2
            row_end   = min(target_h, _FLORENCE_SIZE)
            col_start = min(pad_x, target_w - 1)
            col_end   = min(pad_x + content_w, target_w)
            content   = mask[:, :row_end, col_start:col_end]
            fix2_note = (f"Florence coord fix: portrait {target_w}x{target_h}"
                         f" depad x[{col_start}:{col_end}/{_FLORENCE_SIZE}]")
        mask = processor.rescale_m(content, target_w, target_h, "nearest")
        note_parts.append(fix2_note + f", rescaled to {target_w}x{target_h}")

    if mask.shape[1] != target_h or mask.shape[2] != target_w:
        old_h, old_w = mask.shape[1], mask.shape[2]

        # Fix 1 — Florence2 letterbox correction for masks at 1024×1024:
        # Florence2 pads images to a square before processing. The output mask
        # is in that padded-square space. Detect and remove the padding before
        # rescaling so the detected region lands at the correct position.
        if depad_florence and old_h > 0 and old_w > 0:
            mask_ar = old_w / old_h
            if abs(mask_ar - target_ar) > 0.05 and abs(mask_ar - 1.0) < 0.05:
                if target_ar > 1.0:
                    # Landscape original → Florence padded top/bottom
                    content_h = max(1, round(old_h * target_h / target_w))
                    pad_y     = max(0, (old_h - content_h) // 2)
                    content_h = min(content_h, old_h - pad_y)
                    mask = mask[:, pad_y : pad_y + content_h, :]
                    note_parts.append(
                        f"Florence depad top/bottom y[{pad_y}:{pad_y+content_h}/{old_h}]"
                    )
                else:
                    # Portrait original → Florence padded left/right
                    content_w = max(1, round(old_w * target_w / target_h))
                    pad_x     = max(0, (old_w - content_w) // 2)
                    content_w = min(content_w, old_w - pad_x)
                    mask = mask[:, :, pad_x : pad_x + content_w]
                    note_parts.append(
                        f"Florence depad left/right x[{pad_x}:{pad_x+content_w}/{old_w}]"
                    )

        mask = processor.rescale_m(mask, target_w, target_h, "nearest")
        note_parts.append(
            f"mask resized {old_w}x{old_h} -> {target_w}x{target_h}"
        )

    return mask, image, (" | ".join(note_parts) if note_parts else "mask already matched image")


def _grow_and_feather_mask(mask_2d: torch.Tensor,
                           expand_percent: float,
                           feather_percent: float) -> torch.Tensor:
    mask_np = mask_2d.detach().cpu().numpy().astype(np.float32)
    binary = mask_np > 0.001
    ys, xs = np.nonzero(binary)
    if len(xs) == 0 or len(ys) == 0:
        return mask_2d.clamp(0, 1)

    box_w = max(1, int(xs.max() - xs.min() + 1))
    box_h = max(1, int(ys.max() - ys.min() + 1))
    grow_px = int(round(max(box_w, box_h) * (max(0.0, expand_percent) / 100.0)))
    if grow_px > 0:
        kernel = np.ones((grow_px * 2 + 1, grow_px * 2 + 1), dtype=np.uint8)
        binary = grey_dilation(binary.astype(np.float32), footprint=kernel, mode="reflect") > 0.0

    mask_float = binary.astype(np.float32)
    feather_px = max(0.0, max(box_w, box_h) * (max(0.0, feather_percent) / 100.0))
    if feather_px > 0.0:
        blurred = gaussian_filter(mask_float, sigma=max(0.5, feather_px / 3.0), mode="reflect")
        if blurred.max() > 0:
            mask_float = blurred / blurred.max()
    return torch.from_numpy(mask_float).to(mask_2d.device, dtype=torch.float32).clamp(0, 1)


def _soft_blur_mask(mask_2d: torch.Tensor, blur_percent: float) -> torch.Tensor:
    blur_percent = max(0.0, float(blur_percent))
    if blur_percent <= 0.0:
        return mask_2d.clamp(0, 1)

    mask_np = mask_2d.detach().cpu().numpy().astype(np.float32)
    ys, xs = np.nonzero(mask_np > 0.001)
    if len(xs) == 0 or len(ys) == 0:
        return mask_2d.clamp(0, 1)

    box_w = max(1, int(xs.max() - xs.min() + 1))
    box_h = max(1, int(ys.max() - ys.min() + 1))
    blur_px = max(box_w, box_h) * (blur_percent / 100.0)
    blurred = gaussian_filter(mask_np, sigma=max(0.5, blur_px / 3.0), mode="reflect")
    max_before = float(mask_np.max())
    max_after = float(blurred.max())
    if max_before > 0.0 and max_after > 0.0:
        blurred = blurred * (max_before / max_after)

    return torch.from_numpy(blurred).to(mask_2d.device, dtype=torch.float32).clamp(0, 1)


def _blur_uint8_mask(mask_uint8: np.ndarray, blur_percent: float) -> np.ndarray:
    mask_float = torch.from_numpy(mask_uint8.astype(np.float32) / 255.0)
    blurred = _soft_blur_mask(mask_float, blur_percent)
    return np.clip(blurred.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


def _parse_edit_size(size_value: str) -> tuple[int, int]:
    size_value = _coerce_text_value(size_value)
    match = re.fullmatch(r"(\d+)x(\d+)", size_value)
    if not match:
        raise ValueError(f"Unsupported edit size value: {size_value}")
    return int(match.group(1)), int(match.group(2))


def _fit_aspect_rect_to_bbox(x: int, y: int, w: int, h: int,
                             image_w: int, image_h: int,
                             target_ar: float) -> tuple[int, int, int, int]:
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    if (w / max(1.0, h)) < target_ar:
        rect_h = float(h)
        rect_w = rect_h * target_ar
    else:
        rect_w = float(w)
        rect_h = rect_w / target_ar

    max_rect_w = float(image_w)
    max_rect_h = max_rect_w / target_ar
    if max_rect_h > image_h:
        max_rect_h = float(image_h)
        max_rect_w = max_rect_h * target_ar

    rect_w = min(rect_w, max_rect_w)
    rect_h = min(rect_h, max_rect_h)

    rect_w_i = max(1, int(round(rect_w)))
    rect_h_i = max(1, int(round(rect_h)))
    rect_h_i = max(1, min(image_h, int(round(rect_w_i / target_ar))))
    rect_w_i = max(1, min(image_w, int(round(rect_h_i * target_ar))))

    rect_x = int(round(center_x - rect_w_i / 2.0))
    rect_y = int(round(center_y - rect_h_i / 2.0))
    rect_x = max(0, min(rect_x, image_w - rect_w_i))
    rect_y = max(0, min(rect_y, image_h - rect_h_i))
    return rect_x, rect_y, rect_w_i, rect_h_i
# ===========================================================================
#  NEW NODE 1 — NanoBanana2MaskGen
# ===========================================================================

class NanoBanana2MaskGen:
    """
    Generates a rectangular MASK whose aspect ratio matches the exact output
    of Nano Banana 2 (1K / 2K / 4K in 16:9, 9:16 or 1:1).

    Place the rectangle anywhere on the original image with center_x / center_y.
    The crop_width slider controls how much of the original image is covered
    (height is auto-calculated from the selected aspect ratio).

    If the rectangle would overflow the image boundaries it is shifted inward
    automatically.  If crop_width is wider than the image it is clamped, and
    the height is recalculated accordingly.

    Outputs
    -------
    mask        — binary float mask [B, H, W] with 1 inside the rectangle
    nb2_width   — exact pixel width  that Nano Banana 2 will produce
    nb2_height  — exact pixel height that Nano Banana 2 will produce
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "resolution":   (["1K",   "2K",   "4K"],  {"default": "2K"}),
                "center_x": ("INT", {
                    "default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1,
                    "tooltip": "Horizontal centre of the crop rectangle in pixels."}),
                "center_y": ("INT", {
                    "default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1,
                    "tooltip": "Vertical centre of the crop rectangle in pixels."}),
                "crop_width": ("INT", {
                    "default": 960, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1,
                    "tooltip": ("Width of the crop region on the original image in pixels. "
                                "Height is auto-calculated from the aspect ratio. "
                                "Clamped if larger than the image.")}),
            }
        }

    RETURN_TYPES  = ("MASK", "INT", "INT", "IMAGE")
    RETURN_NAMES  = ("mask", "nb2_width", "nb2_height", "preview_image")
    FUNCTION      = "generate_mask"
    CATEGORY      = "inpaint/nb2"
    DESCRIPTION   = (
        "Creates a positioned rectangle mask matching Nano Banana 2 aspect "
        "ratios.  Connect mask → InpaintCropNB2, nb2_width/nb2_height → NB2 "
        "Crop inputs, and preview_image → any Preview Image node to see the "
        "crop position annotated on the original."
    )

    def generate_mask(self, image, aspect_ratio, resolution,
                      center_x, center_y, crop_width):
        B, H, W, _C = image.shape
        nb2_w, nb2_h = NB2_RESOLUTIONS[aspect_ratio][resolution]
        ar = nb2_w / nb2_h

        # --- calculate crop size on the original image ---
        cw = min(crop_width, W)
        ch = int(round(cw / ar))

        # If height overflows, shrink to fit and recompute width
        if ch > H:
            ch = H
            cw = int(round(ch * ar))
            cw = min(cw, W)
            ch = int(round(cw / ar))

        # --- top-left from centre ---
        x1 = center_x - cw // 2
        y1 = center_y - ch // 2

        # Clamp position so the rectangle stays inside the image
        x1 = max(0, min(x1, W - cw))
        y1 = max(0, min(y1, H - ch))

        mask = torch.zeros(B, H, W, dtype=torch.float32)
        mask[:, y1:y1 + ch, x1:x1 + cw] = 1.0

        # Build preview for both the IMAGE output and the JS canvas widget
        preview_tensor, temp_info = _make_nb2_preview(image[0], y1, x1, ch, cw)
        # Replicate preview for every item in the batch
        preview_batch = preview_tensor.expand(B, -1, -1, -1)

        result = {"result": (mask, nb2_w, nb2_h, preview_batch)}
        if temp_info:
            result["ui"] = {"nb2_preview": [temp_info]}
        return result


class NB2SmartRegionMask:
    """
    Converts any semantic mask into a rectangular NB2-compatible crop mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region_mask": ("MASK",),
                "aspect_ratio": (["auto", "16:9", "9:16", "1:1"], {"default": "auto"}),
                "resolution":   (["1K", "2K", "4K"], {"default": "2K"}),
                "padding_percent": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Expands the detected region before rectangle fitting."}),
                "crop_scale": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Additional scale multiplier applied after aspect-ratio fitting."}),
                "depad_florence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Remove Florence2's internal square letterbox padding before "
                        "resizing the mask. Keep True when the mask comes from "
                        "Florence2Run (kijai). Disable only if your mask is already "
                        "at the exact source image resolution."
                    )}),
            },
            "optional": {
                "region_info": ("STRING",),
            }
        }

    RETURN_TYPES = ("MASK", "INT", "INT", "IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "mask",
        "nb2_width",
        "nb2_height",
        "preview_image",
        "center_x",
        "center_y",
        "crop_width",
        "crop_height",
        "info",
    )
    FUNCTION = "generate_from_region"
    CATEGORY = "inpaint/nb2"
    DESCRIPTION = (
        "Fits a rectangular NB2 crop around a semantic region mask. "
        "Use this after a Florence/SAM-style selector when you want to keep "
        "the existing NB2 Crop and NB2 Stitch workflow."
    )

    def generate_from_region(self, image, region_mask, aspect_ratio, resolution,
                             padding_percent, crop_scale, depad_florence=True, region_info=""):
        image = image.clone()
        region_mask = region_mask.clone()
        processor = CPUProcessorLogic()
        region_mask, image, mask_note = _normalize_mask_to_image(
            region_mask, image, processor, "NB2SmartRegionMask",
            depad_florence=depad_florence
        )

        B, H, W, _ = image.shape
        region_context = _extract_region_context(region_info)
        resolved_aspect_ratio = aspect_ratio
        aspect_ratio_source = "manual"
        if aspect_ratio == "auto":
            resolved_aspect_ratio = region_context.get("recommended_aspect_ratio") or _recommend_aspect_ratio_for_region(
                region_context.get("region_type"),
                region_context.get("bbox"),
            )
            if not resolved_aspect_ratio or resolved_aspect_ratio == "1:1":
                ys, xs = torch.nonzero(region_mask[0] > 0, as_tuple=True)
                if len(xs) > 0 and len(ys) > 0:
                    bbox_w = int(xs.max().item() - xs.min().item() + 1)
                    bbox_h = int(ys.max().item() - ys.min().item() + 1)
                    resolved_aspect_ratio = _aspect_ratio_from_bbox_dims(bbox_w, bbox_h)
                    aspect_ratio_source = "mask_bbox"
            if aspect_ratio_source == "manual":
                aspect_ratio_source = "region_info" if region_context.get("region_type") or region_context.get("recommended_aspect_ratio") else "default"
        nb2_w, nb2_h = NB2_RESOLUTIONS[resolved_aspect_ratio][resolution]
        target_ar = nb2_w / nb2_h

        mask_out = torch.zeros(B, H, W, dtype=torch.float32)
        previews = []
        infos = []
        preview_ui = []

        center_x_out = []
        center_y_out = []
        crop_w_out = []
        crop_h_out = []

        for i in range(B):
            x1, y1, cw, ch = _fit_nb2_rect_to_mask(
                region_mask[i], W, H, target_ar, padding_percent, crop_scale
            )
            mask_out[i, y1:y1 + ch, x1:x1 + cw] = 1.0

            center_x = x1 + cw // 2
            center_y = y1 + ch // 2

            center_x_out.append(int(center_x))
            center_y_out.append(int(center_y))
            crop_w_out.append(int(cw))
            crop_h_out.append(int(ch))

            preview_tensor, temp_info = _make_nb2_preview(image[i], y1, x1, ch, cw)
            previews.append(preview_tensor.squeeze(0))
            if i == 0 and temp_info:
                preview_ui.append(temp_info)

            infos.append({
                "requested_aspect_ratio": aspect_ratio,
                "resolved_aspect_ratio": resolved_aspect_ratio,
                "aspect_ratio_source": aspect_ratio_source,
                "resolution": resolution,
                "padding_percent": padding_percent,
                "crop_scale": crop_scale,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x1 + cw),
                "y2": int(y1 + ch),
                "center_x": int(center_x),
                "center_y": int(center_y),
                "crop_width": int(cw),
                "crop_height": int(ch),
                "nb2_width": int(nb2_w),
                "nb2_height": int(nb2_h),
                "mask_note": mask_note,
            })

        preview_batch = torch.stack(previews, dim=0)
        result = {
            "result": (
                mask_out,
                nb2_w,
                nb2_h,
                preview_batch,
                center_x_out[0],
                center_y_out[0],
                crop_w_out[0],
                crop_h_out[0],
                str(infos[0] if len(infos) == 1 else {
                    "batch_count": len(infos),
                    "first": infos[0],
                }),
            )
        }
        if preview_ui:
            result["ui"] = {"nb2_preview": preview_ui}
        return result


class SmartMaskCrop:
    """
    Crop a local masked edit region so mask-based editors work on a focused area
    instead of the whole image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "context_expand": ("FLOAT", {
                    "default": 1.15, "min": 1.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Grow the detected mask region before cropping."}),
                "use_region_guidance": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use Florence region metadata to override context and target size when available.",
                }),
                "mask_expand_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Extra expansion applied to the edit mask after crop. 0 uses region defaults when guidance is enabled.",
                }),
                "mask_feather_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Softens the edit mask edges after crop. 0 uses region defaults when guidance is enabled.",
                }),
                "resize_mode": (["keep_local_size", "upscale_to_target_if_smaller", "resize_to_target"], {
                    "default": "upscale_to_target_if_smaller"}),
                "target_width": ("INT", {
                    "default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "target_height": ("INT", {
                    "default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos",
                                         "box", "hamming"], {"default": "bilinear"}),
                "upscale_algorithm":   (["nearest", "bilinear", "bicubic", "lanczos",
                                         "box", "hamming"], {"default": "bicubic"}),
                "device_mode": (["gpu (much faster)", "cpu (compatible)"],
                                {"default": "gpu (much faster)"}),
                "depad_florence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Remove Florence2's internal square letterbox padding before "
                        "resizing the mask. Keep True when the mask comes from "
                        "Florence2Run (kijai). Disable only if your mask is already "
                        "at the exact source image resolution."
                    )}),
                "use_region_mask_defaults": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When enabled, 0 mask expand/feather values use Florence "
                        "region defaults. Disable it when you need a hard mask."
                    )}),
            },
            "optional": {
                "region_info": ("STRING",),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "cropped_mask_image", "preview_image", "info")
    FUNCTION = "smart_mask_crop"
    CATEGORY = "inpaint/masked"
    DESCRIPTION = (
        "Crops a focused local region around a mask, keeping a local mask for "
        "mask-based editing models such as GPT Image."
    )

    def smart_mask_crop(self, image, mask, context_expand, use_region_guidance,
                        mask_expand_percent, mask_feather_percent, resize_mode,
                        target_width, target_height, downscale_algorithm,
                        upscale_algorithm, device_mode, depad_florence=True,
                        region_info="", use_region_mask_defaults=True):
        image = image.clone()
        mask = mask.clone()
        if device_mode == "gpu (much faster)":
            device = comfy.model_management.get_torch_device()
            image = image.to(device)
            mask = mask.to(device)
            processor = GPUProcessorLogic()
        else:
            device = torch.device("cpu")
            processor = CPUProcessorLogic()
        mask, image, mask_note = _normalize_mask_to_image(
            mask, image, processor, "SmartMaskCrop",
            depad_florence=depad_florence
        )
        region_context = _extract_region_context(region_info)
        context_expand_effective = float(context_expand)
        target_width_effective = int(target_width)
        target_height_effective = int(target_height)
        mask_expand_effective = float(mask_expand_percent)
        mask_feather_effective = float(mask_feather_percent)

        if use_region_guidance:
            if region_context.get("recommended_context_expand", 0.0) > 0.0:
                context_expand_effective = max(context_expand_effective, float(region_context["recommended_context_expand"]))
            if resize_mode == "resize_to_target" and region_context.get("recommended_edit_size"):
                target_width_effective, target_height_effective = _parse_edit_size(region_context["recommended_edit_size"])
            if (use_region_mask_defaults and mask_expand_effective <= 0.0
                    and region_context.get("recommended_mask_expand_percent", 0.0) > 0.0):
                mask_expand_effective = float(region_context["recommended_mask_expand_percent"])
            if (use_region_mask_defaults and mask_feather_effective <= 0.0
                    and region_context.get("recommended_mask_feather_percent", 0.0) > 0.0):
                mask_feather_effective = float(region_context["recommended_mask_feather_percent"])

        result_stitcher = {
            'downscale_algorithm': downscale_algorithm,
            'upscale_algorithm': upscale_algorithm,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
            'device_mode': device_mode,
        }
        result_image = []
        result_mask = []
        result_mask_image = []
        preview_ui = []
        previews = []
        infos = []

        batch_size = image.shape[0]
        for i in range(batch_size):
            sub_image = image[i:i+1]
            sub_mask = mask[i:i+1]
            image_h = sub_image.shape[1]
            image_w = sub_image.shape[2]

            _, bx, by, bw, bh = processor.batched_findcontextarea_m(sub_mask)
            if bx[0] == -1:
                raise ValueError("mask is empty; Smart Mask Crop requires a non-empty mask.")

            if context_expand_effective > 1.0:
                _, bx, by, bw, bh = processor.batched_growcontextarea_m(
                    sub_mask, bx, by, bw, bh, context_expand_effective
                )

            cur_x = bx[0].item()
            cur_y = by[0].item()
            cur_w = bw[0].item()
            cur_h = bh[0].item()

            target_ar = target_width_effective / max(1, target_height_effective)
            rect_x, rect_y, rect_w, rect_h = _fit_aspect_rect_to_bbox(
                cur_x, cur_y, cur_w, cur_h, image_w, image_h, target_ar
            )

            if resize_mode == "keep_local_size":
                out_w = max(1, int(rect_w))
                out_h = max(1, int(rect_h))
                resize_output = False
            elif resize_mode == "upscale_to_target_if_smaller":
                out_w = int(target_width_effective)
                out_h = int(target_height_effective)
                resize_output = rect_w < out_w or rect_h < out_h
            else:
                out_w = int(target_width_effective)
                out_h = int(target_height_effective)
                resize_output = True

            (canvas_image, cto_x, cto_y, cto_w, cto_h,
             cropped_image, cropped_mask,
             ctc_x, ctc_y, ctc_w, ctc_h) = processor.crop_magic_im(
                sub_image, sub_mask,
                rect_x, rect_y, rect_w, rect_h,
                rect_w, rect_h,
                0,
                downscale_algorithm, upscale_algorithm,
                resize_output=False)

            if resize_output:
                if out_w > ctc_w or out_h > ctc_h:
                    cropped_image = processor.rescale_i(cropped_image, out_w, out_h, upscale_algorithm)
                    cropped_mask = processor.rescale_m(cropped_mask, out_w, out_h, upscale_algorithm)
                else:
                    cropped_image = processor.rescale_i(cropped_image, out_w, out_h, downscale_algorithm)
                    cropped_mask = processor.rescale_m(cropped_mask, out_w, out_h, downscale_algorithm)
            else:
                out_w = int(ctc_w)
                out_h = int(ctc_h)

            result_stitcher['canvas_to_orig_x'].append(cto_x)
            result_stitcher['canvas_to_orig_y'].append(cto_y)
            result_stitcher['canvas_to_orig_w'].append(cto_w)
            result_stitcher['canvas_to_orig_h'].append(cto_h)
            result_stitcher['canvas_image'].append(canvas_image.cpu())
            result_stitcher['cropped_to_canvas_x'].append(ctc_x)
            result_stitcher['cropped_to_canvas_y'].append(ctc_y)
            result_stitcher['cropped_to_canvas_w'].append(ctc_w)
            result_stitcher['cropped_to_canvas_h'].append(ctc_h)
            edit_mask = _grow_and_feather_mask(
                cropped_mask.squeeze(0),
                mask_expand_effective,
                mask_feather_effective,
            ).unsqueeze(0)
            result_stitcher['cropped_mask_for_blend'].append(edit_mask.cpu())

            result_image.append(cropped_image.squeeze(0).cpu())
            result_mask.append(edit_mask.squeeze(0).cpu())
            mask_rgb = torch.stack([edit_mask.squeeze(0).cpu()] * 3, dim=-1)
            result_mask_image.append(mask_rgb)

            preview_tensor, temp_info = _make_nb2_preview(sub_image[0].cpu(), cur_y, cur_x, cur_h, cur_w)
            previews.append(preview_tensor.squeeze(0))
            if i == 0 and temp_info:
                preview_ui.append(temp_info)

            infos.append({
                "context_expand": context_expand_effective,
                "use_region_guidance": bool(use_region_guidance),
                "resize_mode": resize_mode,
                "original_crop_width": int(rect_w),
                "original_crop_height": int(rect_h),
                "target_width": int(out_w),
                "target_height": int(out_h),
                "mask_expand_percent": float(mask_expand_effective),
                "mask_feather_percent": float(mask_feather_effective),
                "use_region_mask_defaults": bool(use_region_mask_defaults),
                "mask_bbox_x": int(cur_x),
                "mask_bbox_y": int(cur_y),
                "mask_bbox_w": int(cur_w),
                "mask_bbox_h": int(cur_h),
                "crop_canvas_x": int(ctc_x),
                "crop_canvas_y": int(ctc_y),
                "crop_canvas_w": int(ctc_w),
                "crop_canvas_h": int(ctc_h),
                "mask_note": mask_note,
            })

        result = {
            "result": (
                result_stitcher,
                torch.stack(result_image, dim=0),
                torch.stack(result_mask, dim=0),
                torch.stack(result_mask_image, dim=0),
                torch.stack(previews, dim=0),
                str(infos[0] if len(infos) == 1 else {"batch_count": len(infos), "first": infos[0]}),
            )
        }
        if preview_ui:
            result["ui"] = {"nb2_preview": preview_ui}
        return result


class SmartMaskStitch:
    """
    Stitch a locally edited masked crop back into the original image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "edited_image": ("IMAGE",),
                "edge_feather_percent": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": "Extra feather applied to the local crop edge."}),
                "result_mask_feather_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Softens the stored local mask used to blend the edited result back."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "smart_mask_stitch"
    CATEGORY = "inpaint/masked"
    DESCRIPTION = (
        "Stitches a locally edited masked crop back into the original image "
        "using the stored local mask as the primary blend."
    )

    def smart_mask_stitch(self, stitcher, edited_image, edge_feather_percent,
                          result_mask_feather_percent=0.0):
        edited_image = edited_image.clone()

        device_mode = stitcher.get('device_mode', 'cpu (compatible)')
        if device_mode == "gpu (much faster)":
            device = comfy.model_management.get_torch_device()
            edited_image = edited_image.to(device)
            processor = GPUProcessorLogic()
        else:
            device = torch.device("cpu")
            processor = CPUProcessorLogic()

        downscale_algorithm = stitcher['downscale_algorithm']
        upscale_algorithm = stitcher['upscale_algorithm']
        canvas_images = [t.to(device) if torch.is_tensor(t) else t
                         for t in stitcher['canvas_image']]
        blend_masks = [t.to(device) if torch.is_tensor(t) else t
                       for t in stitcher['cropped_mask_for_blend']]

        batch_size = edited_image.shape[0]
        n_stitchers = len(stitcher['cropped_to_canvas_x'])
        assert n_stitchers == batch_size or n_stitchers == 1, \
            "Stitch batch size doesn't match image batch size"
        override = (n_stitchers == 1 and batch_size > 1)

        results = []
        for i in range(batch_size):
            idx = 0 if override else i
            one_image = edited_image[i:i+1]
            one_mask = blend_masks[idx]
            if one_mask.ndim == 2:
                one_mask = one_mask.unsqueeze(0)

            ctc_x = stitcher['cropped_to_canvas_x'][idx]
            ctc_y = stitcher['cropped_to_canvas_y'][idx]
            ctc_w = stitcher['cropped_to_canvas_w'][idx]
            ctc_h = stitcher['cropped_to_canvas_h'][idx]
            cto_x = stitcher['canvas_to_orig_x'][idx]
            cto_y = stitcher['canvas_to_orig_y'][idx]
            cto_w = stitcher['canvas_to_orig_w'][idx]
            cto_h = stitcher['canvas_to_orig_h'][idx]
            canvas = canvas_images[idx].clone()

            out = self._stitch_single(
                canvas, one_image, one_mask,
                ctc_x, ctc_y, ctc_w, ctc_h,
                cto_x, cto_y, cto_w, cto_h,
                downscale_algorithm, upscale_algorithm, edge_feather_percent,
                result_mask_feather_percent, device, processor
            )
            results.append(out.squeeze(0))

        return (torch.stack(results, dim=0).cpu(),)

    def _stitch_single(self, canvas_image, edited_image, local_mask,
                       ctc_x, ctc_y, ctc_w, ctc_h,
                       cto_x, cto_y, cto_w, cto_h,
                       downscale_algo, upscale_algo,
                       edge_feather_percent, result_mask_feather_percent,
                       device, processor):
        canvas_image = canvas_image.clone()

        n_channels = edited_image.shape[-1]
        if n_channels == 4:
            alpha_raw = edited_image[..., 3:4]
            rgb = edited_image[..., :3]
        else:
            alpha_raw = None
            rgb = edited_image

        B, h, w, _ = rgb.shape
        if ctc_w > w or ctc_h > h:
            resized_rgb = processor.rescale_i(rgb, ctc_w, ctc_h, upscale_algo)
            resized_mask = processor.rescale_m(local_mask, ctc_w, ctc_h, upscale_algo)
        else:
            resized_rgb = processor.rescale_i(rgb, ctc_w, ctc_h, downscale_algo)
            resized_mask = processor.rescale_m(local_mask, ctc_w, ctc_h, downscale_algo)

        resized_mask = resized_mask.clamp(0, 1)
        if result_mask_feather_percent > 0.0:
            resized_mask = torch.stack([
                _soft_blur_mask(resized_mask[j], result_mask_feather_percent)
                for j in range(resized_mask.shape[0])
            ], dim=0)

        feather_h_px = int(ctc_h * edge_feather_percent / 100.0)
        feather_w_px = int(ctc_w * edge_feather_percent / 100.0)
        feather = make_smoothstep_feather(ctc_h, ctc_w, feather_h_px, feather_w_px, device)
        blend_mask = resized_mask * feather.unsqueeze(0)

        if alpha_raw is not None:
            alpha_m = alpha_raw.squeeze(-1)
            if ctc_w > alpha_m.shape[2] or ctc_h > alpha_m.shape[1]:
                resized_alpha = processor.rescale_m(alpha_m, ctc_w, ctc_h, upscale_algo)
            else:
                resized_alpha = processor.rescale_m(alpha_m, ctc_w, ctc_h, downscale_algo)
            blend_mask = blend_mask * resized_alpha.clamp(0, 1)

        blend_mask = blend_mask.unsqueeze(-1)

        canvas_crop_full = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
        canvas_crop_rgb = canvas_crop_full[..., :3]
        blended_rgb = blend_mask * resized_rgb + (1.0 - blend_mask) * canvas_crop_rgb

        if canvas_image.shape[-1] == 4:
            canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = torch.cat(
                [blended_rgb, canvas_crop_full[..., 3:4]], dim=-1
            )
        else:
            canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended_rgb

        return canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :3]


# ===========================================================================
#  NEW NODE 2 — InpaintCropNB2
# ===========================================================================

class InpaintCropNB2:
    """
    Crops the image around the mask and scales the result to the exact
    resolution expected / produced by Nano Banana 2.

    Select the same aspect_ratio and resolution that you will use in
    NanoBanana2MaskGen.  The node computes nb2_width / nb2_height internally
    from those dropdowns — no INT wiring required.

    The STITCHER output carries all information needed by InpaintStitchNB2
    to composite the generated image back onto the original canvas.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {
                    "default": "16:9",
                    "tooltip": "Must match the aspect_ratio set in NB2 Mask Generator."}),
                "resolution":   (["1K", "2K", "4K"], {
                    "default": "2K",
                    "tooltip": "Must match the resolution set in NB2 Mask Generator."}),
                "context_extend_factor": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 4.0, "step": 0.01,
                    "tooltip": ("Grow the crop region by this factor in every direction before "
                                "aspect-ratio adjustment.  1.0 = crop exactly the masked area.")}),
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos",
                                         "box", "hamming"], {"default": "bilinear"}),
                "upscale_algorithm":   (["nearest", "bilinear", "bicubic", "lanczos",
                                         "box", "hamming"], {"default": "bicubic"}),
                "device_mode": (["gpu (much faster)", "cpu (compatible)"],
                                {"default": "gpu (much faster)"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES  = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES  = ("stitcher",  "cropped_image", "cropped_mask")
    FUNCTION      = "inpaint_crop_nb2"
    CATEGORY      = "inpaint/nb2"
    DESCRIPTION   = (
        "Crops the image to the mask region and scales to the exact Nano "
        "Banana 2 resolution.  Set aspect_ratio and resolution to match "
        "NB2 Mask Generator.  Pair with InpaintStitchNB2 after generation."
    )

    def inpaint_crop_nb2(self, image, aspect_ratio, resolution,
                         context_extend_factor,
                         downscale_algorithm, upscale_algorithm,
                         device_mode, mask=None):
        nb2_width, nb2_height = NB2_RESOLUTIONS[aspect_ratio][resolution]
        image = image.clone()
        if mask is not None:
            mask = mask.clone()

        if device_mode == "gpu (much faster)":
            device = comfy.model_management.get_torch_device()
            image = image.to(device)
            if mask is not None:
                mask = mask.to(device)
            processor = GPUProcessorLogic()
        else:
            device = torch.device("cpu")
            processor = CPUProcessorLogic()

        # Handle missing mask — use a full-coverage mask
        if mask is None:
            mask = torch.ones_like(image[:, :, :, 0])

        mask, image, _mask_note = _normalize_mask_to_image(
            mask, image, processor, "InpaintCropNB2"
        )

        assert image.ndim == 4, f"Expected 4D image tensor, got {image.shape}"
        assert mask.ndim  == 3, f"Expected 3D mask tensor,  got {mask.shape}"

        result_stitcher = {
            'downscale_algorithm':  downscale_algorithm,
            'upscale_algorithm':    upscale_algorithm,
            'blend_pixels':         0,          # feather handled in stitch
            'canvas_to_orig_x':    [],
            'canvas_to_orig_y':    [],
            'canvas_to_orig_w':    [],
            'canvas_to_orig_h':    [],
            'canvas_image':        [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
            'device_mode':         device_mode,
        }
        result_image = []
        result_mask  = []

        batch_size = image.shape[0]
        for i in range(batch_size):
            sub_image = image[i:i+1]
            sub_mask  = mask[i:i+1]

            # Locate the masked area
            _, bx, by, bw, bh = processor.batched_findcontextarea_m(sub_mask)
            if bx[0] == -1:                         # empty mask → full image
                bx[0], by[0] = 0, 0
                bw[0] = sub_image.shape[2]
                bh[0] = sub_image.shape[1]

            # Optional context expansion
            if context_extend_factor > 1.01:
                _, bx, by, bw, bh = processor.batched_growcontextarea_m(
                    sub_mask, bx, by, bw, bh, context_extend_factor)

            cur_x = bx[0].item(); cur_y = by[0].item()
            cur_w = bw[0].item(); cur_h = bh[0].item()

            # Crop and scale to exact NB2 resolution
            (canvas_image, cto_x, cto_y, cto_w, cto_h,
             cropped_image, cropped_mask,
             ctc_x, ctc_y, ctc_w, ctc_h) = processor.crop_magic_im(
                sub_image, sub_mask,
                cur_x, cur_y, cur_w, cur_h,
                nb2_width, nb2_height,
                0,                          # no padding — keep exact NB2 dims
                downscale_algorithm, upscale_algorithm,
                resize_output=True)

            result_stitcher['canvas_to_orig_x'].append(cto_x)
            result_stitcher['canvas_to_orig_y'].append(cto_y)
            result_stitcher['canvas_to_orig_w'].append(cto_w)
            result_stitcher['canvas_to_orig_h'].append(cto_h)
            result_stitcher['canvas_image'].append(canvas_image.cpu())
            result_stitcher['cropped_to_canvas_x'].append(ctc_x)
            result_stitcher['cropped_to_canvas_y'].append(ctc_y)
            result_stitcher['cropped_to_canvas_w'].append(ctc_w)
            result_stitcher['cropped_to_canvas_h'].append(ctc_h)
            result_stitcher['cropped_mask_for_blend'].append(
                torch.ones(1, ctc_h, ctc_w, dtype=torch.float32))   # full coverage

            result_image.append(cropped_image.squeeze(0).cpu())
            result_mask.append(cropped_mask.squeeze(0).cpu())

        result_image = torch.stack(result_image, dim=0)
        result_mask  = torch.stack(result_mask,  dim=0)

        return (result_stitcher, result_image, result_mask)


# ===========================================================================
#  NEW NODE 3 — InpaintStitchNB2
# ===========================================================================

class InpaintStitchNB2:
    """
    Composites a Nano Banana 2 generated image back onto the original canvas.

    Key additions over the original Inpaint Stitch node
    ----------------------------------------------------
    edge_feather_percent
        A smoothstep gradient (0 → 1) is applied from each edge of the
        generated region inward.  The gradient width equals this percentage
        of the crop dimension (e.g. 5 % of 2752 px ≈ 138 px on each side).
        Set to 0 to disable feathering entirely.

    Alpha-channel support
        If the generated image has four channels (RGBA) the alpha channel is
        extracted and multiplied with the feather mask before blending.
        This lets downstream nodes (e.g. a Feather / Matte node) pre-compute
        soft edges and pass them through without needing a separate mask wire.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher":        ("STITCHER",),
                "inpainted_image": ("IMAGE",),
                "edge_feather_percent": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": ("Percentage of each crop edge to feather (smoothstep). "
                                "E.g. 5 means 5 %% of width / height on each side. "
                                "Set to 0 for a hard cut.")}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "inpaint_stitch_nb2"
    CATEGORY      = "inpaint/nb2"
    DESCRIPTION   = (
        "Composites the Nano Banana 2 output back onto the original image "
        "with smoothstep edge feathering and optional alpha-channel blending."
    )

    def inpaint_stitch_nb2(self, stitcher, inpainted_image,
                           edge_feather_percent):
        inpainted_image = inpainted_image.clone()

        device_mode = stitcher.get('device_mode', 'cpu (compatible)')
        if device_mode == "gpu (much faster)":
            device = comfy.model_management.get_torch_device()
            inpainted_image = inpainted_image.to(device)
            processor = GPUProcessorLogic()
        else:
            device = torch.device("cpu")
            processor = CPUProcessorLogic()

        downscale_algorithm = stitcher['downscale_algorithm']
        upscale_algorithm   = stitcher['upscale_algorithm']

        # Move canvas tensors to the working device
        canvas_images = [t.to(device) if torch.is_tensor(t) else t
                         for t in stitcher['canvas_image']]

        batch_size  = inpainted_image.shape[0]
        n_stitchers = len(stitcher['cropped_to_canvas_x'])
        assert n_stitchers == batch_size or n_stitchers == 1, \
            "Stitch batch size doesn't match image batch size"
        override = (n_stitchers == 1 and batch_size > 1)

        results = []
        for i in range(batch_size):
            idx = 0 if override else i
            one_image = inpainted_image[i:i+1]

            ctc_x = stitcher['cropped_to_canvas_x'][idx]
            ctc_y = stitcher['cropped_to_canvas_y'][idx]
            ctc_w = stitcher['cropped_to_canvas_w'][idx]
            ctc_h = stitcher['cropped_to_canvas_h'][idx]
            cto_x = stitcher['canvas_to_orig_x'][idx]
            cto_y = stitcher['canvas_to_orig_y'][idx]
            cto_w = stitcher['canvas_to_orig_w'][idx]
            cto_h = stitcher['canvas_to_orig_h'][idx]
            canvas = canvas_images[idx].clone()

            out = self._stitch_single(
                canvas, one_image,
                ctc_x, ctc_y, ctc_w, ctc_h,
                cto_x, cto_y, cto_w, cto_h,
                downscale_algorithm, upscale_algorithm,
                edge_feather_percent, device, processor)
            results.append(out.squeeze(0))

        result_batch = torch.stack(results, dim=0).cpu()
        return (result_batch,)

    # ------------------------------------------------------------------

    def _stitch_single(self, canvas_image, inpainted_image,
                       ctc_x, ctc_y, ctc_w, ctc_h,
                       cto_x, cto_y, cto_w, cto_h,
                       downscale_algo, upscale_algo,
                       edge_feather_percent, device, processor):
        """
        Composite one generated image onto the canvas with feather + alpha.

        inpainted_image: [1, H, W, 3 or 4]
          3 channels → RGB  (feather mask only)
          4 channels → RGBA (feather mask × alpha channel)
        """
        canvas_image = canvas_image.clone()

        # --- split alpha if present ---
        n_channels = inpainted_image.shape[-1]
        if n_channels == 4:
            alpha_raw = inpainted_image[..., 3:4]   # [1, H, W, 1]
            rgb       = inpainted_image[..., :3]
        else:
            alpha_raw = None
            rgb       = inpainted_image

        # --- scale RGB to crop dimensions ---
        B, h, w, _ = rgb.shape
        if ctc_w > w or ctc_h > h:
            resized_rgb = processor.rescale_i(rgb, ctc_w, ctc_h, upscale_algo)
        else:
            resized_rgb = processor.rescale_i(rgb, ctc_w, ctc_h, downscale_algo)

        # --- build edge-feather mask [ctc_h, ctc_w] ---
        feather_h_px = int(ctc_h * edge_feather_percent / 100.0)
        feather_w_px = int(ctc_w * edge_feather_percent / 100.0)
        feather = make_smoothstep_feather(
            ctc_h, ctc_w, feather_h_px, feather_w_px, device)  # [H, W]

        # --- blend mask: alpha takes priority, feather is the fallback ---
        if alpha_raw is not None:
            # The image already has a pre-computed alpha (e.g. from NB2AddAlpha).
            # Use it directly as the blend mask — do NOT multiply with the stitch
            # feather, because that would make even the centre semi-transparent
            # and shift the visual position of the composite.
            alpha_m = alpha_raw.squeeze(-1)   # [1, H, W]
            if ctc_w > alpha_m.shape[2] or ctc_h > alpha_m.shape[1]:
                resized_alpha = processor.rescale_m(alpha_m, ctc_w, ctc_h, upscale_algo)
            else:
                resized_alpha = processor.rescale_m(alpha_m, ctc_w, ctc_h, downscale_algo)
            blend_mask = resized_alpha.clamp(0, 1)                           # [1, H, W]
        else:
            # No alpha supplied → use the stitch's own edge feather.
            blend_mask = feather.unsqueeze(0).expand(B, -1, -1).clone()     # [B, H, W]

        # [B, H, W, 1] for broadcasting with RGB
        blend_mask = blend_mask.unsqueeze(-1)

        # --- composite ---
        # canvas_image may be RGBA if the original input had alpha.
        # Always blend in RGB space to avoid channel-count mismatches,
        # then output clean RGB regardless of input format.
        canvas_crop_full = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
        canvas_crop_rgb  = canvas_crop_full[..., :3]

        blended_rgb = blend_mask * resized_rgb + (1.0 - blend_mask) * canvas_crop_rgb

        # Write blended RGB back; keep canvas alpha channel if it existed
        if canvas_image.shape[-1] == 4:
            canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = torch.cat(
                [blended_rgb, canvas_crop_full[..., 3:4]], dim=-1)
        else:
            canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended_rgb

        # Always return RGB — downstream nodes don't expect alpha from a stitch
        return canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :3]


# ===========================================================================
#  NEW NODE 4 — NB2AddAlpha
# ===========================================================================

class NB2AddAlpha:
    """
    Converts an RGB image to RGBA by generating a feathered alpha channel.

    The alpha is a smoothstep gradient: 0 at every edge, 1 in the centre.
    Use this after Nano Banana 2 generation when you want a compositable
    layer with soft edges — pipe the RGBA into NB2 Stitch or any compositor.

    feather_percent controls the ramp width as a percentage of each
    dimension (e.g. 5 → 5 % of width on left & right, 5 % of height on
    top & bottom).  Set to 0 for a hard rectangular alpha.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "feather_percent": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": ("Width of the edge fade as %% of image size. "
                                "0 = hard edges, 50 = fully fades to centre.")}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("rgba_image",)
    FUNCTION      = "add_alpha"
    CATEGORY      = "inpaint/nb2"
    DESCRIPTION   = (
        "Adds a smoothstep feathered alpha channel to an RGB image.  "
        "Useful to pre-compute soft edges on the NB2 output before stitching."
    )

    def add_alpha(self, image, feather_percent):
        B, H, W, _C = image.shape
        rgb = image[..., :3]   # ensure we work in RGB even if input is RGBA

        fh = int(H * feather_percent / 100.0)
        fw = int(W * feather_percent / 100.0)
        feather = make_smoothstep_feather(H, W, fh, fw, image.device)  # [H, W]

        # Broadcast to [B, H, W, 1]
        alpha = feather.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)

        rgba = torch.cat([rgb, alpha], dim=-1)  # [B, H, W, 4]
        return (rgba,)


# ===========================================================================
#  NEW NODE 5 — Florence-2 Smart Region Selector (FAL API)
# ===========================================================================

class NB2Florence2RegionSelector:
    """
    Select a semantic region through the external Florence-2 FAL API.

    The API key is never stored in code. Users can either:
    - paste it into the `api_key` input for the current session, or
    - leave `api_key` blank and provide it via an environment variable
      such as FAL_KEY.
    """

    REGION_TYPE_OPTIONS = ["glasses", "face", "upper_body", "lower_body", "full_body", "object"]
    SELECTION_MODE_OPTIONS = ["largest", "merge_all"]
    REGION_QUERY_MAP = {
        "glasses": "eyeglasses",
        "face": "face",
        "upper_body": "upper body",
        "lower_body": "lower body",
        "full_body": "full body person",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region_type": (cls.REGION_TYPE_OPTIONS, {"default": "face"}),
            },
            "optional": {
                "custom_text": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Only used when region_type is object",
                    },
                ),
                "selection_mode": (
                    cls.SELECTION_MODE_OPTIONS,
                    {"default": "largest"},
                ),
                "padding_percent": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                    },
                ),
                "return_rect_mask": ("BOOLEAN", {"default": False}),
                "api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Optional. Leave blank to use FAL_KEY",
                    },
                ),
                "api_key_env_var": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "FAL_KEY",
                        "placeholder": "Environment variable fallback",
                    },
                ),
                "mask_blur_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Soft blur applied to the returned Florence mask.",
                    },
                ),
                "upload_max_dimension": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 512,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Downscale longest image edge before upload. Lower this if FAL closes the connection.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "mask",
        "mask_image",
        "info",
        "center_x",
        "center_y",
        "crop_width",
        "crop_height",
    )
    FUNCTION = "select_region"
    CATEGORY = "inpaint/api"
    DESCRIPTION = (
        "Select one semantic region at a time using Florence-2 through the FAL API. "
        "API key can be provided by input or environment variable."
    )

    def _coerce_text(self, value):
        if value is None or isinstance(value, bool):
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _looks_like_api_key(self, value):
        candidate = self._coerce_text(value)
        if not candidate:
            return False
        if len(candidate) < 24:
            return False
        return ":" in candidate or candidate.startswith("fal_")

    def _looks_like_env_var_name(self, value):
        candidate = self._coerce_text(value)
        if not candidate:
            return False
        if ":" in candidate or any(ch.isspace() for ch in candidate):
            return False
        return candidate.replace("_", "a").isalnum()

    def _resolve_api_key(self, api_key, api_key_env_var):
        direct_key = self._coerce_text(api_key)
        if direct_key:
            if self._looks_like_env_var_name(direct_key) and not self._looks_like_api_key(direct_key):
                env_key = os.getenv(direct_key, "").strip()
                if env_key:
                    logger.warning(
                        "Florence node received an environment variable name in api_key; resolving it from the environment."
                    )
                    return env_key, f"environment:{direct_key}"
            return direct_key, "direct_input"

        env_name = self._coerce_text(api_key_env_var) or "FAL_KEY"

        # If the key was pasted into the env-var field by mistake, treat it as
        # the key directly instead of leaking it back in an error message.
        if self._looks_like_api_key(env_name):
            logger.warning(
                "Florence node received an API key in api_key_env_var; using it as a direct key."
            )
            return env_name, "api_key_env_var_direct_input"

        if not self._looks_like_env_var_name(env_name):
            logger.warning(
                "Florence node received an invalid api_key_env_var value %r; falling back to FAL_KEY.",
                api_key_env_var,
            )
            env_name = "FAL_KEY"

        env_key = os.getenv(env_name, "").strip()
        if env_key:
            return env_key, f"environment:{env_name}"

        raise ValueError(
            f"Missing FAL API key. Paste it into api_key or set the environment variable {env_name}."
        )

    def _get_fal_client(self):
        try:
            import fal_client
        except ImportError as e:
            raise RuntimeError(
                "fal-client is not installed. Install it in ComfyUI's Python environment."
            ) from e
        return fal_client

    def _is_retryable_network_error(self, error):
        text = str(error).lower()
        retry_markers = (
            "winerror 10054",
            "forcibly closed",
            "connection reset",
            "connection aborted",
            "remote host",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "500",
            "internal server error",
            "502",
            "503",
            "504",
        )
        return any(marker in text for marker in retry_markers)

    def _with_retries(self, label, operation, attempts=3):
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if attempt >= attempts or not self._is_retryable_network_error(e):
                    raise
                sleep_seconds = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Florence %s failed on attempt %s/%s: %s. Retrying in %ss.",
                    label,
                    attempt,
                    attempts,
                    str(e),
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise last_error

    def _normalize_image_array(self, image):
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = np.asarray(image)

        if image_np.ndim == 4 and image_np.shape[0] == 1:
            image_np = image_np[0]
        elif image_np.ndim == 3 and image_np.shape[0] in (3, 4):
            image_np = np.transpose(image_np, (1, 2, 0))

        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
            else:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]

        return image_np

    def _build_query(self, region_type, custom_text):
        custom_text = (custom_text or "").strip()
        if region_type == "object":
            if not custom_text:
                raise ValueError("custom_text is required when region_type is object.")
            return custom_text

        if custom_text:
            raise ValueError("custom_text can only be used when region_type is object.")

        return self.REGION_QUERY_MAP[region_type]

    def _prepare_image_for_upload(self, image_tensor, max_dimension=None):
        image_np = self._normalize_image_array(image_tensor)
        image = Image.fromarray(image_np)
        original_size = image.size

        if max_dimension is not None:
            width, height = image.size
            longest_edge = max(width, height)
            if longest_edge > max_dimension:
                scale = max_dimension / float(longest_edge)
                resized_size = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
                image = image.resize(resized_size, Image.LANCZOS)
                logger.info(
                    "Downscaled image before upload from %sx%s to %sx%s",
                    width,
                    height,
                    resized_size[0],
                    resized_size[1],
                )

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return buffered.getvalue(), original_size, image.size

    def _upload_image(self, image_tensor, api_key, max_dimension=None):
        fal_client = self._get_fal_client()
        previous_key = os.environ.get("FAL_KEY")
        os.environ["FAL_KEY"] = api_key
        try:
            img_bytes, original_size, uploaded_size = self._prepare_image_for_upload(
                image_tensor,
                max_dimension=max_dimension,
            )
            image_url = self._with_retries(
                "image upload",
                lambda: fal_client.upload(
                    img_bytes,
                    "image/png",
                ),
            )
            return image_url, original_size, uploaded_size
        finally:
            if previous_key is None:
                os.environ.pop("FAL_KEY", None)
            else:
                os.environ["FAL_KEY"] = previous_key

    def _call_api(self, endpoint, arguments, api_key):
        fal_client = self._get_fal_client()
        previous_key = os.environ.get("FAL_KEY")
        os.environ["FAL_KEY"] = api_key
        try:
            result = self._with_retries(
                f"API call {endpoint}",
                lambda: fal_client.run(endpoint, arguments=arguments),
            )
            logger.debug("FAL API response from %s: %s", endpoint, json.dumps(result))
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to call FAL endpoint {endpoint}: {str(e)}") from e
        finally:
            if previous_key is None:
                os.environ.pop("FAL_KEY", None)
            else:
                os.environ["FAL_KEY"] = previous_key

    def _is_point_pair(self, value):
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and all(isinstance(v, (int, float)) for v in value[:2])
        )

    def _extract_points_recursive(self, value):
        if isinstance(value, dict):
            if "x" in value and "y" in value:
                x = value["x"]
                y = value["y"]
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    return [(float(x), float(y))]
            for nested in value.values():
                points = self._extract_points_recursive(nested)
                if points and len(points) >= 3:
                    return points
            return None

        if isinstance(value, (list, tuple)):
            if len(value) >= 3 and all(self._is_point_pair(item) for item in value):
                return [(float(item[0]), float(item[1])) for item in value]

            collected = []
            for item in value:
                points = self._extract_points_recursive(item)
                if points and len(points) >= 3:
                    return points
                if points and len(points) == 1:
                    collected.extend(points)
            if len(collected) >= 3:
                return collected

        return None

    def _extract_bbox(self, value):
        if isinstance(value, dict):
            if all(key in value for key in ("x1", "y1", "x2", "y2")):
                coords = (value["x1"], value["y1"], value["x2"], value["y2"])
                if all(isinstance(v, (int, float)) for v in coords):
                    return tuple(float(v) for v in coords)
            if all(key in value for key in ("xmin", "ymin", "xmax", "ymax")):
                coords = (value["xmin"], value["ymin"], value["xmax"], value["ymax"])
                if all(isinstance(v, (int, float)) for v in coords):
                    return tuple(float(v) for v in coords)
            if "bbox" in value:
                bbox = self._extract_bbox(value["bbox"])
                if bbox:
                    return bbox
            for nested in value.values():
                bbox = self._extract_bbox(nested)
                if bbox:
                    return bbox
            return None

        if isinstance(value, (list, tuple)) and len(value) >= 4:
            if all(isinstance(v, (int, float)) for v in value[:4]):
                x1, y1, x2, y2 = [float(v) for v in value[:4]]
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2, y2)

        return None

    def _polygon_area(self, points):
        pts = list(points)
        if len(pts) < 3:
            return 0.0
        area = 0.0
        for index, (x1, y1) in enumerate(pts):
            x2, y2 = pts[(index + 1) % len(pts)]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _coerce_polygon_entries(self, result):
        polygons = result.get("results", {}).get("polygons", [])
        output = []
        for entry in polygons:
            points = self._extract_points_recursive(entry)
            if points and len(points) >= 3:
                output.append(points)
        return output

    def _coerce_bbox_entries(self, result):
        bboxes = result.get("results", {}).get("bboxes", [])
        output = []
        for entry in bboxes:
            bbox = self._extract_bbox(entry)
            if bbox:
                output.append(bbox)
        return output

    def _scale_polygons_if_needed(
        self,
        polygons,
        original_size,
        uploaded_size,
    ):
        if not polygons:
            return polygons

        original_w, original_h = original_size
        uploaded_w, uploaded_h = uploaded_size
        if (original_w, original_h) == (uploaded_w, uploaded_h):
            return polygons

        max_x = max(point[0] for polygon in polygons for point in polygon)
        max_y = max(point[1] for polygon in polygons for point in polygon)
        if max_x > uploaded_w + 1 or max_y > uploaded_h + 1:
            return polygons

        scale_x = original_w / float(uploaded_w)
        scale_y = original_h / float(uploaded_h)
        logger.info(
            "Scaling Florence polygon coordinates from uploaded size %sx%s back to original size %sx%s",
            uploaded_w,
            uploaded_h,
            original_w,
            original_h,
        )
        return [
            [(x * scale_x, y * scale_y) for x, y in polygon]
            for polygon in polygons
        ]

    def _scale_bboxes_if_needed(
        self,
        bboxes,
        original_size,
        uploaded_size,
    ):
        if not bboxes:
            return bboxes

        original_w, original_h = original_size
        uploaded_w, uploaded_h = uploaded_size
        if (original_w, original_h) == (uploaded_w, uploaded_h):
            return bboxes

        max_x = max(bbox[2] for bbox in bboxes)
        max_y = max(bbox[3] for bbox in bboxes)
        if max_x > uploaded_w + 1 or max_y > uploaded_h + 1:
            return bboxes

        scale_x = original_w / float(uploaded_w)
        scale_y = original_h / float(uploaded_h)
        logger.info(
            "Scaling Florence bbox coordinates from uploaded size %sx%s back to original size %sx%s",
            uploaded_w,
            uploaded_h,
            original_w,
            original_h,
        )
        return [
            (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            for x1, y1, x2, y2 in bboxes
        ]

    def _render_mask_from_polygons(self, width, height, polygons, selection_mode):
        if selection_mode == "largest":
            polygons = [max(polygons, key=self._polygon_area)]

        mask_image = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask_image)
        for polygon in polygons:
            drawer.polygon(polygon, fill=255)
        return np.array(mask_image, dtype=np.uint8)

    def _render_mask_from_bboxes(self, width, height, bboxes, selection_mode):
        if selection_mode == "largest":
            bboxes = [max(bboxes, key=self._bbox_area)]

        mask_image = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask_image)
        for x1, y1, x2, y2 in bboxes:
            drawer.rectangle((x1, y1, x2, y2), fill=255)
        return np.array(mask_image, dtype=np.uint8)

    def _apply_padding(self, bbox, width, height, padding_percent):
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(round(box_w * (padding_percent / 100.0)))
        pad_y = int(round(box_h * (padding_percent / 100.0)))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    def _mask_bbox(self, mask_uint8):
        ys, xs = np.nonzero(mask_uint8 > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError("Florence did not return a usable region.")
        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )

    def _rect_mask_from_bbox(self, width, height, bbox):
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
        return mask

    def _mask_to_outputs(self, mask_uint8):
        mask_float = mask_uint8.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)
        mask_rgb = np.stack([mask_float] * 3, axis=-1)
        mask_image_tensor = torch.from_numpy(mask_rgb).unsqueeze(0)
        return mask_tensor, mask_image_tensor

    def _call_segmentation(self, image_url, query, api_key):
        return self._call_api(
            "fal-ai/florence-2-large/referring-expression-segmentation",
            {"image_url": image_url, "text_input": query},
            api_key,
        )

    def _call_grounding(self, image_url, query, api_key):
        return self._call_api(
            "fal-ai/florence-2-large/caption-to-phrase-grounding",
            {"image_url": image_url, "text_input": query},
            api_key,
        )

    def select_region(
        self,
        image,
        region_type,
        custom_text="",
        selection_mode="largest",
        padding_percent=8.0,
        return_rect_mask=False,
        api_key="",
        api_key_env_var="FAL_KEY",
        mask_blur_percent=0.0,
        upload_max_dimension=2048,
    ):
        try:
            if not isinstance(image, torch.Tensor):
                raise ValueError("image input must be a ComfyUI IMAGE tensor.")
            if image.ndim != 4:
                raise ValueError(
                    f"Expected IMAGE tensor with shape [B,H,W,C], got {tuple(image.shape)}."
                )
            if image.shape[0] != 1:
                raise ValueError(
                    "NB2Florence2RegionSelector currently supports batch size 1 only."
                )

            resolved_api_key, api_key_source = self._resolve_api_key(api_key, api_key_env_var)
            query = self._build_query(region_type, custom_text)
            image_url, original_size, uploaded_size = self._upload_image(
                image,
                resolved_api_key,
                max_dimension=int(upload_max_dimension),
            )
            image_np = self._normalize_image_array(image[0:1])
            height, width = image_np.shape[:2]

            logger.info(
                "Running Florence selector via FAL with region_type=%s query=%s",
                region_type,
                query,
            )

            mask_uint8 = None
            source = None

            segmentation_result = self._call_segmentation(image_url, query, resolved_api_key)
            polygons = self._coerce_polygon_entries(segmentation_result)
            polygons = self._scale_polygons_if_needed(
                polygons,
                original_size,
                uploaded_size,
            )
            if polygons:
                mask_uint8 = self._render_mask_from_polygons(
                    width, height, polygons, selection_mode
                )
                source = "referring-expression-segmentation"

            if mask_uint8 is None:
                grounding_result = self._call_grounding(image_url, query, resolved_api_key)
                bboxes = self._coerce_bbox_entries(grounding_result)
                bboxes = self._scale_bboxes_if_needed(
                    bboxes,
                    original_size,
                    uploaded_size,
                )
                if not bboxes:
                    raise RuntimeError(
                        "Florence returned no polygons and no bounding boxes for this region."
                    )
                mask_uint8 = self._render_mask_from_bboxes(
                    width, height, bboxes, selection_mode
                )
                source = "caption-to-phrase-grounding"

            bbox = self._mask_bbox(mask_uint8)
            padded_bbox = self._apply_padding(bbox, width, height, padding_percent)

            if return_rect_mask:
                output_mask_uint8 = self._rect_mask_from_bbox(width, height, padded_bbox)
            else:
                output_mask_uint8 = mask_uint8.copy()
            if mask_blur_percent > 0.0:
                output_mask_uint8 = _blur_uint8_mask(output_mask_uint8, mask_blur_percent)

            center_x = int(round((padded_bbox[0] + padded_bbox[2]) / 2.0))
            center_y = int(round((padded_bbox[1] + padded_bbox[3]) / 2.0))
            crop_width = int(padded_bbox[2] - padded_bbox[0])
            crop_height = int(padded_bbox[3] - padded_bbox[1])

            mask_tensor, mask_image_tensor = self._mask_to_outputs(output_mask_uint8)
            region_edit_hints = _get_region_edit_hints(region_type)
            recommended_aspect_ratio = region_edit_hints["aspect_ratio"] or _recommend_aspect_ratio_for_region(region_type, padded_bbox)
            info = {
                "region_type": region_type,
                "query": query,
                "source": source,
                "selection_mode": selection_mode,
                "padding_percent": padding_percent,
                "mask_blur_percent": float(mask_blur_percent),
                "upload_max_dimension": int(upload_max_dimension),
                "api_key_source": api_key_source,
                "recommended_aspect_ratio": recommended_aspect_ratio,
                "recommended_edit_size": region_edit_hints["edit_size"],
                "recommended_mask_expand_percent": float(region_edit_hints["mask_expand_percent"]),
                "recommended_mask_feather_percent": float(region_edit_hints["mask_feather_percent"]),
                "recommended_context_expand": float(region_edit_hints["context_expand"]),
                "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
                "uploaded_size": {"width": int(uploaded_size[0]), "height": int(uploaded_size[1])},
                "bbox": {
                    "x1": int(padded_bbox[0]),
                    "y1": int(padded_bbox[1]),
                    "x2": int(padded_bbox[2]),
                    "y2": int(padded_bbox[3]),
                },
                "center_x": center_x,
                "center_y": center_y,
                "crop_width": crop_width,
                "crop_height": crop_height,
            }

            return (
                mask_tensor,
                mask_image_tensor,
                json.dumps(info),
                center_x,
                center_y,
                crop_width,
                crop_height,
            )
        except Exception as e:
            error_summary = _summarize_remote_error(e)
            logger.error("Florence region selection failed: %s", error_summary)
            raise RuntimeError(f"Florence region selection failed: {error_summary}") from e


class NB2OpenAIImageEdit:
    """
    Edit an image with GPT Image 2 through FAL using an optional mask.
    """

    MODEL_OPTIONS = ["openai/gpt-image-2/edit"]
    QUALITY_OPTIONS = ["auto", "low", "medium", "high"]
    SIZE_MODE_OPTIONS = ["auto_from_input", "max_from_input_aspect", "preset", "custom", "auto_from_region", "manual"]
    SIZE_OPTIONS = [
        "auto",
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
        "1024x768",
        "1024x1024",
        "1024x1536",
        "1920x1080",
        "2560x1440",
        "3840x2160",
    ]
    BACKGROUND_OPTIONS = ["auto", "opaque", "transparent"]
    FORMAT_OPTIONS = ["png", "webp", "jpeg"]
    MODERATION_OPTIONS = ["auto", "low"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Retouch only the masked region. Preserve the rest of the image.",
                }),
                "model": (cls.MODEL_OPTIONS, {"default": "openai/gpt-image-2/edit"}),
                "quality": (cls.QUALITY_OPTIONS, {"default": "high"}),
                "size_mode": (cls.SIZE_MODE_OPTIONS, {"default": "auto_from_input"}),
                "size": (cls.SIZE_OPTIONS, {"default": "auto"}),
                "background": (cls.BACKGROUND_OPTIONS, {"default": "auto"}),
                "output_format": (cls.FORMAT_OPTIONS, {"default": "png"}),
                "output_compression": ("INT", {"default": 90, "min": 0, "max": 100, "step": 1}),
                "moderation": (cls.MODERATION_OPTIONS, {"default": "auto"}),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Optional. Leave blank to use FAL_KEY",
                }),
                "api_key_env_var": ("STRING", {
                    "multiline": False,
                    "default": "FAL_KEY",
                    "placeholder": "FAL environment variable fallback",
                }),
                "openai_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Optional. Leave blank to use OPENAI_API_KEY",
                }),
                "openai_api_key_env_var": ("STRING", {
                    "multiline": False,
                    "default": "OPENAI_API_KEY",
                    "placeholder": "OpenAI environment variable fallback",
                }),
            },
            "optional": {
                "mask_image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "region_info": ("STRING",),
                "custom_width": ("INT", {"default": 3840, "min": 512, "max": 3840, "step": 16}),
                "custom_height": ("INT", {"default": 2160, "min": 512, "max": 3840, "step": 16}),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "edit_image"
    CATEGORY = "inpaint/api"
    DESCRIPTION = (
        "Edits an image through FAL's GPT Image 2 edit endpoint. "
        "Supports optional masks and safe auto sizing from the input crop."
    )

    def _looks_like_openai_api_key(self, value):
        candidate = _coerce_text_value(value)
        if not candidate:
            return False
        if len(candidate) < 20:
            return False
        return candidate.startswith("sk-") or candidate.startswith("org-") or candidate.startswith("proj_")

    def _looks_like_fal_api_key(self, value):
        candidate = _coerce_text_value(value)
        if not candidate:
            return False
        if len(candidate) < 24:
            return False
        return ":" in candidate or candidate.startswith("fal_")

    def _looks_like_env_var_name(self, value):
        candidate = _coerce_text_value(value)
        if not candidate:
            return False
        if any(ch.isspace() for ch in candidate) or ":" in candidate:
            return False
        return candidate.replace("_", "a").isalnum()

    def _resolve_api_key(self, api_key, api_key_env_var, looks_like_key, default_env_name, label):
        direct_key = _coerce_text_value(api_key)
        if direct_key:
            if self._looks_like_env_var_name(direct_key) and not looks_like_key(direct_key):
                env_key = os.getenv(direct_key, "").strip()
                if env_key:
                    logger.warning(
                        "%s image node received an environment variable name in api_key; resolving it from the environment.",
                        label,
                    )
                    return env_key, f"environment:{direct_key}"
            return direct_key, "direct_input"

        env_name = _coerce_text_value(api_key_env_var) or default_env_name
        if looks_like_key(env_name):
            logger.warning(
                "%s image node received an API key in api_key_env_var; using it as a direct key.",
                label,
            )
            return env_name, "api_key_env_var_direct_input"

        if not self._looks_like_env_var_name(env_name):
            logger.warning(
                "%s image node received an invalid api_key_env_var value %r; falling back to %s.",
                label,
                api_key_env_var,
                default_env_name,
            )
            env_name = default_env_name

        env_key = os.getenv(env_name, "").strip()
        if env_key:
            return env_key, f"environment:{env_name}"

        raise ValueError(
            f"Missing {label} API key. Paste it into api_key or set the environment variable {env_name}."
        )

    def _image_tensor_to_png_bytes(self, image_tensor):
        image_np = NB2Florence2RegionSelector()._normalize_image_array(image_tensor)
        mode = "RGBA" if image_np.shape[-1] == 4 else "RGB"
        image = Image.fromarray(image_np[..., :4] if mode == "RGBA" else image_np[..., :3], mode=mode)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue(), image.size

    def _mask_tensor_to_png_bytes(self, mask_image, target_size):
        mask_np = NB2Florence2RegionSelector()._normalize_image_array(mask_image)
        if mask_np.shape[-1] >= 3:
            mask_gray = np.max(mask_np[..., :3], axis=-1).astype(np.uint8)
        else:
            mask_gray = mask_np[..., 0].astype(np.uint8)
        mask = Image.fromarray(mask_gray, mode="L")
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.NEAREST)
        mask_rgba = mask.convert("RGBA")
        mask_rgba.putalpha(mask)
        buf = io.BytesIO()
        mask_rgba.save(buf, format="PNG")
        return buf.getvalue()

    def _mask_bbox(self, mask_image):
        mask_np = NB2Florence2RegionSelector()._normalize_image_array(mask_image)
        if mask_np.shape[-1] >= 3:
            mask_gray = np.max(mask_np[..., :3], axis=-1)
        else:
            mask_gray = mask_np[..., 0]
        ys, xs = np.nonzero(mask_gray > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def _max_size_from_input_aspect(self, input_size):
        source_w, source_h = input_size
        source_w = max(1, int(source_w))
        source_h = max(1, int(source_h))
        max_edge = 3840
        max_pixels = 8294400

        scale = min(
            max_edge / float(source_w),
            max_edge / float(source_h),
            math.sqrt(max_pixels / float(source_w * source_h)),
        )
        target_w = max(16, int(math.floor(source_w * scale / 16.0) * 16))
        target_h = max(16, int(math.floor(source_h * scale / 16.0) * 16))
        return {"width": target_w, "height": target_h}

    def _normalize_custom_size(self, width, height):
        width = max(512, min(3840, int(width)))
        height = max(512, min(3840, int(height)))
        width = max(512, int(math.floor(width / 16.0) * 16))
        height = max(512, int(math.floor(height / 16.0) * 16))

        max_pixels = 8294400
        ratio = max(width / float(height), height / float(width))
        if ratio > 3.0:
            if width > height:
                width = int(math.floor((height * 3.0) / 16.0) * 16)
            else:
                height = int(math.floor((width * 3.0) / 16.0) * 16)

        pixels = width * height
        if pixels > max_pixels:
            scale = math.sqrt(max_pixels / float(pixels))
            width = max(512, int(math.floor(width * scale / 16.0) * 16))
            height = max(512, int(math.floor(height * scale / 16.0) * 16))

        return {"width": width, "height": height}

    def _resolve_size(self, size_mode, size, region_info, mask_image, input_size=None,
                      custom_width=3840, custom_height=2160):
        if size_mode == "auto_from_input":
            return "auto", "input_image"
        if size_mode == "max_from_input_aspect":
            if not input_size:
                return size, "manual_fallback"
            return self._max_size_from_input_aspect(input_size), "max_from_input_aspect"
        if size_mode == "custom":
            return self._normalize_custom_size(custom_width, custom_height), "custom"
        if size_mode == "preset":
            return size, "preset"
        if size_mode != "auto_from_region":
            return size, "manual"

        region_context = _extract_region_context(region_info)
        if region_context.get("recommended_edit_size"):
            return region_context["recommended_edit_size"], "region_info"

        bbox = region_context.get("bbox")
        region_type = region_context.get("region_type")
        if bbox or region_type:
            aspect_ratio = _recommend_aspect_ratio_for_region(region_type, bbox)
            return _recommend_edit_size_for_aspect_ratio(aspect_ratio), "region_info"

        if mask_image is not None:
            mask_bbox = self._mask_bbox(mask_image)
            if mask_bbox:
                aspect_ratio = _recommend_aspect_ratio_for_region("", mask_bbox)
                return _recommend_edit_size_for_aspect_ratio(aspect_ratio), "mask_bbox"

        return size, "manual_fallback"

    def _format_image_size(self, resolved_size):
        if isinstance(resolved_size, dict):
            return {
                "width": int(resolved_size["width"]),
                "height": int(resolved_size["height"]),
            }
        size_value = _coerce_text_value(resolved_size)
        if not size_value or size_value == "auto":
            return "auto"
        if size_value in {
            "square_hd",
            "square",
            "portrait_4_3",
            "portrait_16_9",
            "landscape_4_3",
            "landscape_16_9",
        }:
            return size_value
        match = re.fullmatch(r"(\d+)x(\d+)", size_value)
        if not match:
            raise ValueError(f"Unsupported image size value: {resolved_size}")
        return {
            "width": int(match.group(1)),
            "height": int(match.group(2)),
        }

    def _get_fal_client(self):
        try:
            import fal_client
        except ImportError as e:
            raise RuntimeError(
                "fal-client is not installed. Install it in ComfyUI's Python environment."
            ) from e
        return fal_client

    def _is_retryable_network_error(self, error):
        text = str(error).lower()
        retry_markers = (
            "winerror 10054",
            "forcibly closed",
            "connection reset",
            "connection aborted",
            "remote host",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "500",
            "internal server error",
            "502",
            "503",
            "504",
        )
        return any(marker in text for marker in retry_markers)

    def _with_retries(self, label, operation, attempts=3):
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if attempt >= attempts or not self._is_retryable_network_error(e):
                    raise
                sleep_seconds = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "GPT Image %s failed on attempt %s/%s: %s. Retrying in %ss.",
                    label,
                    attempt,
                    attempts,
                    str(e),
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise last_error

    def _upload_to_fal(self, data_bytes, mime_type, fal_api_key):
        fal_client = self._get_fal_client()
        previous_key = os.environ.get("FAL_KEY")
        os.environ["FAL_KEY"] = fal_api_key
        try:
            return self._with_retries(
                "image upload",
                lambda: fal_client.upload(data_bytes, mime_type),
            )
        finally:
            if previous_key is None:
                os.environ.pop("FAL_KEY", None)
            else:
                os.environ["FAL_KEY"] = previous_key

    def _call_fal(self, endpoint, arguments, fal_api_key):
        fal_client = self._get_fal_client()
        previous_key = os.environ.get("FAL_KEY")
        os.environ["FAL_KEY"] = fal_api_key
        try:
            result = self._with_retries(
                f"API call {endpoint}",
                lambda: fal_client.run(endpoint, arguments=arguments),
            )
            logger.debug("FAL GPT Image response from %s: %s", endpoint, json.dumps(result))
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to call FAL endpoint {endpoint}: {str(e)}") from e
        finally:
            if previous_key is None:
                os.environ.pop("FAL_KEY", None)
            else:
                os.environ["FAL_KEY"] = previous_key

    def _extract_result_image_url(self, result):
        if not isinstance(result, dict):
            return None
        candidates = result.get("images") or result.get("data") or []
        for item in candidates:
            if isinstance(item, dict):
                url = _coerce_text_value(item.get("url") or item.get("image_url"))
                if url:
                    return url
        return _coerce_text_value(result.get("image_url"))

    def _decode_image_result(self, image_url):
        response = requests.get(image_url, timeout=300)
        response.raise_for_status()
        pil_image = Image.open(io.BytesIO(response.content))
        pil_image.load()
        if pil_image.mode not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGBA" if "A" in pil_image.getbands() else "RGB")
        image_np = np.asarray(pil_image).astype(np.float32) / 255.0
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        return torch.from_numpy(image_np).unsqueeze(0)

    def edit_image(
        self,
        image_1,
        prompt,
        model,
        quality,
        size_mode,
        size,
        background,
        output_format,
        output_compression,
        moderation,
        api_key,
        api_key_env_var,
        openai_api_key="",
        openai_api_key_env_var="OPENAI_API_KEY",
        mask_image=None,
        image_2=None,
        custom_width=3840,
        custom_height=2160,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        region_info="",
    ):
        try:
            fal_api_key, fal_api_key_source = self._resolve_api_key(
                api_key,
                api_key_env_var,
                self._looks_like_fal_api_key,
                "FAL_KEY",
                "FAL",
            )
            resolved_openai_api_key, openai_api_key_source = self._resolve_api_key(
                openai_api_key,
                openai_api_key_env_var,
                self._looks_like_openai_api_key,
                "OPENAI_API_KEY",
                "OpenAI",
            )

            input_images = [
                image_1,
                image_2,
                image_3,
                image_4,
                image_5,
                image_6,
                image_7,
                image_8,
            ]
            image_urls = []
            image_size = None
            for index, input_image in enumerate(input_images, start=1):
                if input_image is None:
                    continue
                image_bytes, current_size = self._image_tensor_to_png_bytes(input_image)
                if index == 1:
                    image_size = current_size
                image_urls.append(self._upload_to_fal(image_bytes, "image/png", fal_api_key))
            if image_size is None:
                raise ValueError("image_1 is required.")

            resolved_size, size_source = self._resolve_size(
                size_mode,
                size,
                region_info,
                mask_image,
                input_size=image_size,
                custom_width=custom_width,
                custom_height=custom_height,
            )

            mask_url = None
            if mask_image is not None:
                mask_bytes = self._mask_tensor_to_png_bytes(mask_image, image_size)
                mask_url = self._upload_to_fal(mask_bytes, "image/png", fal_api_key)

            prompt_sent = _coerce_text_value(prompt)
            image_size_sent = self._format_image_size(resolved_size)
            arguments = {
                "prompt": prompt_sent,
                "image_urls": image_urls,
                "openai_api_key": resolved_openai_api_key,
                "image_size": image_size_sent,
                "background": background,
                "output_format": output_format,
                "moderation": moderation,
            }
            if quality != "auto":
                arguments["quality"] = quality
            if mask_url:
                arguments["mask_url"] = mask_url
            if output_format in ("jpeg", "webp"):
                arguments["output_compression"] = int(output_compression)

            result = self._call_fal(model, arguments, fal_api_key)
            image_url = self._extract_result_image_url(result)
            if not image_url:
                raise RuntimeError("FAL GPT Image edit returned no output image URL.")

            output_image = self._decode_image_result(image_url)
            output_height = int(output_image.shape[1])
            output_width = int(output_image.shape[2])
            info = {
                "model": model,
                "quality": quality,
                "prompt_sent": prompt_sent,
                "size_mode": size_mode,
                "resolved_size": resolved_size,
                "image_size_sent": image_size_sent,
                "size_source": size_source,
                "input_width": int(image_size[0]),
                "input_height": int(image_size[1]),
                "output_width": output_width,
                "output_height": output_height,
                "background": background,
                "output_format": output_format,
                "moderation": moderation,
                "fal_api_key_source": fal_api_key_source,
                "openai_api_key_source": openai_api_key_source,
                "mask_used": mask_image is not None,
                "reference_image_count": max(0, len(image_urls) - 1),
                "reference_image_used": len(image_urls) > 1,
                "output_image_url": image_url,
                "usage": result.get("usage") if isinstance(result, dict) else None,
            }
            return (output_image, json.dumps(info))
        except Exception as e:
            error_summary = _summarize_remote_error(e)
            logger.error("OpenAI image edit failed: %s", error_summary)
            raise RuntimeError(f"OpenAI image edit failed: {error_summary}") from e


class NB2NanoBanana2Edit(NB2OpenAIImageEdit):
    """
    Edit images with Nano Banana 2 through FAL, with per-node API key inputs.
    """

    ASPECT_RATIO_OPTIONS = [
        "auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1",
        "4:5", "3:4", "2:3", "9:16", "4:1", "1:4", "8:1", "1:8",
    ]
    RESOLUTION_OPTIONS = ["0.5K", "1K", "2K", "4K"]
    OUTPUT_FORMAT_OPTIONS = ["png", "jpeg", "webp"]
    SAFETY_TOLERANCE_OPTIONS = ["1", "2", "3", "4", "5", "6"]
    THINKING_LEVEL_OPTIONS = ["none", "minimal", "high"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Edit Image 1 using the connected reference images. Preserve the original identity, pose, lighting, and framing unless explicitly requested.",
                }),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "aspect_ratio": (cls.ASPECT_RATIO_OPTIONS, {"default": "auto"}),
                "resolution": (cls.RESOLUTION_OPTIONS, {"default": "2K"}),
                "output_format": (cls.OUTPUT_FORMAT_OPTIONS, {"default": "png"}),
                "safety_tolerance": (cls.SAFETY_TOLERANCE_OPTIONS, {"default": "4"}),
                "limit_generations": ("BOOLEAN", {"default": True}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "thinking_level": (cls.THINKING_LEVEL_OPTIONS, {"default": "none"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Optional. Leave blank to use FAL_KEY",
                }),
                "api_key_env_var": ("STRING", {
                    "multiline": False,
                    "default": "FAL_KEY",
                    "placeholder": "FAL environment variable fallback",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "edit_image"
    CATEGORY = "inpaint/api"
    DESCRIPTION = (
        "Edits one base image with up to five references through FAL's "
        "Nano Banana 2 edit endpoint."
    )

    def _extract_result_image_urls(self, result):
        if not isinstance(result, dict):
            return []
        urls = []
        candidates = result.get("images") or result.get("data") or []
        for item in candidates:
            if isinstance(item, dict):
                url = _coerce_text_value(item.get("url") or item.get("image_url"))
                if url:
                    urls.append(url)
            elif isinstance(item, str):
                url = _coerce_text_value(item)
                if url:
                    urls.append(url)
        fallback = _coerce_text_value(result.get("image_url"))
        if fallback:
            urls.append(fallback)
        return urls

    def edit_image(
        self,
        image_1,
        prompt,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        num_images=1,
        aspect_ratio="auto",
        resolution="2K",
        output_format="png",
        safety_tolerance="4",
        limit_generations=True,
        enable_web_search=False,
        thinking_level="none",
        seed=-1,
        sync_mode=False,
        api_key="",
        api_key_env_var="FAL_KEY",
    ):
        try:
            fal_api_key, fal_api_key_source = self._resolve_api_key(
                api_key,
                api_key_env_var,
                self._looks_like_fal_api_key,
                "FAL_KEY",
                "FAL",
            )

            input_images = [image_1, image_2, image_3, image_4, image_5, image_6]
            image_urls = []
            for input_image in input_images:
                if input_image is None:
                    continue
                image_bytes, _ = self._image_tensor_to_png_bytes(input_image)
                image_urls.append(self._upload_to_fal(image_bytes, "image/png", fal_api_key))
            if not image_urls:
                raise ValueError("image_1 is required.")

            arguments = {
                "prompt": _coerce_text_value(prompt),
                "image_urls": image_urls,
                "num_images": int(num_images),
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "resolution": resolution,
                "safety_tolerance": str(safety_tolerance),
                "limit_generations": bool(limit_generations),
                "enable_web_search": bool(enable_web_search),
                "sync_mode": bool(sync_mode),
            }
            if seed != -1:
                arguments["seed"] = int(seed)
            if thinking_level != "none":
                arguments["thinking_level"] = thinking_level

            result = self._call_fal("fal-ai/nano-banana-2/edit", arguments, fal_api_key)
            image_urls_out = self._extract_result_image_urls(result)
            if not image_urls_out:
                raise RuntimeError("FAL Nano Banana 2 edit returned no output image URL.")

            output_images = [self._decode_image_result(url) for url in image_urls_out]
            first_shape = output_images[0].shape
            if all(image.shape == first_shape for image in output_images):
                output_batch = torch.cat(output_images, dim=0)
            else:
                output_batch = output_images[0]

            info = {
                "endpoint": "fal-ai/nano-banana-2/edit",
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "num_images": int(num_images),
                "input_image_count": len(image_urls),
                "output_image_count": len(image_urls_out),
                "fal_api_key_source": fal_api_key_source,
                "safety_tolerance": str(safety_tolerance),
                "limit_generations": bool(limit_generations),
                "enable_web_search": bool(enable_web_search),
                "thinking_level": thinking_level,
                "output_image_urls": image_urls_out,
                "description": result.get("description") if isinstance(result, dict) else None,
            }
            if output_batch.shape[0] != len(image_urls_out):
                info["warning"] = "Output images had different sizes; returned only the first image."

            return (output_batch.cpu(), json.dumps(info))
        except Exception as e:
            error_summary = _summarize_remote_error(e)
            logger.error("Nano Banana 2 edit failed: %s", error_summary)
            raise RuntimeError(f"Nano Banana 2 edit failed: {error_summary}") from e


# ===========================================================================
#  ComfyUI registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "NanoBanana2MaskGen":  NanoBanana2MaskGen,
    "NB2SmartRegionMask":  NB2SmartRegionMask,
    "SmartMaskCrop":       SmartMaskCrop,
    "SmartMaskStitch":     SmartMaskStitch,
    "InpaintCropNB2":      InpaintCropNB2,
    "InpaintStitchNB2":    InpaintStitchNB2,
    "NB2AddAlpha":         NB2AddAlpha,
    "NB2Florence2RegionSelector": NB2Florence2RegionSelector,
    "NB2OpenAIImageEdit": NB2OpenAIImageEdit,
    "NB2NanoBanana2Edit": NB2NanoBanana2Edit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana2MaskGen":  "🎯 NB2 Mask Generator",
    "NB2SmartRegionMask":  "🧠 NB2 Smart Region",
    "SmartMaskCrop":       "🪄 Smart Mask Crop",
    "SmartMaskStitch":     "🪄 Smart Mask Stitch",
    "InpaintCropNB2":      "✂️ NB2 Crop",
    "InpaintStitchNB2":    "✂️ NB2 Stitch",
    "NB2AddAlpha":         "🔲 NB2 Add Alpha",
    "NB2Florence2RegionSelector": "Florence-2 Smart Region Selector (FAL API)",
    "NB2OpenAIImageEdit": "OpenAI GPT Image Edit",
    "NB2NanoBanana2Edit": "Nano Banana 2 Edit (FAL API)",
}
