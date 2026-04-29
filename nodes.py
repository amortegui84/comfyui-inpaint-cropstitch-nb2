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
import math
import nodes
import numpy as np
import os
import uuid
import torch
import torch.nn.functional as TF
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, grey_dilation, binary_closing, binary_fill_holes
from abc import ABC, abstractmethod

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
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
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
                             padding_percent, crop_scale, depad_florence=True):
        image = image.clone()
        region_mask = region_mask.clone()
        processor = CPUProcessorLogic()
        region_mask, image, mask_note = _normalize_mask_to_image(
            region_mask, image, processor, "NB2SmartRegionMask",
            depad_florence=depad_florence
        )

        B, H, W, _ = image.shape
        nb2_w, nb2_h = NB2_RESOLUTIONS[aspect_ratio][resolution]
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
                "aspect_ratio": aspect_ratio,
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
                "resize_mode": (["keep_local_size", "resize_to_target"], {
                    "default": "resize_to_target"}),
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

    def smart_mask_crop(self, image, mask, context_expand, resize_mode,
                        target_width, target_height, downscale_algorithm,
                        upscale_algorithm, device_mode, depad_florence=True):
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

            _, bx, by, bw, bh = processor.batched_findcontextarea_m(sub_mask)
            if bx[0] == -1:
                raise ValueError("mask is empty; Smart Mask Crop requires a non-empty mask.")

            if context_expand > 1.0:
                _, bx, by, bw, bh = processor.batched_growcontextarea_m(
                    sub_mask, bx, by, bw, bh, context_expand
                )

            cur_x = bx[0].item()
            cur_y = by[0].item()
            cur_w = bw[0].item()
            cur_h = bh[0].item()

            if resize_mode == "keep_local_size":
                out_w = max(1, int(cur_w))
                out_h = max(1, int(cur_h))
                resize_output = False
            else:
                out_w = int(target_width)
                out_h = int(target_height)
                resize_output = True

            (canvas_image, cto_x, cto_y, cto_w, cto_h,
             cropped_image, cropped_mask,
             ctc_x, ctc_y, ctc_w, ctc_h) = processor.crop_magic_im(
                sub_image, sub_mask,
                cur_x, cur_y, cur_w, cur_h,
                out_w, out_h,
                0,
                downscale_algorithm, upscale_algorithm,
                resize_output=resize_output)

            result_stitcher['canvas_to_orig_x'].append(cto_x)
            result_stitcher['canvas_to_orig_y'].append(cto_y)
            result_stitcher['canvas_to_orig_w'].append(cto_w)
            result_stitcher['canvas_to_orig_h'].append(cto_h)
            result_stitcher['canvas_image'].append(canvas_image.cpu())
            result_stitcher['cropped_to_canvas_x'].append(ctc_x)
            result_stitcher['cropped_to_canvas_y'].append(ctc_y)
            result_stitcher['cropped_to_canvas_w'].append(ctc_w)
            result_stitcher['cropped_to_canvas_h'].append(ctc_h)
            result_stitcher['cropped_mask_for_blend'].append(cropped_mask.cpu())

            result_image.append(cropped_image.squeeze(0).cpu())
            result_mask.append(cropped_mask.squeeze(0).cpu())
            mask_rgb = torch.stack([cropped_mask.squeeze(0).cpu()] * 3, dim=-1)
            result_mask_image.append(mask_rgb)

            preview_tensor, temp_info = _make_nb2_preview(sub_image[0].cpu(), cur_y, cur_x, cur_h, cur_w)
            previews.append(preview_tensor.squeeze(0))
            if i == 0 and temp_info:
                preview_ui.append(temp_info)

            infos.append({
                "context_expand": context_expand,
                "resize_mode": resize_mode,
                "target_width": int(out_w),
                "target_height": int(out_h),
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

    def smart_mask_stitch(self, stitcher, edited_image, edge_feather_percent):
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
                downscale_algorithm, upscale_algorithm,
                edge_feather_percent, device, processor
            )
            results.append(out.squeeze(0))

        return (torch.stack(results, dim=0).cpu(),)

    def _stitch_single(self, canvas_image, edited_image, local_mask,
                       ctc_x, ctc_y, ctc_w, ctc_h,
                       cto_x, cto_y, cto_w, cto_h,
                       downscale_algo, upscale_algo,
                       edge_feather_percent, device, processor):
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana2MaskGen":  "🎯 NB2 Mask Generator",
    "NB2SmartRegionMask":  "🧠 NB2 Smart Region",
    "SmartMaskCrop":       "🪄 Smart Mask Crop",
    "SmartMaskStitch":     "🪄 Smart Mask Stitch",
    "InpaintCropNB2":      "✂️ NB2 Crop",
    "InpaintStitchNB2":    "✂️ NB2 Stitch",
    "NB2AddAlpha":         "🔲 NB2 Add Alpha",
}
