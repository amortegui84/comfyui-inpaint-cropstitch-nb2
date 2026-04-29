# ComfyUI-Inpaint-CropStitch-NB2

Fork of [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) by `lquesada`, adapted for post-production retouching workflows using Nano Banana 2 and local masked editing flows.

All credits for the original crop/stitch engine go to [lquesada](https://github.com/lquesada).

## Demo

[![Demo video](https://img.youtube.com/vi/aqXyEMK6grk/maxresdefault.jpg)](https://www.youtube.com/watch?v=aqXyEMK6grk)

## Why These Nodes Exist

Traditional inpainting workflows require a mask: you draw the area you want to regenerate and the model fills it in.

Nano Banana 2 is different. It is a generation model, not a true inpainting model. It does not accept a mask directly. It takes a clean image crop and generates a new image at a fixed resolution.

These nodes bridge that gap in two ways:

1. `NB2 crop/stitch path`
   Use a semantic region or manual region to define a rectangular crop, generate only that crop, and stitch it back into the original image.

2. `local mask crop/stitch path`
   Use a semantic mask to crop a focused local edit region, run a true mask-based editor such as GPT Image, then stitch the result back into the original image.

## Resolution Design

Nano Banana 2 produces images at fixed resolutions. These nodes are built around those exact sizes:

| Aspect ratio | 1K | 2K | 4K |
|---|---|---|---|
| 16:9 | 1376 x 768 | 2752 x 1536 | 5504 x 3072 |
| 9:16 | 768 x 1376 | 1536 x 2752 | 3072 x 5504 |
| 1:1 | 1024 x 1024 | 2048 x 2048 | 4096 x 4096 |

Choosing the right resolution:

- Use `1K` for small touch-ups, faces, or small objects.
- Use `2K` for medium-size retouching regions.
- Use `4K` when you need maximum detail or the retouch area is large.

Mixing generation resolution and crop size is valid. You can generate at a higher resolution than the crop area and let the stitch step downscale the result cleanly before compositing.

Example:

```text
crop_width = 2752  (2K-sized area on the original)
generate at 4K     -> NB2 outputs 5504x3072
stitch             -> downscales 5504x3072 to 2752x1536, composites back
```

## Nodes

### NB2 Mask Generator

Interactive node that generates a rectangular mask at an exact position on the original image, with an aspect ratio that matches a Nano Banana 2 output resolution.

Inputs:

| Name | Type | Description |
|---|---|---|
| image | IMAGE | Source image used for dimensions and preview |
| aspect_ratio | choice | `16:9`, `9:16`, `1:1` |
| resolution | choice | `1K`, `2K`, `4K` |
| center_x | INT | Horizontal center of the crop rectangle |
| center_y | INT | Vertical center of the crop rectangle |
| crop_width | INT | Width of the crop rectangle on the source image |

Outputs:

- `mask`
- `nb2_width`
- `nb2_height`
- `preview_image`

### NB2 Smart Region

Automatic NB2 rectangle fitting from a semantic mask.

Use this when you already have a region mask from another node such as:

- Florence-2 Smart Region Selector
- SAM or segmentation nodes
- any external object-selection pipeline

Inputs:

| Name | Type | Description |
|---|---|---|
| image | IMAGE | Source image |
| region_mask | MASK | Semantic mask to fit |
| aspect_ratio | choice | `16:9`, `9:16`, `1:1` |
| resolution | choice | `1K`, `2K`, `4K` |
| padding_percent | FLOAT | Expands the detected region before rectangle fitting |
| crop_scale | FLOAT | Additional scale multiplier after fitting |

Outputs:

- `mask`
- `nb2_width`
- `nb2_height`
- `preview_image`
- `center_x`
- `center_y`
- `crop_width`
- `crop_height`
- `info`

Notes:

- `NB2 Smart Region` accepts `MASK` tensors in either `[H, W]` or `[B, H, W]` form.
- This node is for the rectangular NB2 workflow, not for exact mask editing.

### NB2 Crop

Crops the source image around the mask region and scales the result to the exact NB2 resolution.

Inputs:

| Name | Type | Description |
|---|---|---|
| image | IMAGE | Original source image |
| mask | MASK | Mask from `NB2 Mask Generator` or `NB2 Smart Region` |
| aspect_ratio | choice | Must match the upstream NB2 region node |
| resolution | choice | Must match the upstream NB2 region node |
| context_extend_factor | FLOAT | Extra context growth before aspect-ratio fit |
| downscale_algorithm | choice | Used when crop is larger than target |
| upscale_algorithm | choice | Used when crop is smaller than target |
| device_mode | choice | CPU or GPU execution |

Outputs:

- `stitcher`
- `cropped_image`
- `cropped_mask`

### NB2 Stitch

Composites the NB2-generated image back onto the original canvas.

Inputs:

| Name | Type | Description |
|---|---|---|
| stitcher | STITCHER | Coordinate data from `NB2 Crop` |
| inpainted_image | IMAGE | RGB or RGBA output from Nano Banana 2 |
| edge_feather_percent | FLOAT | Extra edge blend width |

Outputs:

- `image`

### NB2 Add Alpha

Converts an RGB image to RGBA by generating a feathered alpha channel. Useful when you want soft-edge RGBA output for compositing.

Inputs:

| Name | Type | Description |
|---|---|---|
| image | IMAGE | RGB or RGBA input |
| feather_percent | FLOAT | Edge fade width as a percentage |

Outputs:

- `rgba_image`

### Smart Mask Crop

Local masked-edit crop for models that really use a mask.

Use this when a selector finds a small region like a face, shirt, watch, sleeve, or object and you do not want to send the full image into a masked editor.

Inputs:

| Name | Type | Description |
|---|---|---|
| image | IMAGE | Source image |
| mask | MASK | Local semantic mask |
| context_expand | FLOAT | Grows the detected region before crop |
| resize_mode | choice | `keep_local_size` or `resize_to_target` |
| target_width | INT | Used when resizing the local crop |
| target_height | INT | Used when resizing the local crop |
| downscale_algorithm | choice | Resize down algorithm |
| upscale_algorithm | choice | Resize up algorithm |
| device_mode | choice | CPU or GPU execution |

Outputs:

- `stitcher`
- `cropped_image`
- `cropped_mask`
- `cropped_mask_image`
- `preview_image`
- `info`

Important:

- `cropped_mask_image` is the output you can wire directly into `GPT Image 2 Edit -> mask_image`.
- This node keeps the edit localized around the selected region instead of sending the whole image to the editor.

### Smart Mask Stitch

Pastes a locally edited masked crop back into the original image using the stored local mask as the primary blend.

Inputs:

| Name | Type | Description |
|---|---|---|
| stitcher | STITCHER | Coordinate data from `Smart Mask Crop` |
| edited_image | IMAGE | Output of the local masked editor |
| edge_feather_percent | FLOAT | Extra edge feather for the crop boundary |

Outputs:

- `image`

## Recommended Workflows

### 1. NB2 Region Retouch Workflow

```text
selector or manual region
    -> NB2 Smart Region or NB2 Mask Generator
    -> NB2 Crop
    -> Nano Banana 2 generation
    -> NB2 Stitch
```

Use this when the destination model does not consume an exact mask and you want the existing crop/stitch technique.

### 2. Local Masked Edit Workflow

```text
Florence-2 Smart Region Selector
    -> mask
    -> Smart Mask Crop
    -> cropped_image -> GPT Image 2 Edit
    -> cropped_mask_image -> GPT Image 2 Edit.mask_image
    -> Smart Mask Stitch
```

Use this when the selected area is small and you want the editing model to work on a focused crop instead of the entire image.

## Compatibility Notes

- `Florence-2 Smart Region Selector` currently supports batch size `1` only.
- `NB2 Smart Region` accepts `MASK` tensors in `[H, W]` or `[B, H, W]`.
- `Smart Mask Crop` and `Smart Mask Stitch` reuse the same crop/stitch coordinate logic so the local edit can be pasted back consistently.

## Tips

- For `face`, start with `padding_percent` around `10` to `18`.
- For `upper_body`, start with `8` to `15`.
- For `object`, start with `5` to `12`.
- For local masked editing, keep `context_expand` around `1.1` to `1.25` so the edited region has enough context without becoming too diffuse.
- For stitch feathering, `3` to `8` is usually enough.

## Workflow

A ready-to-use ComfyUI workflow is included in this repo.  
[Download inpainting_workflow.json](workflows/inpainting_workflow.json)

Additional sample workflows for the new smart-region flows:

- [01_nb2_smart_region_face_roundtrip.json](workflows/01_nb2_smart_region_face_roundtrip.json)
- [02_local_mask_edit_face_gpt_image2.json](workflows/02_local_mask_edit_face_gpt_image2.json)
- [03_local_mask_edit_object_template.json](workflows/03_local_mask_edit_object_template.json)

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/amortegui84/comfyui-inpaint-cropstitch-nb2
```

Restart ComfyUI after updating.

## Update From GitHub

Inside the repo folder:

```bash
git pull origin master
```

If you are updating from your ComfyUI install:

```bash
cd ComfyUI/custom_nodes/comfyui-inpaint-cropstitch-nb2
git pull origin master
```

## License

Apache 2.0 - see [LICENSE](LICENSE). Original work copyright 2024 lquesada. Modifications copyright 2025 amortegui84.
