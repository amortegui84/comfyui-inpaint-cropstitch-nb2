# ComfyUI-Inpaint-CropStitch-NB2

Fork of [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) by `lquesada`, adapted for post-production retouching with Nano Banana 2 and local masked editing.

All credits for the original crop/stitch engine go to [lquesada](https://github.com/lquesada).

## Demo

[![Demo video](https://img.youtube.com/vi/aqXyEMK6grk/maxresdefault.jpg)](https://www.youtube.com/watch?v=aqXyEMK6grk)

---

## Why These Nodes Exist

Traditional inpainting requires a mask: you draw the area to regenerate and the model fills it in.

Nano Banana 2 is different — it is a generation model, not a true inpainting model. It does not accept a mask directly. It takes a clean image crop and generates a new image at a fixed resolution.

These nodes bridge that gap in two ways:

1. **NB2 crop/stitch path** — use a semantic region or manual region to define a rectangular crop, generate only that crop with NB2, and stitch it back into the original image.
2. **Local mask crop/stitch path** — use a semantic mask to crop a focused local edit region, run a true mask-based editor such as GPT Image, then stitch the result back.

---

## Resolution Design

Nano Banana 2 produces images at fixed resolutions. These nodes are built around those exact sizes:

| Aspect ratio | 1K | 2K | 4K |
|---|---|---|---|
| 16:9 | 1376 × 768 | 2752 × 1536 | 5504 × 3072 |
| 9:16 | 768 × 1376 | 1536 × 2752 | 3072 × 5504 |
| 1:1 | 1024 × 1024 | 2048 × 2048 | 4096 × 4096 |

- Use `1K` for small touch-ups, faces, or small objects.
- Use `2K` for medium-size retouching regions.
- Use `4K` when you need maximum detail or the retouch area is large.

You can mix generation resolution and crop size. Example:

```
crop_width = 2752  (2K-sized area on the original)
generate at 4K     -> NB2 outputs 5504×3072
stitch             -> downscales 5504×3072 to 2752×1536, composites back
```

---

## Nodes

### Florence-2 Smart Region Selector (FAL API)

External Florence-2 region selector integrated into this repo. It calls FAL's Florence API, returns a ComfyUI mask, and is intended to feed `NB2 Smart Region` or `Smart Mask Crop`.

| Input | Type | Description |
|---|---|---|
| image | IMAGE | Source image |
| region_type | choice | `glasses`, `face`, `upper_body`, `lower_body`, `full_body`, `object` |
| custom_text | STRING | Required only when `region_type = object` |
| selection_mode | choice | `largest` or `merge_all` |
| padding_percent | FLOAT | Expands the detected region bbox for downstream crop sizing |
| return_rect_mask | BOOLEAN | Return a rectangular bbox mask instead of the raw semantic mask |
| api_key | STRING | Optional direct API key input. Leave blank if using an env var |
| api_key_env_var | STRING | Env var name fallback, default `FAL_KEY` |

Outputs: `mask`, `mask_image`, `info`, `center_x`, `center_y`, `crop_width`, `crop_height`

Built-in region defaults: `glasses -> 16:9`, `face -> 1:1`, `upper_body -> 1:1`, `lower_body -> 1:1`, `full_body -> 9:16`. These hints are stored in the `info` output and can drive downstream auto sizing.

Security notes:

- The repo does not store any API key.
- All bundled workflows leave `api_key` empty.
- For safer usage, prefer setting `FAL_KEY` in the environment and keep `api_key` blank.
- If you paste a key into the node and save the workflow yourself, ComfyUI may persist that widget value into the workflow JSON.

Typical flow:
```
NB2Florence2RegionSelector -> mask -> NB2 Smart Region -> NB2 Crop -> Nano Banana 2 -> NB2 Stitch
```

---

### NB2 Mask Generator

Interactive node that generates a rectangular mask at an exact position on the original image, with an aspect ratio that matches an NB2 resolution.

| Input | Type | Description |
|---|---|---|
| image | IMAGE | Source image used for dimensions and preview |
| aspect_ratio | choice | `auto`, `16:9`, `9:16`, `1:1` |
| resolution | choice | `1K`, `2K`, `4K` |
| center_x | INT | Horizontal center of the crop rectangle |
| center_y | INT | Vertical center of the crop rectangle |
| crop_width | INT | Width of the crop rectangle on the source image |

Outputs: `mask`, `nb2_width`, `nb2_height`, `preview_image`

---

### NB2 Smart Region

Automatic NB2 rectangle fitting from a semantic mask. Use this when you already have a region mask from Florence2, SAM, or any segmentation node.

| Input | Type | Description |
|---|---|---|
| image | IMAGE | Source image |
| region_mask | MASK | Semantic mask to fit |
| aspect_ratio | choice | `auto`, `16:9`, `9:16`, `1:1` |
| resolution | choice | `1K`, `2K`, `4K` |
| padding_percent | FLOAT | Expands the detected region before rectangle fitting |
| crop_scale | FLOAT | Additional scale multiplier after fitting |
| depad_florence | BOOLEAN | Remove Florence2's internal square letterbox padding (default `True`) |

Outputs: `mask`, `nb2_width`, `nb2_height`, `preview_image`, `center_x`, `center_y`, `crop_width`, `crop_height`, `info`

When `aspect_ratio = auto`, the node resolves the crop shape from Florence region metadata first, then falls back to the mask bbox if no region hint is available.

> **depad_florence** — Florence2 pads images to a square internally before processing. Without this correction the detected bbox shifts sideways or vertically on non-square images. Leave `True` when the mask comes from `Florence2Run (kijai)`. Set to `False` only if your mask is already at the exact source image resolution (e.g. from a hand-drawn mask or SAM).

Typical flow:
```
NB2Florence2RegionSelector or Florence2Run (kijai) -> mask -> NB2 Smart Region -> NB2 Crop -> Nano Banana 2 -> NB2 Stitch
```

---

### OpenAI GPT Image Edit

External GPT Image 2 editor integrated into this repo through FAL. It calls `openai/gpt-image-2/edit`, accepts an optional mask, and can either preserve the input crop with `auto` or use a documented preset size when you explicitly request it.

| Input | Type | Description |
|---|---|---|
| image_1 | IMAGE | Base image to edit |
| prompt | STRING | Edit instruction |
| model | choice | `openai/gpt-image-2/edit` |
| quality | choice | `auto`, `low`, `medium`, `high` |
| size_mode | choice | `auto_from_input`, `auto_from_region`, or `manual` |
| size | choice | `auto`, `1024x768`, `1024x1024`, `1024x1536`, `1920x1080`, `2560x1440`, `3840x2160` |
| background | choice | `auto`, `opaque`, `transparent` |
| output_format | choice | `png`, `webp`, `jpeg` |
| output_compression | INT | Used for `webp` and `jpeg` outputs |
| moderation | choice | `auto` or `low` |
| api_key | STRING | Optional direct FAL API key input |
| api_key_env_var | STRING | FAL env var fallback, default `FAL_KEY` |
| openai_api_key | STRING | Optional direct OpenAI API key input passed through to FAL |
| openai_api_key_env_var | STRING | OpenAI env var fallback, default `OPENAI_API_KEY` |
| mask_image | IMAGE | Optional mask image. If connected, the node converts it to an alpha mask automatically |
| region_info | STRING | Optional Florence `info` output used only when `size_mode = auto_from_region` |

Outputs: `images`, `info`

Typical flow:
```
NB2Florence2RegionSelector -> mask -> Smart Mask Crop -> OpenAI GPT Image Edit -> Smart Mask Stitch
```

Recommended default:

- Use `size_mode = auto_from_input` for masked local edits. This preserves the crop size inferred by FAL and avoids accidental rescaling.
- Use `auto_from_region` only when you explicitly want Florence's aspect hint to drive a preset size.

---

### NB2 Crop

Crops the source image around the mask region and scales to the exact NB2 resolution.

| Input | Type | Description |
|---|---|---|
| image | IMAGE | Original source image |
| mask | MASK | Mask from `NB2 Mask Generator` or `NB2 Smart Region` |
| aspect_ratio | choice | Must match the upstream region node |
| resolution | choice | Must match the upstream region node |
| context_extend_factor | FLOAT | Extra context growth before aspect-ratio fit |
| downscale_algorithm | choice | Used when crop is larger than target |
| upscale_algorithm | choice | Used when crop is smaller than target |
| device_mode | choice | CPU or GPU execution |

Outputs: `stitcher`, `cropped_image`, `cropped_mask`

---

### NB2 Stitch

Composites the NB2-generated image back onto the original canvas.

| Input | Type | Description |
|---|---|---|
| stitcher | STITCHER | Coordinate data from `NB2 Crop` |
| inpainted_image | IMAGE | RGB or RGBA output from Nano Banana 2 |
| edge_feather_percent | FLOAT | Extra edge blend width |

Outputs: `image`

---

### Smart Mask Crop

Local masked-edit crop for models that accept a real mask (e.g. GPT Image). Use this when the selected region is small and you do not want to send the full image to the editor.

| Input | Type | Description |
|---|---|---|
| image | IMAGE | Source image |
| mask | MASK | Local semantic mask |
| context_expand | FLOAT | Grows the detected region before cropping |
| use_region_guidance | BOOLEAN | Reuse Florence region metadata for context and target-size defaults |
| mask_expand_percent | FLOAT | Extra expansion applied to the edit mask after crop |
| mask_feather_percent | FLOAT | Softens the edit mask edges before sending to the editor |
| resize_mode | choice | `keep_local_size` or `resize_to_target` |
| target_width | INT | Used when `resize_to_target` |
| target_height | INT | Used when `resize_to_target` |
| downscale_algorithm | choice | Resize-down algorithm |
| upscale_algorithm | choice | Resize-up algorithm |
| device_mode | choice | CPU or GPU |
| depad_florence | BOOLEAN | Remove Florence2 letterbox padding (default `True`) |
| region_info | STRING | Optional Florence `info` output for shared aspect, mask, and sizing hints |

Outputs: `stitcher`, `cropped_image`, `cropped_mask`, `cropped_mask_image`, `preview_image`, `info`

> Wire `cropped_mask_image` directly into `GPT Image 2 Edit → mask_image`.

Typical flow:
```
NB2Florence2RegionSelector or Florence2Run (kijai) -> mask -> Smart Mask Crop -> GPT Image 2 Edit -> Smart Mask Stitch
```

---

### Smart Mask Stitch

Pastes a locally edited masked crop back into the original image using the stored local mask as the primary blend.

| Input | Type | Description |
|---|---|---|
| stitcher | STITCHER | Coordinate data from `Smart Mask Crop` |
| edited_image | IMAGE | Output of the local masked editor |
| edge_feather_percent | FLOAT | Extra edge feather for the crop boundary |

Outputs: `image`

---

### NB2 Add Alpha

Converts an RGB image to RGBA by generating a feathered alpha channel. Useful for soft-edge RGBA output for compositing.

| Input | Type | Description |
|---|---|---|
| image | IMAGE | RGB or RGBA input |
| feather_percent | FLOAT | Edge fade width as a percentage |

Outputs: `rgba_image`

---

## Recommended Workflows

### Path 1 — NB2 region retouch

```
NB2Florence2RegionSelector or Florence2Run (kijai) or NB2 Mask Generator
    -> NB2 Smart Region (or direct mask)
    -> NB2 Crop
    -> Nano Banana 2 generation
    -> NB2 Stitch
```

### Path 2 — Local masked edit (GPT Image, etc.)

```
NB2Florence2RegionSelector or Florence2Run (kijai)
    -> mask
    -> Smart Mask Crop
        -> cropped_image        -> GPT Image 2 Edit
        -> cropped_mask_image   -> GPT Image 2 Edit (mask_image)
    -> Smart Mask Stitch
```

### Tips

- For `face`: `padding_percent` 10–18, `context_expand` 1.1–1.2
- For `upper_body`: `padding_percent` 8–15
- For small objects: `padding_percent` 5–12
- Stitch feathering: `edge_feather_percent` 3–8 is usually enough

---

## Example Workflows

| File | Description |
|---|---|
| `inpainting_workflow.json` | Original NB2 crop/stitch workflow |
| `01_nb2_smart_region_face_roundtrip.json` | FAL Florence-2 → NB2 Smart Region → NB2 round-trip |
| `02_local_mask_edit_face_gpt_image2.json` | FAL Florence-2 → Smart Mask Crop → GPT Image 2 |
| `03_local_mask_edit_object_template.json` | Object local mask edit template |
| `04_nb2_upper_body_template.json` | Upper body NB2 template |

Load any `.json` via **ComfyUI → Load** (drag & drop or File > Open).

---

## Installation

### Option A — ComfyUI Manager (recommended)

Search for **comfyui-inpaint-cropstitch-nb2** in the Manager and click Install. Restart ComfyUI.

### Option B — Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/amortegui84/comfyui-inpaint-cropstitch-nb2
```

Restart ComfyUI after cloning.

### Updating

```bash
cd ComfyUI/custom_nodes/comfyui-inpaint-cropstitch-nb2
git pull
```

Restart ComfyUI after updating.

### Python dependencies

This repo now includes external Florence-2 and GPT Image edit nodes through FAL. They need `fal-client` plus the normal HTTP/image dependencies in the same Python environment ComfyUI uses.

```bash
python -m pip install fal-client requests pillow numpy
```

Restart ComfyUI after installing dependencies.

### Git LFS note

This repo tracks `assets/demo.mp4` with Git LFS. The nodes and workflows still work without that demo file, but if you want the full asset after `git clone` or `git pull`, install Git LFS once on the machine:

```bash
git lfs install
```

### Optional dependency — Florence2

Optional only if you also want the fully local `Florence2Run` path from kijai. That path runs **fully local — no API key needed**.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-Florence2
```

Or search **ComfyUI-Florence2** in the Manager.

---

## License

Apache 2.0 — see [LICENSE](LICENSE). Original work copyright 2024 lquesada. Modifications copyright 2025 amortegui84.
