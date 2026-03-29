# ComfyUI-Inpaint-CropStitch-NB2

Fork of [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) by **lquesada**, adapted for use with the **Nano Banana 2** generation model.

> All credits for the original crop/stitch engine go to [lquesada](https://github.com/lquesada).
> This repository only adds the three nodes described below and does not modify the original node classes.

---

## What's new

### 🎯 NB2 Mask Generator
Generates a rectangular mask at an exact position on the original image.
The mask aspect ratio always matches the Nano Banana 2 output for the selected quality level.

| Aspect ratio | 1K | 2K | 4K |
|---|---|---|---|
| 16:9 | 1376 × 768 | 2752 × 1536 | 5504 × 3072 |
| 9:16 | 768 × 1376 | 1536 × 2752 | 3072 × 5504 |
| 1:1  | 1024 × 1024 | 2048 × 2048 | 4096 × 4096 |

**Inputs**
| Name | Type | Description |
|---|---|---|
| image | IMAGE | Original image (used only for dimensions) |
| aspect_ratio | choice | 16:9 · 9:16 · 1:1 |
| resolution | choice | 1K · 2K · 4K |
| center_x | INT | Horizontal centre of the crop rectangle (pixels) |
| center_y | INT | Vertical centre of the crop rectangle (pixels) |
| crop_width | INT | Width of the rectangle on the original image. Height is auto-calculated. |

**Outputs** — `mask`, `nb2_width`, `nb2_height`

---

### ✂️ NB2 Crop
Crops the original image around the masked area and **scales the result to the exact NB2 resolution** so that Nano Banana 2 receives a pixel-perfect input.

Wire `nb2_width` and `nb2_height` from **NB2 Mask Generator** to guarantee the crop matches NB2's expected dimensions.
If the original image is smaller than the target resolution the crop is scaled up automatically.

**Outputs** — `stitcher`, `cropped_image`, `cropped_mask`

---

### ✂️ NB2 Stitch
Composites the NB2-generated image back onto the original canvas.

Two blending mechanisms work together:

1. **Edge feathering** (`edge_feather_percent`)
   A smoothstep gradient runs from 0 (at the very boundary of the generated region) to 1 (a few pixels inside). This hides the hard cut and makes the composite look natural.
   The ramp width is expressed as a percentage of the crop dimension — e.g. `5` means 5 % of the width and 5 % of the height on each side.
   Set to `0` for a hard cut.

2. **Alpha channel** (automatic)
   If the generated image has four channels (RGBA) the alpha is extracted and multiplied with the feather mask before blending.
   This lets you pre-compute soft edges upstream (e.g. with a dedicated matte/feather node) and pass them through without needing an extra mask wire.

---

## Workflow

```
[original image]
      │
NB2 Mask Generator  ←  aspect_ratio · resolution · center_x · center_y · crop_width
      │ mask   nb2_width   nb2_height
NB2 Crop
      │ stitcher   cropped_image
[Nano Banana 2]     ← receives cropped_image (no mask required)
      │ generated_image  (RGB or RGBA)
NB2 Stitch          ← edge_feather_percent
      │
[final composited image]
```

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/amortegui84/comfyui-inpaint-cropstitch-nb2
```

Restart ComfyUI.  No additional Python packages are required beyond those already used by ComfyUI.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
Original work © 2024 lquesada.  Modifications © 2025 amortegui84.
