# ComfyUI-Inpaint-CropStitch-NB2

Fork of [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) by **lquesada**, adapted for post-production retouching workflows using **Nano Banana 2**.

> All credits for the original crop/stitch engine go to [lquesada](https://github.com/lquesada).

---

## Why these nodes exist

Traditional inpainting workflows require a mask — you draw the area you want to regenerate and the model fills it in. **Nano Banana 2 is a generation model, not an inpainting model**: it does not accept a mask. It takes a clean image and generates a new one at a fixed resolution.

These nodes bridge that gap. The idea is to use NB2 as a **post-production retouching tool**: isolate a region of an existing image, send it to NB2 for regeneration at the best possible quality, and stitch the result seamlessly back onto the original canvas — all without ever drawing a mask by hand.

This makes it possible to automate the retouching of images through NB2 in a repeatable, non-destructive pipeline.

---

## Resolution design

Nano Banana 2 produces images at fixed resolutions. These nodes are built around those exact sizes:

| Aspect ratio | 1K | 2K | 4K |
|---|---|---|---|
| 16:9 | 1376 × 768 | 2752 × 1536 | 5504 × 3072 |
| 9:16 | 768 × 1376 | 1536 × 2752 | 3072 × 5504 |
| 1:1  | 1024 × 1024 | 2048 × 2048 | 4096 × 4096 |

**Choosing the right resolution:**

- Use **1K** for small touch-ups, faces, or objects that occupy a limited portion of the frame.
- Use **2K** for medium-sized regions — a good balance between detail and processing time.
- Use **4K** when you need maximum quality, or when the region to retouch is large relative to the original image.

**Mixing generation resolution and crop size:**

The crop size on the original image (`crop_width`) and the NB2 generation resolution are independent. You can generate at a higher resolution than the crop area — NB2 will produce finer detail and the stitch will downscale it with a quality algorithm before compositing. For example:

```
crop_width = 2752  (2K area on the original)
generate at 4K     → NB2 outputs 5504×3072
stitch             → downscales 5504×3072 → 2752×1536, composites
```

This is valid and safe. The stitch always detects whether the generated image is larger or smaller than the crop region and applies the appropriate scaling algorithm automatically.

---

## Nodes

### 🎯 NB2 Mask Generator

Interactive node that generates a rectangular mask at an exact position on the original image, with an aspect ratio that matches a Nano Banana 2 output resolution. No manual mask drawing needed.

**Interactive canvas widget** — connect an image and the preview loads immediately (no re-run needed):

| Control | Action |
|---|---|
| Drag | Move crop centre → updates `center_x` / `center_y` |
| Scroll wheel | Resize crop area → updates `crop_width` |
| Arrow keys | Nudge centre 1 px |
| Shift + Arrow | Nudge centre 10 px |

When you change `resolution` or `aspect_ratio`, `crop_width` is automatically set to the NB2 output width for that selection (clamped to the image width), so the rectangle immediately shows the real capture area.

**Inputs**

| Name | Type | Description |
|---|---|---|
| image | IMAGE | Source image (used for dimensions and canvas preview) |
| aspect_ratio | choice | 16:9 · 9:16 · 1:1 |
| resolution | choice | 1K · 2K · 4K |
| center_x | INT | Horizontal centre of the crop rectangle (pixels) |
| center_y | INT | Vertical centre of the crop rectangle (pixels) |
| crop_width | INT | Width of the rectangle on the source image. Height is computed from the aspect ratio. |

**Outputs** — `MASK`, `nb2_width (INT)`, `nb2_height (INT)`, `preview (IMAGE)`

---

### ✂️ NB2 Crop

Crops the source image around the masked region and scales the result to the **exact NB2 resolution**, so Nano Banana 2 receives a pixel-perfect input regardless of the original image size.

Wire `nb2_width` and `nb2_height` from **NB2 Mask Generator** to guarantee the crop matches NB2's expected dimensions.

**Inputs**

| Name | Type | Description |
|---|---|---|
| image | IMAGE | Original source image |
| mask | MASK | Mask from NB2 Mask Generator |
| nb2_width | INT | Target width — wire from NB2 Mask Generator |
| nb2_height | INT | Target height — wire from NB2 Mask Generator |
| downscale_algorithm | choice | Algorithm used when the crop is larger than the target |
| upscale_algorithm | choice | Algorithm used when the crop is smaller than the target |

**Outputs** — `stitcher (STITCHER)`, `cropped_image (IMAGE)`, `cropped_mask (MASK)`

---

### ✂️ NB2 Stitch

Composites the NB2-generated image back onto the original canvas.

The stitch **automatically handles any size mismatch** between the generated image and the crop region. Whether NB2 produced a 1K, 2K, or 4K image, the stitch detects the difference and applies the correct scaling algorithm (upscale or downscale) before compositing. You do not need to pre-resize the generated image.

**Blending behaviour:**

- **RGB input** — `edge_feather_percent` generates a smoothstep gradient from 0 at the crop boundary to 1 a few pixels inside, producing a natural blend. Set to `0` for a hard cut.
- **RGBA input** — the alpha channel is used directly as the blend mask; `edge_feather_percent` is ignored. This avoids double-feathering. Use `NB2 Add Alpha` upstream if you want custom soft-edge control.

> For the standard workflow you do **not** need `NB2 Add Alpha`. Just set `edge_feather_percent > 0` on the Stitch node. `NB2 Add Alpha` is only needed if you want to pass RGBA to a different compositor.

**Inputs**

| Name | Type | Description |
|---|---|---|
| stitcher | STITCHER | Coordinate data from NB2 Crop |
| inpainted_image | IMAGE | RGB or RGBA output from Nano Banana 2 |
| edge_feather_percent | FLOAT | Edge blend ramp width as % of crop size (0 = hard cut) |

**Outputs** — `IMAGE`

---

### 🔲 NB2 Add Alpha

Converts an RGB image to RGBA by generating a feathered alpha channel. Useful when you want soft-edge RGBA output for a compositor other than NB2 Stitch.

For the standard NB2 retouching workflow, this node is not needed — use `edge_feather_percent` on the Stitch instead.

**Inputs**

| Name | Type | Description |
|---|---|---|
| image | IMAGE | RGB (or RGBA) input |
| feather_percent | FLOAT | Edge fade width as % of image size (0 = hard rectangular alpha) |

**Outputs** — `IMAGE` (RGBA)

---

## Standard workflow

```
[original image]  (any resolution)
      │
      ├─────────────────────────────────────────────────────┐
      │                                                     │
      ▼                                                     │
NB2 Mask Generator                                         │
  aspect_ratio · resolution · center_x / center_y          │
  crop_width (auto-set when resolution changes)            │
      │                                                     │
      │  mask   nb2_width   nb2_height                      │
      ▼                                                     │
NB2 Crop ────────────────────────────────────────────── (canvas)
      │
      │  stitcher      cropped_image (exact NB2 resolution)
      │                      │
      │               [Nano Banana 2]
      │                      │  generated_image (RGB)
      │                      │  (can be any NB2 resolution —
      │                      │   stitch auto-rescales)
      │                      ▼
      └──────────────► NB2 Stitch ◄── edge_feather_percent
                            │
                     [retouched image]
                     (same size as original)
```

---

## Tips

- **Aspect ratio first**: set `aspect_ratio` to match the shape of the region you want to retouch, then choose `resolution`. The crop rectangle in the preview updates automatically.
- **Position with the canvas**: drag the rectangle on the preview to place the crop centre, scroll to resize. No need to run the pipeline to see the position.
- **Quality vs. speed**: generating at 4K on a 2K-sized crop gives more NB2 detail at the cost of processing time. The stitch downscales cleanly. For fast iteration, use 1K or 2K.
- **Feather**: `edge_feather_percent` between 3 and 8 is usually enough for a natural blend. Increase it if the original and generated areas have very different lighting.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/amortegui84/comfyui-inpaint-cropstitch-nb2
```

Restart ComfyUI. No additional Python packages are required beyond those already used by ComfyUI.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
Original work © 2024 lquesada. Modifications © 2025 amortegui84.
