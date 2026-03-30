# ComfyUI-Inpaint-CropStitch-NB2

Fork of [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) by **lquesada**, adapted for use with the **Nano Banana 2** generation model.

> All credits for the original crop/stitch engine go to [lquesada](https://github.com/lquesada).

---

## Nodes

### 🎯 NB2 Mask Generator

Interactive node that generates a rectangular mask at an exact position on the original image.
The mask aspect ratio always matches a Nano Banana 2 output resolution.

**Interactive canvas widget** — connect an image and the preview loads immediately (no re-run needed):

| Control | Action |
|---|---|
| Drag | Move crop centre → updates `center_x` / `center_y` |
| Scroll wheel | Resize crop area → updates `crop_width` |
| Arrow keys | Nudge centre 1 px |
| Shift + Arrow | Nudge centre 10 px |

Changing `resolution` or `aspect_ratio` automatically resizes the crop rectangle to the NB2 output width for that selection (clamped to the image width), so you can immediately see what area will be captured.

**NB2 output sizes:**

| Aspect ratio | 1K | 2K | 4K |
|---|---|---|---|
| 16:9 | 1376 × 768 | 2752 × 1536 | 5504 × 3072 |
| 9:16 | 768 × 1376 | 1536 × 2752 | 3072 × 5504 |
| 1:1  | 1024 × 1024 | 2048 × 2048 | 4096 × 4096 |

**Inputs**

| Name | Type | Description |
|---|---|---|
| image | IMAGE | Source image (used for dimensions and canvas preview) |
| aspect_ratio | choice | 16:9 · 9:16 · 1:1 |
| resolution | choice | 1K · 2K · 4K |
| center_x | INT | Horizontal centre of the crop rectangle (pixels) |
| center_y | INT | Vertical centre of the crop rectangle (pixels) |
| crop_width | INT | Width of the rectangle on the source image. Height is computed from aspect ratio. |

**Outputs** — `MASK`, `nb2_width (INT)`, `nb2_height (INT)`, `preview (IMAGE)`

---

### ✂️ NB2 Crop

Crops the source image around the masked region and **scales the result to the exact NB2 resolution**, so Nano Banana 2 receives a pixel-perfect input.

Wire `nb2_width` and `nb2_height` from **NB2 Mask Generator** to guarantee the crop matches NB2's expected dimensions.
If the source image is smaller than the target resolution, the crop is scaled up automatically.

**Inputs**

| Name | Type | Description |
|---|---|---|
| image | IMAGE | Original source image |
| mask | MASK | Mask from NB2 Mask Generator |
| nb2_width | INT | Target width from NB2 Mask Generator |
| nb2_height | INT | Target height from NB2 Mask Generator |
| downscale_algorithm | choice | Algorithm for downscaling |
| upscale_algorithm | choice | Algorithm for upscaling |

**Outputs** — `stitcher (STITCHER)`, `cropped_image (IMAGE)`, `cropped_mask (MASK)`

---

### ✂️ NB2 Stitch

Composites the NB2-generated image back onto the original canvas using a smoothstep edge feather.

**Blending behaviour:**

- **RGB input** — `edge_feather_percent` controls a smoothstep ramp that runs from 0 at the crop boundary to 1 a few pixels inside. Set to `0` for a hard cut.
- **RGBA input** — the alpha channel is used **directly** as the blend mask; `edge_feather_percent` is ignored. This avoids double-feathering. Use `NB2 Add Alpha` upstream if you want custom alpha control.

> For the standard workflow you do **not** need `NB2 Add Alpha` — `edge_feather_percent` on the Stitch node already produces soft edges. `NB2 Add Alpha` is only needed if you want RGBA output for a different compositor.

**Inputs**

| Name | Type | Description |
|---|---|---|
| stitcher | STITCHER | From NB2 Crop |
| inpainted_image | IMAGE | RGB or RGBA output from Nano Banana 2 |
| edge_feather_percent | FLOAT | Edge feather ramp width as % of crop size (0 = hard cut) |

**Outputs** — `IMAGE`

---

### 🔲 NB2 Add Alpha

Converts an RGB image to RGBA by generating a feathered alpha channel.

Use this only when you need RGBA output for a downstream compositor other than NB2 Stitch.
For the standard NB2 → Stitch workflow, just use `edge_feather_percent` on the Stitch node instead.

**Inputs**

| Name | Type | Description |
|---|---|---|
| image | IMAGE | RGB (or RGBA) input |
| feather_percent | FLOAT | Edge fade width as % of image size (0 = hard rectangular alpha) |

**Outputs** — `IMAGE` (RGBA)

---

## Standard workflow

```
[original image]
      │
      ├─────────────────────────────────────────────┐
      │                                             │
NB2 Mask Generator  ←  aspect_ratio                │
                        resolution                  │
                        center_x / center_y         │
                        crop_width                  │
      │                                             │
      │  mask  nb2_width  nb2_height                │
      ▼                                             │
NB2 Crop ──────────────────────────────────────────┘
      │                                             │
      │  stitcher    cropped_image                  │
      │                  │                          │
      │           [Nano Banana 2]                   │
      │                  │ generated_image (RGB)    │
      │                  ▼                          │
      └──────────► NB2 Stitch ◄── edge_feather_percent
                        │
                  [composited image]
```

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
