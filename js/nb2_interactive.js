/**
 * NB2 Interactive Mask Generator widget
 *
 * The canvas inside the node shows the connected image with a blue
 * crop-rectangle overlay so you can visually position the mask.
 *
 * Image loading — two paths:
 *   1. On connection  → tries app.nodeOutputs to find a cached image from
 *                       the upstream node immediately (no re-run needed).
 *   2. After execution → receives the annotated preview via ui.nb2_preview.
 *
 * Controls (click the canvas to focus it):
 *   Drag          → move crop centre  (updates center_x / center_y)
 *   Scroll wheel  → resize crop_width (maintains aspect ratio)
 *   Arrow keys    → nudge 1 px  |  Shift+Arrow → nudge 10 px
 */

import { app } from "../../scripts/app.js";

// ── helpers ─────────────────────────────────────────────────────────────────

function nb2GetW(node, name) { return node.widgets?.find(w => w.name === name); }
function nb2SetW(node, name, val) { const w = nb2GetW(node, name); if (w) w.value = val; }
function nb2AR(str) {
    if (str === "16:9") return 16 / 9;
    if (str === "9:16") return 9 / 16;
    return 1.0;
}

/** Try to get a cached image URL for a given node id from app.nodeOutputs */
function getCachedImageUrl(nodeId) {
    const out = app.nodeOutputs?.[nodeId];
    if (!out) return null;

    // Standard IMAGE output: { images: [{filename, subfolder, type}] }
    const imgInfo = out.images?.[0];
    if (!imgInfo) return null;

    return app.api.apiURL(
        `/view?filename=${encodeURIComponent(imgInfo.filename)}`
        + `&subfolder=${encodeURIComponent(imgInfo.subfolder ?? "")}`
        + `&type=${imgInfo.type ?? "output"}`
        + `&t=${Date.now()}`
    );
}

// ── extension ───────────────────────────────────────────────────────────────

app.registerExtension({
    name: "Comfy.NB2.InteractiveMaskGen",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "NanoBanana2MaskGen") return;

        // ── onCreate ──────────────────────────────────────────────────────
        const origOnCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnCreated?.apply(this, arguments);
            attachNB2Canvas(this);
        };

        // ── onConnectionsChange: load from cache when image is connected ──
        const origOnConnChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
            origOnConnChange?.apply(this, arguments);

            // type 1 = input slot; index 0 = "image" (first input)
            if (type !== 1 || index !== 0 || !connected || !linkInfo) return;

            const sourceNodeId = String(linkInfo.origin_id);
            const domW = this.widgets?.find(w => w.name === "_nb2_canvas");
            if (!domW?._nb2Load) return;

            // Try immediately from nodeOutputs cache
            const url = getCachedImageUrl(sourceNodeId);
            if (url) {
                const img = new Image();
                img.onload = () => domW._nb2Load(img);
                img.src = url;
                return;
            }

            // Not cached yet — poll once after a short delay (handles
            // cases where the upstream node finishes shortly after connection)
            setTimeout(() => {
                const url2 = getCachedImageUrl(sourceNodeId);
                if (url2) {
                    const img = new Image();
                    img.onload = () => domW._nb2Load(img);
                    img.src = url2;
                }
            }, 800);
        };

        // ── onExecuted: load annotated preview sent by Python ─────────────
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);

            const pData = message?.nb2_preview?.[0];
            if (!pData) return;

            const domW = this.widgets?.find(w => w.name === "_nb2_canvas");
            if (!domW?._nb2Load) return;

            const url = app.api.apiURL(
                `/view?filename=${encodeURIComponent(pData.filename)}`
                + `&subfolder=${encodeURIComponent(pData.subfolder ?? "")}`
                + `&type=${pData.type ?? "temp"}`
                + `&t=${Date.now()}`
            );
            const img = new Image();
            img.onload = () => domW._nb2Load(img);
            img.src = url;
        };
    },
});

// ── canvas widget factory ────────────────────────────────────────────────────

function attachNB2Canvas(node) {
    const CVS_H = 260;

    // ── DOM element ───────────────────────────────────────────────────────
    const cvs = document.createElement("canvas");
    cvs.height = CVS_H;
    Object.assign(cvs.style, {
        display:      "block",
        width:        "100%",
        background:   "#181818",
        border:       "1px solid #2a2a2a",
        borderRadius: "4px",
        cursor:       "crosshair",
        boxSizing:    "border-box",
        outline:      "none",
    });
    cvs.setAttribute("tabindex", "0");

    const ctx = cvs.getContext("2d");

    // ── state ─────────────────────────────────────────────────────────────
    let img      = null;
    let imgW     = 1, imgH = 1;
    let dragging = false;
    let drag     = {};
    let rInfo    = null;

    // ── render ────────────────────────────────────────────────────────────
    function render() {
        const W = Math.max(cvs.offsetWidth || 320, 10);
        cvs.width = W;
        const H = CVS_H;

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#181818";
        ctx.fillRect(0, 0, W, H);

        if (!img) {
            ctx.fillStyle = "#444";
            ctx.font      = "13px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("Connect an image — preview loads automatically", W / 2, H / 2 - 10);
            ctx.fillStyle = "#2e2e2e";
            ctx.font      = "11px sans-serif";
            ctx.fillText("Drag · Scroll to resize · Arrow keys to nudge", W / 2, H / 2 + 12);
            return;
        }

        // letterbox
        const pad = 8;
        const mxW = W - pad * 2, mxH = H - pad * 2;
        const iar = imgW / imgH;
        let dW = mxW, dH = mxW / iar;
        if (dH > mxH) { dH = mxH; dW = mxH * iar; }
        const dX = pad + (mxW - dW) / 2;
        const dY = pad + (mxH - dH) / 2;
        ctx.drawImage(img, dX, dY, dW, dH);

        // current values
        const cx  = nb2GetW(node, "center_x")?.value  ?? Math.round(imgW / 2);
        const cy  = nb2GetW(node, "center_y")?.value  ?? Math.round(imgH / 2);
        const cw  = nb2GetW(node, "crop_width")?.value ?? 100;
        const arS = nb2GetW(node, "aspect_ratio")?.value ?? "16:9";
        const ch  = Math.round(cw / nb2AR(arS));

        const scX = dW / imgW, scY = dH / imgH;
        const rX  = dX + (cx - cw / 2) * scX;
        const rY  = dY + (cy - ch / 2) * scY;
        const rW  = cw * scX, rH = ch * scY;

        // filled rect
        ctx.fillStyle = "rgba(0,140,255,0.15)";
        ctx.fillRect(rX, rY, rW, rH);
        // border
        ctx.strokeStyle = "rgba(0,200,255,0.9)";
        ctx.lineWidth   = 2;
        ctx.strokeRect(rX, rY, rW, rH);
        // centre cross
        const ccX = dX + cx * scX, ccY = dY + cy * scY;
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth   = 1.5;
        ctx.beginPath();
        ctx.moveTo(ccX - 10, ccY); ctx.lineTo(ccX + 10, ccY);
        ctx.moveTo(ccX, ccY - 10); ctx.lineTo(ccX, ccY + 10);
        ctx.stroke();
        // label
        ctx.fillStyle = "rgba(0,210,255,0.9)";
        ctx.font      = "bold 11px monospace";
        ctx.textAlign = "left";
        const labelY  = Math.max(rY + 15, dY + 15);
        ctx.fillText(`${cw} × ${ch} px  |  ${arS}`, rX + 4, labelY);

        rInfo = { dX, dY, scX, scY };
    }

    // ── img coords helper ─────────────────────────────────────────────────
    function toImg(clientX, clientY) {
        if (!rInfo) return { ix: 0, iy: 0 };
        const r  = cvs.getBoundingClientRect();
        const px = (clientX - r.left) * (cvs.width  / r.width);
        const py = (clientY - r.top)  * (cvs.height / r.height);
        return { ix: (px - rInfo.dX) / rInfo.scX, iy: (py - rInfo.dY) / rInfo.scY };
    }

    // ── pointer events ────────────────────────────────────────────────────
    cvs.addEventListener("mousedown", e => {
        if (e.button !== 0 || !img) return;
        e.preventDefault(); cvs.focus();
        const { ix, iy } = toImg(e.clientX, e.clientY);
        dragging = true;
        drag = {
            ix0: ix, iy0: iy,
            cx0: nb2GetW(node, "center_x")?.value ?? Math.round(imgW / 2),
            cy0: nb2GetW(node, "center_y")?.value ?? Math.round(imgH / 2),
        };
    });

    cvs.addEventListener("mousemove", e => {
        if (!dragging) return;
        e.preventDefault();
        const { ix, iy } = toImg(e.clientX, e.clientY);
        nb2SetW(node, "center_x", Math.max(0, Math.min(imgW, Math.round(drag.cx0 + ix - drag.ix0))));
        nb2SetW(node, "center_y", Math.max(0, Math.min(imgH, Math.round(drag.cy0 + iy - drag.iy0))));
        render();
        app.graph.setDirtyCanvas(true, true);
    });

    cvs.addEventListener("mouseup",    () => { dragging = false; });
    cvs.addEventListener("mouseleave", () => { dragging = false; });

    // scroll → resize
    cvs.addEventListener("wheel", e => {
        e.preventDefault();
        const w = nb2GetW(node, "crop_width");
        if (!w) return;
        const step = Math.max(8, Math.round((w.value ?? 100) * 0.04));
        w.value = Math.max(64, Math.min(imgW, w.value + (e.deltaY > 0 ? -step : step)));
        render();
        app.graph.setDirtyCanvas(true, true);
    }, { passive: false });

    // arrow keys
    cvs.addEventListener("keydown", e => {
        const n  = e.shiftKey ? 10 : 1;
        const cxW = nb2GetW(node, "center_x");
        const cyW = nb2GetW(node, "center_y");
        let changed = false;
        if (e.key === "ArrowLeft"  && cxW) { cxW.value = Math.max(0,    cxW.value - n); changed = true; }
        if (e.key === "ArrowRight" && cxW) { cxW.value = Math.min(imgW, cxW.value + n); changed = true; }
        if (e.key === "ArrowUp"    && cyW) { cyW.value = Math.max(0,    cyW.value - n); changed = true; }
        if (e.key === "ArrowDown"  && cyW) { cyW.value = Math.min(imgH, cyW.value + n); changed = true; }
        if (changed) { e.preventDefault(); render(); app.graph.setDirtyCanvas(true, true); }
    });

    // re-render when dropdowns change
    ["aspect_ratio", "resolution"].forEach(name => {
        const w = nb2GetW(node, name);
        if (!w) return;
        const orig = w.callback;
        w.callback = function (...a) { orig?.apply(this, a); setTimeout(render, 20); };
    });

    // ── addDOMWidget ──────────────────────────────────────────────────────
    const domW = node.addDOMWidget("_nb2_canvas", "nb2canvas", cvs, {
        getValue:  () => "",
        setValue:  () => {},
        serialize: false,
    });

    // public API consumed by onConnectionsChange and onExecuted
    domW._nb2Load = (newImg) => {
        img  = newImg;
        imgW = newImg.naturalWidth;
        imgH = newImg.naturalHeight;
        // Set centre defaults to image centre on first connection
        const cxW = nb2GetW(node, "center_x");
        const cyW = nb2GetW(node, "center_y");
        if (cxW && cxW.value === 512) cxW.value = Math.round(imgW / 2);
        if (cyW && cyW.value === 512) cyW.value = Math.round(imgH / 2);
        render();
        app.graph.setDirtyCanvas(true, true);
    };
    domW._nb2Render = render;

    // initial blank frame
    render();
}
