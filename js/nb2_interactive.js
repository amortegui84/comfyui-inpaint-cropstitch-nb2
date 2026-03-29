/**
 * NB2 Interactive Mask Generator widget
 *
 * After the first execution the node shows an annotated preview of the
 * original image with a blue crop rectangle overlaid.
 *
 * Controls:
 *   Drag          → move the crop centre (updates center_x / center_y)
 *   Scroll wheel  → resize crop_width (maintains aspect ratio)
 *   Arrow keys    → nudge center_x / center_y by 1 px (when the canvas
 *                   is focused / hovered)
 */

import { app } from "../../scripts/app.js";

// ── helpers ─────────────────────────────────────────────────────────────────

function getW(node, name) {
    return node.widgets?.find(w => w.name === name);
}

function setW(node, name, val) {
    const w = getW(node, name);
    if (w) w.value = val;
}

function getAR(str) {
    if (str === "16:9") return 16 / 9;
    if (str === "9:16") return 9 / 16;
    return 1.0;   // 1:1
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
            const node = this;

            // ── DOM canvas ────────────────────────────────────────────────
            const cvs = document.createElement("canvas");
            cvs.height = 250;
            Object.assign(cvs.style, {
                display:       "block",
                width:         "100%",
                background:    "#181818",
                border:        "1px solid #333",
                borderRadius:  "4px",
                cursor:        "crosshair",
                boxSizing:     "border-box",
                outline:       "none",
            });
            cvs.setAttribute("tabindex", "0");   // keyboard focus

            const ctx = cvs.getContext("2d");

            // ── state ─────────────────────────────────────────────────────
            let img      = null;
            let imgW     = 1, imgH = 1;
            let dragging = false;
            let drag     = {};
            let rInfo    = null;   // last render info (image rect on canvas)

            // ── render ────────────────────────────────────────────────────
            function render() {
                const W = cvs.offsetWidth  || 320;
                const H = cvs.height;
                cvs.width = W;

                ctx.clearRect(0, 0, W, H);
                ctx.fillStyle = "#181818";
                ctx.fillRect(0, 0, W, H);

                if (!img) {
                    ctx.fillStyle = "#555";
                    ctx.font      = "13px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("▶  Run once to enable interactive preview", W / 2, H / 2 - 10);
                    ctx.fillStyle = "#3a3a3a";
                    ctx.font      = "11px sans-serif";
                    ctx.fillText("Drag · Scroll to resize · Arrow keys to nudge", W / 2, H / 2 + 12);
                    return;
                }

                // letterbox fit
                const pad = 8;
                const mxW = W - pad * 2, mxH = H - pad * 2;
                const iar = imgW / imgH;
                let dW = mxW, dH = mxW / iar;
                if (dH > mxH) { dH = mxH; dW = mxH * iar; }
                const dX = pad + (mxW - dW) / 2;
                const dY = pad + (mxH - dH) / 2;

                ctx.drawImage(img, dX, dY, dW, dH);

                // current widget values
                const cx  = getW(node, "center_x")?.value  ?? Math.round(imgW / 2);
                const cy  = getW(node, "center_y")?.value  ?? Math.round(imgH / 2);
                const cw  = getW(node, "crop_width")?.value ?? 100;
                const arS = getW(node, "aspect_ratio")?.value ?? "16:9";
                const ar  = getAR(arS);
                const ch  = Math.round(cw / ar);

                const scX = dW / imgW, scY = dH / imgH;

                const rX = dX + (cx - cw / 2) * scX;
                const rY = dY + (cy - ch / 2) * scY;
                const rW = cw * scX;
                const rH = ch * scY;

                // fill
                ctx.fillStyle = "rgba(0,140,255,0.15)";
                ctx.fillRect(rX, rY, rW, rH);
                // border
                ctx.strokeStyle = "rgba(0,200,255,0.95)";
                ctx.lineWidth   = 2;
                ctx.strokeRect(rX, rY, rW, rH);
                // centre cross
                const ccX = dX + cx * scX, ccY = dY + cy * scY;
                ctx.strokeStyle = "rgba(255,255,255,0.85)";
                ctx.lineWidth   = 1.5;
                ctx.beginPath();
                ctx.moveTo(ccX - 10, ccY); ctx.lineTo(ccX + 10, ccY);
                ctx.moveTo(ccX, ccY - 10); ctx.lineTo(ccX, ccY + 10);
                ctx.stroke();
                // size label
                ctx.fillStyle = "rgba(0,210,255,0.9)";
                ctx.font      = "bold 11px monospace";
                ctx.textAlign = "left";
                ctx.fillText(`${cw} × ${ch} px`, rX + 4, Math.max(rY + 14, dY + 14));

                rInfo = { dX, dY, dW, dH, scX, scY };
            }

            // ── coord helper ──────────────────────────────────────────────
            function toImg(clientX, clientY) {
                if (!rInfo) return { ix: 0, iy: 0 };
                const r  = cvs.getBoundingClientRect();
                const px = (clientX - r.left) * (cvs.width  / r.width);
                const py = (clientY - r.top)  * (cvs.height / r.height);
                return {
                    ix: (px - rInfo.dX) / rInfo.scX,
                    iy: (py - rInfo.dY) / rInfo.scY,
                };
            }

            // ── mouse events ──────────────────────────────────────────────
            cvs.addEventListener("mousedown", e => {
                if (e.button !== 0 || !img) return;
                e.preventDefault();
                cvs.focus();
                const { ix, iy } = toImg(e.clientX, e.clientY);
                dragging = true;
                drag = {
                    ix0: ix, iy0: iy,
                    cx0: getW(node, "center_x")?.value ?? Math.round(imgW / 2),
                    cy0: getW(node, "center_y")?.value ?? Math.round(imgH / 2),
                };
            });

            cvs.addEventListener("mousemove", e => {
                if (!dragging) return;
                e.preventDefault();
                const { ix, iy } = toImg(e.clientX, e.clientY);
                const newCx = Math.max(0, Math.min(imgW, Math.round(drag.cx0 + ix - drag.ix0)));
                const newCy = Math.max(0, Math.min(imgH, Math.round(drag.cy0 + iy - drag.iy0)));
                setW(node, "center_x", newCx);
                setW(node, "center_y", newCy);
                render();
                app.graph.setDirtyCanvas(true, true);
            });

            cvs.addEventListener("mouseup",    () => { dragging = false; });
            cvs.addEventListener("mouseleave", () => { dragging = false; });

            // scroll → resize crop_width
            cvs.addEventListener("wheel", e => {
                e.preventDefault();
                const w = getW(node, "crop_width");
                if (!w) return;
                const step = Math.max(8, Math.round((w.value ?? 100) * 0.04));
                w.value = Math.max(64, Math.min(imgW, w.value + (e.deltaY > 0 ? -step : step)));
                render();
                app.graph.setDirtyCanvas(true, true);
            }, { passive: false });

            // arrow keys → nudge centre
            cvs.addEventListener("keydown", e => {
                const nudge = e.shiftKey ? 10 : 1;
                const cxW = getW(node, "center_x");
                const cyW = getW(node, "center_y");
                let changed = false;
                if (e.key === "ArrowLeft"  && cxW) { cxW.value = Math.max(0, cxW.value - nudge); changed = true; }
                if (e.key === "ArrowRight" && cxW) { cxW.value = Math.min(imgW, cxW.value + nudge); changed = true; }
                if (e.key === "ArrowUp"    && cyW) { cyW.value = Math.max(0, cyW.value - nudge); changed = true; }
                if (e.key === "ArrowDown"  && cyW) { cyW.value = Math.min(imgH, cyW.value + nudge); changed = true; }
                if (changed) {
                    e.preventDefault();
                    render();
                    app.graph.setDirtyCanvas(true, true);
                }
            });

            // re-render when aspect_ratio dropdown changes
            const arWidget = getW(node, "aspect_ratio");
            if (arWidget) {
                const origCb = arWidget.callback;
                arWidget.callback = function (...args) {
                    origCb?.apply(this, args);
                    setTimeout(render, 20);
                };
            }

            // ── add DOM widget ────────────────────────────────────────────
            const domW = node.addDOMWidget(
                "_nb2_canvas", "nb2canvas", cvs,
                { getValue: () => "", setValue: () => {}, serialize: false }
            );

            // expose for onExecuted
            domW._nb2Load = (newImg) => {
                img  = newImg;
                imgW = newImg.naturalWidth;
                imgH = newImg.naturalHeight;
                // default centres to image centre on first load
                const cxW = getW(node, "center_x");
                const cyW = getW(node, "center_y");
                if (cxW && cxW.value === 512) cxW.value = Math.round(imgW / 2);
                if (cyW && cyW.value === 512) cyW.value = Math.round(imgH / 2);
                render();
                app.graph.setDirtyCanvas(true, true);
            };
            domW._nb2Render = render;
        };

        // ── onExecuted: load preview from server ──────────────────────────
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
            const newImg = new Image();
            newImg.onload = () => domW._nb2Load(newImg);
            newImg.src    = url;
        };
    },
});
