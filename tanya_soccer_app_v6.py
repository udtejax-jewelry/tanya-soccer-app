# tanya_soccer_app_v5.py
import os
import io
import random
from collections import deque
from typing import List, Tuple
from numpy.typing import NDArray

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import streamlit as st
from dotenv import load_dotenv


# -------- Gemini SDK (pure image compose tab) --------
try:
    from google import genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

# -------- Optional high-quality cutout --------
try:
    from rembg import remove as _rembg_remove
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False


# =====================================================
# App + constants
# =====================================================
load_dotenv()
st.set_page_config(page_title="Tanya Soccer", page_icon="⚽", layout="wide")
st.title("⚽ Tanya Soccer v6")

FloatArr = NDArray[np.float32]

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if "top_cards_pool" not in st.session_state:
    st.session_state.top_cards_pool = []  # type: ignore[var-annotated]
if "bot_cards_pool" not in st.session_state:
    st.session_state.bot_cards_pool = []  # type: ignore[var-annotated]

# ---------- Hard-coded pallet calibration (your happy settings) ----------
# Slanted main face corners (screen pixels): TL, TR, BR, BL
HARD_CORNERS = np.array([
    [235.0, 416.0],   # TL_x, TL_y
    [2100.0, 200.0],  # TR_x, TR_y
    [2300.0, 1865.0], # BR_x, BR_y
    [588.0, 2132.0],  # BL_x, BL_y
], dtype=np.float32)

COLS_DEFAULT      = 6
TOP_ROWS_DEFAULT  = 2  # discs
BOT_ROWS_DEFAULT  = 2  # charms
FACE_INSET_PCT    = 0.94   # shrink whole face toward center
CELL_INSET_PCT    = 0.98   # shrink each cell quad a bit
CELL_WIDTH_SCALE  = 0.90
CELL_HEIGHT_SCALE = 0.95
PRESERVE_ASPECT   = True


# =====================================================
# Utilities
# =====================================================
def open_rgba(file) -> Image.Image:
    """Open image as RGBA without altering colors."""
    if file is None:
        return None  # type: ignore[return-value]
    img = Image.open(file)
    if img.mode not in ("RGBA", "RGB", "L", "LA"):
        img = img.convert("RGBA")
    else:
        img = img.convert("RGBA")
    return img

def shrink_quad(quad: np.ndarray, pad: float) -> np.ndarray:
    """Pad < 1.0 shrinks quad toward its centroid."""
    c = quad.mean(axis=0, keepdims=True)
    return c + (quad - c) * float(pad)

def perspective_coeffs(src_pts, dst_pts):
    """
    Compute coefficients for PIL.Image.transform(Image.PERSPECTIVE).
    PIL expects a mapping from OUTPUT->INPUT, so pass src=dst_local, dst=src_rect.
    """
    matrix = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        matrix.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        matrix.append([0, 0, 0, x, y, 1, -v*x, -v*y])
    A = np.asarray(matrix, dtype=np.float64)
    B = np.asarray(dst_pts, dtype=np.float64).reshape(8)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return res

def warp_rect_to_quad(base_rgba: Image.Image, src_rgba: Image.Image, quad_xy: np.ndarray):
    """
    Warp src_rgba rectangle to quad_xy on base_rgba using PIL perspective,
    preserving the source's per-pixel alpha AND clipping to the quad.
    """
    # local bbox for quad
    minx = int(np.floor(quad_xy[:, 0].min()))
    maxx = int(np.ceil(quad_xy[:, 0].max()))
    miny = int(np.floor(quad_xy[:, 1].min()))
    maxy = int(np.ceil(quad_xy[:, 1].max()))
    W = max(1, maxx - minx)
    H = max(1, maxy - miny)

    # quad in local coords
    dst_local = quad_xy.copy()
    dst_local[:, 0] -= minx
    dst_local[:, 1] -= miny

    # src rectangle coords
    sw, sh = src_rgba.size
    src_rect = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)

    # OUTPUT->INPUT mapping for PIL
    coeffs = perspective_coeffs(dst_local, src_rect)

    # Warp the RGBA source
    warped = src_rgba.transform((W, H), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC)

    # Build a polygon mask for the quad (local coords)
    poly_mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(poly_mask).polygon(dst_local.flatten().tolist(), fill=255)

    # Combine source alpha with polygon mask (preserve transparency)
    src_alpha = warped.split()[-1] if warped.mode == "RGBA" else Image.new("L", (W, H), 255)
    final_mask = ImageChops.multiply(src_alpha, poly_mask)

    # Paste using the combined mask (no black boxes)
    base_rgba.paste(warped, (minx, miny), final_mask)

def bilinear_point(quad: FloatArr, u: float, v: float) -> FloatArr:
    # quad order: TL, TR, BR, BL
    tl, tr, br, bl = quad
    top = tl + (tr - tl) * u
    bot = bl + (br - bl) * u
    return top + (bot - top) * v

def grid_quads(quad: FloatArr, cols: int, rows: int) -> list[FloatArr]:
    """Split a quad by bilinear interpolation into rows*cols sub-quads."""
    quads: list[FloatArr] = []
    for r in range(rows):
        v0 = r / rows
        v1 = (r + 1) / rows
        for c in range(cols):
            u0 = c / cols
            u1 = (c + 1) / cols
            q = np.stack([
                bilinear_point(quad, u0, v0),  # TL
                bilinear_point(quad, u1, v0),  # TR
                bilinear_point(quad, u1, v1),  # BR
                bilinear_point(quad, u0, v1),  # BL
            ], axis=0).astype(np.float32)
            quads.append(q)
    return quads

def scale_quad_about_center(q: np.ndarray, sx: float, sy: float) -> np.ndarray:
    c = q.mean(axis=0, keepdims=True)
    # Build local axes from TL->TR (x) and TL->BL (y)
    tl, tr, br, bl = q
    ax = tr - tl
    ay = bl - tl
    # normalize to unit frame
    M = np.stack([ax, ay], axis=1)  # 2x2
    if np.linalg.det(M) == 0:
        return q
    Minv = np.linalg.inv(M)
    local = (q - tl) @ Minv  # 4x2 -> local uv
    # scale in local space around 0.5,0.5
    local -= 0.5
    local[:, 0] *= sx
    local[:, 1] *= sy
    local += 0.5
    # back to world
    out = tl + local @ M
    # recenter
    return c + (out - c)


# =====================================================
# Smart background cutout (edge-connected; keeps interior shapes)
# =====================================================
def _has_meaningful_alpha(img: Image.Image) -> bool:
    if img.mode in ("LA", "RGBA"):
        a = np.asarray(img.split()[-1])
        return a.min() < 255
    return False

def _border_stats(img: Image.Image, border=10):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w, _ = arr.shape
    top = arr[:border, :, :]
    bot = arr[-border:, :, :]
    left = arr[:, :border, :]
    right = arr[:, -border:, :]
    border_pixels = np.concatenate(
        [top.reshape(-1, 3), bot.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0
    )
    mean = border_pixels.mean(axis=0)
    std = border_pixels.std(axis=0).mean()
    return mean, std

def remove_edge_connected_bg(img: Image.Image, tol: int = 24, feather: int = 1, border_band: int = 10) -> Image.Image:
    """Erase ONLY the background similar to the border color AND connected to the image edges."""
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    rgb = arr[..., :3].astype(np.int16)
    h, w, _ = rgb.shape

    # sample border color (median is robust)
    top = rgb[:border_band, :, :]
    bot = rgb[-border_band:, :, :]
    lef = rgb[:, :border_band, :]
    rig = rgb[:, -border_band:, :]
    ref = np.median(
        np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3), lef.reshape(-1, 3), rig.reshape(-1, 3)], 0),
        axis=0
    )

    # pixels similar to border color
    d = np.sqrt(((rgb - ref) ** 2).sum(-1))
    near = d <= tol

    # flood fill from edges over "near"
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        if near[0, x]:     seen[0, x] = True;     q.append((0, x))
        if near[h-1, x]:   seen[h-1, x] = True;   q.append((h-1, x))
    for y in range(h):
        if near[y, 0]:     seen[y, 0] = True;     q.append((y, 0))
        if near[y, w-1]:   seen[y, w-1] = True;   q.append((y, w-1))

    while q:
        y, x = q.popleft()
        for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and near[ny, nx]:
                seen[ny, nx] = True; q.append((ny, nx))

    # delete only the connected background (keep interior whites)
    alpha = arr[..., 3].copy()
    alpha[seen] = 0
    a_img = Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(feather)) if feather > 0 else Image.fromarray(alpha, "L")

    out = rgba.copy()
    out.putalpha(a_img)
    return out

# (kept as ultimate fallback)
def _remove_by_color(img: Image.Image, bg_rgb, tol=28, feather=1):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    d = np.sqrt(((arr - bg_rgb) ** 2).sum(axis=-1))
    mask = (d > tol).astype(np.uint8) * 255
    a = Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(feather))
    out = img.convert("RGBA")
    out.putalpha(a)
    return out

def auto_cutout_smart(img: Image.Image, feather_px: int = 1) -> Image.Image:
    """
    1) If input already has transparency, keep it.
    2) If border is uniform, remove only edge-connected background (protects interior whites).
    3) Else try rembg.
    4) Fallback to edge-connected removal again with slightly tighter tolerance.
    """
    if _has_meaningful_alpha(img):
        return img.convert("RGBA")

    _, std = _border_stats(img, border=10)
    if std < 6:
        return remove_edge_connected_bg(img, tol=24, feather=feather_px, border_band=10)

    if _HAS_REMBG:
        try:
            buf = io.BytesIO(); img.save(buf, format="PNG")
            cut = _rembg_remove(buf.getvalue())
            out = Image.open(io.BytesIO(cut)).convert("RGBA")
            if feather_px > 0:
                a = out.split()[-1].filter(ImageFilter.GaussianBlur(feather_px))
                out.putalpha(a)
            return out
        except Exception:
            pass

    # last resort (still safer than plain color-thresholding)
    return remove_edge_connected_bg(img, tol=22, feather=feather_px, border_band=10)


# =====================================================
# Compose Tab (Gemini) — place jewelry on card
# =====================================================
def compose_tab():
    st.subheader("Compose card (Gemini) — optional")
    st.caption("Send a card + jewelry (+ optional reference) to Gemini, then auto-cutout the background with edge-connected removal.")

    # Sidebar config for Gemini key
    with st.sidebar:
        st.markdown("### Gemini")
        global GEMINI_KEY
        if not GEMINI_KEY:
            st.warning("Add GEMINI_API_KEY to .env or paste it here.")
            GEMINI_KEY = st.text_input("GEMINI_API_KEY", type="password")
        client = genai.Client(api_key=GEMINI_KEY) if (_HAS_GEMINI and GEMINI_KEY) else None

    # Inputs
    cols = st.columns(3)
    with cols[0]:
        up_card = st.file_uploader("Card template (required)", type=["png","jpg","jpeg"], key="compose_card")
        card = open_rgba(up_card) if up_card else None
        if card: st.image(card, caption="Card", use_container_width=True)
    with cols[1]:
        up_jewel = st.file_uploader("Jewelry (required)", type=["png","jpg","jpeg"], key="compose_jewel")
        jewel = open_rgba(up_jewel) if up_jewel else None
        if jewel: st.image(jewel, caption="Jewelry", use_container_width=True)
    with cols[2]:
        up_ref = st.file_uploader("Reference (optional)", type=["png","jpg","jpeg"], key="compose_ref")
        ref = open_rgba(up_ref) if up_ref else None
        if ref: st.image(ref, caption="Reference", use_container_width=True)

    DEFAULT_PROMPT = (
        "Place this jewelry onto this card. Remove the background of the jewelry without changing the structure or color. Place it within the correct placement area and align naturally with the hole/guide. Keep colors accurate. Do not let the jewelry fall outside the card."
    )
    prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=100)

    if st.button("✨ Compose with Gemini", use_container_width=True, disabled=(client is None or not card or not jewel)):
        if client is None:
            st.error("No Gemini client configured.")
            return
        with st.spinner("Calling Gemini…"):
            contents = [prompt]
            if ref is not None:
                contents += ["Use the next image only as placement/scale reference.", ref]
            else:
                contents += ["No reference provided. Use the geometry and prompt."]
            contents += ["This is the card template:", card,
                         "This is the jewelry to remove background and place:", jewel]
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash-image-preview",
                    contents=contents,
                )
            except Exception as e:
                st.exception(e)
                return

        out_img = None
        if getattr(resp, "candidates", None):
            for part in resp.candidates[0].content.parts:
                if getattr(part, "inline_data", None) is not None:
                    try:
                        out_img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        break
                    except Exception:
                        pass
        if out_img is None:
            st.error("Gemini returned no image.")
            return

        # Safe cutout (edge-connected)
        out_img = auto_cutout_smart(out_img, feather_px=1)

        st.success("Done")
        st.image(out_img, caption="Composed card (transparent background preserved correctly)", use_container_width=True)
        buf = io.BytesIO(); out_img.save(buf, format="PNG")
        st.download_button("Download PNG", buf.getvalue(), "composed_card.png", "image/png", use_container_width=True)

        # Add to pools quickly
        add_top = st.checkbox("Add this to TOP pool", value=False)
        add_bot = st.checkbox("Add this to BOTTOM pool", value=False)
        if add_top:
            st.session_state.top_cards_pool.append(out_img.copy())
        if add_bot:
            st.session_state.bot_cards_pool.append(out_img.copy())


# =====================================================
# Place Tab — pallet mock
# =====================================================
def place_tab():
    st.subheader("Place cards on pallet")

    # Pallet image
    cA, cB = st.columns([2, 1])
    with cA:
        up_pallet = st.file_uploader("Upload pallet base (PNG/JPG).", type=["png", "jpg", "jpeg"], key="pallet")
        if up_pallet:
            pallet = open_rgba(up_pallet)
        else:
            st.warning("Please upload the half-pallet base image.")
            pallet = None

    # Face calibration (with lock)
    with cB:
        st.markdown("#### Face calibration")
        unlock = st.checkbox("Unlock fine-tune corners", value=False)
        q = HARD_CORNERS.copy()
        tlx = st.number_input("TL_x", value=float(q[0, 0]), step=1.0, disabled=not unlock)
        tly = st.number_input("TL_y", value=float(q[0, 1]), step=1.0, disabled=not unlock)
        trx = st.number_input("TR_x", value=float(q[1, 0]), step=1.0, disabled=not unlock)
        try_ = st.number_input("TR_y", value=float(q[1, 1]), step=1.0, disabled=not unlock)
        brx = st.number_input("BR_x", value=float(q[2, 0]), step=1.0, disabled=not unlock)
        bry = st.number_input("BR_y", value=float(q[2, 1]), step=1.0, disabled=not unlock)
        blx = st.number_input("BL_x", value=float(q[3, 0]), step=1.0, disabled=not unlock)
        bly = st.number_input("BL_y", value=float(q[3, 1]), step=1.0, disabled=not unlock)
        quads_face = np.array([[tlx, tly], [trx, try_], [brx, bry], [blx, bly]], dtype=np.float32)


    st.markdown("#### Global spacing / scaling")
    c1, c2, c3 = st.columns(3)
    with c1:
        cols = st.number_input("Columns (total)", 1, 12, COLS_DEFAULT, 1)
        top_rows = st.number_input("Top rows (discs)", 0, 6, TOP_ROWS_DEFAULT, 1)
        bot_rows = st.number_input("Bottom rows (charms)", 0, 6, BOT_ROWS_DEFAULT, 1)
        preserve = st.checkbox("Preserve card aspect (letterbox)", value=PRESERVE_ASPECT)
    with c2:
        face_inset = st.slider("Face inset %", 0.80, 1.00, FACE_INSET_PCT, 0.01)
        cell_inset = st.slider("Inset inside each cell", 0.90, 1.00, CELL_INSET_PCT, 0.01)
    with c3:
        scale_x = st.slider("Cell width scale ×", 0.70, 1.20, CELL_WIDTH_SCALE, 0.01)
        scale_y = st.slider("Cell height scale ×", 0.70, 1.20, CELL_HEIGHT_SCALE, 0.01)

    st.markdown("#### Card pools (multiple different designs supported)")
    cT, cBtm = st.columns(2)
    with cT:
        ups_top = st.file_uploader("Top zone cards (multiple)", type=["png", "jpg", "jpeg"],
                                   accept_multiple_files=True, key="cards_top")
        if ups_top:
            st.session_state.top_cards_pool.extend([auto_cutout_smart(open_rgba(f)) for f in ups_top])
        st.caption(f"Top pool size: {len(st.session_state.top_cards_pool)}")
        if st.button("Clear TOP pool", key="clear_top"):
            st.session_state.top_cards_pool = []
            st.rerun()
        # thumbnails
        if st.session_state.top_cards_pool:
            th_cols = st.columns(min(6, len(st.session_state.top_cards_pool)))
            for i, im in enumerate(st.session_state.top_cards_pool[:6]):
                with th_cols[i]:
                    st.image(im, caption=f"{i+1}", use_container_width=True)

    with cBtm:
        ups_bot = st.file_uploader("Bottom zone cards (multiple)", type=["png", "jpg", "jpeg"],
                                   accept_multiple_files=True, key="cards_bot")
        if ups_bot:
            st.session_state.bot_cards_pool.extend([auto_cutout_smart(open_rgba(f)) for f in ups_bot])
        st.caption(f"Bottom pool size: {len(st.session_state.bot_cards_pool)}")
        if st.button("Clear BOTTOM pool", key="clear_bot"):
            st.session_state.bot_cards_pool = []
            st.rerun()
        if st.session_state.bot_cards_pool:
            th_cols = st.columns(min(6, len(st.session_state.bot_cards_pool)))
            for i, im in enumerate(st.session_state.bot_cards_pool[:6]):
                with th_cols[i]:
                    st.image(im, caption=f"{i+1}", use_container_width=True)

    shuffle_choice = st.checkbox("Shuffle cards within each zone", value=False)

    build = st.button("🖼️ Build pallet", type="primary", use_container_width=True)
    if not build:
        return

    if pallet is None:
        st.error("Please upload the pallet image.")
        return

    if (top_rows > 0 and len(st.session_state.top_cards_pool) == 0) or \
       (bot_rows > 0 and len(st.session_state.bot_cards_pool) == 0):
        st.error("Please add at least one card into each zone's pool (Top/Bottom).")
        return

    with st.spinner("Compositing…"):
        out = pallet.copy().convert("RGBA")

        # 1) Build inner face quad
        face = shrink_quad(quads_face, pad=float(face_inset))

        total_rows = int(top_rows + bot_rows)
        if total_rows <= 0:
            st.error("Total rows is zero.")
            return

        # 2) All cell quads on face
        all_cells = grid_quads(face, cols=int(cols), rows=int(total_rows))

        def paste_zone(cell_quads: List[np.ndarray], imgs: List[Image.Image], label: str):
            if not cell_quads or not imgs:
                return
            seq = imgs[:]
            if shuffle_choice:
                random.shuffle(seq)
            # cycle images
            idx = 0
            for q in cell_quads:
                # Inset inside each cell and apply anisotropic scale
                q2 = shrink_quad(q, pad=float(cell_inset))
                q3 = scale_quad_about_center(q2, sx=float(scale_x), sy=float(scale_y))

                card = seq[idx % len(seq)]
                idx += 1

                # Letterbox preserve aspect: resize source beforehand within a box proportional to the cell
                if preserve:
                    # Estimate cell size in pixels
                    w_est = (np.linalg.norm(q3[1] - q3[0]) + np.linalg.norm(q3[2] - q3[3])) * 0.5
                    h_est = (np.linalg.norm(q3[3] - q3[0]) + np.linalg.norm(q3[2] - q3[1])) * 0.5
                    w_est = max(1, int(w_est))
                    h_est = max(1, int(h_est))
                    cw, ch = card.size
                    k = min(w_est / cw, h_est / ch)
                    pre = card.resize((max(1, int(cw * k)), max(1, int(ch * k))), Image.LANCZOS)
                    warp_rect_to_quad(out, pre, q3)
                else:
                    warp_rect_to_quad(out, card, q3)

        # 3) Split cells into top zone and bottom zone (row-major)
        row_quads = [all_cells[i*cols:(i+1)*cols] for i in range(total_rows)]
        top_quads = [q for r in row_quads[:int(top_rows)] for q in r]
        bot_quads = [q for r in row_quads[int(top_rows):] for q in r]

        paste_zone(top_quads, st.session_state.top_cards_pool, "TOP")
        paste_zone(bot_quads, st.session_state.bot_cards_pool, "BOTTOM")

    st.success("Done")
    st.image(out, use_container_width=True)
    buf = io.BytesIO(); out.save(buf, format="PNG")
    st.download_button("Download PNG", buf.getvalue(), "tanya_soccer_pallet.png", "image/png", use_container_width=True)


# =====================================================
# Run
# =====================================================
tab1, tab2 = st.tabs(["🧩 Compose cards (Gemini)", "🖼️ Place on pallet"])

with tab1:
    compose_tab()

with tab2:
    place_tab()
