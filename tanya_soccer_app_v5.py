import os
import io
import hashlib
import random
from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import streamlit as st
from dotenv import load_dotenv

# Optional Gemini
try:
    from google import genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

# Optional high-quality cutout
try:
    from rembg import remove as _rembg_remove
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False

FloatArr = NDArray[np.float32]

# =====================================================
# App + constants
# =====================================================
load_dotenv()
st.set_page_config(page_title="Tanya Soccer", page_icon="⚽", layout="wide")
st.title("⚽ Tanya Soccer v5")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(APP_DIR, "assets")

def first_existing(*names: str) -> Optional[str]:
    for n in names:
        p = os.path.join(ASSETS_DIR, n)
        if os.path.exists(p):
            return p
    return None

DEFAULT_PALLET_PATH = first_existing("football_palette.png", "football_palette.jpg")
DEFAULT_CARD_PATH   = first_existing("football_card.png", "football_card.jpg")

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Session state
for key, default in [
    ("top_cards_pool", []),
    ("bot_cards_pool", []),
    ("seen_top_hashes", set()),
    ("seen_bot_hashes", set()),
]:
    if key not in st.session_state:
        st.session_state[key] = default  # type: ignore[assignment]

# ---------- Hard-coded pallet calibration (your happy settings) ----------
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
def open_rgba(file_or_path) -> Image.Image:
    img = Image.open(file_or_path)
    return img.convert("RGBA")

def shrink_quad(quad: FloatArr, pad: float) -> FloatArr:
    c = quad.mean(axis=0, keepdims=True)
    return (c + (quad - c) * float(pad)).astype(np.float32)

def bilinear_point(quad: FloatArr, u: float, v: float) -> FloatArr:
    tl, tr, br, bl = quad
    top = tl + (tr - tl) * u
    bot = bl + (br - bl) * u
    return (top + (bot - top) * v).astype(np.float32)

def grid_quads(quad: FloatArr, cols: int, rows: int) -> List[FloatArr]:
    quads: List[FloatArr] = []
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

def perspective_coeffs(src_pts: np.ndarray, dst_pts: np.ndarray):
    matrix = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        matrix.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        matrix.append([0, 0, 0, x, y, 1, -v*x, -v*y])
    A = np.asarray(matrix, dtype=np.float64)
    B = np.asarray(dst_pts, dtype=np.float64).reshape(8)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.tolist())

def warp_rect_to_quad(base_rgba: Image.Image, src_rgba: Image.Image, quad_xy: FloatArr):
    minx = int(np.floor(quad_xy[:, 0].min()))
    maxx = int(np.ceil(quad_xy[:, 0].max()))
    miny = int(np.floor(quad_xy[:, 1].min()))
    maxy = int(np.ceil(quad_xy[:, 1].max()))
    W = max(1, maxx - minx)
    H = max(1, maxy - miny)

    dst_local = quad_xy.copy()
    dst_local[:, 0] -= minx
    dst_local[:, 1] -= miny

    sw, sh = src_rgba.size
    src_rect = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
    coeffs = perspective_coeffs(dst_local, src_rect)

    warped = src_rgba.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    poly_mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(poly_mask).polygon(dst_local.flatten().tolist(), fill=255)

    src_alpha = warped.split()[-1] if warped.mode == "RGBA" else Image.new("L", (W, H), 255)
    final_mask = ImageChops.multiply(src_alpha, poly_mask)

    base_rgba.paste(warped, (minx, miny), final_mask)

def scale_quad_about_center(q: FloatArr, sx: float, sy: float) -> FloatArr:
    c = q.mean(axis=0, keepdims=True)
    tl, tr, br, bl = q
    ax = tr - tl
    ay = bl - tl
    M = np.stack([ax, ay], axis=1)
    if np.linalg.det(M) == 0:
        return q
    Minv = np.linalg.inv(M)
    local = (q - tl) @ Minv
    local -= 0.5
    local[:, 0] *= sx
    local[:, 1] *= sy
    local += 0.5
    out = tl + local @ M
    return (c + (out - c)).astype(np.float32)

# =====================================================
# Smart background cutout
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

def _remove_by_color(img: Image.Image, bg_rgb, tol=28, feather=1):
    import numpy as _np
    arr = _np.asarray(img.convert("RGB")).astype(_np.float32)
    d = _np.sqrt(((arr - bg_rgb) ** 2).sum(axis=-1))
    mask = (d > tol).astype(_np.uint8) * 255
    a = Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(feather))
    out = img.convert("RGBA")
    out.putalpha(a)
    return out

def auto_cutout_smart(img: Image.Image, feather_px: int = 1) -> Image.Image:
    if _has_meaningful_alpha(img):
        return img.convert("RGBA")
    mean, std = _border_stats(img, border=10)
    if std < 6:
        return _remove_by_color(img, mean, tol=28, feather=feather_px)
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
    return _remove_by_color(img, np.array([252, 252, 252], dtype=np.float32), tol=12, feather=feather_px)

# =====================================================
# Compose Tab (Gemini)  — place jewelry on card
# =====================================================
def compose_tab():
    st.subheader("Compose card (Gemini) — optional")
    st.caption("Send a card + jewelry (+ optional reference) to Gemini, then auto-cutout the background.")

    with st.sidebar:
        st.markdown("### Gemini")
        global GEMINI_KEY
        if not GEMINI_KEY:
            st.warning("Add GEMINI_API_KEY to .env or paste it here.")
            GEMINI_KEY = st.text_input("GEMINI_API_KEY", type="password")
        client = genai.Client(api_key=GEMINI_KEY) if (_HAS_GEMINI and GEMINI_KEY) else None

    cols = st.columns(3)
    with cols[0]:
        up_card = st.file_uploader("Card template (required)", type=["png","jpg","jpeg"], key="compose_card")
        if up_card:
            card = open_rgba(up_card)
        elif DEFAULT_CARD_PATH:
            st.info(f"Using default card: **{os.path.basename(DEFAULT_CARD_PATH)}** (assets/)")
            card = open_rgba(DEFAULT_CARD_PATH)
        else:
            card = None
            st.warning("Please upload a card template.")
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
        "Place this jewelry onto this card. Remove the background of the jewelry without changing the "
        "structure or color. Place it within the correct placement area and align naturally with the hole/guide."
    )
    prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=100)

    if st.button("✨ Compose with Gemini", disabled=(client is None or not card or not jewel)):
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

        out_img = auto_cutout_smart(out_img, feather_px=1)

        st.success("Done")
        st.image(out_img, caption="Composed card (transparent background)", use_container_width=True)
        buf = io.BytesIO(); out_img.save(buf, format="PNG")
        st.download_button("Download PNG", buf.getvalue(), "composed_card.png", "image/png")

        add_top = st.checkbox("Add this to TOP pool", value=False)
        add_bot = st.checkbox("Add this to BOTTOM pool", value=False)
        if add_top:
            st.session_state.top_cards_pool.append(out_img.copy())
        if add_bot:
            st.session_state.bot_cards_pool.append(out_img.copy())

# =====================================================
# Helpers for place tab
# =====================================================
def scale_pre_letterbox(card: Image.Image, q: FloatArr) -> Image.Image:
    w_est = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) * 0.5
    h_est = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) * 0.5
    w_est = max(1, int(w_est))
    h_est = max(1, int(h_est))
    cw, ch = card.size
    k = min(w_est / cw, h_est / ch)
    return card.resize((max(1, int(cw * k)), max(1, int(ch * k))), Image.LANCZOS)

def cycle_sequence(pool_imgs: List[Image.Image], unique_take: int, total_slots: int) -> List[Image.Image]:
    if total_slots <= 0 or not pool_imgs:
        return []
    take = max(1, min(unique_take, len(pool_imgs)))
    picked = pool_imgs[:take]
    out: List[Image.Image] = []
    i = 0
    for _ in range(total_slots):
        out.append(picked[i % len(picked)])
        i += 1
    return out

def file_digest(upload) -> str:
    # Stable hash for an UploadedFile
    b = bytes(upload.getbuffer())
    return hashlib.sha1(b).hexdigest()

def uploaded_to_image(upload) -> Image.Image:
    b = bytes(upload.getbuffer())
    return open_rgba(io.BytesIO(b))

def add_files_to_pool(uploads, pool_key: str, seen_key: str) -> Tuple[int, int]:
    """Add only new files (by hash) to the given pool."""
    if not uploads:
        return (0, 0)
    added = 0
    skipped = 0
    seen = st.session_state[seen_key]
    for up in uploads:
        h = file_digest(up)
        if h in seen:
            skipped += 1
            continue
        im = uploaded_to_image(up)
        im = auto_cutout_smart(im)
        st.session_state[pool_key].append(im)
        seen.add(h)
        added += 1
    return (added, skipped)

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
            st.info(f"Using uploaded pallet: **{getattr(up_pallet, 'name', 'pallet')}**")
        elif DEFAULT_PALLET_PATH:
            st.info(f"Using default pallet from assets: **{os.path.basename(DEFAULT_PALLET_PATH)}**")
            pallet = open_rgba(DEFAULT_PALLET_PATH)
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

    # ===== Pools & thumbnails (de-duplicated) =====
    st.markdown("### Card pools (multiple different designs supported)")
    cT, cBtm = st.columns(2)

    with cT:
        ups_top = st.file_uploader("Top zone cards (multiple)", type=["png", "jpg", "jpeg"],
                                   accept_multiple_files=True, key="cards_top")
        added, skipped = add_files_to_pool(ups_top, "top_cards_pool", "seen_top_hashes")
        if added or skipped:
            st.caption(f"Top uploads → added **{added}**, skipped duplicates **{skipped}**.")
        top_pool_size = len(st.session_state.top_cards_pool)
        st.caption(f"Top pool size: **{top_pool_size}**")

        if st.session_state.top_cards_pool:
            st.caption(f"Top pool thumbnails (total: **{top_pool_size}**)")
            tcols = st.columns(min(5, top_pool_size))
            for i, im in enumerate(st.session_state.top_cards_pool[:5]):
                with tcols[i % len(tcols)]:
                    st.image(im, use_container_width=True)

        if st.button("Clear TOP pool"):
            st.session_state.top_cards_pool = []
            st.session_state.seen_top_hashes = set()
            st.rerun()

    with cBtm:
        ups_bot = st.file_uploader("Bottom zone cards (multiple)", type=["png", "jpg", "jpeg"],
                                   accept_multiple_files=True, key="cards_bot")
        added_b, skipped_b = add_files_to_pool(ups_bot, "bot_cards_pool", "seen_bot_hashes")
        if added_b or skipped_b:
            st.caption(f"Bottom uploads → added **{added_b}**, skipped duplicates **{skipped_b}**.")
        bot_pool_size = len(st.session_state.bot_cards_pool)
        st.caption(f"Bottom pool size: **{bot_pool_size}**")

        if st.session_state.bot_cards_pool:
            st.caption(f"Bottom pool thumbnails (total: **{bot_pool_size}**)")
            bcols = st.columns(min(5, bot_pool_size))
            for i, im in enumerate(st.session_state.bot_cards_pool[:5]):
                with bcols[i % len(bcols)]:
                    st.image(im, use_container_width=True)

        if st.button("Clear BOTTOM pool"):
            st.session_state.bot_cards_pool = []
            st.session_state.seen_bot_hashes = set()
            st.rerun()

    shuffle_choice = st.checkbox("Shuffle cards within each zone", value=False)

    # ===== Layout summary & selection per zone =====
    total_rows = int(top_rows + bot_rows)
    top_slots = int(cols) * int(top_rows)
    bot_slots = int(cols) * int(bot_rows)

    st.info(
        f"**Layout summary** — columns: **{cols}**, top rows: **{top_rows}**, bottom rows: **{bot_rows}**, "
        f"**top slots**: **{top_slots}**, **bottom slots**: **{bot_slots}**.\n\n"
        f"• **Top pool** has **{top_pool_size}** items for **{top_slots}** slots → repeats occur if fewer.\n"
        f"• **Bottom pool** has **{bot_pool_size}** items for **{bot_slots}** slots → repeats occur if fewer."
    )

    st.markdown("### Selection per zone")
    # TOP
    if top_slots == 0:
        take_top = 0
        st.caption("Top zone disabled (0 rows).")
    else:
        top_max_take = min(top_pool_size, top_slots)
        if top_max_take <= 0:
            take_top = 0
            st.warning(f"Top pool is empty — add cards to fill {top_slots} slots.")
        elif top_max_take == 1:
            take_top = 1
            st.caption("Top: only one design available → it will repeat for all slots.")
        else:
            take_top = st.slider(
                "How many **unique TOP** cards to use (rest will repeat)",
                min_value=1,
                max_value=int(top_max_take),
                value=int(top_max_take),
                key="take_top_slider",
            )
        st.caption(f"Top: **{take_top}/{top_slots}** selected • pool size: {top_pool_size}")

    # BOTTOM
    if bot_slots == 0:
        take_bot = 0
        st.caption("Bottom zone disabled (0 rows).")
    else:
        bot_max_take = min(bot_pool_size, bot_slots)
        if bot_max_take <= 0:
            take_bot = 0
            st.warning(f"Bottom pool is empty — add cards to fill {bot_slots} slots.")
        elif bot_max_take == 1:
            take_bot = 1
            st.caption("Bottom: only one design available → it will repeat for all slots.")
        else:
            take_bot = st.slider(
                "How many **unique BOTTOM** cards to use (rest will repeat)",
                min_value=1,
                max_value=int(bot_max_take),
                value=int(bot_max_take),
                key="take_bot_slider",
            )
        st.caption(f"Bottom: **{take_bot}/{bot_slots}** selected • pool size: {bot_pool_size}")

    build = st.button("🖼️ Build pallet")
    if not build:
        return

    if pallet is None:
        st.error("Please provide the pallet image.")
        return

    if total_rows <= 0:
        st.error("Total rows is zero.")
        return

    if top_slots > 0 and (take_top == 0 or top_pool_size == 0):
        st.error("Top zone has slots but nothing selected/available.")
        return
    if bot_slots > 0 and (take_bot == 0 or bot_pool_size == 0):
        st.error("Bottom zone has slots but nothing selected/available.")
        return

    with st.spinner("Compositing…"):
        out = pallet.copy().convert("RGBA")

        face = shrink_quad(quads_face, pad=float(face_inset))

        all_cells = grid_quads(face, cols=int(cols), rows=int(total_rows))
        row_quads = [all_cells[i*int(cols):(i+1)*int(cols)] for i in range(total_rows)]
        top_quads = [q for r in row_quads[:int(top_rows)] for q in r]
        bot_quads = [q for r in row_quads[int(top_rows):] for q in r]

        top_seq = cycle_sequence(st.session_state.top_cards_pool, take_top, len(top_quads))
        bot_seq = cycle_sequence(st.session_state.bot_cards_pool, take_bot, len(bot_quads))

        if shuffle_choice and top_seq:
            random.shuffle(top_seq)
        if shuffle_choice and bot_seq:
            random.shuffle(bot_seq)

        def paste_run(quads: List[FloatArr], seq: List[Image.Image]):
            for q, card in zip(quads, seq):
                q2 = shrink_quad(q, pad=float(cell_inset))
                q3 = scale_quad_about_center(q2, sx=float(scale_x), sy=float(scale_y))
                if preserve:
                    pre = scale_pre_letterbox(card, q3)
                    warp_rect_to_quad(out, pre, q3)
                else:
                    warp_rect_to_quad(out, card, q3)

        paste_run(top_quads, top_seq)
        paste_run(bot_quads, bot_seq)

    st.success("Done")
    st.image(out, use_container_width=True)
    buf = io.BytesIO(); out.save(buf, format="PNG")
    st.download_button("Download PNG", buf.getvalue(), "tanya_soccer_pallet.png", "image/png")

# =====================================================
# Run
# =====================================================
tab1, tab2 = st.tabs(["🧩 Compose cards (Gemini)", "🖼️ Place on pallet"])

with tab1:
    compose_tab()

with tab2:
    place_tab()

