#!/usr/bin/env python3
"""
GitHub Profile Banner Generator
================================
Generates dark.svg and light.svg animated banners for VIVEKVRON's GitHub profile.

Pipeline:
  1. Portrait → Floyd-Steinberg dither → dot coordinates
  2. Logo shapes → sampled dot coordinates
  3. Optimal transport matching between logos
  4. SVG assembly with SMIL animation (intro shimmer + logo morph loop)

Source of truth: .npy files + this script. The SVGs are derived artifacts.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_closing, binary_fill_holes, label as ndlabel
import math
import os
import sys
import json
import hashlib

# ============================================================
# CONFIGURATION
# ============================================================
PORTRAIT_W, PORTRAIT_H = 300, 340
BANNER_W, BANNER_H = 1180, 610
DOT_SIZE = 2  # px per dot in SVG

# Portrait placement within banner
PORTRAIT_LEFT = 60
PORTRAIT_TOP = 95
PORTRAIT_PANEL_W = int(BANNER_W * 0.38)  # ~448px

# Colors
COLORS = {
    "dark": {
        "bg": "#0A101F",
        "panel_bg": "#0F1629",
        "title_bar": "#151B2C",
        "dot": "#A78BFA",
        "chrome": "#22D3EE",
        "accent": "#108981",
        "text": "#C8D0E0",
        "text_dim": "#6B7394",
        "text_value": "#E2E8F0",
        "live_bg": "#DC2626",
        "live_text": "#FFFFFF",
        "pill_bg": "#22D3EE",
        "pill_text": "#0A101F",
        "border": "#1E293B",
    },
    "light": {
        "bg": "#F0F2F5",
        "panel_bg": "#FFFFFF",
        "title_bar": "#E5E7EB",
        "dot": "#7C3AED",
        "chrome": "#089182",
        "accent": "#108981",
        "text": "#374151",
        "text_dim": "#9CA3AF",
        "text_value": "#1F2937",
        "live_bg": "#DC2626",
        "live_text": "#FFFFFF",
        "pill_bg": "#089182",
        "pill_text": "#FFFFFF",
        "border": "#D1D5DB",
    },
}

# Animation timing (seconds)
INTRO_TOTAL = 3.2
INTRO_FADE = 2.0
LOOP_DURATION = 14.2
PORTRAIT_HOLD = 3.0
LOGO_HOLD = 2.0
TRANSITION = 1.3

# Counts
NUM_INTRO_GROUPS = 60
NUM_DRIFT_BANDS = 94
NUM_TRAVELLERS = 900
NOISE_SIGMA = 4.0

# Info panel data
INFO_ROWS = [
    ("Subject", "Vivek V Ron"),
    ("Role", "ML Engineer"),
    ("Origin", "Hubballi, KA, IN"),
    ("Education", "B.E. CSE, MSEC"),
    ("Status", "Building intelligent web apps"),
    ("ToolChain", "VS Code, Git, GCS, Maven, K8s"),
    ("Core.Lang", "Python, Java, C, C++"),
    ("Core.Frontend", "Web UI / Minimalist"),
    ("Core.Backend", "Flask, Spring Boot 3"),
    ("Core.Database", "Vector Databases"),
    ("Core.Infra", "Google Cloud Console"),
    ("Grid.Mail", "vivekvron@gmail.com"),
    ("Grid.Portfolio", "coming soon"),
    ("Grid.LinkedIn", "vivek-ron"),
    ("Grid.GitHub", "VIVEKVRON"),
    ("Grid.Facebook", "Vivek Ron"),
]

FONT_FAMILY = "'Courier New', 'Consolas', monospace"

# ============================================================
# PORTRAIT PROCESSING
# ============================================================
def load_and_crop_portrait(image_path):
    """Load portrait, crop head+shoulders (not tight face), resize to PORTRAIT_W x PORTRAIT_H."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # For this photo: crop to upper ~80% to get head+shoulders with coffee cup
    # Avoid tight face crop (the spec says "over-zoomed reads aggressive")
    crop_top = int(h * 0.0)
    crop_bottom = int(h * 0.92)
    crop_left = int(w * 0.08)
    crop_right = int(w * 0.92)
    img = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Resize to fit portrait area, maintaining aspect ratio then center-crop
    img_ratio = img.width / img.height
    target_ratio = PORTRAIT_W / PORTRAIT_H

    if img_ratio > target_ratio:
        # Wider than target: fit height, crop width
        new_h = PORTRAIT_H
        new_w = int(new_h * img_ratio)
    else:
        # Taller than target: fit width, crop height
        new_w = PORTRAIT_W
        new_h = int(new_w / img_ratio)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - PORTRAIT_W) // 2
    top = (new_h - PORTRAIT_H) // 2
    img = img.crop((left, top, left + PORTRAIT_W, top + PORTRAIT_H))

    return img


def enhance_portrait(img):
    """Apply autocontrast, contrast boost, and unsharp mask per spec."""
    # Autocontrast with cutoff=1
    img = ImageOps.autocontrast(img, cutoff=1)
    # Contrast 1.3x
    img = ImageEnhance.Contrast(img).enhance(1.3)
    # UnsharpMask(radius=3, percent=140)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    return img


def floyd_steinberg_dither_serpentine(gray_array):
    """
    1-bit Floyd-Steinberg dither with serpentine (boustrophedon) scan order.
    Input: 2D float array [0, 255]. Output: binary array (0 or 255).
    """
    h, w = gray_array.shape
    img = gray_array.astype(np.float64)
    out = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        if y % 2 == 0:
            # Left to right
            x_range = range(w)
            dx_right, dx_bl, dx_b, dx_br = 1, -1, 0, 1
        else:
            # Right to left
            x_range = range(w - 1, -1, -1)
            dx_right, dx_bl, dx_b, dx_br = -1, 1, 0, -1

        for x in x_range:
            old_val = img[y, x]
            new_val = 255.0 if old_val >= 128.0 else 0.0
            out[y, x] = int(new_val)
            error = old_val - new_val

            # Distribute error
            if 0 <= x + dx_right < w:
                img[y, x + dx_right] += error * 7.0 / 16.0
            if y + 1 < h:
                if 0 <= x + dx_bl < w:
                    img[y + 1, x + dx_bl] += error * 3.0 / 16.0
                img[y + 1, x + dx_b] += error * 5.0 / 16.0
                if 0 <= x + dx_br < w:
                    img[y + 1, x + dx_br] += error * 1.0 / 16.0

    return out


def segment_subject(rgb_img):
    """
    Segment the subject from background for dark mode.
    Uses color-distance thresholding + morphological cleanup.
    Returns binary mask: True = subject.
    """
    arr = np.array(rgb_img, dtype=np.float64)

    # Sample background from corners (top-left, top-right, bottom-left, bottom-right)
    corners = []
    margin = 15
    for cy in [margin, arr.shape[0] - margin]:
        for cx in [margin, arr.shape[1] - margin]:
            corners.append(arr[cy - margin:cy + margin, cx - margin:cx + margin].reshape(-1, 3))
    bg_samples = np.vstack(corners)
    bg_color = np.median(bg_samples, axis=0)

    # Color distance from background
    dist = np.sqrt(np.sum((arr - bg_color) ** 2, axis=2))

    # Threshold (adaptive: use Otsu-like on the distance map)
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(dist)
    mask = dist > thresh * 0.7  # Slightly permissive to keep edges

    # Morphological cleanup
    struct = np.ones((7, 7), dtype=bool)
    mask = binary_closing(mask, structure=struct, iterations=3)
    mask = binary_fill_holes(mask)

    # Keep only the largest connected component
    labeled, num_features = ndlabel(mask)
    if num_features > 1:
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    return mask


def process_portrait(image_path):
    """
    Full portrait processing pipeline.
    Returns:
        dots_light: (N, 2) array of (x, y) dot coordinates for light mode
        dots_dark:  (M, 2) array of (x, y) dot coordinates for dark mode (subject only)
    """
    print("[1/6] Loading and cropping portrait...")
    img = load_and_crop_portrait(image_path)

    print("[2/6] Enhancing portrait...")
    img_enhanced = enhance_portrait(img)

    print("[3/6] Segmenting subject for dark mode...")
    subject_mask = segment_subject(img_enhanced)

    print("[4/6] Converting to grayscale and dithering...")
    gray = np.array(img_enhanced.convert("L"), dtype=np.float64)

    # --- LIGHT MODE ---
    # Dots where dark (standard dithering: dark areas → dots on light background)
    # Use gamma to control density: target ~20k dots (not subsampled)
    TARGET_DOTS = 20000
    TOLERANCE = 3000

    gray_light = _adjust_density(gray.copy(), TARGET_DOTS, TOLERANCE, invert=False)
    dithered_light = floyd_steinberg_dither_serpentine(gray_light)
    light_ys, light_xs = np.where(dithered_light == 0)
    dots_light = np.column_stack([light_xs, light_ys])

    # --- DARK MODE ---
    # CRITICAL FIX: Invert grayscale BEFORE dithering.
    # This makes bright skin → dark → dots, dark hair → bright → no dots.
    # On a dark background, the lit subject pops out correctly.
    # Without inversion, you get a photo-negative effect.
    gray_inv = 255.0 - gray

    gray_dark = _adjust_density(gray_inv.copy(), TARGET_DOTS, TOLERANCE, invert=False)
    dithered_dark = floyd_steinberg_dither_serpentine(gray_dark)

    # Mask to subject only (dots where dithered == 0, within subject)
    from scipy.ndimage import binary_erosion
    dark_ys, dark_xs = np.where((dithered_dark == 0) & subject_mask)
    dots_dark = np.column_stack([dark_xs, dark_ys])

    # Hard-clear error-diffusion bleed at mask edge
    eroded_mask = binary_erosion(subject_mask, iterations=2)
    dark_keep = eroded_mask[dots_dark[:, 1], dots_dark[:, 0]]
    dots_dark = dots_dark[dark_keep]

    print(f"    Light mode dots: {len(dots_light)}")
    print(f"    Dark mode dots: {len(dots_dark)}")

    return dots_light, dots_dark, img_enhanced


def _adjust_density(gray, target, tolerance, invert=False):
    """
    Adjust image brightness via gamma to make Floyd-Steinberg produce
    approximately `target` dark dots. This preserves dither quality
    (unlike random subsampling which destroys the error diffusion pattern).
    """
    best_gray = gray.copy()
    best_count = None

    for gamma in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]:
        adjusted = 255.0 * np.power(np.clip(gray / 255.0, 0, 1), gamma)
        dithered = floyd_steinberg_dither_serpentine(adjusted.copy())
        count = np.sum(dithered == 0)

        if best_count is None or abs(count - target) < abs(best_count - target):
            best_count = count
            best_gray = adjusted.copy()
            best_gamma = gamma

        if abs(count - target) < tolerance:
            break

    print(f"    Gamma {best_gamma:.1f} -> {best_count} dots (target {target})")
    return best_gray


# ============================================================
# LOGO DOT GENERATION
# ============================================================
def create_python_logo(size=200):
    """Create a simplified Python logo as a binary mask."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size * 0.42

    # Two intertwined halves forming a plus/cross shape
    # Top-left snake body (horizontal bar + vertical bar)
    bar_w = r * 0.45
    bar_h = r * 0.85

    # Upper half (blue in original) - left portion + top
    # Vertical bar left side
    draw.rounded_rectangle(
        [cx - bar_w, cy - bar_h, cx, cy + bar_w * 0.3],
        radius=int(bar_w * 0.3), fill=255
    )
    # Horizontal bar top
    draw.rounded_rectangle(
        [cx - bar_w * 0.3, cy - bar_h, cx + bar_h, cy - bar_h + bar_w],
        radius=int(bar_w * 0.3), fill=255
    )

    # Lower half (yellow in original) - right portion + bottom
    # Vertical bar right side
    draw.rounded_rectangle(
        [cx, cy - bar_w * 0.3, cx + bar_w, cy + bar_h],
        radius=int(bar_w * 0.3), fill=255
    )
    # Horizontal bar bottom
    draw.rounded_rectangle(
        [cx - bar_h, cy + bar_h - bar_w, cx + bar_w * 0.3, cy + bar_h],
        radius=int(bar_w * 0.3), fill=255
    )

    # Eyes (circles)
    eye_r = bar_w * 0.18
    draw.ellipse([cx - bar_w * 0.55, cy - bar_h * 0.7,
                   cx - bar_w * 0.55 + eye_r * 2, cy - bar_h * 0.7 + eye_r * 2], fill=0)
    draw.ellipse([cx + bar_w * 0.55 - eye_r * 2, cy + bar_h * 0.7 - eye_r * 2,
                   cx + bar_w * 0.55, cy + bar_h * 0.7], fill=0)

    return np.array(img) > 128


def create_java_logo(size=200):
    """Create a simplified Java coffee cup logo as a binary mask."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cx = size // 2

    # Coffee cup body
    cup_top = int(size * 0.35)
    cup_bottom = int(size * 0.85)
    cup_left = int(size * 0.25)
    cup_right = int(size * 0.65)
    draw.rounded_rectangle(
        [cup_left, cup_top, cup_right, cup_bottom],
        radius=int(size * 0.05), fill=255
    )

    # Cup handle (right side arc)
    handle_l = cup_right - 2
    handle_t = cup_top + int(size * 0.08)
    handle_r = int(size * 0.78)
    handle_b = cup_top + int(size * 0.32)
    draw.arc([handle_l, handle_t, handle_r, handle_b], -90, 90, fill=255, width=int(size * 0.04))

    # Steam lines (three wavy lines above cup)
    for i, offset_x in enumerate([-0.08, 0.0, 0.08]):
        sx = cx + int(size * offset_x)
        sy_start = cup_top - int(size * 0.03)
        sy_end = cup_top - int(size * 0.3)
        # Draw simple curved steam
        for sy in range(sy_end, sy_start, 2):
            t = (sy - sy_end) / (sy_start - sy_end)
            wave = math.sin(t * math.pi * 2.5) * size * 0.03
            draw.rectangle([sx + int(wave) - 1, sy, sx + int(wave) + 1, sy + 2], fill=255)

    # Base/saucer
    saucer_y = cup_bottom + int(size * 0.02)
    draw.rounded_rectangle(
        [cup_left - int(size * 0.05), saucer_y,
         cup_right + int(size * 0.05), saucer_y + int(size * 0.04)],
        radius=2, fill=255
    )

    return np.array(img) > 128


def create_gcloud_logo(size=200):
    """Create a simplified Google Cloud logo as a binary mask."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2

    # Google Cloud logo: a hexagonal cloud shape made of overlapping elements
    # Simplified: a cloud shape with the GCP hexagon

    # Main cloud body (large circle + smaller circles on top and sides)
    main_r = size * 0.25
    draw.ellipse([cx - main_r, cy - main_r * 0.7, cx + main_r, cy + main_r * 0.9], fill=255)

    # Left bump
    lr = main_r * 0.6
    draw.ellipse([cx - main_r * 1.1, cy - lr * 0.5, cx - main_r * 1.1 + lr * 2, cy + lr * 1.2], fill=255)

    # Right bump
    draw.ellipse([cx + main_r * 0.3, cy - lr * 0.5, cx + main_r * 0.3 + lr * 2, cy + lr * 1.2], fill=255)

    # Top bump
    tr = main_r * 0.55
    draw.ellipse([cx - tr, cy - main_r * 0.9, cx + tr, cy - main_r * 0.9 + tr * 2], fill=255)

    # Base (flat bottom)
    draw.rectangle([cx - main_r * 0.9, cy + main_r * 0.3, cx + main_r * 0.9, cy + main_r * 0.6], fill=255)

    # Inner hexagon outline (GCP branding)
    hex_r = main_r * 0.35
    hex_points = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        hx = cx + hex_r * math.cos(angle)
        hy = cy - main_r * 0.1 + hex_r * math.sin(angle)
        hex_points.append((hx, hy))
    draw.polygon(hex_points, outline=0, fill=None)
    # Draw thicker outline to carve out the hexagon
    for i in range(6):
        x1, y1 = hex_points[i]
        x2, y2 = hex_points[(i + 1) % 6]
        draw.line([(x1, y1), (x2, y2)], fill=0, width=int(size * 0.02))

    return np.array(img) > 128


def sample_dots_from_mask(mask, n_dots, rng=None):
    """Sample n_dots uniformly from True pixels in a binary mask."""
    if rng is None:
        rng = np.random.default_rng(42)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty mask — no pixels to sample from")
    indices = rng.choice(len(xs), size=min(n_dots, len(xs)), replace=False)
    return np.column_stack([xs[indices], ys[indices]])


def generate_logo_dots():
    """
    Generate dot coordinate arrays for each logo, scaled to portrait coordinate space.
    Returns dict: {"python": (N,2), "java": (N,2), "gcloud": (N,2)}
    """
    print("[5/6] Generating logo dot patterns...")
    rng = np.random.default_rng(2024)
    logo_size = 200

    logos = {
        "python": create_python_logo(logo_size),
        "java": create_java_logo(logo_size),
        "gcloud": create_gcloud_logo(logo_size),
    }

    logo_dots = {}
    for name, mask in logos.items():
        dots = sample_dots_from_mask(mask, NUM_TRAVELLERS, rng)
        # Scale to portrait coordinate space: center in portrait area
        # Logo occupies roughly 60% of portrait width, centered
        logo_w = PORTRAIT_W * 0.55
        logo_h = PORTRAIT_H * 0.55
        dots_f = dots.astype(np.float64)
        dots_f[:, 0] = (dots_f[:, 0] / logo_size) * logo_w + (PORTRAIT_W - logo_w) / 2
        dots_f[:, 1] = (dots_f[:, 1] / logo_size) * logo_h + (PORTRAIT_H - logo_h) / 2
        logo_dots[name] = dots_f
        print(f"    {name}: {len(dots_f)} dots")

    return logo_dots


# ============================================================
# OPTIMAL TRANSPORT
# ============================================================
def compute_optimal_transport(source_dots, target_dots):
    """
    Match dots between source and target using optimal transport
    (linear sum assignment on Euclidean distance cost matrix).
    Returns permutation indices for target.
    """
    n = min(len(source_dots), len(target_dots))
    src = source_dots[:n]
    tgt = target_dots[:n]

    # Cost matrix: Euclidean distance
    # For large n, compute in chunks to avoid memory issues
    if n <= 1500:
        cost = np.sqrt(
            (src[:, 0:1] - tgt[:, 0:1].T) ** 2 +
            (src[:, 1:2] - tgt[:, 1:2].T) ** 2
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        return col_ind
    else:
        # Fallback: nearest-neighbor greedy (less optimal but feasible)
        from scipy.spatial import KDTree
        tree = KDTree(tgt)
        used = set()
        col_ind = np.zeros(n, dtype=int)
        for i in range(n):
            dists, indices = tree.query(src[i], k=min(20, n))
            for idx in indices:
                if idx not in used:
                    col_ind[i] = idx
                    used.add(idx)
                    break
        return col_ind


def compute_all_transports(logo_dots):
    """Compute optimal transport between consecutive logos and back to first."""
    print("[6/6] Computing optimal transport matchings...")
    names = ["python", "java", "gcloud"]
    transports = {}

    for i in range(len(names)):
        src_name = names[i]
        tgt_name = names[(i + 1) % len(names)]
        key = f"{src_name}_to_{tgt_name}"
        print(f"    {key}...")
        perm = compute_optimal_transport(logo_dots[src_name], logo_dots[tgt_name])
        transports[key] = perm

    return transports


# ============================================================
# ANIMATION GROUPING
# ============================================================
def create_intro_groups(dots, n_groups=NUM_INTRO_GROUPS, rng=None):
    """
    Assign each dot to one of n_groups for intro fade-in animation.
    Groups must be spatially scattered (not clustered) — verified by evenness metric.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(dots)
    # Pure random assignment ensures spatial scatter
    groups = rng.integers(0, n_groups, size=n)

    # Verify evenness: compute std of group centroids normalized by portrait size
    evenness = compute_evenness_metric(dots, groups, n_groups)
    print(f"    Intro group evenness metric: {evenness:.4f} (target <= 0.05)")

    return groups, evenness


def compute_evenness_metric(dots, groups, n_groups):
    """
    Evenness metric: std of group centroids / portrait diagonal.
    Low = groups are spread evenly. High = groups are spatially clustered.
    """
    centroids = []
    for g in range(n_groups):
        mask = groups == g
        if np.sum(mask) > 0:
            cx = np.mean(dots[mask, 0])
            cy = np.mean(dots[mask, 1])
            centroids.append([cx, cy])

    if len(centroids) < 2:
        return 1.0

    centroids = np.array(centroids)
    # Ideal centroid would be the overall centroid
    overall_cx = np.mean(dots[:, 0])
    overall_cy = np.mean(dots[:, 1])

    # Measure how far group centroids deviate from overall centroid
    # relative to portrait size
    diag = math.sqrt(PORTRAIT_W ** 2 + PORTRAIT_H ** 2)
    dists = np.sqrt((centroids[:, 0] - overall_cx) ** 2 + (centroids[:, 1] - overall_cy) ** 2)
    return np.std(dists) / diag


def create_drift_bands(dots, centroid, n_bands=NUM_DRIFT_BANDS, noise_sigma=NOISE_SIGMA, rng=None):
    """
    Create drift bands for the portrait dissolve animation.
    Each band translates toward centroid during transition.

    Key: add 2D positional jitter BEFORE computing distance to break
    concentric-ring patterns into organic boundaries.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    # 2D positional jitter: perturb each dot's position before computing
    # distance to centroid. This breaks concentric rings much better than
    # 1D noise on the distance scalar.
    # sigma scaled to ~8% of portrait diagonal for organic boundaries
    diag = math.sqrt(PORTRAIT_W ** 2 + PORTRAIT_H ** 2)
    jitter_sigma = diag * 0.08  # ~36px for 300x340

    jittered_x = dots[:, 0] + rng.normal(0, jitter_sigma, size=len(dots))
    jittered_y = dots[:, 1] + rng.normal(0, jitter_sigma, size=len(dots))

    # Drift value = distance from jittered position to centroid
    dx = jittered_x - centroid[0]
    dy = jittered_y - centroid[1]
    drift_vals = np.sqrt(dx ** 2 + dy ** 2)

    # Sort and assign to bands
    sorted_indices = np.argsort(drift_vals)
    bands = np.zeros(len(dots), dtype=int)
    band_size = len(dots) // n_bands
    for b in range(n_bands):
        start = b * band_size
        end = start + band_size if b < n_bands - 1 else len(dots)
        bands[sorted_indices[start:end]] = b

    # Verify straight-boundary metric
    boundary_metric = compute_boundary_metric(dots, bands, n_bands)
    print(f"    Drift band boundary metric: {boundary_metric:.4f} (target <= 0.01)")

    return bands, boundary_metric


def compute_boundary_metric(dots, bands, n_bands):
    """
    Measure how 'straight' the band boundaries are using linear regression R².
    For each band, find boundary dots and compute R² of a line fit.
    Mean R² across bands: ~0.01 = organic, ~0.17 = grid-like.
    """
    # Build adjacency: for each dot, find which band its nearest neighbors are in
    from scipy.spatial import KDTree
    tree = KDTree(dots)
    
    r2_values = []
    for b in range(n_bands):
        band_mask = bands == b
        band_dots = dots[band_mask]
        if len(band_dots) < 10:
            continue
        
        # Find boundary dots: dots in this band that have a neighbor in a different band
        # Query nearest 8 neighbors for each dot in this band
        dists, indices = tree.query(band_dots, k=min(9, len(dots)))
        boundary_dots = []
        for i in range(len(band_dots)):
            neighbor_bands = bands[indices[i, 1:]]  # skip self
            if np.any(neighbor_bands != b):
                boundary_dots.append(band_dots[i])
        
        if len(boundary_dots) < 5:
            continue
        
        boundary_dots = np.array(boundary_dots)
        # Compute R² of linear fit to boundary points
        x = boundary_dots[:, 0]
        y = boundary_dots[:, 1]
        
        # Try both x~y and y~x, take the max (catches both horizontal and vertical lines)
        r2_xy = 0.0
        r2_yx = 0.0
        
        if np.std(x) > 1e-6:
            corr = np.corrcoef(x, y)[0, 1]
            r2_xy = corr ** 2 if not np.isnan(corr) else 0.0
        if np.std(y) > 1e-6:
            corr = np.corrcoef(y, x)[0, 1]
            r2_yx = corr ** 2 if not np.isnan(corr) else 0.0
        
        r2_values.append(max(r2_xy, r2_yx))
    
    if not r2_values:
        return 0.0
    return float(np.mean(r2_values))


# ============================================================
# SVG BUILDER
# ============================================================
def dots_to_path(dots, dot_size=DOT_SIZE):
    """Convert dot coordinates to SVG <path> d attribute. Each dot is a small rect."""
    parts = []
    for x, y in dots:
        xi, yi = int(x), int(y)
        parts.append(f"M{xi},{yi}h{dot_size}v{dot_size}h-{dot_size}z")
    return "".join(parts)


def build_info_panel_svg(theme, x_start, y_start):
    """Build the right-side SYSTEM.INFO panel as SVG elements."""
    c = COLORS[theme]
    elements = []
    row_height = 23
    font_size = 14
    header_size = 13
    char_width = 8.4  # approximate monospace char width at 14px

    # Panel dimensions
    panel_w = BANNER_W - x_start - 40
    total_leader_chars = 38  # total chars for label + leaders + value per row

    # Header: SYSTEM.INFO
    elements.append(
        f'<text x="{x_start}" y="{y_start}" font-family="{FONT_FAMILY}" '
        f'font-size="{header_size}" fill="{c["chrome"]}" font-weight="bold">'
        f'SYSTEM.INFO</text>'
    )
    # Underline
    elements.append(
        f'<line x1="{x_start}" y1="{y_start + 6}" x2="{x_start + panel_w}" '
        f'y2="{y_start + 6}" stroke="{c["chrome"]}" stroke-width="1" '
        f'stroke-dasharray="2,2" opacity="0.4"/>'
    )

    y = y_start + row_height + 10

    for label, value in INFO_ROWS:
        # Compute dotted leaders
        label_chars = len(label)
        value_chars = len(value)
        available = total_leader_chars - label_chars - value_chars
        leaders = " " + "." * max(available - 2, 2) + " "

        full_text = label + leaders + value

        # Row with textLength for alignment
        text_length = panel_w - 10
        elements.append(
            f'<text x="{x_start}" y="{y}" font-family="{FONT_FAMILY}" '
            f'font-size="{font_size}" textLength="{text_length}" '
            f'lengthAdjust="spacingAndGlyphs">'
            f'<tspan fill="{c["text_dim"]}">{label}</tspan>'
            f'<tspan fill="{c["text_dim"]}">{_escape_xml(leaders)}</tspan>'
            f'<tspan fill="{c["text_value"]}">{_escape_xml(value)}</tspan>'
            f'</text>'
        )
        y += row_height

    # LIVE badge (pulsing)
    y += 10
    live_x = x_start
    live_y = y
    badge_w = 52
    badge_h = 22
    elements.append(
        f'<g>'
        f'<rect x="{live_x}" y="{live_y - 15}" width="{badge_w}" height="{badge_h}" '
        f'rx="4" fill="{c["live_bg"]}">'
        f'<animate attributeName="opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite"/>'
        f'</rect>'
        f'<circle cx="{live_x + 14}" cy="{live_y - 4}" r="3.5" fill="{c["live_text"]}">'
        f'<animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite"/>'
        f'</circle>'
        f'<text x="{live_x + 24}" y="{live_y}" font-family="{FONT_FAMILY}" '
        f'font-size="12" fill="{c["live_text"]}" font-weight="bold">LIVE</text>'
        f'</g>'
    )

    # Handle pill
    pill_x = live_x + badge_w + 20
    pill_text = "@VIVEKVRON"
    pill_w = len(pill_text) * 9 + 20
    pill_h = 26
    elements.append(
        f'<rect x="{pill_x}" y="{live_y - 17}" width="{pill_w}" height="{pill_h}" '
        f'rx="13" fill="{c["pill_bg"]}"/>'
        f'<text x="{pill_x + 10}" y="{live_y}" font-family="{FONT_FAMILY}" '
        f'font-size="14" fill="{c["pill_text"]}" font-weight="bold">{pill_text}</text>'
    )

    return "\n".join(elements)


def _escape_xml(s):
    """Escape XML special characters."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_portrait_animation_svg(dots, intro_groups, drift_bands, logo_dots, transports, theme):
    """
    Build the animated portrait + traveller layers as SVG.
    Two independent layers:
    1. Portrait dots with intro fade-in + drift dissolve on loop
    2. Traveller dots that morph between logos
    """
    c = COLORS[theme]
    elements = []
    n_dots = len(dots)

    # ---- INTRO ANIMATION LAYER (duplicate for fade-in) ----
    # Each intro group fades in at a staggered time
    intro_element_parts = []
    for g in range(NUM_INTRO_GROUPS):
        mask = intro_groups == g
        group_dots = dots[mask]
        if len(group_dots) == 0:
            continue

        path_d = dots_to_path(group_dots)
        # Stagger: each group starts at a slightly different time within INTRO_FADE duration
        begin_time = (g / NUM_INTRO_GROUPS) * INTRO_FADE
        fade_dur = INTRO_FADE / 3  # each group fades in over 1/3 of total fade time

        intro_element_parts.append(
            f'<path d="{path_d}" fill="{c["dot"]}" shape-rendering="crispEdges" opacity="0">'
            f'<animate attributeName="opacity" from="0" to="1" '
            f'begin="{begin_time:.3f}s" dur="{fade_dur:.3f}s" fill="freeze"/>'
            f'</path>'
        )

    elements.append(f'<g id="intro-layer">')
    elements.extend(intro_element_parts)
    elements.append('</g>')

    # ---- PORTRAIT LAYER (persistent, with drift animation on loop) ----
    # Compute centroid of first logo for drift direction
    first_logo_centroid = np.mean(logo_dots["python"], axis=0)
    drift_pct = 0.42  # translate 42% toward centroid

    portrait_parts = []
    for b in range(NUM_DRIFT_BANDS):
        mask = drift_bands == b
        band_dots = dots[mask]
        if len(band_dots) == 0:
            continue

        path_d = dots_to_path(band_dots)
        band_centroid = np.mean(band_dots, axis=0)

        # Drift vector: from band centroid toward first logo centroid
        dx = (first_logo_centroid[0] - band_centroid[0]) * drift_pct
        dy = (first_logo_centroid[1] - band_centroid[1]) * drift_pct

        # Compute keyTimes for the loop (uneven spacing)
        # Portrait: 0 -> portrait_end
        # Trans1: portrait_end -> trans1_end
        # Logo1: trans1_end -> logo1_end
        # Trans2: logo1_end -> trans2_end
        # Logo2: trans2_end -> logo2_end
        # Trans3: logo2_end -> trans3_end
        # Logo3: trans3_end -> logo3_end
        # Trans_back: logo3_end -> 1.0

        t_portrait = PORTRAIT_HOLD / LOOP_DURATION
        t_trans = TRANSITION / LOOP_DURATION
        t_logo = LOGO_HOLD / LOOP_DURATION

        # keyTimes: portrait_hold, trans_out, logo1, trans_in, logo2, trans_out2, logo3, trans_back
        kt = [0]
        t = 0
        t += t_portrait; kt.append(round(t, 4))  # end of portrait hold
        t += t_trans; kt.append(round(t, 4))      # end of transition to logo1
        t += t_logo; kt.append(round(t, 4))       # end of logo1 hold
        t += t_trans; kt.append(round(t, 4))      # end of transition to logo2
        t += t_logo; kt.append(round(t, 4))       # end of logo2 hold
        t += t_trans; kt.append(round(t, 4))      # end of transition to logo3
        t += t_logo; kt.append(round(t, 4))       # end of logo3 hold
        t += t_trans; kt.append(min(1.0, round(t, 4)))  # end (back to portrait)

        kt_str = ";".join(str(k) for k in kt)

        # Opacity: visible during portrait, fade out during first transition, hidden during logos, fade back
        opacity_vals = "1;1;0;0;0;0;0;0;1"

        # Translation: 0 during portrait, drift during transitions, stay drifted during logos, return
        translate_vals = (
            f"0,0;0,0;{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};"
            f"{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};0,0"
        )

        portrait_parts.append(
            f'<g>'
            f'<path d="{path_d}" fill="{c["dot"]}" shape-rendering="crispEdges"/>'
            f'<animateTransform attributeName="transform" type="translate" '
            f'values="{translate_vals}" keyTimes="{kt_str}" '
            f'dur="{LOOP_DURATION}s" begin="{INTRO_TOTAL}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{opacity_vals}" '
            f'keyTimes="{kt_str}" dur="{LOOP_DURATION}s" begin="{INTRO_TOTAL}s" '
            f'repeatCount="indefinite"/>'
            f'</g>'
        )

    elements.append(f'<g id="portrait-layer" opacity="0">')
    elements.append(f'<animate attributeName="opacity" from="0" to="1" begin="0s" dur="0.01s" fill="freeze"/>')
    elements.extend(portrait_parts)
    elements.append('</g>')

    # ---- TRAVELLER LAYER (logo morphing) ----
    elements.append(build_traveller_layer(logo_dots, transports, c))

    return "\n".join(elements)


def build_traveller_layer(logo_dots, transports, colors):
    """
    Build the ~900 traveller dots that morph between logos.
    Each dot has its own position animation following optimal transport.
    Opacity: hidden during portrait phase, visible during logo phases.
    """
    parts = ['<g id="traveller-layer">']

    logos = ["python", "java", "gcloud"]
    n = NUM_TRAVELLERS

    # Get dot positions for each logo (in order, after transport matching)
    # Start with python dots in original order
    positions = [logo_dots["python"][:n].copy()]

    # Apply transports to get matched positions
    perm_pj = transports["python_to_java"]
    java_matched = logo_dots["java"][:n][perm_pj[:n]]
    positions.append(java_matched)

    perm_jg = transports["java_to_gcloud"]
    gcloud_matched = logo_dots["gcloud"][:n][perm_jg[:n]]
    positions.append(gcloud_matched)

    perm_gp = transports["gcloud_to_python"]
    python_return = logo_dots["python"][:n][perm_gp[:n]]
    positions.append(python_return)

    # Compute keyTimes for the loop
    t_portrait = PORTRAIT_HOLD / LOOP_DURATION
    t_trans = TRANSITION / LOOP_DURATION
    t_logo = LOGO_HOLD / LOOP_DURATION

    kt = [0]
    t = 0
    t += t_portrait; kt.append(round(t, 4))
    t += t_trans; kt.append(round(t, 4))
    t += t_logo; kt.append(round(t, 4))
    t += t_trans; kt.append(round(t, 4))
    t += t_logo; kt.append(round(t, 4))
    t += t_trans; kt.append(round(t, 4))
    t += t_logo; kt.append(round(t, 4))
    t += t_trans; kt.append(min(1.0, round(t, 4)))
    kt_str = ";".join(str(k) for k in kt)

    # Opacity: hidden during portrait, visible during logos
    # 0;0;1;1;1;1;1;1;0  (hidden at portrait hold, appear at first logo, hide at return)
    opacity_vals = "0;0;1;1;1;1;1;1;0"

    for i in range(n):
        # Position values at each keyframe
        # portrait_hold: at python pos (hidden anyway)
        # trans_to_logo1: move to python pos
        # logo1_hold: at python pos
        # trans_to_logo2: move to java pos
        # logo2_hold: at java pos
        # trans_to_logo3: move to gcloud pos
        # logo3_hold: at gcloud pos
        # trans_back: move to python return pos
        # end: at python return pos (hidden)
        p0 = positions[0][i]  # python
        p1 = positions[1][i]  # java (matched)
        p2 = positions[2][i]  # gcloud (matched)
        p3 = positions[3][i]  # python return (matched)

        x_vals = f"{p0[0]:.1f};{p0[0]:.1f};{p0[0]:.1f};{p0[0]:.1f};{p1[0]:.1f};{p1[0]:.1f};{p2[0]:.1f};{p2[0]:.1f};{p3[0]:.1f}"
        y_vals = f"{p0[1]:.1f};{p0[1]:.1f};{p0[1]:.1f};{p0[1]:.1f};{p1[1]:.1f};{p1[1]:.1f};{p2[1]:.1f};{p2[1]:.1f};{p3[1]:.1f}"

        dot_size = DOT_SIZE + 1  # traveller dots slightly larger

        parts.append(
            f'<rect x="{p0[0]:.1f}" y="{p0[1]:.1f}" width="{dot_size}" height="{dot_size}" '
            f'fill="{colors["chrome"]}" shape-rendering="crispEdges" opacity="0">'
            f'<animate attributeName="x" values="{x_vals}" keyTimes="{kt_str}" '
            f'dur="{LOOP_DURATION}s" begin="{INTRO_TOTAL}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y" values="{y_vals}" keyTimes="{kt_str}" '
            f'dur="{LOOP_DURATION}s" begin="{INTRO_TOTAL}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{opacity_vals}" keyTimes="{kt_str}" '
            f'dur="{LOOP_DURATION}s" begin="{INTRO_TOTAL}s" repeatCount="indefinite"/>'
            f'</rect>'
        )

    parts.append('</g>')
    return "\n".join(parts)


def build_terminal_chrome(theme):
    """Build the terminal window frame (title bar, border, traffic lights)."""
    c = COLORS[theme]
    return f"""<rect width="{BANNER_W}" height="{BANNER_H}" rx="12" fill="{c['bg']}" stroke="{c['border']}" stroke-width="1"/>
<rect width="{BANNER_W}" height="36" rx="12" fill="{c['title_bar']}"/>
<rect y="24" width="{BANNER_W}" height="12" fill="{c['title_bar']}"/>
<circle cx="24" cy="18" r="6" fill="#FF5F56"/>
<circle cx="46" cy="18" r="6" fill="#FFBD2E"/>
<circle cx="68" cy="18" r="6" fill="#27C93F"/>
<text x="{BANNER_W // 2}" y="23" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="13" fill="{c['text_dim']}">profile.sh-live</text>
<line x1="0" y1="36" x2="{BANNER_W}" y2="36" stroke="{c['border']}" stroke-width="1"/>"""


def build_portrait_frame(theme):
    """Build the VISUAL.MAP label and portrait frame."""
    c = COLORS[theme]
    frame_x = PORTRAIT_LEFT - 10
    frame_y = 50
    frame_w = PORTRAIT_W + 20
    frame_h = PORTRAIT_H + 60

    return f"""<text x="{PORTRAIT_LEFT}" y="{frame_y + 20}" font-family="{FONT_FAMILY}" font-size="{13}" fill="{c['chrome']}" font-weight="bold">VISUAL.MAP</text>
<rect x="{frame_x}" y="{frame_y + 28}" width="{frame_w}" height="{frame_h - 28}" rx="4" fill="none" stroke="{c['chrome']}" stroke-width="1" stroke-dasharray="4,4" opacity="0.3"/>"""


def assemble_banner(dots, intro_groups, drift_bands, logo_dots, transports, theme):
    """Assemble the complete banner SVG."""
    c = COLORS[theme]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{BANNER_W}" height="{BANNER_H}" viewBox="0 0 {BANNER_W} {BANNER_H}">',
        '<defs>',
        f'<clipPath id="portrait-clip"><rect x="{PORTRAIT_LEFT}" y="{PORTRAIT_TOP}" width="{PORTRAIT_W}" height="{PORTRAIT_H}"/></clipPath>',
        '</defs>',
        '',
        '<!-- Terminal chrome -->',
        build_terminal_chrome(theme),
        '',
        '<!-- Portrait frame -->',
        build_portrait_frame(theme),
        '',
        '<!-- Portrait and animation layers (clipped to frame) -->',
        f'<g clip-path="url(#portrait-clip)" transform="translate({PORTRAIT_LEFT},{PORTRAIT_TOP})">',
        build_portrait_animation_svg(dots, intro_groups, drift_bands, logo_dots, transports, theme),
        '</g>',
        '',
        '<!-- Info panel -->',
        build_info_panel_svg(theme, PORTRAIT_PANEL_W + 30, 70),
        '',
        '</svg>',
    ]

    return "\n".join(svg_parts)


# ============================================================
# METRICS & VERIFICATION
# ============================================================
def print_metrics(dots_light, dots_dark, intro_groups_l, intro_groups_d,
                  drift_bands_l, drift_bands_d, evenness_l, evenness_d,
                  boundary_l, boundary_d):
    """Print all verification metrics."""
    print("\n" + "=" * 60)
    print("VERIFICATION METRICS")
    print("=" * 60)
    print(f"Light mode dots:     {len(dots_light):,}")
    print(f"Dark mode dots:      {len(dots_dark):,}")
    print(f"Traveller dots:      {NUM_TRAVELLERS}")
    print(f"Intro groups:        {NUM_INTRO_GROUPS}")
    print(f"Drift bands:         {NUM_DRIFT_BANDS}")
    print(f"Evenness (light):    {evenness_l:.4f}  {'✓' if evenness_l <= 0.05 else '✗ PATCHY'}")
    print(f"Evenness (dark):     {evenness_d:.4f}  {'✓' if evenness_d <= 0.05 else '✗ PATCHY'}")
    print(f"Boundary (light):    {boundary_l:.4f}  {'✓' if boundary_l <= 0.01 else '✗ GRID-LIKE'}")
    print(f"Boundary (dark):     {boundary_d:.4f}  {'✓' if boundary_d <= 0.01 else '✗ GRID-LIKE'}")
    print(f"Ink coverage (light): {len(dots_light) / (PORTRAIT_W * PORTRAIT_H) * 100:.1f}%")
    print(f"Ink coverage (dark):  {len(dots_dark) / (PORTRAIT_W * PORTRAIT_H) * 100:.1f}%")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    assets_dir = os.path.join(project_dir, "assets")
    output_dir = os.path.join(project_dir, "output")
    data_dir = os.path.join(project_dir, "data")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    portrait_path = os.path.join(assets_dir, "portrait.jpg")
    if not os.path.exists(portrait_path):
        print(f"ERROR: Portrait not found at {portrait_path}")
        sys.exit(1)

    # 1. Process portrait
    dots_light, dots_dark, enhanced_img = process_portrait(portrait_path)

    # 2. Generate logo dots
    logo_dots = generate_logo_dots()

    # 3. Compute optimal transport
    transports = compute_all_transports(logo_dots)

    # 4. Create animation groups
    print("\nCreating animation groups...")
    rng = np.random.default_rng(42)

    print("  Light mode:")
    intro_groups_l, evenness_l = create_intro_groups(dots_light, rng=rng)
    first_logo_centroid = np.mean(logo_dots["python"], axis=0)
    drift_bands_l, boundary_l = create_drift_bands(dots_light, first_logo_centroid, rng=rng)

    print("  Dark mode:")
    rng2 = np.random.default_rng(43)
    intro_groups_d, evenness_d = create_intro_groups(dots_dark, rng=rng2)
    drift_bands_d, boundary_d = create_drift_bands(dots_dark, first_logo_centroid, rng=rng2)

    # 5. Save data files (source of truth)
    print("\nSaving data files...")
    np.save(os.path.join(data_dir, "portrait_dots_light.npy"), dots_light)
    np.save(os.path.join(data_dir, "portrait_dots_dark.npy"), dots_dark)
    for name, d in logo_dots.items():
        np.save(os.path.join(data_dir, f"logo_dots_{name}.npy"), d)

    # 6. Assemble SVGs
    print("\nAssembling dark.svg...")
    dark_svg = assemble_banner(dots_dark, intro_groups_d, drift_bands_d, logo_dots, transports, "dark")
    dark_path = os.path.join(output_dir, "dark.svg")
    with open(dark_path, "w", encoding="utf-8") as f:
        f.write(dark_svg)
    dark_size = os.path.getsize(dark_path)
    print(f"  dark.svg: {dark_size:,} bytes ({dark_size / 1024:.0f} KB)")

    print("\nAssembling light.svg...")
    light_svg = assemble_banner(dots_light, intro_groups_l, drift_bands_l, logo_dots, transports, "light")
    light_path = os.path.join(output_dir, "light.svg")
    with open(light_path, "w", encoding="utf-8") as f:
        f.write(light_svg)
    light_size = os.path.getsize(light_path)
    print(f"  light.svg: {light_size:,} bytes ({light_size / 1024:.0f} KB)")

    # 7. Print metrics
    print_metrics(dots_light, dots_dark, intro_groups_l, intro_groups_d,
                  drift_bands_l, drift_bands_d, evenness_l, evenness_d,
                  boundary_l, boundary_d)

    print(f"\n✓ Done! Files saved to: {output_dir}")
    print("⚠ Remember: verify animations in a browser, not cairosvg")


if __name__ == "__main__":
    main()
