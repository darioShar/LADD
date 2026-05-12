"""Generate README animations for Co-LADD and Di-LADD.

The script intentionally keeps the visuals schematic: it illustrates the
paired data/latent channels and the forward/backward diffusion processes
without depending on any training code.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib import patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ASSET_DIR = Path(__file__).resolve().parent
TOKENS = list("LADD!")
MASK = "[mask]"
W, H = 11.7, 6.7
DPI = 120
FPS_MS = 70
ENCODE_FRAMES = 23
FORWARD_FRAMES = 36
BACKWARD_FRAMES = FORWARD_FRAMES
DISCRETE_BACKWARD_FRAMES = 46
DATA_MASK_ORDER = np.array([2, 0, 4, 1, 3])
LATENT_MASK_ORDER = np.array([1, 0, 2])
DATA_MASK_TIMES = np.array([0.15, 0.30, 0.47, 0.66, 0.84])
LATENT_MASK_TIMES = np.array([0.28, 0.58, 0.86])
MDM_REVEAL_GROUPS = [[2], [0], [4], [1], [3]]
LADD_REVEAL_GROUPS = [[2], [0, 4], [1, 3]]
DI_LATENT_REVEAL_GROUPS = [[1], [0], [2]]
PLOT_X_RANGE = (-2.10, 2.10)
DENSITY_P0_STD = 1e-4
DENSITY_ALPHA_REVEALED = 0.7
DENSITY_ALPHA_UNREVEALED = 0.20
VP_BETA_MIN = 0.45
VP_BETA_MAX = 5.00
DATA_PANEL = (0.42, 0.92, 4.95, 4.72)
LATENT_PANEL = (5.74, 0.92, 5.48, 4.72)
MDM_DATA_PANEL = (3.35, 0.92, 4.95, 4.72)
CHANNEL_CAPTION_Y = 0.62
ENCODER_LABEL_Y = 0.82
FRAME_CENTER_X = W / 2.0
BOTTLENECK_X = 5.56
BOTTLENECK_Y = 1.18


PALETTE = {
    "ink": "#18202b",
    "muted": "#667085",
    "paper": "#fbfaf7",
    "panel": "#ffffff",
    "grid": "#e6e8ee",
    "data": "#246bfe",
    "latent_a": "#f05a7e",
    "latent_b": "#13a89e",
    "mask": "#2a2f3a",
    "gold": "#f5b84b",
    "violet": "#7957d5",
}


def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ordered_thresholds(order: np.ndarray, times: np.ndarray) -> np.ndarray:
    out = np.empty_like(times)
    out[order] = times
    return out


def step_state(progress: float, num_steps: int) -> tuple[int, float]:
    progress = float(np.clip(progress, 0.0, 1.0))
    # Hold t=1, every intermediate jump, and t=0 for the same amount of time.
    step = int(np.floor(progress * (num_steps + 1)))
    step = min(step, num_steps)
    return step, step / num_steps


def revealed_from_groups(groups: list[list[int]], step: int) -> set[int]:
    revealed: set[int] = set()
    for group in groups[:step]:
        revealed.update(group)
    return revealed


def num_frames(discrete_backward: bool = False) -> int:
    backward_frames = DISCRETE_BACKWARD_FRAMES if discrete_backward else BACKWARD_FRAMES
    return ENCODE_FRAMES + FORWARD_FRAMES + backward_frames


def cycle_progress(frame: int, discrete_backward: bool = False) -> tuple[str, float]:
    backward_frames = DISCRETE_BACKWARD_FRAMES if discrete_backward else BACKWARD_FRAMES
    if frame < ENCODE_FRAMES:
        return "encode", smoothstep(frame / max(ENCODE_FRAMES - 1, 1))
    if frame < ENCODE_FRAMES + FORWARD_FRAMES:
        local_frame = frame - ENCODE_FRAMES
        return "forward", smoothstep(local_frame / max(FORWARD_FRAMES - 1, 1))
    local_frame = frame - ENCODE_FRAMES - FORWARD_FRAMES
    return "backward", local_frame / max(backward_frames - 1, 1)


def add_panel(ax, xy, width, height, title, accent):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.08",
        linewidth=1.6,
        edgecolor=accent,
        facecolor=PALETTE["panel"],
        alpha=0.97,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.22,
        y + height - 0.33,
        title,
        ha="left",
        va="center",
        fontsize=13,
        fontweight=700,
        color=PALETTE["ink"],
    )


def arrow(ax, start, end, color, lw=2.4, alpha=1.0, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=15,
            linewidth=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def text_with_glow(ax, x, y, text, color, size, weight=700, ha="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        fontweight=weight,
        color=color,
        path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.75)],
    )


def blend(start, end, amount):
    return (1.0 - amount) * np.asarray(start) + amount * np.asarray(end)


def token_box(ax, x, y, token, face, edge, text_color, scale=1.0, alpha=1.0):
    width = 0.66 * scale if token != MASK else 0.76 * scale
    height = 0.54 * scale
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.065",
        linewidth=1.35,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        token,
        ha="center",
        va="center",
        fontsize=(8.5 if token == MASK else 15) * scale,
        fontweight=800,
        color=text_color,
        alpha=alpha,
    )


def draw_tokens(ax, xs, y, amount_masked):
    thresholds = ordered_thresholds(DATA_MASK_ORDER, DATA_MASK_TIMES)
    for x, token, threshold in zip(xs, TOKENS, thresholds, strict=True):
        masked = amount_masked >= threshold
        token_box(
            ax,
            x,
            y,
            MASK if masked else token,
            PALETTE["mask"] if masked else "#eef4ff",
            "#111827" if masked else PALETTE["data"],
            "white" if masked else PALETTE["data"],
            scale=0.86,
            alpha=0.98,
        )


def draw_generation_tokens(ax, xs, y, revealed: set[int]):
    for idx, (x, token) in enumerate(zip(xs, TOKENS, strict=True)):
        masked = idx not in revealed
        token_box(
            ax,
            x,
            y,
            MASK if masked else token,
            PALETTE["mask"] if masked else "#eef4ff",
            "#111827" if masked else PALETTE["data"],
            "white" if masked else PALETTE["data"],
            scale=0.86,
            alpha=0.98,
        )


def draw_clean_tokens(ax, xs, y, alpha=0.98):
    for x, token in zip(xs, TOKENS, strict=True):
        token_box(
            ax,
            x,
            y,
            token,
            "#eef4ff",
            PALETTE["data"],
            PALETTE["data"],
            scale=0.86,
            alpha=alpha,
        )


def beta_t(t: np.ndarray | float) -> np.ndarray | float:
    return VP_BETA_MIN + t * (VP_BETA_MAX - VP_BETA_MIN)


def beta_integral(t: np.ndarray | float) -> np.ndarray | float:
    return VP_BETA_MIN * t + 0.5 * (VP_BETA_MAX - VP_BETA_MIN) * np.asarray(t) ** 2


def alpha_t(t: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * beta_integral(t))


def sigma2_t(t: np.ndarray | float) -> np.ndarray | float:
    alpha = alpha_t(t)
    return np.maximum(1.0 - alpha**2, 0.0)


def component_var_t(t: np.ndarray | float) -> np.ndarray | float:
    alpha = alpha_t(t)
    return alpha**2 * DENSITY_P0_STD**2 + sigma2_t(t)


def forward_sde_path(t: np.ndarray, start: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty_like(t)
    x[0] = start
    for i in range(1, len(t)):
        transition_alpha = alpha_t(t[i]) / alpha_t(t[i - 1])
        transition_std = np.sqrt(max(1.0 - transition_alpha**2, 0.0))
        x[i] = transition_alpha * x[i - 1] + transition_std * rng.normal()
    return x


def mixture_score(x: float, t: float, starts: np.ndarray) -> float:
    alpha = float(alpha_t(t))
    var = float(component_var_t(t))
    means = alpha * starts
    log_weights = -0.5 * ((x - means) ** 2) / var
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()
    component_scores = -(x - means) / var
    return float(np.sum(weights * component_scores))


def probability_flow_velocity(x: float, t: float, starts: np.ndarray) -> float:
    beta = float(beta_t(t))
    return -0.5 * beta * (x + mixture_score(x, t, starts))


def probability_flow_ode_path(t: np.ndarray, start: float, starts: np.ndarray) -> np.ndarray:
    x = np.empty_like(t)
    x[0] = start
    for i in range(1, len(t)):
        t0 = float(t[i - 1])
        dt = float(t[i] - t[i - 1])
        x0 = float(x[i - 1])
        k1 = probability_flow_velocity(x0, t0, starts)
        k2 = probability_flow_velocity(x0 + 0.5 * dt * k1, t0 + 0.5 * dt, starts)
        k3 = probability_flow_velocity(x0 + 0.5 * dt * k2, t0 + 0.5 * dt, starts)
        k4 = probability_flow_velocity(x0 + dt * k3, t0 + dt, starts)
        x[i] = x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def map_x(value: np.ndarray | float, plot_x0: float, plot_x1: float) -> np.ndarray | float:
    lo, hi = PLOT_X_RANGE
    mapped = plot_x0 + (np.asarray(value) - lo) / (hi - lo) * (plot_x1 - plot_x0)
    return float(mapped) if np.ndim(mapped) == 0 else mapped


def draw_density(
    ax,
    plot_x0: float,
    plot_x1: float,
    plot_y0: float,
    plot_y1: float,
    starts: np.ndarray,
    phase: str,
    progress: float,
):
    x = np.linspace(PLOT_X_RANGE[0], PLOT_X_RANGE[1], 240)
    t = np.linspace(0, 1, 180)
    xx, tt = np.meshgrid(x, t)
    var = component_var_t(tt)
    density = np.zeros_like(xx)
    for start in starts:
        mean = alpha_t(tt) * start
        density += np.exp(-0.5 * ((xx - mean) ** 2) / var) / np.sqrt(2.0 * np.pi * var)
    density *= 0.5
    density = np.log1p(density)
    lo = np.quantile(density, 0.08)
    hi = np.quantile(density, 0.992)
    density = np.clip((density - lo) / (hi - lo), 0.0, 1.0)
    if phase == "forward":
        revealed = tt <= progress
    elif phase == "backward":
        revealed = tt >= 1.0 - progress
    else:
        revealed = np.zeros_like(tt, dtype=bool)
    alpha = np.where(revealed, DENSITY_ALPHA_REVEALED, DENSITY_ALPHA_UNREVEALED)

    ax.imshow(
        density,
        extent=(plot_x0, plot_x1, plot_y0, plot_y1),
        origin="lower",
        cmap=colormaps["viridis"],
        alpha=alpha,
        aspect="auto",
        zorder=1.4,
    )
    ax.text(
        plot_x1 - 0.10,
        plot_y1 - 0.24,
        r"$p_t$",
        ha="right",
        va="center",
        fontsize=11,
        color="#355f2a",
        fontweight=900,
    )


def draw_continuous_latents(
    ax,
    phase,
    progress,
    use_stochastic_forward=False,
    show_density=False,
    show_initial_latents=True,
    discrete_backward=False,
):
    del use_stochastic_forward
    x0 = np.array([-0.78, 0.64])
    colors = [PALETTE["latent_a"], PALETTE["latent_b"]]
    labels = [r"$z_1$", r"$z_2$"]

    plot_x0, plot_x1 = 6.68, 10.78
    plot_y0, plot_y1 = 1.36, 4.98

    if phase == "backward" and discrete_backward:
        _, progress = step_state(progress, len(LADD_REVEAL_GROUPS))

    if show_density and phase != "encode":
        draw_density(ax, plot_x0, plot_x1, plot_y0, plot_y1, x0, phase, progress)

    for frac in np.linspace(0, 1, 6):
        y = plot_y0 + frac * (plot_y1 - plot_y0)
        ax.plot([plot_x0, plot_x1], [y, y], color=PALETTE["grid"], lw=0.8, zorder=2)
    ax.plot([plot_x0, plot_x0], [plot_y0, plot_y1], color="#bac2d1", lw=1.2, zorder=3)
    ax.plot([plot_x0, plot_x1], [plot_y0, plot_y0], color="#bac2d1", lw=1.2, zorder=3)
    if phase == "encode":
        return

    ax.text(11.03, plot_y0 + 0.12, "t=0", ha="left", va="center", fontsize=10, color=PALETTE["muted"])
    ax.text(11.03, plot_y1 - 0.12, "t=1", ha="left", va="center", fontsize=10, color=PALETTE["muted"])

    ts = np.linspace(0, 1, 140)
    if phase == "encode":
        current_t = 0.0
        trail_t = ts[:1]
    else:
        current_t = progress if phase == "forward" else 1.0 - progress
        trail_t = ts[ts <= current_t] if phase == "forward" else ts[ts >= current_t]
    forward_paths = [forward_sde_path(ts, start, 1100 + i) for i, start in enumerate(x0)]
    reverse_paths = [probability_flow_ode_path(ts, start, x0) for start in x0]

    for i, (color, label) in enumerate(zip(colors, labels, strict=True)):
        forward_path = forward_paths[i]
        reverse_path = reverse_paths[i]
        xs = forward_path if phase == "forward" else reverse_path
        ax.plot(
            map_x(xs, plot_x0, plot_x1),
            plot_y0 + ts * (plot_y1 - plot_y0),
            color=color,
            lw=2.0,
            alpha=0.25,
        )
        if trail_t.size:
            source = forward_path if phase == "forward" else reverse_path
            xt = np.interp(trail_t, ts, source)
            ax.plot(
                map_x(xt, plot_x0, plot_x1),
                plot_y0 + trail_t * (plot_y1 - plot_y0),
                color=color,
                lw=3.4,
                alpha=0.95,
            )
        source = forward_path if phase == "forward" else reverse_path
        x_now = np.interp(current_t, ts, source)
        px = map_x(x_now, plot_x0, plot_x1)
        py = plot_y0 + current_t * (plot_y1 - plot_y0)
        ax.scatter(px, py, s=115, color=color, edgecolor="white", linewidth=1.8, zorder=6)
        ax.text(px + 0.12, py + 0.12, label, fontsize=12, color=color, fontweight=800)

    if phase == "forward":
        arrow(ax, (10.92, 1.42), (10.92, 4.80), PALETTE["violet"], lw=2.0)
    elif phase == "backward":
        arrow(ax, (10.92, 4.80), (10.92, 1.42), PALETTE["violet"], lw=2.0)

    if not show_initial_latents:
        return


def draw_discrete_latents(ax, phase, progress, alpha=0.98, discrete_backward=False):
    colors = ["#ff4d6d", "#37b24d", "#7b2cbf"]
    xs = np.array([7.15, 8.55, 9.95])
    y0, y1 = 1.34, 4.65
    thresholds = ordered_thresholds(LATENT_MASK_ORDER, LATENT_MASK_TIMES)
    amount = progress if phase == "forward" else 1.0 - progress
    revealed: set[int] | None = None

    if phase == "encode":
        amount = 0.0
    elif phase == "backward" and discrete_backward:
        step, step_fraction = step_state(progress, len(DI_LATENT_REVEAL_GROUPS))
        amount = 1.0 - step_fraction
        revealed = revealed_from_groups(DI_LATENT_REVEAL_GROUPS, step)

    if phase != "encode":
        arrow(ax, (10.74, 1.36), (10.74, 4.68), PALETTE["violet"], lw=2.0, alpha=0.9)
        ax.text(10.89, y0 + 0.10, "t=0", ha="left", va="center", fontsize=10, color=PALETTE["muted"])
        ax.text(10.89, y1 - 0.10, "t=1", ha="left", va="center", fontsize=10, color=PALETTE["muted"])

    for j, x in enumerate(xs):
        masked = (j not in revealed) if revealed is not None else amount >= thresholds[j]
        token = MASK if masked else f"c{j + 1}"
        face = PALETTE["mask"] if masked else colors[j]
        edge = "#111827" if masked else "white"
        token_box(ax, x, y0 + amount * (y1 - y0), token, face, edge, "white", scale=0.88, alpha=alpha)


def draw_encoder_arrow(ax, start_x: float, end_x: float, y: float):
    arrow(ax, (start_x, y), (end_x, y), PALETTE["gold"], lw=2.2)
    ax.text(
        0.5 * (start_x + end_x),
        ENCODER_LABEL_Y,
        "encoder",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["muted"],
        fontweight=700,
    )


def draw_encoding_motion(ax, model: str, token_xs: np.ndarray, progress: float):
    colors = [PALETTE["latent_a"], PALETTE["latent_b"]] if model == "co" else ["#ff4d6d", "#37b24d", "#7b2cbf"]
    token_points = [(x, 1.18) for x in token_xs]
    compress = smoothstep(min(progress / 0.50, 1.0))
    expand = smoothstep(max((progress - 0.42) / 0.58, 0.0))
    for start in token_points:
        ax.plot(
            [start[0], BOTTLENECK_X],
            [start[1], BOTTLENECK_Y],
            color="#98a2b3",
            lw=1.0,
            alpha=0.22 * (1.0 - expand),
            zorder=6,
        )
    for start in token_points:
        x, y = blend(start, (BOTTLENECK_X, BOTTLENECK_Y), compress)
        ax.scatter(x, y, s=42, color="#98a2b3", edgecolor="white", linewidth=1.0, alpha=0.70 * (1.0 - expand), zorder=8)

    ax.scatter(
        BOTTLENECK_X,
        BOTTLENECK_Y,
        s=150,
        color="#667085",
        edgecolor="white",
        linewidth=1.5,
        alpha=min(1.0, 1.25 * progress) * (1.0 - 0.65 * expand),
        zorder=9,
    )

    if model == "co":
        latent_axis_x0, latent_axis_x1 = 6.68, 10.78
        end_points = [(map_x(-0.78, latent_axis_x0, latent_axis_x1), 1.08), (map_x(0.64, latent_axis_x0, latent_axis_x1), 1.08)]
    else:
        end_points = [(7.15, 1.34), (8.55, 1.34), (9.95, 1.34)]

    for color, end in zip(colors, end_points, strict=True):
        x, y = blend((BOTTLENECK_X, BOTTLENECK_Y), end, expand)
        ax.scatter(x, y, s=100, color=color, edgecolor="white", linewidth=1.5, alpha=0.82 * expand, zorder=10)


def draw_coupled_channels(ax, progress: float):
    y_top, y_bottom = 4.52, 1.56
    y = y_top - progress * (y_top - y_bottom)
    arrow(ax, (4.95, y), (6.05, y), "#475467", lw=2.2, style="<->")
    ax.text(
        5.50,
        y + 0.46,
        "coupled channels",
        ha="center",
        va="center",
        fontsize=10,
        color="#475467",
        fontweight=800,
    )


def draw_common_scene(
    model: str,
    phase: str,
    progress: float,
    use_stochastic_forward=False,
    show_density=False,
    discrete_backward=False,
):
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0, 6.7)
    ax.axis("off")

    model_name = {"mdm": "MDM", "co": "Co-LADD", "di": "Di-LADD"}[model]
    subtitle = {
        "mdm": "Masked Discrete Diffusion",
        "co": "Continuous Latent Diffusion",
        "di": "Discrete Latent Diffusion",
    }[model]
    phase_label = {
        "encode": "No Encoding" if model == "mdm" else "Encode data",
        "forward": "Forward process",
        "backward": "Backward process",
    }[phase]
    ax.text(0.34, 6.27, model_name, ha="left", va="center", fontsize=24, fontweight=950, color=PALETTE["ink"])
    ax.text(2.78, 6.27, subtitle, ha="left", va="center", fontsize=12.5, fontweight=750, color=PALETTE["muted"])
    badge = FancyBboxPatch(
        (8.48, 6.01),
        3.10,
        0.46,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=1.1,
        edgecolor="#f1cf89",
        facecolor="#fff4da",
        alpha=0.98,
    )
    ax.add_patch(badge)
    if phase == "encode":
        badge_text = phase_label
    elif phase == "forward":
        badge_text = f"{phase_label}  t={progress:.2f}"
    elif discrete_backward:
        groups = MDM_REVEAL_GROUPS if model == "mdm" else LADD_REVEAL_GROUPS
        _, step_fraction = step_state(progress, len(groups))
        badge_text = f"{phase_label}  t={1 - step_fraction:.2f}"
    else:
        badge_text = f"{phase_label}  t={1 - progress:.2f}"
    text_with_glow(ax, 10.03, 6.24, badge_text, PALETTE["gold"], 10.6)

    data_x, data_y, data_w, data_h = MDM_DATA_PANEL if model == "mdm" else DATA_PANEL
    latent_x, latent_y, latent_w, latent_h = LATENT_PANEL
    add_panel(ax, (data_x, data_y), data_w, data_h, "data channel", PALETTE["data"])
    if model != "mdm":
        add_panel(ax, (latent_x, latent_y), latent_w, latent_h, "latent channel", PALETTE["violet"])

    y_bottom, y_top = 1.36, 4.74
    token_xs = np.linspace(data_x + 0.84, data_x + data_w - 0.87, len(TOKENS))
    data_arrow_x = data_x + 0.44
    data_time_x = data_arrow_x - 0.14

    if phase == "encode":
        amount_masked = 0.0
        current_y = y_bottom
    elif phase == "forward":
        amount_masked = progress
        current_y = y_bottom + progress * (y_top - y_bottom)
        arrow(ax, (data_arrow_x, y_bottom), (data_arrow_x, y_top), PALETTE["data"], lw=2.0)
        ax.text(data_time_x, y_bottom + 0.10, "t=0", ha="right", va="center", fontsize=10, color=PALETTE["muted"])
        ax.text(data_time_x, y_top - 0.10, "t=1", ha="right", va="center", fontsize=10, color=PALETTE["muted"])
    else:
        if discrete_backward:
            groups = MDM_REVEAL_GROUPS if model == "mdm" else LADD_REVEAL_GROUPS
            step, step_fraction = step_state(progress, len(groups))
            revealed = revealed_from_groups(groups, step)
            amount_masked = 1.0 - step_fraction
            current_y = y_top - step_fraction * (y_top - y_bottom)
        else:
            amount_masked = 1.0 - progress
            current_y = y_top - progress * (y_top - y_bottom)
        arrow(ax, (data_arrow_x, y_top), (data_arrow_x, y_bottom), PALETTE["data"], lw=2.0)
        ax.text(data_time_x, y_bottom + 0.10, "t=0", ha="right", va="center", fontsize=10, color=PALETTE["muted"])
        ax.text(data_time_x, y_top - 0.10, "t=1", ha="right", va="center", fontsize=10, color=PALETTE["muted"])

    data_caption = {
        "encode": "No Encoding" if model == "mdm" else "encode clean tokens",
        "forward": "categorical masking",
        "backward": "sample clean tokens",
    }[phase]
    data_center_x = data_x + 0.5 * data_w
    ax.text(data_center_x, CHANNEL_CAPTION_Y, data_caption, ha="center", va="center", fontsize=11, color=PALETTE["muted"], fontweight=800)
    if phase == "encode":
        draw_clean_tokens(ax, token_xs, current_y)
        if model == "mdm":
            ax.text(data_center_x, 3.18, "No Encoding", ha="center", va="center", fontsize=20, color=PALETTE["muted"], fontweight=900)
    elif phase == "backward" and discrete_backward:
        draw_generation_tokens(ax, token_xs, current_y, revealed)
    else:
        draw_tokens(ax, token_xs, current_y, amount_masked)

    if model == "mdm":
        return fig

    if model == "co":
        latent_axis_x0, latent_axis_x1 = 6.68, 10.78
        if phase == "encode":
            draw_encoder_arrow(ax, token_xs[-1] + 0.48, 6.30, 1.08)
            draw_encoding_motion(ax, model, token_xs, progress)
        ax.plot([latent_axis_x0, latent_axis_x1], [1.08, 1.08], color="#bac2d1", lw=1.2)
        marker_alpha = 1.0
        if phase == "encode":
            marker_alpha = smoothstep(max((progress - 0.42) / 0.58, 0.0))
        elif phase == "backward":
            marker_alpha = smoothstep((progress - 0.84) / 0.16)
        for value, color, label in [(-0.78, PALETTE["latent_a"], "-0.8"), (0.64, PALETTE["latent_b"], "0.6")]:
            x = map_x(value, latent_axis_x0, latent_axis_x1)
            if marker_alpha > 0.02:
                ax.scatter(x, 1.08, s=135, color=color, edgecolor="white", linewidth=1.8, zorder=5, alpha=marker_alpha)
                ax.text(x, 1.26, label, ha="center", va="center", fontsize=10, color=color, fontweight=800, alpha=marker_alpha)
        latent_caption = {
            "encode": "encode latents",
            "forward": "forward noising",
            "backward": "sample latents with ODE flow",
        }[phase]
        ax.text(8.62, CHANNEL_CAPTION_Y, latent_caption, ha="center", va="center", fontsize=11, color=PALETTE["muted"], fontweight=800)
        ax.text(8.62, 1.44, "continuous latents", ha="center", va="center", fontsize=10, color=PALETTE["muted"])
        draw_continuous_latents(ax, phase, progress, use_stochastic_forward, show_density, discrete_backward=discrete_backward)
    else:
        if phase == "encode":
            draw_encoder_arrow(ax, token_xs[-1] + 0.48, 6.30, 1.34)
            draw_encoding_motion(ax, model, token_xs, progress)
        latent_caption = {
            "encode": "encode latent tokens",
            "forward": "categorical masking",
            "backward": "sample clean latents",
        }[phase]
        ax.text(8.62, CHANNEL_CAPTION_Y, latent_caption, ha="center", va="center", fontsize=11, color=PALETTE["muted"], fontweight=800)
        latent_alpha = smoothstep(max((progress - 0.42) / 0.58, 0.0)) if phase == "encode" else 0.98
        draw_discrete_latents(ax, phase, progress, alpha=latent_alpha, discrete_backward=discrete_backward)

    if phase == "backward":
        coupled_progress = progress
        if discrete_backward:
            _, coupled_progress = step_state(progress, len(LADD_REVEAL_GROUPS))
        draw_coupled_channels(ax, coupled_progress)

    return fig


def fig_to_pil(fig) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


def save_gif(model: str, name: str, use_stochastic_forward=False, show_density=False, discrete_backward=False):
    frames = []
    for frame_idx in range(num_frames(discrete_backward)):
        phase, progress = cycle_progress(frame_idx, discrete_backward)
        fig = draw_common_scene(model, phase, progress, use_stochastic_forward, show_density, discrete_backward)
        frame = fig_to_pil(fig).convert("RGB")
        frame.putpixel((0, 0), (248, 248, 240 + (frame_idx % 12)))
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    out = ASSET_DIR / name
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=FPS_MS,
        loop=0,
        optimize=False,
    )
    return out


def save_contact_sheet(gif_path: Path):
    gif = Image.open(gif_path)
    n_frames = getattr(gif, "n_frames", 1)
    indices = np.linspace(0, n_frames - 1, 6, dtype=int)
    thumbs = []
    for idx in indices:
        gif.seek(idx)
        frame = gif.convert("RGB").resize((330, 186), Image.Resampling.LANCZOS)
        thumbs.append(frame)
    sheet = Image.new("RGB", (990, 372), "#fbfaf7")
    for i, frame in enumerate(thumbs):
        sheet.paste(frame, ((i % 3) * 330, (i // 3) * 186))
    sheet.save(gif_path.with_suffix(".contact.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-stochastic-forward",
        action="store_true",
        help="deprecated compatibility flag; Co-LADD always uses the exact forward SDE",
    )
    parser.add_argument("--show-density", action="store_true", help="show the continuous latent density p_t")
    parser.add_argument("--all-variants", action="store_true", help="write all Co-LADD comparison variants")
    parser.add_argument("--contact-sheets", action="store_true", help="write six-frame PNG sheets for visual QA")
    parser.add_argument(
        "--discrete-backward",
        action="store_true",
        help="use the few-step backward animation and write files with a _discrete_steps suffix",
    )
    args = parser.parse_args()

    if args.all_variants or not args.show_density:
        jobs = [
            ("mdm", "mdm_process.gif", False, False),
            ("co", "coladd_process.gif", False, False),
            ("co", "coladd_process_density.gif", False, True),
            ("di", "diladd_process.gif", False, False),
        ]
    else:
        jobs = [
            ("mdm", "mdm_process.gif", False, False),
            ("co", "coladd_process_density.gif", False, True),
            ("di", "diladd_process.gif", False, False),
        ]

    for model, filename, use_stochastic_forward, show_density in jobs:
        if args.discrete_backward:
            filename = filename.removesuffix(".gif") + "_discrete_steps.gif"
        path = save_gif(model, filename, use_stochastic_forward, show_density, args.discrete_backward)
        if args.contact_sheets:
            save_contact_sheet(path)
        print(path)


if __name__ == "__main__":
    main()
