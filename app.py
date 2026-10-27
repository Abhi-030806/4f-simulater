
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from optics import (
    create_grid,
    create_object_a,
    create_object_b,
    create_filter_mask,
    propagate_4f,
    verify_superposition,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="4f Optics - Assignment 16",
    page_icon="🔬",
    layout="wide",
)

st.title("4f Optical Imaging — Assignment 16")
st.caption(
    "Explore the input objects, Fourier-plane filtering, "
    "reconstructed image, and verification of superposition."
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.header("Settings")

    obj_name = st.radio(
        "Input object",
        ["Object A", "Object B"],
    )

    filt_name = st.selectbox(
        "Spatial filter",
        [
            "All-pass",
            "Low-pass",
            "High-pass",
            "Horizontal slit",
            "Vertical slit",
        ],
    )

    param = 5.0

    if filt_name in ("Low-pass", "High-pass"):
        param = st.slider(
            "Cutoff frequency (cycles/mm)",
            0.5, 30.0, 5.0, 0.5,
        )

    elif filt_name in ("Horizontal slit", "Vertical slit"):
        param = st.slider(
            "Slit width (cycles/mm)",
            0.5, 30.0, 4.0, 0.5,
        )

    focal_mm = st.slider(
        "Lens focal length (mm)",
        50, 500, 200, 25,
    )

    n = st.select_slider(
        "Grid size",
        options=[512, 1024, 2048],
        value=1024,
    )

    L_mm = st.slider(
        "Window size (mm)",
        6.0, 12.0, 8.0, 1.0,
    )

    norm_out = st.checkbox(
        "Normalize output intensity",
        value=True,
    )

    log_fourier = st.checkbox(
        "Log scale for Fourier intensity",
        value=True,
    )

    cmap = st.selectbox(
        "Plot colour map",
        ["gray", "viridis", "inferno", "cividis", "magma"],
    )


# ============================================================
# OPTICAL SIMULATION
# ============================================================
L = L_mm * 1e-3
lam = 600e-9
f = focal_mm * 1e-3

grid = create_grid(
    N=n,
    L=L,
    wavelength=lam,
    focal_length=f,
)

if obj_name == "Object A":
    mask = create_object_a(grid)
    masks = [mask]
else:
    mask, masks, centers = create_object_b(grid)

if filt_name == "All-pass":
    H = create_filter_mask(grid, "all-pass")

elif filt_name in ("Low-pass", "High-pass"):
    H = create_filter_mask(
        grid,
        filt_name.lower(),
        cutoff_freq=param * 1e3,
    )

elif filt_name == "Horizontal slit":
    H = create_filter_mask(
        grid,
        "horizontal-slit",
        slit_width=param * 1e3,
    )

else:
    H = create_filter_mask(
        grid,
        "vertical-slit",
        slit_width=param * 1e3,
    )

res = propagate_4f(mask, grid, H)

# Physical coordinate extents
extent = [
    -L_mm / 2,
    L_mm / 2,
    -L_mm / 2,
    L_mm / 2,
]

fmax = grid.f_max / 1e3
freq_extent = [-fmax, fmax, -fmax, fmax]
zoom_freq = min(15.0, fmax)


# ============================================================
# PLOTTING HELPERS
# ============================================================
def make_figure(rows=1, cols=1):
    """Create consistently sized, tightly arranged figures."""
    return plt.subplots(
        rows,
        cols,
        figsize=(5.5 * cols, 4.5 * rows),
        constrained_layout=True,
        squeeze=False,
        dpi=150,
    )


def format_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=9)


def show_img(
    ax,
    data,
    title,
    ext=extent,
    xlabel="x (mm)",
    ylabel="y (mm)",
    log=False,
    fixed_range=None,
    interpolation="lanczos",
):
    """Display an image with consistent color scaling."""

    data = np.asarray(data, dtype=float)

    if log:
        peak = np.max(np.abs(data))
        if peak > 0:
            data = (
                np.log10(
                    1 + 10000 * np.maximum(data, 0) / peak
                )
                / np.log10(10001)
            )

    kwargs = {}

    if fixed_range is not None:
        kwargs["vmin"] = fixed_range[0]
        kwargs["vmax"] = fixed_range[1]

    im = ax.imshow(
        data,
        extent=ext,
        origin="lower",
        cmap=cmap,
        interpolation=interpolation,
        aspect="equal",
        **kwargs,
    )

    format_axes(ax, title, xlabel, ylabel)

    ax.figure.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    return im


def object_limits(data, padding=0.20):
    """Zoom to the nonzero object region with proportional padding."""

    data = np.abs(data)
    peak = np.max(data)

    if peak <= 0:
        return extent

    rows, cols = np.where(data > 0.01 * peak)

    if len(rows) == 0:
        return extent

    x_min = grid.x[cols.min()] * 1e3
    x_max = grid.x[cols.max()] * 1e3
    y_min = grid.y[rows.min()] * 1e3
    y_max = grid.y[rows.max()] * 1e3

    dx = max(x_max - x_min, grid.dx * 1e3)
    dy = max(y_max - y_min, grid.dx * 1e3)

    px = padding * dx
    py = padding * dy

    return [
        max(-L_mm / 2, x_min - px),
        min(L_mm / 2, x_max + px),
        max(-L_mm / 2, y_min - py),
        min(L_mm / 2, y_max + py),
    ]


def apply_limits(ax, limits):
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])


def positive_max(data):
    """Return a safe positive maximum for color normalization."""
    return max(float(np.max(np.abs(data))), 1e-15)


# Use the same physical zoom for all spatial object plots.
spatial_zoom = object_limits(res.U_in)

# Output intensity used for visualization.
out = np.array(res.I_out, dtype=float, copy=True)

if norm_out:
    peak_out = np.max(out)
    if peak_out > 0:
        out /= peak_out


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Object and setup",
        "Fourier plane",
        "Reconstructed image",
        "Superposition",
    ]
)


# ============================================================
# TAB 1: INPUT OBJECT
# ============================================================
with tab1:
    st.subheader("Input object")

    c1, c2 = st.columns([1, 1.5], gap="large")

    with c1:
        st.write(f"**Object:** {obj_name}")
        st.write("**Wavelength:** 600 nm")
        st.write(f"**Focal length:** {focal_mm} mm")
        st.write(f"**Total 4f length:** {4 * focal_mm} mm")
        st.write(f"**Sampling pitch:** {grid.dx * 1e6:.2f} µm")
        st.write(
            f"**Nyquist frequency:** "
            f"{grid.f_max / 1e3:.2f} cycles/mm"
        )

        st.write(
            "Object A: 2 mm square, 1 mm circular aperture, "
            "and opaque triangular region."
        )

        st.write("Object B: four translated copies of Object A.")

    with c2:
        fig, ax = make_figure()

        show_img(
            ax[0, 0],
            res.I_in,
            f"{obj_name} — Input intensity",
            fixed_range=(0, positive_max(res.I_in)),
        )

        apply_limits(ax[0, 0], spatial_zoom)

        st.pyplot(
            fig,
            use_container_width=True,
            dpi=180,
        )
        plt.close(fig)


# ============================================================
# TAB 2: FOURIER PLANE
# ============================================================
with tab2:
    st.subheader("Fourier-plane filtering")

    st.write(
        "The first lens forms the Fourier transform. "
        "The spatial filter selects the frequencies transmitted "
        "through the Fourier plane."
    )

    fig, ax = make_figure(2, 2)

    # Input intensity
    show_img(
        ax[0, 0],
        res.I_in,
        "Input intensity",
        fixed_range=(0, positive_max(res.I_in)),
    )
    apply_limits(ax[0, 0], spatial_zoom)

    # Unfiltered Fourier intensity
    show_img(
        ax[0, 1],
        res.I_fourier_unfiltered,
        "Unfiltered Fourier intensity",
        ext=freq_extent,
        xlabel="fx (cycles/mm)",
        ylabel="fy (cycles/mm)",
        log=log_fourier,
    )

    # Filter transmission: 0 = blocked, 1 = transmitted
    show_img(
        ax[1, 0],
        H,
        f"Filter: {filt_name}",
        ext=freq_extent,
        xlabel="fx (cycles/mm)",
        ylabel="fy (cycles/mm)",
        fixed_range=(0, 1),
        interpolation="nearest",
    )

    # Filtered Fourier intensity
    show_img(
        ax[1, 1],
        res.I_fourier_filtered,
        "Filtered Fourier intensity",
        ext=freq_extent,
        xlabel="fx (cycles/mm)",
        ylabel="fy (cycles/mm)",
        log=log_fourier,
    )

    for a in (ax[0, 1], ax[1, 0], ax[1, 1]):
        a.set_xlim(-zoom_freq, zoom_freq)
        a.set_ylim(-zoom_freq, zoom_freq)

    st.pyplot(fig, use_container_width=True, dpi=180)
    plt.close(fig)


# ============================================================
# TAB 3: RECONSTRUCTED IMAGE
# ============================================================
with tab3:
    st.subheader("Input and reconstructed image")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        fig, ax = make_figure()

        show_img(
            ax[0, 0],
            res.I_in,
            "Input intensity",
            fixed_range=(0, positive_max(res.I_in)),
        )

        apply_limits(ax[0, 0], spatial_zoom)

        st.pyplot(fig, use_container_width=True, dpi=180)
        plt.close(fig)

    with c2:
        fig, ax = make_figure()

        show_img(
            ax[0, 0],
            out,
            "Reconstructed intensity",
            fixed_range=(0, positive_max(out)),
        )

        apply_limits(ax[0, 0], spatial_zoom)

        st.pyplot(fig, use_container_width=True, dpi=180)
        plt.close(fig)

    # Intensity cross-section
    row = int(np.argmin(np.abs(grid.y)))

    if obj_name == "Object B":
        row = int(np.argmin(np.abs(grid.y - 1e-3)))

    fig, ax = make_figure()

    ax[0, 0].plot(
        grid.x * 1e3,
        res.I_in[row],
        label="Input",
        linewidth=2,
    )

    ax[0, 0].plot(
        grid.x * 1e3,
        out[row],
        label="Output",
        linewidth=2,
    )

    ax[0, 0].set_xlim(spatial_zoom[0], spatial_zoom[1])
    ax[0, 0].set_xlabel("x (mm)")
    ax[0, 0].set_ylabel("Intensity")
    ax[0, 0].set_title("Intensity cross-section")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend()

    st.pyplot(fig, use_container_width=True, dpi=180)
    plt.close(fig)

    explanations = {
        "Low-pass": (
            "Low-pass filtering suppresses high spatial frequencies, "
            "which tends to smooth sharp edges."
        ),
        "High-pass": (
            "High-pass filtering suppresses low spatial frequencies "
            "and emphasizes rapid spatial variations."
        ),
        "Horizontal slit": (
            "The horizontal slit passes a narrow band along fx, "
            "restricting frequencies along fy."
        ),
        "Vertical slit": (
            "The vertical slit passes a narrow band along fy, "
            "restricting frequencies along fx."
        ),
        "All-pass": (
            "The all-pass filter transmits all sampled frequencies. "
            "The output should reproduce the input, subject to "
            "sampling, propagation, and numerical scaling."
        ),
    }

    st.info(explanations[filt_name])


# ============================================================
# TAB 4: SUPERPOSITION
# ============================================================
with tab4:
    st.subheader("Principle of superposition")

    st.write(
        "For a linear optical system, the output field of the combined "
        "object should equal the sum of the individual output fields. "
        "Intensities generally do not add directly because coherent "
        "fields can interfere."
    )

    if obj_name == "Object B":
        combined = mask
        individual = masks
    else:
        combined, individual, _ = create_object_b(grid)

    sup = verify_superposition(
        individual,
        combined,
        grid,
        H,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Relative field error",
        f"{sup.rel_field_error:.3e}",
    )

    c2.metric(
        "Maximum field error",
        f"{sup.max_abs_field_error:.3e}",
    )

    c3.metric(
        "Relative intensity discrepancy",
        f"{sup.rel_intensity_discrepancy:.3e}",
    )

    if sup.passed:
        st.success("Field superposition test: PASS")
    else:
        st.warning("Field superposition test: CHECK")

    st.write(f"Tolerance: {sup.tolerance:.1e}")

    fig, ax = make_figure(2, 3)

    show_img(
        ax[0, 0],
        np.abs(sup.U_B_out),
        "Combined field |U_B|",
        fixed_range=(0, positive_max(sup.U_B_out)),
    )

    show_img(
        ax[0, 1],
        np.abs(sup.U_sum_out),
        "Sum of fields |sum(U_i)|",
        fixed_range=(0, positive_max(sup.U_sum_out)),
    )

    show_img(
        ax[0, 2],
        sup.diff_field,
        "Absolute field difference",
        fixed_range=(0, positive_max(sup.diff_field)),
    )

    show_img(
        ax[1, 0],
        sup.I_B,
        "Combined intensity",
        fixed_range=(0, positive_max(sup.I_B)),
    )

    show_img(
        ax[1, 1],
        sup.I_incoherent,
        "Sum of individual intensities",
        fixed_range=(0, positive_max(sup.I_incoherent)),
    )

    show_img(
        ax[1, 2],
        sup.diff_interference,
        "Interference term",
        fixed_range=(0, positive_max(sup.diff_interference)),
    )

    st.pyplot(fig, use_container_width=True, dpi=180)
    plt.close(fig)
