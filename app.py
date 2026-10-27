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

st.set_page_config(page_title="4f Optics - Assignment 16", layout="wide")
st.title("4f Optical Imaging — Assignment 16")
st.caption("Change the object and filter, then inspect the Fourier plane, output image, and superposition test.")

with st.sidebar:
    st.header("Settings")
    obj_name = st.radio("Input object", ["Object A", "Object B"])
    filt_name = st.selectbox(
        "Spatial filter",
        ["All-pass", "Low-pass", "High-pass", "Horizontal slit", "Vertical slit"],
    )

    param = 5.0
    if filt_name in ("Low-pass", "High-pass"):
        param = st.slider("Cutoff frequency (cycles/mm)", 0.5, 30.0, 5.0, 0.5)
    elif filt_name in ("Horizontal slit", "Vertical slit"):
        param = st.slider("Slit width (cycles/mm)", 0.5, 30.0, 4.0, 0.5)

    focal_mm = st.slider("Lens focal length (mm)", 50, 500, 200, 25)
    n = st.select_slider("Grid size", options=[256, 512, 1024], value=512)
    L_mm = st.slider("Window size (mm)", 6.0, 12.0, 8.0, 1.0)
    norm_out = st.checkbox("Normalize output intensity", value=True)
    log_fourier = st.checkbox("Log scale for Fourier intensity", value=True)
    cmap = st.selectbox("Plot colour map", ["gray", "viridis", "inferno", "cividis", "magma"])

L = L_mm * 1e-3
lam = 600e-9
f = focal_mm * 1e-3
grid = create_grid(N=n, L=L, wavelength=lam, focal_length=f)

if obj_name == "Object A":
    mask = create_object_a(grid)
    masks = [mask]
else:
    mask, masks, centers = create_object_b(grid)

if filt_name == "All-pass":
    H = create_filter_mask(grid, "all-pass")
elif filt_name in ("Low-pass", "High-pass"):
    H = create_filter_mask(grid, filt_name.lower(), cutoff_freq=param * 1e3)
elif filt_name == "Horizontal slit":
    H = create_filter_mask(grid, "horizontal-slit", slit_width=param * 1e3)
else:
    H = create_filter_mask(grid, "vertical-slit", slit_width=param * 1e3)

res = propagate_4f(mask, grid, H)
extent = [-L_mm / 2, L_mm / 2, -L_mm / 2, L_mm / 2]
fmax = grid.f_max / 1e3
zoom = min(15.0, fmax)
freq_extent = [-fmax, fmax, -fmax, fmax]

def show_img(ax, data, title, ext=extent, xlabel="x (mm)", ylabel="y (mm)", log=False):
    if log:
        data = np.log10(1 + 10000 * data / (np.max(data) + 1e-15)) / np.log10(10001)
    im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Object and setup", "Fourier plane", "Reconstructed image", "Superposition"]
)

with tab1:
    st.subheader("Input object")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Object: {obj_name}")
        st.write("Wavelength: 600 nm")
        st.write(f"Focal length: {focal_mm} mm")
        st.write(f"Total 4f length: {4 * focal_mm} mm")
        st.write(f"Sampling pitch: {grid.dx * 1e6:.2f} µm")
        st.write(f"Nyquist frequency: {grid.f_max / 1e3:.2f} cycles/mm")
        st.write("Object A: 2 mm square, 1 mm circular aperture, opaque triangular region.")
        st.write("Object B: four translated copies of Object A.")
    with c2:
        fig, ax = plt.subplots()
        show_img(ax, res.U_in.real, f"{obj_name} transmission")
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    st.subheader("Fourier-plane filtering")
    st.write("The first lens forms the Fourier transform. The filter removes selected spatial frequencies.")
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    show_img(ax[0, 0], res.I_in, "Input intensity")
    show_img(ax[0, 1], res.I_fourier_unfiltered, "Unfiltered Fourier intensity",
             ext=freq_extent, xlabel="fx (cycles/mm)", ylabel="fy (cycles/mm)", log=log_fourier)
    show_img(ax[1, 0], H, f"Filter: {filt_name}",
             ext=freq_extent, xlabel="fx (cycles/mm)", ylabel="fy (cycles/mm)")
    show_img(ax[1, 1], res.I_fourier_filtered, "Filtered Fourier intensity",
             ext=freq_extent, xlabel="fx (cycles/mm)", ylabel="fy (cycles/mm)", log=log_fourier)
    for a in ax.flat:
        a.set_xlim(-zoom, zoom)
        a.set_ylim(-zoom, zoom)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Input and reconstructed image")
    out = res.I_out.copy()
    if norm_out:
        out /= np.max(out) + 1e-15

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        show_img(ax, res.I_in, "Input intensity")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        show_img(ax, out, "Reconstructed intensity")
        st.pyplot(fig)
        plt.close(fig)

    row = n // 2
    if obj_name == "Object B":
        row = int(np.argmin(np.abs(grid.y - 1e-3)))
    fig, ax = plt.subplots()
    ax.plot(grid.x * 1e3, res.I_in[row], label="Input")
    ax.plot(grid.x * 1e3, out[row], label="Output")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Intensity cross-section at y = {grid.y[row] * 1e3:.2f} mm")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    if filt_name == "Low-pass":
        st.write("Low-pass filtering removes high spatial frequencies, so sharp edges become smoother.")
    elif filt_name == "High-pass":
        st.write("High-pass filtering removes low spatial frequencies and emphasizes rapid changes and edges.")
    elif filt_name == "Horizontal slit":
        st.write("The horizontal slit passes frequencies along fx but restricts fy, reducing vertical detail.")
    elif filt_name == "Vertical slit":
        st.write("The vertical slit passes frequencies along fy but restricts fx, reducing horizontal detail.")
    else:
        st.write("The all-pass filter keeps all sampled frequencies, so the output should reproduce the input, subject to numerical sampling.")

with tab4:
    st.subheader("Principle of superposition")
    st.write("For coherent fields, the combined output should equal the sum of the outputs from the four individual objects. Intensities do not generally add directly because of interference.")

    if obj_name == "Object B":
        combined = mask
        individual = masks
    else:
        combined, individual, _ = create_object_b(grid)

    sup = verify_superposition(individual, combined, grid, H)

    c1, c2, c3 = st.columns(3)
    c1.metric("Relative field error", f"{sup.rel_field_error:.3e}")
    c2.metric("Maximum field error", f"{sup.max_abs_field_error:.3e}")
    c3.metric("Relative intensity discrepancy", f"{sup.rel_intensity_discrepancy:.3e}")

    st.write("Field superposition test:", "PASS" if sup.passed else "CHECK")
    st.write(f"Tolerance: {sup.tolerance:.1e}")

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    show_img(ax[0, 0], np.abs(sup.U_B_out), "Combined field |U_B|")
    show_img(ax[0, 1], np.abs(sup.U_sum_out), "Sum of fields |sum(U_i)|")
    show_img(ax[0, 2], sup.diff_field, "Absolute field difference")
    show_img(ax[1, 0], sup.I_B, "Combined intensity")
    show_img(ax[1, 1], sup.I_incoherent, "Sum of individual intensities")
    show_img(ax[1, 2], sup.diff_interference, "Interference term")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)