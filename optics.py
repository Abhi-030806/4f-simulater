from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class OpticalGrid:
    N: int
    L: float
    dx: float
    dy: float
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    FX: np.ndarray
    FY: np.ndarray
    wavelength: float
    focal_length: float

    @property
    def f_max(self):
        return 1.0 / (2.0 * self.dx)

    @property
    def df(self):
        return 1.0 / self.L

    @property
    def fourier_plane_X(self):
        return self.wavelength * self.focal_length * self.FX

    @property
    def fourier_plane_Y(self):
        return self.wavelength * self.focal_length * self.FY


def create_grid(N=512, L=8e-3, wavelength=600e-9, focal_length=0.2):
    if N <= 0:
        raise ValueError("N must be positive")

    dx = L / N
    dy = dx

    x = (np.arange(N) - N // 2) * dx
    y = (np.arange(N) - N // 2) * dy
    X, Y = np.meshgrid(x, y)

    f = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fx = f
    fy = f
    FX, FY = np.meshgrid(fx, fy)

    return OpticalGrid(
        N, L, dx, dy, x, y, X, Y, fx, fy, FX, FY,
        wavelength, focal_length
    )


def point_in_triangle(px, py, v1, v2, v3):
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
    d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3)
    d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1)

    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return ~(neg & pos)


def get_equilateral_triangle_vertices(center, radius, orientation_deg=0):
    xc, yc = center
    a = np.radians([orientation_deg, orientation_deg + 120, orientation_deg + 240])

    v1 = (xc + radius * np.cos(a[0]), yc + radius * np.sin(a[0]))
    v2 = (xc + radius * np.cos(a[1]), yc + radius * np.sin(a[1]))
    v3 = (xc + radius * np.cos(a[2]), yc + radius * np.sin(a[2]))

    return v1, v2, v3


def create_object_a(
    grid,
    square_size=2e-3,
    circle_diam=1e-3,
    triangle_radius=None,
    orientation_deg=0,
    center=(0, 0)
):
    xc, yc = center

    if triangle_radius is None:
        triangle_radius = circle_diam / 2

    dx = grid.X - xc
    dy = grid.Y - yc

    sq = (np.abs(dx) <= square_size / 2) & (np.abs(dy) <= square_size / 2)
    cir = dx**2 + dy**2 <= (circle_diam / 2)**2

    v1, v2, v3 = get_equilateral_triangle_vertices(
        center, triangle_radius, orientation_deg
    )

    tri = point_in_triangle(grid.X, grid.Y, v1, v2, v3)

    mask = np.zeros_like(grid.X, dtype=float)
    mask[sq & cir & ~tri] = 1.0

    return mask


def create_object_b(
    grid,
    square_size=2e-3,
    circle_diam=1e-3,
    triangle_radius=None,
    spacing=2e-3,
    orientations=(0, 90, 270, 180)
):
    if triangle_radius is None:
        triangle_radius = circle_diam / 2

    s = spacing / 2
    centers = [
        (-s, s),
        (s, s),
        (-s, -s),
        (s, -s)
    ]

    masks = []

    for c, ori in zip(centers, orientations):
        m = create_object_a(
            grid,
            square_size,
            circle_diam,
            triangle_radius,
            ori,
            c
        )
        masks.append(m)

    combined = np.maximum.reduce(masks)

    return combined, masks, centers


def fourier_transform_2d(field, dx, dy):
    return dx * dy * np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(field))
    )


def inverse_fourier_transform_2d(field, dx, dy):
    return (1 / (dx * dy)) * np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(field))
    )


def create_filter_mask(grid, filter_type, cutoff_freq=None, slit_width=None):
    ft = filter_type.strip().lower()

    if ft in ("all-pass", "none", "unfiltered"):
        return np.ones_like(grid.FX)

    if ft == "zero":
        return np.zeros_like(grid.FX)

    r = np.sqrt(grid.FX**2 + grid.FY**2)

    if ft in ("low-pass", "lowpass", "lp"):
        if cutoff_freq is None or cutoff_freq < 0:
            raise ValueError("Invalid cutoff frequency")
        return (r <= cutoff_freq).astype(float)

    if ft in ("high-pass", "highpass", "hp"):
        if cutoff_freq is None or cutoff_freq < 0:
            raise ValueError("Invalid cutoff frequency")
        return (r >= cutoff_freq).astype(float)

    if ft in ("horizontal-slit", "horizontal_slit", "h_slit"):
        if slit_width is None or slit_width < 0:
            raise ValueError("Invalid slit width")
        return (np.abs(grid.FY) <= slit_width / 2).astype(float)

    if ft in ("vertical-slit", "vertical_slit", "v_slit"):
        if slit_width is None or slit_width < 0:
            raise ValueError("Invalid slit width")
        return (np.abs(grid.FX) <= slit_width / 2).astype(float)

    raise ValueError(f"Unknown filter type: {filter_type}")


@dataclass
class SimulationResults:
    U_in: np.ndarray
    U_fourier_unfiltered: np.ndarray
    filter_mask: np.ndarray
    U_fourier_filtered: np.ndarray
    U_out: np.ndarray
    I_in: np.ndarray
    I_fourier_unfiltered: np.ndarray
    I_fourier_filtered: np.ndarray
    I_out: np.ndarray


def propagate_4f(
    mask,
    grid,
    filter_mask,
    incident_amplitude=1.0 + 0j,
    include_phase_factors=False
):
    U_in = incident_amplitude * mask
    U_f = fourier_transform_2d(U_in, grid.dx, grid.dy)

    if include_phase_factors:
        k = 2 * np.pi / grid.wavelength
        p = np.exp(1j * k * grid.focal_length) / (
            1j * grid.wavelength * grid.focal_length
        )
        U_f *= p

    U_ff = U_f * filter_mask

    if include_phase_factors:
        U_ff = U_ff / p

    U_out = inverse_fourier_transform_2d(U_ff, grid.dx, grid.dy)

    return SimulationResults(
        U_in=U_in,
        U_fourier_unfiltered=U_f,
        filter_mask=filter_mask,
        U_fourier_filtered=U_ff if not include_phase_factors else U_f * filter_mask,
        U_out=U_out,
        I_in=np.abs(U_in)**2,
        I_fourier_unfiltered=np.abs(U_f)**2,
        I_fourier_filtered=np.abs(U_f * filter_mask)**2,
        I_out=np.abs(U_out)**2
    )


@dataclass
class SuperpositionResults:
    U_B_out: np.ndarray
    U_sum_out: np.ndarray
    I_B: np.ndarray
    I_incoherent: np.ndarray
    I_coherent_sum: np.ndarray
    diff_field: np.ndarray
    diff_intensity_incoh: np.ndarray
    diff_interference: np.ndarray
    rel_field_error: float
    max_abs_field_error: float
    rel_intensity_discrepancy: float
    max_abs_intensity_discrepancy: float
    passed: bool
    tolerance: float
    individual_results: List[SimulationResults]


def verify_superposition(
    individual_masks,
    combined_mask,
    grid,
    filter_mask,
    tolerance=1e-11,
    incident_amplitude=1.0 + 0j
):
    eps = 1e-15

    res_b = propagate_4f(
        combined_mask, grid, filter_mask, incident_amplitude
    )

    U_b = res_b.U_out
    I_b = res_b.I_out

    results = []
    fields = []
    intensities = []

    for m in individual_masks:
        res = propagate_4f(m, grid, filter_mask, incident_amplitude)
        results.append(res)
        fields.append(res.U_out)
        intensities.append(res.I_out)

    U_sum = np.sum(fields, axis=0)
    I_sum = np.abs(U_sum)**2
    I_incoh = np.sum(intensities, axis=0)

    diff_u = np.abs(U_b - U_sum)
    diff_i = np.abs(I_b - I_incoh)
    diff_cross = I_sum - I_incoh

    rel_u = np.linalg.norm(U_b - U_sum) / (np.linalg.norm(U_b) + eps)
    max_u = np.max(diff_u)

    rel_i = np.linalg.norm(I_b - I_incoh) / (np.linalg.norm(I_b) + eps)
    max_i = np.max(diff_i)

    return SuperpositionResults(
        U_B_out=U_b,
        U_sum_out=U_sum,
        I_B=I_b,
        I_incoherent=I_incoh,
        I_coherent_sum=I_sum,
        diff_field=diff_u,
        diff_intensity_incoh=diff_i,
        diff_interference=diff_cross,
        rel_field_error=float(rel_u),
        max_abs_field_error=float(max_u),
        rel_intensity_discrepancy=float(rel_i),
        max_abs_intensity_discrepancy=float(max_i),
        passed=bool(rel_u < tolerance),
        tolerance=tolerance,
        individual_results=results
    )