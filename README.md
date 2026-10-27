# Simulation of Optical Imaging Using a 4f System (Assignment 16)

An interactive, scientific computational physics application implementing a full **4f Fourier optics system** to simulate optical imaging, 2D spatial frequency filtering, and rigorous numerical verification of the **Principle of Superposition**.

Developed in Python using **NumPy**, **SciPy**, **Matplotlib**, and **Streamlit** based on the geometric and physical requirements of **Assignment #16**.

---

## 1. Project Overview & Physical Principles

A classic **4f optical imaging system** (Fourier processor) consists of two thin positive lenses of equal focal length $f$ arranged along a common optical axis:

```
[Object Plane] ────(f)────► [Lens 1] ────(f)────► [Fourier Plane] ────(f)────► [Lens 2] ────(f)────► [Observation Screen]
    (z = 0)                   (z = f)             & Spatial Filter          (z = 3f)                 (z = 4f)
                                                       (z = 2f)
```

- **Total optical path length**: $4f$.
- **Separation between lenses**: $2f$.
- **Wavelength**: $\lambda = 600\text{ nm}$ (monochromatic, spatially and temporally coherent plane wave).

### Optical Propagation Mathematics

1. **Object Plane ($z = 0$)**:
   The input optical field immediately behind a binary amplitude mask $t(x, y) \in \{0, 1\}$ illuminated normally by a unit-amplitude plane wave ($U_0 = 1$) is:
   $$U_{\text{in}}(x, y) = U_0 \cdot t(x, y)$$

2. **Fourier Transform by Lens 1 ($z = 2f$)**:
   Because the object is located precisely at the front focal plane of Lens 1, the quadratic phase curvature across the back focal plane vanishes identically (Goodman, *Introduction to Fourier Optics*, §5.2). The field in the Fourier plane is the exact two-dimensional Fourier transform of the object field:
   $$U_F(f_x, f_y) = \iint_{-\infty}^{\infty} U_{\text{in}}(x, y) \exp\left[-i 2\pi (f_x x + f_y y)\right] dx\,dy$$
   where the spatial frequencies $(f_x, f_y)$ map to physical Fourier plane coordinates $(X, Y)$ via:
   $$X = \lambda f f_x, \qquad Y = \lambda f f_y$$

3. **Fourier-Plane Spatial Filtering ($z = 2f$)**:
   A filter with complex amplitude transmission function $H(f_x, f_y)$ modulates the **complex optical field**:
   $$U_{F, \text{filtered}}(f_x, f_y) = H(f_x, f_y) \cdot U_F(f_x, f_y)$$

4. **Image Reconstruction by Lens 2 ($z = 4f$)**:
   Lens 2 performs an inverse Fourier transform, forming the reconstructed output complex field at the observation screen:
   $$U_{\text{out}}(x, y) = \iint_{-\infty}^{\infty} U_{F, \text{filtered}}(f_x, f_y) \exp\left[+i 2\pi (f_x x + f_y y)\right] df_x\,df_y$$

5. **Screen Intensity**:
   Square-law optical sensors measure the irradiance (intensity):
   $$I_{\text{out}}(x, y) = |U_{\text{out}}(x, y)|^2$$

---

## 2. Assignment Geometry: Object A & Object B

Both objects are constructed parametrically from exact vector and geometric definitions (no manual raster painting):

### Object A
- **Outer boundary**: Square of side length $2.0\text{ mm}$, opaque background ($t = 0$).
- **Circular aperture**: Centered disk of diameter $1.0\text{ mm}$ (radius $r = 0.5\text{ mm}$), transparent ($t = 1$).
- **Equilateral triangle**: Centered and inscribed inside the circle (circumradius $R = 0.5\text{ mm}$), opaque ($t = 0$).
- **Orientation**: Primary vertex points along $+x$ (to the right, $\theta_0 = 0^\circ$).
- **Transmission mask**:
  $$t_A(x, y) = \begin{cases} 1, & \text{inside circle and outside triangle} \\ 0, & \text{otherwise} \end{cases}$$

### Object B
- Consists of **four translated copies** of Object A arranged in a $2 \times 2$ grid inside a $4.0\text{ mm} \times 4.0\text{ mm}$ square region:
  - **Top-Left** (Sub-object 1): Center $(-1.0\text{ mm}, +1.0\text{ mm})$, triangle points **Right** ($\rightarrow$, $0^\circ$)
  - **Top-Right** (Sub-object 2): Center $(+1.0\text{ mm}, +1.0\text{ mm})$, triangle points **Up** ($\uparrow$, $90^\circ$)
  - **Bottom-Left** (Sub-object 3): Center $(-1.0\text{ mm}, -1.0\text{ mm})$, triangle points **Down** ($\downarrow$, $270^\circ$)
  - **Bottom-Right** (Sub-object 4): Center $(+1.0\text{ mm}, -1.0\text{ mm})$, triangle points **Left** ($\leftarrow$, $180^\circ$)
- Because the four constituent masks $t_i(x, y)$ are mutually disjoint with zero overlap:
  $$t_B(x, y) = \max_{i=1,\ldots,4} t_i(x, y) \equiv \sum_{i=1}^4 t_i(x, y)$$

---

## 3. Four Interactive Spatial Filters

The application implements four spatial frequency filters in the Fourier plane:

| Filter | Formula $H(f_x, f_y)$ | Adjustable Parameter | Physical Effect |
| :--- | :--- | :--- | :--- |
| **Low-Pass** | $1$ if $\sqrt{f_x^2 + f_y^2} \le f_c$, else $0$ | Cutoff frequency $f_c$ (cycles/mm) | Rejects high frequencies; smooths fine details and rounds sharp corners (demonstrates Abbe resolution limit & Gibbs ringing). |
| **High-Pass** | $0$ if $\sqrt{f_x^2 + f_y^2} < f_c$, else $1$ | Cutoff frequency $f_c$ (cycles/mm) | Rejects central DC peak; flat areas turn black while sharp boundaries glow brightly (optical edge detection). |
| **Horizontal Slit** | $1$ if $\|f_y\| \le w_y / 2$, else $0$ | Slit width $w_y$ (cycles/mm) | Passes all horizontal frequencies $f_x$, restricts vertical frequencies; vertical features blur while horizontal edges remain sharp. |
| **Vertical Slit** | $1$ if $\|f_x\| \le w_x / 2$, else $0$ | Slit width $w_x$ (cycles/mm) | Passes all vertical frequencies $f_y$, restricts horizontal frequencies; horizontal features blur while vertical edges remain sharp. |
| **All-Pass** | $1$ everywhere | None | Unfiltered baseline reproducing input object to machine precision. |

---

## 4. Principle of Superposition: Numerical Verification

A central objective of Assignment 16 is testing the **Principle of Superposition**.

### Complex Field Superposition
Let $\mathcal{P}_H$ denote the linear operator representing propagation through Lens 1, spatial filtering by $H$, and inverse propagation through Lens 2. Because $\mathcal{P}_H$ is linear:
$$\mathcal{P}_H\left\{\sum_{i=1}^4 U_i^{\text{in}}\right\} = \sum_{i=1}^4 \mathcal{P}_H\{U_i^{\text{in}}\} \implies U_B^{\text{out}} = \sum_{i=1}^4 U_i^{\text{out}} = U_{\text{sum}}$$

We evaluate the relative field error:
$$\epsilon_U = \frac{\|U_B^{\text{out}} - U_{\text{sum}}\|_2}{\|U_B^{\text{out}}\|_2 + \varepsilon}$$
In our simulation, $\epsilon_U < 10^{-14}$, confirming linear superposition to floating-point precision.

### Coherent vs. Incoherent Intensity
For intensities, the square of the summed field includes interference cross-terms:
$$I_{\text{coherent}} = \left|\sum_{i=1}^4 U_i\right|^2 = \sum_{i=1}^4 |U_i|^2 + \sum_{i \ne j} U_i U_j^* = I_{\text{incoherent}} + I_{\text{interference}}$$
- If spatial filtering causes the fields of adjacent apertures to diffract and spread into adjacent regions, the overlapping tails interfere coherently.
- Consequently, $I_{\text{coherent}} \ne I_{\text{incoherent}}$.
- The dashboard displays the difference map $D_I = |I_B - I_{\text{incoherent}}|$ and interference term map $2 \sum_{i<j} \text{Re}(U_i U_j^*)$, providing direct visual proof that **intensities do not obey superposition in a coherent optical system**.

---

## 5. Numerical Conventions & Coordinate Sampling

- **Computational domain**: Square window of side length $L = 8.0\text{ mm}$ ($8 \times 10^{-3}\text{ m}$).
- **Sampling resolution**: $N \in \{256, 512, 1024\}$ (default: $N = 512$).
- **Pixel pitch**: $\Delta x = \Delta y = L / N = 15.625\ \mu\text{m}$ (at $N=512$).
- **Spatial frequency range**: $f_x, f_y \in [-f_{\text{Nyq}}, f_{\text{Nyq}})$ with $f_{\text{Nyq}} = 1 / (2\Delta x) = 32.0\text{ cycles/mm}$ and step $\Delta f = 1/L = 0.125\text{ cycles/mm}$.
- **Fourier plane scaling**: Physical coordinates $X = \lambda f f_x$, $Y = \lambda f f_y$. At $\lambda = 600\text{ nm}$ and $f = 200\text{ mm}$, $1\text{ cycle/mm} \leftrightarrow 0.12\text{ mm}$.
- **FFT normalization**: Continuous 2D Fourier integral approximated as:
  $$\tilde{U}(f_x, f_y) \approx \Delta x \Delta y \cdot \operatorname{fftshift}\left(\operatorname{fft2}\left(\operatorname{ifftshift}(U)\right)\right)$$
  $$\operatorname{recon}(x, y) \approx \frac{1}{\Delta x \Delta y} \cdot \operatorname{fftshift}\left(\operatorname{ifft2}\left(\operatorname{ifftshift}(\tilde{U})\right)\right)$$
  ensuring strict roundtrip identity without numerical scaling drift.

---

## 6. Installation & Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12.
- A virtual environment is recommended.

### Linux / macOS
```bash
# 1. Clone or navigate into the repository directory
cd assignment16

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest tests/test_optics.py -v

# 5. Launch interactive Streamlit dashboard
streamlit run app.py
```

### Windows (PowerShell)
```powershell
# 1. Navigate into repository
cd assignment16

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest tests/test_optics.py -v

# 5. Launch interactive Streamlit dashboard
streamlit run app.py
```

---

## 7. Assignment Requirement Mapping

| Assignment Requirement | Implementation in Application | Automated Test in `test_optics.py` |
| :--- | :--- | :--- |
| **Object A Geometry** (2 mm square, 1 mm circle, triangle) | `optics.create_object_a()` | `test_object_a_geometry`, `test_amplitude_mask_is_binary` |
| **Object B Geometry** (4 translated copies, 2x2, distinct orientations) | `optics.create_object_b()` | `test_object_b_geometry_and_orientations` |
| **Ideal 4f Optical System** ($f, 2f, 4f$ distances, $\lambda=600$ nm) | Tab 1 diagram & metrics; `optics.create_grid()` | `test_coordinate_grid_and_frequencies` |
| **2D FFT Optical Propagation & Reconstruction** | `optics.fourier_transform_2d()`, `optics.propagate_4f()` | `test_fft_ifft_roundtrip`, `test_all_pass_filter` |
| **Complex Field Spatial Filtering** | `optics.create_filter_mask()` applied to complex $U_F$ | `test_zero_filter`, `test_filter_change_affects_output` |
| **Four Required Interactive Filters** (Low-pass, High-pass, H-slit, V-slit) | Tab 2 & Tab 3; interactive parameter sliders | `test_lowpass_highpass_complementarity`, `test_slit_filters_orientation_and_width` |
| **Superposition Verification: Complex Field** | Tab 4 scorecard & difference map $D_U(x, y)$ | `test_complex_field_superposition` |
| **Superposition Verification: Intensity vs. Coherent** | Tab 4 side-by-side $I_B$ vs $I_{\text{incoherent}}$ & interference map | `test_coherent_vs_incoherent_intensity` |
| **Numerical Stability & No NaNs** | Safe boundaries, epsilon guards, zero-division safeguards | `test_numerical_stability` |

---

## 8. Model Assumptions & Limitations

1. **Paraxial Scalar Diffraction**: The simulation assumes scalar wave theory under paraxial Fresnel/Fraunhofer approximations. Polarization, vector diffraction, and off-axis non-paraxial effects are not modeled.
2. **Thin-Lens Model**: Lenses are treated as ideal phase transformers without spherical or chromatic aberrations.
3. **Periodic FFT Boundaries**: Discrete FFT implies periodic domain continuation; the field of view $L = 8\text{ mm}$ provides ample zero-padding to eliminate boundary wrap-around.

