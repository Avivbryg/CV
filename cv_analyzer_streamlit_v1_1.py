# cv_analyzer_streamlit_v1_1.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

from scipy.signal import find_peaks

st.set_page_config(page_title="CV Analyzer v1.1", layout="wide")


# =========================== Robust MPT reader ===========================
def read_mpt_bytes(uploaded_file):
    """Robust MPT reader for Streamlit UploadedFile (BioLogic .MPT)."""
    raw = uploaded_file.getvalue()  # important: don't call .read() twice
    text = raw.decode("latin1", errors="ignore")
    lines = text.splitlines()

    # Find header line containing column names
    header_index = None
    for i, line in enumerate(lines):
        if "Ewe/V" in line and "<I>/mA" in line:
            header_index = i
            break
    if header_index is None:
        raise ValueError("Could not find MPT header containing 'Ewe/V' and '<I>/mA'.")

    header = lines[header_index].split("\t")

    # Parse data rows
    rows = []
    for line in lines[header_index + 1:]:
        parts = line.split("\t")
        try:
            rows.append([float(x.replace(",", ".")) for x in parts])
        except:
            continue  # skip non-numeric lines

    data = np.array(rows)
    if data.size == 0:
        raise ValueError("No numeric data rows found after header.")

    # Helper to find column index by partial names
    def find(*names):
        for name in names:
            for i, h in enumerate(header):
                if name in h:
                    return i
        return None

    idx_E   = find("Ewe/V", "Ewe")
    idx_I   = find("<I>/mA", "I/mA")
    idx_t   = find("time/s", "time")
    idx_cyc = find("cycle number", "cycle", "Cycle", "Ns")

    if idx_E is None or idx_I is None:
        raise ValueError("Missing 'E' or 'I' columns in MPT file.")

    E = data[:, idx_E]
    I = data[:, idx_I]
    t = data[:, idx_t] if idx_t is not None else np.arange(len(E))
    cyc = data[:, idx_cyc].astype(int) if idx_cyc is not None else np.zeros(len(E), int)

    return E, I, t, cyc


# =========================== Reference conversion ===========================
REF_TABLE = {
    "RHE": 0.0,
    "Ag/AgCl 3M KCl": 0.205,
    "Ag/AgCl sat. KCl": 0.197,
    "Hg/HgO 1M KOH": 0.098,
    "Hg/HgO 0.1M KOH": 0.165,
    "SCE": 0.241,
}

def convert_reference(E, ref_type, pH):
    """Convert measured E vs the chosen ref to RHE scale."""
    if ref_type == "RHE":
        return E
    return E + REF_TABLE.get(ref_type, 0.0) + 0.059 * float(pH)


# =========================== OER subtraction (overlap only) ===========================
def subtract_oer(E, I):
    """
    Create corrected I only in the forward anodic overlap region:
    - Split curve at vertex (max E).
    - Interpolate backward anodic branch onto forward anodic E grid.
    - Subtract to estimate OER-only current.
    Returns (I_corr, overlap_mask) or (None, None) if not enough overlap.
    """
    if len(E) < 5:
        return None, None

    idx_sw = np.argmax(E)
    fw_mask = np.arange(len(E)) <= idx_sw
    bw_mask = ~fw_mask

    Ef, If = E[fw_mask], I[fw_mask]
    Eb, Ib = E[bw_mask], I[bw_mask]

    pos_f = If > 0
    pos_b = Ib > 0
    if np.sum(pos_f) < 5 or np.sum(pos_b) < 5:
        return None, None

    # Forward anodic (positive) region
    Ef_pos = Ef[pos_f]
    If_pos = If[pos_f]
    fw_idx_all = np.where(fw_mask)[0]
    fw_pos_idx = fw_idx_all[pos_f]

    # Backward anodic (positive) region
    Eb_pos = Eb[pos_b]
    Ib_pos = Ib[pos_b]

    # Sort both by E
    of = np.argsort(Ef_pos)
    ob = np.argsort(Eb_pos)
    Ef_pos = Ef_pos[of]; If_pos = If_pos[of]; fw_pos_idx = fw_pos_idx[of]
    Eb_pos = Eb_pos[ob]; Ib_pos = Ib_pos[ob]

    E_low  = max(Ef_pos.min(), Eb_pos.min())
    E_high = min(Ef_pos.max(), Eb_pos.max())
    if E_high <= E_low:
        return None, None

    in_win = (Ef_pos >= E_low) & (Ef_pos <= E_high)
    if np.sum(in_win) < 3:
        return None, None

    I_back_interp = np.interp(Ef_pos[in_win], Eb_pos, Ib_pos)

    I_corr = I.copy()
    I_corr[fw_pos_idx[in_win]] = If_pos[in_win] - I_back_interp

    overlap_mask = np.zeros_like(I, dtype=bool)
    overlap_mask[fw_pos_idx[in_win]] = True

    return I_corr, overlap_mask


# =========================== Q integration (same length as input) ===========================
def integrate_Q(I, t):
    """
    Cumulative trapezoidal charge with same length as input arrays,
    starting at 0 at the first sample.
    Units: mA*s (if I in mA, t in s).
    """
    if len(I) < 2:
        return np.zeros_like(I)
    incr = 0.5 * (I[1:] + I[:-1]) * np.diff(t)
    Q = np.concatenate([[0.0], np.cumsum(incr)])
    return Q


# =========================== Peak detection & ΔEpp ===========================
def detect_peaks(E, Y, distance=10, prominence=None):
    """
    Detect anodic (maxima) and cathodic (minima) peaks on Y(E).
    Returns two lists of tuples: [(E_peak, Y_peak), ...]
    """
    idx_max, _ = find_peaks(Y, distance=distance, prominence=prominence)
    idx_min, _ = find_peaks(-Y, distance=distance, prominence=prominence)
    anodic   = [(E[i], Y[i]) for i in idx_max]
    cathodic = [(E[i], Y[i]) for i in idx_min]
    return anodic, cathodic


def pair_peaks_for_dEpp(anodic, cathodic):
    """
    For each anodic peak, find the cathodic peak closest in potential.
    Returns list of (E_an, E_ca, dEpp).
    """
    pairs = []
    if not anodic or not cathodic:
        return pairs
    cath_E = np.array([e for e, _ in cathodic])
    for e_an, _ in anodic:
        idx = np.argmin(np.abs(cath_E - e_an))
        e_ca = cathodic[idx][0]
        pairs.append((e_an, e_ca, abs(e_an - e_ca)))
    return pairs


# =========================== Scan rate ===========================
def compute_scan_rate(E, t):
    """
    Estimate scan rate as median(|dE/dt|) over the cycle.
    Returns V/s.
    """
    if len(E) < 3:
        return np.nan
    dE = np.gradient(E)
    dt = np.gradient(t)
    v = np.abs(dE / dt)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.nanmedian(v))


# =========================== ECSA ===========================
def compute_ecsa(E, I, t, e_low, e_high, C_spec):
    """
    Integrate Q in [e_low, e_high] (RHE scale), then ECSA = Q_dl / C_spec.
    C_spec in F/cm^2 (be careful with units!).
    If I is in mA and t in s -> Q in mA*s; convert to Coulomb (C) by 1e-3.
    """
    mask = (E >= e_low) & (E <= e_high)
    if np.sum(mask) < 2:
        return 0.0, 0.0
    Q = integrate_Q(I[mask], t[mask])  # mA*s
    Q_dl_mAs = Q[-1] if Q.size else 0.0
    Q_dl_C = Q_dl_mAs * 1e-3  # C
    if C_spec <= 0:
        return Q_dl_C, np.nan
    ecsa_cm2 = Q_dl_C / C_spec  # cm^2
    return Q_dl_C, ecsa_cm2


# =========================== UI ===========================
st.title("🔬 CV Analyzer – Streamlit v1.1")
st.caption("Multi-file CV analysis with RHE conversion, OER correction, peaks & ΔEpp, scan rate, ECSA, and Q vs E.")

uploaded = st.file_uploader("📁 Upload BioLogic .MPT files", accept_multiple_files=True)

if not uploaded:
    st.stop()

# Sidebar controls
st.sidebar.header("⚙ Global Settings")
ref = st.sidebar.selectbox("Reference electrode", list(REF_TABLE.keys()))
pH = st.sidebar.number_input("pH", min_value=0.0, max_value=14.5, value=14.0)
line_width = st.sidebar.slider("Line width", 1, 6, 2)
font_size = st.sidebar.slider("Font size", 8, 24, 12)
show_legend = st.sidebar.checkbox("Show legend", value=True)

st.sidebar.markdown("---")
mode_q = st.sidebar.checkbox("Plot Q vs E (integrated Q)")
mode_oer = st.sidebar.checkbox("Subtract OER in overlap region")
st.sidebar.markdown("---")
show_peaks = st.sidebar.checkbox("Show peaks (max/min)")
show_dEpp = st.sidebar.checkbox("Show ΔEpp (peak-to-peak)")
peak_distance = st.sidebar.slider("Peak min distance (samples)", 5, 100, 15)
st.sidebar.markdown("---")
ecsa_on = st.sidebar.checkbox("ECSA mode")
ecsa_win_low = st.sidebar.number_input("ECSA window – low (V_RHE)", value=0.10, step=0.01, format="%.2f")
ecsa_win_high = st.sidebar.number_input("ECSA window – high (V_RHE)", value=0.20, step=0.01, format="%.2f")
# Specific capacitance in F/cm^2. (e.g., 0.04 F/m^2 = 0.004 mF/cm^2; many works use ~40–60 µF/cm^2 = 4e-5–6e-5 F/cm^2)
C_spec_input = st.sidebar.number_input("Specific capacitance Cₛ (F/cm²)", value=4e-5, format="%.6f")

# Figure
fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_id = 0

# Collected data for CSV export
export_rows = []

for upfile in uploaded:
    st.subheader(f"📄 File: {upfile.name}")
    try:
        E_raw, I_raw, t_raw, cyc_raw = read_mpt_bytes(upfile)
    except Exception as e:
        st.error(f"{upfile.name}: {e}")
        continue

    file_label = st.text_input(f"Rename file label ({upfile.name})", upfile.name, key=f"filelabel_{upfile.name}")

    cycles = sorted(set(cyc_raw))
    default_cycles = [cycles[0]] if cycles else []
    chosen_cycles = st.multiselect(
        f"Select cycles for {upfile.name}",
        options=cycles,
        default=default_cycles,
        key=f"cycles_{upfile.name}"
    )

    for c in chosen_cycles:
        mask = (cyc_raw == c)
        if not np.any(mask):
            continue

        E = E_raw[mask]
        I = I_raw[mask]
        t = t_raw[mask]

        # Convert to RHE
        E_rhe = convert_reference(E, ref, pH)

        # Per-cycle label
        cycle_label = st.text_input(f"Rename Cycle {c}", f"Cycle {c}", key=f"cyclelabel_{upfile.name}_{c}")

        # Scan rate
        v_vs = compute_scan_rate(E_rhe, t)  # V/s
        v_mVs = v_vs * 1000 if np.isfinite(v_vs) else float("nan")
        st.caption(f"• {file_label} – {cycle_label}: estimated scan rate ≈ **{v_mVs:.1f} mV/s**")

        # OER subtraction on current
        I_corr = None
        overlap_mask = None
        if mode_oer:
            I_corr, overlap_mask = subtract_oer(E_rhe, I)

        # Choose plotting channel (I or Q)
        if mode_q:
            Y = integrate_Q(I, t)  # mA*s, same length as E
            y_label = "Q (mA·s)"
            if I_corr is not None:
                Y_corr = integrate_Q(I_corr, t)
            else:
                Y_corr = None
        else:
            Y = I
            y_label = "Current (mA)"
            Y_corr = I_corr if I_corr is not None else None

        # Plot RAW
        color = colors[color_id]
        ax.plot(E_rhe, Y, color=color, linewidth=line_width, alpha=0.95,
                label=f"{file_label} – {cycle_label} (raw)")

        # Plot corrected region only (if exists)
        if Y_corr is not None and overlap_mask is not None and np.any(overlap_mask):
            ax.plot(E_rhe[overlap_mask], Y_corr[overlap_mask],
                    linestyle="--", color=color, linewidth=line_width + 1,
                    label=f"{file_label} – {cycle_label} (corrected)")

        # Peak detection & ΔEpp (on what is plotted)
        anodic = cathodic = []
        if show_peaks:
            anodic, cathodic = detect_peaks(E_rhe, Y, distance=peak_distance, prominence=None)
            for e_pk, y_pk in anodic:
                ax.plot(e_pk, y_pk, "ro", ms=6)
                ax.text(e_pk, y_pk, f"{e_pk:.3f} V", fontsize=font_size, color="red")
            for e_pk, y_pk in cathodic:
                ax.plot(e_pk, y_pk, "bo", ms=6)
                ax.text(e_pk, y_pk, f"{e_pk:.3f} V", fontsize=font_size, color="blue")

        if show_dEpp:
            pairs = pair_peaks_for_dEpp(anodic, cathodic)
            # Display the smallest ΔEpp (closest pair) for clarity
            if pairs:
                e_an, e_ca, dE = sorted(pairs, key=lambda x: x[2])[0]
                y_top = np.nanmax(Y)
                ax.text(e_an, y_top, f"ΔEpp = {dE:.3f} V", fontsize=font_size, color="purple")

        # ECSA (Q in window divided by C_spec)
        if ecsa_on:
            Q_dl_C, ecsa_cm2 = compute_ecsa(E_rhe, I, t, ecsa_win_low, ecsa_win_high, C_spec_input)
            st.caption(f"• ECSA window [{ecsa_win_low:.2f},{ecsa_win_high:.2f}] V_RHE → Q_dl = {Q_dl_C:.3e} C,  ECSA ≈ **{ecsa_cm2:.2f} cm²**")

        # Collect for CSV export
        # Store both raw and corrected (if exists)
        df = pd.DataFrame({
            "file": file_label,
            "cycle": c,
            "E_RHE_V": E_rhe,
            "I_mA": I,
            "t_s": t,
            "Y_plotted": Y,                     # I or Q depending on mode
            "I_corr_mA": (I_corr if I_corr is not None else np.full_like(I, np.nan)),
            "Y_corr_plotted": (Y_corr if Y_corr is not None else np.full_like(Y, np.nan)),
        })
        export_rows.append(df)

        color_id = (color_id + 1) % len(colors)

# Final axes formatting
ax.set_xlabel("E vs RHE (V)", fontsize=font_size)
ax.set_ylabel(y_label, fontsize=font_size)
ax.grid(True, alpha=0.35)
ax.tick_params(labelsize=font_size)
if show_legend:
    ax.legend(fontsize=font_size)

st.pyplot(fig)

# Downloads
if export_rows:
    out_df = pd.concat(export_rows, ignore_index=True)
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button("💾 Download data CSV", csv_buf.getvalue(), file_name="cv_export_v1_1.csv", mime="text/csv")

png_buf = io.BytesIO()
fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
st.download_button("📥 Download plot PNG", png_buf.getvalue(), file_name="cv_plot_v1_1.png", mime="image/png")
