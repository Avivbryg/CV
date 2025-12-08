##############################################
#   CV Analyzer – Streamlit v1.3
#   With Savitzky–Golay smoothing upgrade
##############################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="CV Analyzer v1.3", layout="wide")


# =========================== MPT READER ===========================
def read_mpt_bytes(uploaded_file):
    raw = uploaded_file.getvalue()
    text = raw.decode("latin1", errors="ignore")
    lines = text.splitlines()

    header_index = None
    for i, line in enumerate(lines):
        if "Ewe/V" in line and "<I>/mA" in line:
            header_index = i
            break
    if header_index is None:
        raise ValueError("Header with 'Ewe/V' and '<I>/mA' not found.")

    header = lines[header_index].split("\t")

    rows = []
    for line in lines[header_index + 1:]:
        parts = line.split("\t")
        try:
            rows.append([float(x.replace(",", ".")) for x in parts])
        except:
            continue

    data = np.array(rows)
    if data.size == 0:
        raise ValueError("No numeric data detected.")

    def find(*names):
        for name in names:
            for i, h in enumerate(header):
                if name in h:
                    return i
        return None

    idx_E   = find("Ewe/V", "Ewe")
    idx_I   = find("<I>/mA", "I/mA")
    idx_t   = find("time/s", "time")
    idx_cyc = find("cycle", "Cycle", "Ns")

    if idx_E is None or idx_I is None:
        raise ValueError("Missing E or I columns.")

    E = data[:, idx_E]
    I = data[:, idx_I]
    t = data[:, idx_t] if idx_t is not None else np.arange(len(E))
    cyc = data[:, idx_cyc].astype(int) if idx_cyc is not None else np.zeros_like(E, int)

    return E, I, t, cyc


# =========================== REFERENCE CONVERSION ===========================
REF_TABLE = {
    "RHE": 0.0,
    "Ag/AgCl 3M KCl": 0.205,
    "Ag/AgCl sat. KCl": 0.197,
    "Hg/HgO 1M KOH": 0.098,
    "Hg/HgO 0.1M KOH": 0.165,
    "SCE": 0.241,
}

def convert_reference(E, ref_type, pH):
    if ref_type == "RHE":
        return E
    return E + REF_TABLE.get(ref_type, 0.0) + 0.059 * float(pH)


# =========================== SMOOTHING ===========================
def smooth_signal(Y, window=31, poly=3):
    if len(Y) < window:
        return Y
    if window % 2 == 0:
        window += 1
    try:
        return savgol_filter(Y, window_length=window, polyorder=poly)
    except:
        return Y


# =========================== OER CORRECTION ===========================
def subtract_oer(E, I):
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

    Ef_pos = Ef[pos_f]
    If_pos = If[pos_f]
    fw_idx_all = np.where(fw_mask)[0]
    fw_pos_idx = fw_idx_all[pos_f]

    Eb_pos = Eb[pos_b]
    Ib_pos = Ib[pos_b]

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

    overlap_mask = np.zeros_like(I, bool)
    overlap_mask[fw_pos_idx[in_win]] = True

    return I_corr, overlap_mask


# =========================== Q INTEGRATION ===========================
def integrate_Q(I, t):
    if len(I) < 2:
        return np.zeros_like(I)
    incr = 0.5 * (I[1:] + I[:-1]) * np.diff(t)
    Q = np.concatenate([[0.0], np.cumsum(incr)])
    return Q


# =========================== PEAKS + ΔEpp ===========================
def detect_peaks_cv(E, Y, distance=10):
    idx_max, _ = find_peaks(Y, distance=distance)
    idx_min, _ = find_peaks(-Y, distance=distance)
    anodic   = [(E[i], Y[i]) for i in idx_max]
    cathodic = [(E[i], Y[i]) for i in idx_min]
    return anodic, cathodic


def pair_peaks(anodic, cathodic):
    pairs = []
    if not anodic or not cathodic:
        return pairs
    cat_E = np.array([e for e, _ in cathodic])
    for e_an, _ in anodic:
        idx = np.argmin(abs(cat_E - e_an))
        e_ca = cathodic[idx][0]
        pairs.append((e_an, e_ca, abs(e_an - e_ca)))
    return pairs


# =========================== UI ===========================
st.title("🔬 CV Analyzer – Streamlit v1.3 (Smoothed Peaks)")
uploaded = st.file_uploader("Upload .MPT files", accept_multiple_files=True)

if not uploaded:
    st.stop()

# ---------------- SETTINGS ----------------
st.sidebar.header("⚙ Settings")
ref = st.sidebar.selectbox("Reference", list(REF_TABLE.keys()))
pH = st.sidebar.number_input("pH", 0.0, 14.5, 14.0)
line_width = st.sidebar.slider("Line Width", 1, 6, 2)
font_size = st.sidebar.slider("Font Size", 8, 24, 12)
show_legend = st.sidebar.checkbox("Show Legend", True)

st.sidebar.markdown("### OER Correction")
mode_oer = st.sidebar.checkbox("Subtract OER (overlap only)")

st.sidebar.markdown("### Q Plot")
mode_q = st.sidebar.checkbox("Plot Q instead of I")

# ---------------- SMOOTHING ----------------
st.sidebar.markdown("### Smoothing (Savitzky–Golay)")
smooth_on = st.sidebar.checkbox("Smooth before peak detection", True)
smooth_window = st.sidebar.slider("Window", 5, 101, 31, step=2)
smooth_poly = st.sidebar.slider("Polynomial Order", 1, 5, 3)

# ---------------- PEAKS ----------------
show_peaks = st.sidebar.checkbox("Show Peaks")
peak_distance = st.sidebar.slider("Peak Min Distance", 5, 100, 20)
show_dEpp = st.sidebar.checkbox("Show ΔEpp")

# ---------------- FIGURE ----------------
fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_id = 0

export_rows = []

# ================= ANALYSIS =================
for up in uploaded:
    st.subheader(f"📄 {up.name}")

    try:
        E_raw, I_raw, t_raw, cyc_raw = read_mpt_bytes(up)
    except Exception as e:
        st.error(str(e))
        continue

    file_label = st.text_input(f"Rename file ({up.name})", up.name, key=f"fl_{up.name}")

    cycles = sorted(set(cyc_raw))
    chosen_cycles = st.multiselect(
        f"Select cycles ({up.name})",
        cycles,
        default=cycles[:1],
        key=f"clist_{up.name}"
    )

    for c in chosen_cycles:
        mask = cyc_raw == c
        E = E_raw[mask]
        I = I_raw[mask]
        t = t_raw[mask]
        E_rhe = convert_reference(E, ref, pH)

        cycle_label = st.text_input(f"Rename Cycle {c}", f"Cycle {c}", key=f"clab_{up.name}_{c}")

        I_corr = None
        overlap_mask = None
        if mode_oer:
            I_corr, overlap_mask = subtract_oer(E_rhe, I)

        if mode_q:
            Y = integrate_Q(I, t)
            Y_corr = integrate_Q(I_corr, t) if I_corr is not None else None
            y_label = "Q (mA·s)"
        else:
            Y = I
            Y_corr = I_corr if I_corr is not None else None
            y_label = "I (mA)"

        # --------- SMOOTHING FOR PEAKS ---------
        if smooth_on:
            Y_for_peaks = smooth_signal(Y, window=smooth_window, poly=smooth_poly)
        else:
            Y_for_peaks = Y

        # --------- PLOT RAW ---------
        color = colors[color_id]
        ax.plot(E_rhe, Y, color=color, linewidth=line_width,
                label=f"{file_label} – {cycle_label} (raw)")

        # --------- OPTIONAL SMOOTHED PLOT ---------
        if smooth_on:
            ax.plot(E_rhe, Y_for_peaks, color=color, linewidth=1,
                    alpha=0.5, linestyle=":",
                    label=f"{file_label} – {cycle_label} (smoothed)")

        # --------- CORRECTED REGION ---------
        if Y_corr is not None and overlap_mask is not None and np.any(overlap_mask):
            ax.plot(E_rhe[overlap_mask], Y_corr[overlap_mask],
                    linestyle="--", linewidth=line_width+1,
                    color=color,
                    label=f"{file_label} – {cycle_label} (corrected)")

        # --------- PEAK DETECTION ---------
        if show_peaks:
            anodic, cathodic = detect_peaks_cv(E_rhe, Y_for_peaks, distance=peak_distance)

            for e_pk, y_pk in anodic:
                ax.plot(e_pk, y_pk, "ro", ms=6)
            for e_pk, y_pk in cathodic:
                ax.plot(e_pk, y_pk, "bo", ms=6)

            if show_dEpp:
                pairs = pair_peaks(anodic, cathodic)
                if pairs:
                    e_an, e_ca, dE = sorted(pairs, key=lambda x: x[2])[0]
                    ax.text(e_an, max(Y), f"ΔEpp={dE:.3f} V", fontsize=font_size,
                            color="purple")

        df = pd.DataFrame({
            "file": file_label,
            "cycle": c,
            "E_RHE_V": E_rhe,
            "I_mA": I,
            "t_s": t,
            "Y_plot": Y,
            "Y_smooth": Y_for_peaks,
            "I_corr_mA": I_corr if I_corr is not None else np.nan,
        })
        export_rows.append(df)

        color_id = (color_id + 1) % len(colors)


# ================= FINAL FIG =================
ax.set_xlabel("E vs RHE (V)", fontsize=font_size)
ax.set_ylabel(y_label, fontsize=font_size)
ax.grid(True)
if show_legend:
    ax.legend(fontsize=font_size)

st.pyplot(fig)

# ================= EXPORTS =================
if export_rows:
    out = pd.concat(export_rows, ignore_index=True)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button("📥 Download CSV", buf.getvalue(), "cv_export_v1_3.csv")

png_buf = io.BytesIO()
fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
st.download_button("📥 Download PNG", png_buf.getvalue(), "cv_plot_v1_3.png", mime="image/png")
