להלן קובץ אחד, מלא ומוכן להרצה: **CV Analyzer v1.2** ל-Streamlit — כולל:

* העלאת כמה קבצי ‎.MPT‎ + בחירת מחזורים לכל קובץ
* המרה ל-RHE
* החסרת OER (רק באזור חפיפה אנודית קדימה)
* סימון פיקים אוטומטי **+ עריכה ידנית** (הסרה/הוספה)
* חישוב ΔEpp בין פיק אנודי לקתודי (על רשימת הפיקים הסופית לאחר עריכה)
* חישוב מהירות סריקה לכל מחזור
* **מצב ECSA** תקף **רק** כשנבחרו בדיוק 3 מחזורים **עם 3 מהירויות סריקה שונות**; ECSA מחישוב השיפוע של Q מול מהירות סריקה
* תיקון Q-vs-E (אינטגרציה מצטברת עם אותו אורך ושרשור נכון לציר E)
* שינוי שמות לקובץ/מחזור, שליטה בעובי קו/גודל גופן/מקרא
* הורדת PNG + CSV (כולל עמודות RAW/Corrected/Q)

> הוסף ל-`requirements.txt`:
> `streamlit` `matplotlib` `numpy` `pandas` `scipy`

הרצה:

```bash
streamlit run cv_analyzer_streamlit_v1_2.py
```

```python
# cv_analyzer_streamlit_v1_2.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from scipy.signal import find_peaks

st.set_page_config(page_title="CV Analyzer v1.2", layout="wide")


# =========================== Robust MPT reader ===========================
def read_mpt_bytes(uploaded_file):
    """Robust MPT reader for Streamlit UploadedFile (BioLogic .MPT)."""
    raw = uploaded_file.getvalue()  # don't call .read() twice
    text = raw.decode("latin1", errors="ignore")
    lines = text.splitlines()

    # Find header
    header_index = None
    for i, line in enumerate(lines):
        if "Ewe/V" in line and "<I>/mA" in line:
            header_index = i
            break
    if header_index is None:
        raise ValueError("Could not find header with 'Ewe/V' and '<I>/mA'.")

    header = lines[header_index].split("\t")

    # Parse data
    rows = []
    for line in lines[header_index + 1:]:
        parts = line.split("\t")
        try:
            rows.append([float(x.replace(",", ".")) for x in parts])
        except:
            continue

    data = np.array(rows)
    if data.size == 0:
        raise ValueError("No numeric rows found after header.")

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
        raise ValueError("Missing E or I columns in file.")

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
    """Convert measured E vs selected reference to RHE scale."""
    if ref_type == "RHE":
        return E
    return E + REF_TABLE.get(ref_type, 0.0) + 0.059 * float(pH)


# =========================== OER subtraction (overlap only) ===========================
def subtract_oer(E, I):
    """
    Correct only the forward anodic overlap region:
    - Split at vertex (max E).
    - Interpolate backward anodic branch onto forward anodic E grid.
    - Subtract to estimate OER-only current.
    Returns (I_corr, overlap_mask) or (None, None).
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

    overlap_mask = np.zeros_like(I, dtype=bool)
    overlap_mask[fw_pos_idx[in_win]] = True

    return I_corr, overlap_mask


# =========================== Q integration (same length) ===========================
def integrate_Q(I, t):
    """
    Cumulative trapezoidal charge with same length as inputs.
    Units: mA*s (for I in mA, t in s).
    """
    if len(I) < 2:
        return np.zeros_like(I)
    incr = 0.5 * (I[1:] + I[:-1]) * np.diff(t)
    Q = np.concatenate([[0.0], np.cumsum(incr)])
    return Q


# =========================== Peaks & ΔEpp ===========================
def detect_peaks(E, Y, distance=10, prominence=None):
    """Return lists of (E_peak, Y_peak) for maxima (anodic) and minima (cathodic)."""
    idx_max, _ = find_peaks(Y, distance=distance, prominence=prominence)
    idx_min, _ = find_peaks(-Y, distance=distance, prominence=prominence)
    anodic   = [(E[i], Y[i]) for i in idx_max]
    cathodic = [(E[i], Y[i]) for i in idx_min]
    return anodic, cathodic


def pair_peaks_for_dEpp(anodic, cathodic):
    """
    Pair each anodic with closest cathodic by potential.
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
    """Estimate scan rate as median(|dE/dt|) [V/s]."""
    if len(E) < 3:
        return np.nan
    dE = np.gradient(E)
    dt = np.gradient(t)
    v = np.abs(dE / dt)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.nanmedian(v))


# =========================== ECSA helpers ===========================
def compute_Qdl_in_window(E_rhe, I, t, e_low, e_high):
    """
    Integrate Q within [e_low, e_high] on RHE scale.
    Returns Q_dl in Coulombs.
    """
    mask = (E_rhe >= e_low) & (E_rhe <= e_high)
    if np.sum(mask) < 2:
        return 0.0
    Q_mAs = integrate_Q(I[mask], t[mask])  # mA*s
    Q_dl_mAs = Q_mAs[-1] if Q_mAs.size else 0.0
    return Q_dl_mAs * 1e-3  # C


# =========================== Peak editor UI ===========================
def peak_table_editor(label, peaks, E, Y):
    """
    Streamlit UI: show detected peaks + allow manual removal or addition.
    peaks: list[(E_peak, Y_peak)]
    Returns final list of (E_peak, Y_peak).
    """
    with st.expander(f"✏️ Peaks editor – {label}", expanded=False):
        edited = []

        if len(peaks) == 0:
            st.info("No peaks detected. You can add peaks manually below.")

        # existing peaks with keep/remove
        for i, (e_pk, y_pk) in enumerate(peaks):
            c1, c2, c3 = st.columns([1, 1, 0.6])
            c1.write(f"**E:** {e_pk:.4f} V")
            c2.write(f"**Y:** {y_pk:.3e}")
            keep = c3.checkbox("keep", value=True, key=f"{label}_pk_{i}")
            if keep:
                edited.append((e_pk, y_pk))

        st.markdown("---")
        st.markdown("**Add peak manually** (enter potential; value on Y will be interpolated):")
        cl, cr = st.columns([1, 0.6])
        new_E = cl.number_input("E_peak (V)", value=0.0, key=f"{label}_newE")
        add_btn = cr.button("Add peak", key=f"{label}_addbtn")
        if add_btn:
            new_Y = float(np.interp(new_E, E, Y))
            edited.append((new_E, new_Y))
            st.success(f"Added peak at {new_E:.4f} V  (Y≈{new_Y:.3e})")

        # Sort by E for neatness
        edited.sort(key=lambda x: x[0])
        # Show final list
        if edited:
            df_peaks = pd.DataFrame({"E_peak_V": [e for e, _ in edited],
                                     "Y_peak":   [y for _, y in edited]})
            st.dataframe(df_peaks, use_container_width=True)
        return edited


# =========================== UI ===========================
st.title("🔬 CV Analyzer – Streamlit v1.2")
st.caption("Multi-file CV with RHE conversion, OER correction, peak editing & ΔEpp, scan-rate, and ECSA (3 cycles with 3 distinct scan rates).")

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
show_peaks = st.sidebar.checkbox("Show peaks (max/min) + editor")
show_dEpp = st.sidebar.checkbox("Show ΔEpp (peak-to-peak)")
peak_distance = st.sidebar.slider("Peak min distance (samples)", 5, 100, 15)

st.sidebar.markdown("---")
ecsa_on = st.sidebar.checkbox("ECSA mode (requires 3 cycles with 3 distinct scan rates)")
ecsa_win_low = st.sidebar.number_input("ECSA window – low (V_RHE)", value=0.10, step=0.01, format="%.2f")
ecsa_win_high = st.sidebar.number_input("ECSA window – high (V_RHE)", value=0.20, step=0.01, format="%.2f")
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

    # File label
    file_label = st.text_input(f"Rename file label ({upfile.name})", upfile.name, key=f"filelabel_{upfile.name}")

    # Prepare cycles list + scan rates per cycle in file
    cycles = sorted(set(cyc_raw))
    cycle_scanrates = {}
    for c_test in cycles:
        msk = (cyc_raw == c_test)
        v_vs = compute_scan_rate(convert_reference(E_raw[msk], ref, pH), t_raw[msk])
        cycle_scanrates[c_test] = v_vs

    default_cycles = [cycles[0]] if cycles else []
    chosen_cycles = st.multiselect(
        f"Select cycles for {upfile.name}",
        options=cycles,
        default=default_cycles,
        key=f"cycles_{upfile.name}"
    )

    # ---------- ECSA section (file-level rule) ----------
    if ecsa_on:
        if len(chosen_cycles) != 3:
            st.warning("ECSA mode requires exactly **3 selected cycles** with **3 distinct scan rates**.")
        else:
            sr_list = [cycle_scanrates[c] for c in chosen_cycles]
            sr_unique = len(set([round(v, 6) for v in sr_list if np.isfinite(v)]))
            if sr_unique != 3:
                rates_str = ", ".join(f"{v*1000:.1f} mV/s" if np.isfinite(v) else "NaN" for v in sr_list)
                st.warning(f"ECSA requires **3 distinct scan rates**. Detected: {rates_str}")
            else:
                # Compute Q vs scan rate regression
                st.markdown(f"### ECSA Analysis – {file_label}")
                Q_vals_C = []
                sr_vals_Vs = []
                for c_ecsa in chosen_cycles:
                    msk_ecsa = (cyc_raw == c_ecsa)
                    E_c = convert_reference(E_raw[msk_ecsa], ref, pH)
                    I_c = I_raw[msk_ecsa]
                    t_c = t_raw[msk_ecsa]
                    v_c = cycle_scanrates[c_ecsa]  # V/s
                    sr_vals_Vs.append(v_c)
                    Q_dl_C = compute_Qdl_in_window(E_c, I_c, t_c, ecsa_win_low, ecsa_win_high)
                    Q_vals_C.append(Q_dl_C)

                # Linear regression Q[C] = slope * v[V/s] + b  → slope = Cdl [F]
                slope, intercept = np.polyfit(sr_vals_Vs, Q_vals_C, 1)
                Cdl_F = slope
                ECSA_cm2 = Cdl_F / C_spec_input if C_spec_input > 0 else np.nan

                rates_show = ", ".join(f"{v*1000:.0f} mV/s" for v in sr_vals_Vs)
                st.success(
                    f"Scan rates: {rates_show}  \n"
                    f"Cdl = **{Cdl_F:.3e} F**,  ECSA = **{ECSA_cm2:.2f} cm²**  "
                    f"(window [{ecsa_win_low:.2f},{ecsa_win_high:.2f}] V_RHE)"
                )

    # ---------- Plot selected cycles ----------
    for c in chosen_cycles:
        mask = (cyc_raw == c)
        if not np.any(mask):
            continue

        E = E_raw[mask]
        I = I_raw[mask]
        t = t_raw[mask]
        E_rhe = convert_reference(E, ref, pH)

        cycle_label = st.text_input(f"Rename Cycle {c}", f"Cycle {c}", key=f"cyclelabel_{upfile.name}_{c}")

        # Scan rate show
        v_vs = cycle_scanrates.get(c, np.nan)
        v_mVs = v_vs * 1000 if np.isfinite(v_vs) else float("nan")
        st.caption(f"• {file_label} – {cycle_label}: estimated scan rate ≈ **{v_mVs:.1f} mV/s**")

        # OER subtraction on I
        I_corr = None
        overlap_mask = None
        if mode_oer:
            I_corr, overlap_mask = subtract_oer(E_rhe, I)

        # Choose plotting channel
        if mode_q:
            Y = integrate_Q(I, t)        # mA*s
            Y_corr = integrate_Q(I_corr, t) if I_corr is not None else None
            y_label = "Q (mA·s)"
        else:
            Y = I
            Y_corr = I_corr if I_corr is not None else None
            y_label = "Current (mA)"

        # Plot RAW
        color = colors[color_id]
        ax.plot(E_rhe, Y, color=color, linewidth=line_width, alpha=0.95,
                label=f"{file_label} – {cycle_label} (raw)")

        # Plot corrected region only
        if Y_corr is not None and overlap_mask is not None and np.any(overlap_mask):
            ax.plot(E_rhe[overlap_mask], Y_corr[overlap_mask],
                    linestyle="--", color=color, linewidth=line_width + 1,
                    label=f"{file_label} – {cycle_label} (corrected)")

        # Peaks + editor (on what is plotted)
        edited_anodic, edited_cathodic = [], []
        if show_peaks:
            autod_anodic, autod_cathodic = detect_peaks(E_rhe, Y, distance=peak_distance, prominence=None)
            edited_anodic = peak_table_editor(f"{file_label} – {cycle_label} (anodic)", autod_anodic, E_rhe, Y)
            edited_cathodic = peak_table_editor(f"{file_label} – {cycle_label} (cathodic)", autod_cathodic, E_rhe, Y)

            # draw final peaks
            for e_pk, y_pk in edited_anodic:
                ax.plot(e_pk, y_pk, "ro", ms=6)
            for e_pk, y_pk in edited_cathodic:
                ax.plot(e_pk, y_pk, "bo", ms=6)

        # ΔEpp on edited peaks
        if show_dEpp and (edited_anodic or edited_cathodic):
            pairs = pair_peaks_for_dEpp(edited_anodic, edited_cathodic)
            if pairs:
                # show smallest ΔEpp (closest in potential)
                e_an, e_ca, dE = sorted(pairs, key=lambda x: x[2])[0]
                y_top = np.nanmax(Y)
                ax.text(e_an, y_top, f"ΔEpp = {dE:.3f} V", fontsize=font_size, color="purple")

        # Collect for CSV export
        df = pd.DataFrame({
            "file": file_label,
            "cycle": c,
            "E_RHE_V": E_rhe,
            "I_mA": I,
            "t_s": t,
            "Y_plotted": Y,  # I or Q depending on mode
            "I_corr_mA": (I_corr if I_corr is not None else np.full_like(I, np.nan)),
            "Y_corr_plotted": (Y_corr if Y_corr is not None else np.full_like(Y, np.nan)),
        })
        export_rows.append(df)

        color_id = (color_id + 1) % len(colors)

# ---------- Final formatting ----------
ax.set_xlabel("E vs RHE (V)", fontsize=font_size)
ax.set_ylabel(y_label, fontsize=font_size)
ax.grid(True, alpha=0.35)
ax.tick_params(labelsize=font_size)
if show_legend:
    ax.legend(fontsize=font_size)

st.pyplot(fig)

# ---------- Downloads ----------
if export_rows:
    out_df = pd.concat(export_rows, ignore_index=True)
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button("💾 Download data CSV", csv_buf.getvalue(),
                       file_name="cv_export_v1_2.csv", mime="text/csv")

png_buf = io.BytesIO()
fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
st.download_button("📥 Download plot PNG", png_buf.getvalue(),
                   file_name="cv_plot_v1_2.png", mime="image/png")
```

רוצה שאוסיף גם אפשרות לבחור האם חישוב הפיקים/ΔEpp יתבצע על העקומה המתוקנת (ב-overlap) או על ה-RAW, וגם יצוא טבלת פיקים מסודרת (CSV נפרד)?
