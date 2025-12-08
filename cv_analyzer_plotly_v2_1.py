# cv_analyzer_plotly_v2_1.py
# Streamlit + Plotly CV Analyzer (fixed build order, single interactive plot)
# - Tap-to-add/remove peaks
# - No smoothing
# - Auto-peak on RAW and/or OER-corrected (overlap only)
# - Explicit "Editing Target" selector (File / Cycle / RAW|CORR)
# - Clicks are applied AFTER the full figure is rendered

import streamlit as st
import numpy as np
import pandas as pd
import io
import plotly.graph_objects as go
from scipy.signal import find_peaks
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="CV Analyzer v2.1 (Plotly Interactive)", layout="wide")

# ====================== Robust MPT reader ======================
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
        raise ValueError("Could not find header with 'Ewe/V' and '<I>/mA'.")

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
        raise ValueError("Missing E or I columns.")

    E = data[:, idx_E]
    I = data[:, idx_I]
    t = data[:, idx_t] if idx_t is not None else np.arange(len(E))
    cyc = data[:, idx_cyc].astype(int) if idx_cyc is not None else np.zeros_like(E, int)

    return E, I, t, cyc

# ====================== Reference conversion ======================
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

# ====================== OER subtraction (overlap only) ======================
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

    overlap_mask = np.zeros_like(I, dtype=bool)
    overlap_mask[fw_pos_idx[in_win]] = True

    return I_corr, overlap_mask

# ====================== Q integration ======================
def integrate_Q(I, t):
    if len(I) < 2:
        return np.zeros_like(I)
    incr = 0.5 * (I[1:] + I[:-1]) * np.diff(t)
    Q = np.concatenate([[0.0], np.cumsum(incr)])
    return Q

# ====================== Peak detection helpers ======================
def detect_peaks(E, Y, distance=10, prominence=None):
    idx_max, _ = find_peaks(Y, distance=distance, prominence=prominence)
    idx_min, _ = find_peaks(-Y, distance=distance, prominence=prominence)
    anodic   = [(E[i], Y[i]) for i in idx_max]
    cathodic = [(E[i], Y[i]) for i in idx_min]
    return anodic, cathodic

def pair_peaks_for_dEpp(anodic, cathodic):
    pairs = []
    if not anodic or not cathodic:
        return pairs
    cat_E = np.array([e for e, _ in cathodic])
    for e_an, _ in anodic:
        idx = np.argmin(np.abs(cat_E - e_an))
        e_ca = cathodic[idx][0]
        pairs.append((e_an, e_ca, abs(e_an - e_ca)))
    return pairs

def nearest_peak_index(peaks, x_click, y_click, e_tol=0.02, y_tol_ratio=0.15):
    """Return index of nearest peak to (x_click,y_click) if within tolerances, else None."""
    if not peaks:
        return None
    E_arr = np.array([e for e, _ in peaks])
    Y_arr = np.array([y for _, y in peaks])
    y_range = max(1e-9, (np.nanmax(Y_arr) - np.nanmin(Y_arr)))
    y_tol = y_range * y_tol_ratio
    d = np.abs(E_arr - x_click) + np.abs(Y_arr - y_click) / max(1e-9, y_tol)
    idx = int(np.argmin(d))
    if np.abs(E_arr[idx] - x_click) <= e_tol:
        return idx
    return None

# ====================== Session helpers ======================
def pk_key(file_label, cycle_label, source):
    return f"peaks::{file_label}::{cycle_label}::{source}"

def get_peaks_state(file_label, cycle_label, source):
    k = pk_key(file_label, cycle_label, source)
    if k not in st.session_state:
        st.session_state[k] = {"anodic": [], "cathodic": []}
    return st.session_state[k]

def set_peaks_state(file_label, cycle_label, source, anodic, cathodic):
    st.session_state[pk_key(file_label, cycle_label, source)] = {"anodic": anodic, "cathodic": cathodic}

# ====================== UI ======================
st.title("🔬 CV Analyzer – Plotly v2.1 (Tap to add/remove peaks)")
st.caption("הגרף נבנה במלואו ואז מוצג בלחיצה אחת. בחר יעד עריכה (File/Cycle/RAW|CORR) כדי שהקליק יוחל על העקומה הנכונה.")

uploaded = st.file_uploader("📁 העלה קבצי BioLogic ‎.MPT", accept_multiple_files=True)
if not uploaded:
    st.stop()

st.sidebar.header("⚙ Global Settings")
ref = st.sidebar.selectbox("Reference electrode", list(REF_TABLE.keys()))
pH = st.sidebar.number_input("pH", min_value=0.0, max_value=14.5, value=14.0)
line_width = st.sidebar.slider("Line width", 1, 6, 2)
show_legend = st.sidebar.checkbox("Show legend", True)

st.sidebar.markdown("---")
plot_channel = st.sidebar.radio("Plot channel", ["Current (I vs E)", "Charge (Q vs E)"], index=0)
mode_oer = st.sidebar.checkbox("Subtract OER (overlap only)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Peaks")
auto_on_raw = st.sidebar.checkbox("Auto-find on RAW", True)
auto_on_corr = st.sidebar.checkbox("Auto-find on OER-corrected", True)
peak_distance = st.sidebar.slider("Auto peak min distance (samples)", 5, 120, 20)
show_dEpp = st.sidebar.checkbox("Show ΔEpp", True)
delete_e_tolerance = st.sidebar.slider("Delete: E tolerance (V)", 0.002, 0.100, 0.02)

# נבנה את כל ה־figure פעם אחת:
fig = go.Figure()
csv_rows = []
editing_targets = []  # נאסוף כאן אפשרויות 'יעד עריכה' (file|cycle|source)
raw_color = "#1f77b4"
corr_color = "#ff7f0e"

# === שלב 1: קלט קבצים + בחירת מחזורים ===
file_cycle_data = []  # נשמור מידע כדי להשתמש בו אחרי הצגת הגרף

for up in uploaded:
    st.subheader(f"📄 {up.name}")

    try:
        E_all, I_all, t_all, cyc_all = read_mpt_bytes(up)
    except Exception as e:
        st.error(f"{up.name}: {e}")
        continue

    file_label = st.text_input(f"Rename file label ({up.name})", up.name, key=f"filelabel_{up.name}")
    cycles = sorted(set(cyc_all))
    chosen_cycles = st.multiselect(
        f"Select cycles for {up.name}",
        options=cycles,
        default=cycles[:1],
        key=f"cycles_{up.name}"
    )

    for c in chosen_cycles:
        mask = (cyc_all == c)
        if not np.any(mask):
            continue

        E = E_all[mask]
        I = I_all[mask]
        t = t_all[mask]
        E_rhe = convert_reference(E, ref, pH)
        cycle_label = st.text_input(f"Rename Cycle {c}", f"Cycle {c}", key=f"cyclelabel_{up.name}_{c}")

        # OER correction
        I_corr, overlap_mask = (None, None)
        if mode_oer:
            I_corr, overlap_mask = subtract_oer(E_rhe, I)

        # Channel selection
        if plot_channel.startswith("Charge"):
            Y_raw = integrate_Q(I, t)
            Y_corr = integrate_Q(I_corr, t) if I_corr is not None else None
            yaxis_title = "Q (mA·s)"
        else:
            Y_raw = I
            Y_corr = I_corr if I_corr is not None else None
            yaxis_title = "I (mA)"

        # --- Auto peaks init (RAW) ---
        raw_state = get_peaks_state(file_label, cycle_label, "RAW")
        if auto_on_raw and not raw_state["anodic"] and not raw_state["cathodic"]:
            an_raw, ca_raw = detect_peaks(E_rhe, Y_raw, distance=peak_distance)
            set_peaks_state(file_label, cycle_label, "RAW", an_raw, ca_raw)
            raw_state = get_peaks_state(file_label, cycle_label, "RAW")

        # --- Auto peaks init (CORR) ---
        corr_state = get_peaks_state(file_label, cycle_label, "CORR")
        if (auto_on_corr and (Y_corr is not None)
                and not corr_state["anodic"] and not corr_state["cathodic"]):
            an_c, ca_c = detect_peaks(E_rhe, Y_corr, distance=peak_distance)
            # שמירה רק בתחום ה־overlap אם קיים:
            if overlap_mask is not None and np.any(overlap_mask):
                e_min, e_max = E_rhe[overlap_mask].min(), E_rhe[overlap_mask].max()
                an_c = [(e, y) for (e, y) in an_c if e_min <= e <= e_max]
                ca_c = [(e, y) for (e, y) in ca_c if e_min <= e <= e_max]
            set_peaks_state(file_label, cycle_label, "CORR", an_c, ca_c)
            corr_state = get_peaks_state(file_label, cycle_label, "CORR")

        # --- Add traces to figure (RAW curve) ---
        fig.add_trace(go.Scatter(
            x=E_rhe, y=Y_raw, mode="lines",
            line=dict(color=raw_color, width=line_width),
            name=f"{file_label} – {cycle_label} (RAW)"
        ))

        # --- Add corrected region ---
        if Y_corr is not None and overlap_mask is not None and np.any(overlap_mask):
            fig.add_trace(go.Scatter(
                x=E_rhe[overlap_mask], y=Y_corr[overlap_mask], mode="lines",
                line=dict(color=corr_color, width=line_width, dash="dash"),
                name=f"{file_label} – {cycle_label} (CORR overlap)"
            ))

        # --- Add peak markers (RAW) ---
        if raw_state["anodic"]:
            fig.add_trace(go.Scatter(
                x=[e for e, _ in raw_state["anodic"]],
                y=[y for _, y in raw_state["anodic"]],
                mode="markers", marker=dict(color="red", size=9, symbol="circle"),
                name=f"{file_label} – {cycle_label} RAW anodic"
            ))
        if raw_state["cathodic"]:
            fig.add_trace(go.Scatter(
                x=[e for e, _ in raw_state["cathodic"]],
                y=[y for _, y in raw_state["cathodic"]],
                mode="markers", marker=dict(color="blue", size=9, symbol="triangle-down"),
                name=f"{file_label} – {cycle_label} RAW cathodic"
            ))

        # --- Add peak markers (CORR) ---
        if Y_corr is not None:
            if corr_state["anodic"]:
                fig.add_trace(go.Scatter(
                    x=[e for e, _ in corr_state["anodic"]],
                    y=[y for _, y in corr_state["anodic"]],
                    mode="markers", marker=dict(color="red", size=10, symbol="x"),
                    name=f"{file_label} – {cycle_label} CORR anodic"
                ))
            if corr_state["cathodic"]:
                fig.add_trace(go.Scatter(
                    x=[e for e, _ in corr_state["cathodic"]],
                    y=[y for _, y in corr_state["cathodic"]],
                    mode="markers", marker=dict(color="blue", size=10, symbol="x-thin"),
                    name=f"{file_label} – {cycle_label} CORR cathodic"
                ))

        # --- ΔEpp annotation (prefer CORR, fallback RAW) ---
        if show_dEpp:
            if Y_corr is not None and (corr_state["anodic"] or corr_state["cathodic"]):
                pairs = pair_peaks_for_dEpp(corr_state["anodic"], corr_state["cathodic"])
                yref = np.nanmax(Y_corr) if np.any(np.isfinite(Y_corr)) else np.nanmax(Y_raw)
            else:
                pairs = pair_peaks_for_dEpp(raw_state["anodic"], raw_state["cathodic"])
                yref = np.nanmax(Y_raw)
            if pairs:
                e_an, e_ca, dE = sorted(pairs, key=lambda x: x[2])[0]
                fig.add_annotation(x=e_an, y=float(yref), text=f"ΔEpp = {dE:.3f} V",
                                   showarrow=False, font=dict(color="purple", size=12))

        # נשמור נתונים כדי לעבוד עם הקליק אח"כ:
        file_cycle_data.append({
            "file_label": file_label,
            "cycle_label": cycle_label,
            "E": E_rhe,
            "Y_raw": Y_raw,
            "Y_corr": Y_corr,
            "has_corr_overlap": (overlap_mask is not None and np.any(overlap_mask)),
            "overlap_range": (float(E_rhe[overlap_mask].min()), float(E_rhe[overlap_mask].max())) if (overlap_mask is not None and np.any(overlap_mask)) else None
        })

        # לאיסוף להורדה
        df = pd.DataFrame({
            "file": file_label,
            "cycle": c,
            "E_RHE_V": E_rhe,
            "I_or_Q_raw": Y_raw,
            "I_or_Q_corr": (Y_corr if Y_corr is not None else np.full_like(Y_raw, np.nan)),
        })
        csv_rows.append(df)

# === שלב 2: הגדרות figure כלליות והצגה האינטראקטיבית (פעם אחת בלבד!) ===
fig.update_layout(
    xaxis_title="E vs RHE (V)",
    yaxis_title=("Q (mA·s)" if plot_channel.startswith("Charge") else "I (mA)"),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="closest"
)

st.markdown("### Interactive Plot")
click = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    key="cv_main_plot"
)

# === שלב 3: בוחר יעד עריכה + החלת הקליק על היעד שנבחר ===
st.markdown("#### Editing Target")
# בנה רשימת אפשרויות: "File – Cycle – RAW" / "File – Cycle – CORR"
options = []
for row in file_cycle_data:
    options.append(f"{row['file_label']} – {row['cycle_label']} – RAW")
    if row["Y_corr"] is not None:
        options.append(f"{row['file_label']} – {row['cycle_label']} – CORR")

if not options:
    st.stop()

default_opt = options[0]
edit_choice = st.selectbox("בחר יעד עריכה (הקליקים יחולו על זה):", options, index=0)

# מצבי הוספה/מחיקה:
edit_mode = st.radio("Click mode", ["Add anodic (red)", "Add cathodic (blue)", "Delete nearest"], index=0, horizontal=True)
reset_btn = st.button("Reset peaks for selected target")

# פירוק הבחירה:
def parse_choice(choice_str):
    # "file – cycle – RAW/CORR"
    parts = choice_str.rsplit(" – ", 2)
    if len(parts) != 3:
        return None, None, None
    return parts[0], parts[1], parts[2]  # file_label, cycle_label, source_str

sel_file, sel_cycle, sel_source = parse_choice(edit_choice)
if sel_file is None:
    st.stop()

# איפוס פיקים ליעד שנבחר
if reset_btn:
    set_peaks_state(sel_file, sel_cycle, sel_source, [], [])
    st.experimental_rerun()

# החלת הקליק (אם היה)
if click:
    pt = click[0]
    x_click = float(pt.get("x"))
    y_click = float(pt.get("y"))

    state = get_peaks_state(sel_file, sel_cycle, sel_source)
    anodic_peaks = list(state["anodic"])
    cathodic_peaks = list(state["cathodic"])

    if edit_mode.startswith("Add anodic"):
        anodic_peaks.append((x_click, y_click))
        set_peaks_state(sel_file, sel_cycle, sel_source, anodic_peaks, cathodic_peaks)
        st.experimental_rerun()
    elif edit_mode.startswith("Add cathodic"):
        cathodic_peaks.append((x_click, y_click))
        set_peaks_state(sel_file, sel_cycle, sel_source, anodic_peaks, cathodic_peaks)
        st.experimental_rerun()
    else:
        idx_a = nearest_peak_index(anodic_peaks, x_click, y_click, e_tol=delete_e_tolerance)
        idx_c = nearest_peak_index(cathodic_peaks, x_click, y_click, e_tol=delete_e_tolerance)
        # העדפה למחיקה לפי הקרבה ב-E
        if idx_a is not None and idx_c is not None:
            if abs(anodic_peaks[idx_a][0] - x_click) <= abs(cathodic_peaks[idx_c][0] - x_click):
                anodic_peaks.pop(idx_a)
            else:
                cathodic_peaks.pop(idx_c)
        elif idx_a is not None:
            anodic_peaks.pop(idx_a)
        elif idx_c is not None:
            cathodic_peaks.pop(idx_c)
        set_peaks_state(sel_file, sel_cycle, sel_source, anodic_peaks, cathodic_peaks)
        st.experimental_rerun()

# === הורדות ===
if csv_rows:
    out_df = pd.concat(csv_rows, ignore_index=True)
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button("💾 Download data CSV", csv_buf.getvalue(), file_name="cv_export_v2_1.csv", mime="text/csv")
