import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

st.set_page_config(page_title="CV Analyzer", layout="wide")


# =================== Robust MPT reader ===================
def read_mpt_bytes(uploaded_file):
    """Robust MPT reader for Streamlit file_uploader (BioLogic MPT)."""
    
    raw = uploaded_file.getvalue()
    text = raw.decode("latin1", errors="ignore")
    lines = text.splitlines()

    # Find header
    header_index = None
    for i, line in enumerate(lines):
        if "Ewe/V" in line and "<I>/mA" in line:
            header_index = i
            break

    if header_index is None:
        raise ValueError("Could not find MPT header containing Ewe/V and <I>/mA")

    header = lines[header_index].split("\t")

    # Parse data rows
    rows = []
    for line in lines[header_index + 1:]:
        parts = line.split("\t")
        try:
            rows.append([float(x.replace(",", ".")) for x in parts])
        except:
            continue

    data = np.array(rows)

    # Find column indices
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
        raise ValueError("Missing E or I columns in MPT file")

    E = data[:, idx_E]
    I = data[:, idx_I]
    t = data[:, idx_t] if idx_t is not None else np.arange(len(E))
    cyc = data[:, idx_cyc].astype(int) if idx_cyc is not None else np.zeros(len(E), int)

    return E, I, t, cyc



# =================== Reference conversion ===================
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
    return E + REF_TABLE.get(ref_type, 0) + 0.059 * pH



# =================== OER subtraction ===================
def subtract_oer(E, I):
    """Return corrected I only in forward anodic overlap region."""
    idx_sw = np.argmax(E)
    fw_mask = np.arange(len(E)) <= idx_sw
    bw_mask = ~fw_mask

    Ef, If = E[fw_mask], I[fw_mask]
    Eb, Ib = E[bw_mask], I[bw_mask]

    pos_f = If > 0
    pos_b = Ib > 0

    if np.sum(pos_f) < 5 or np.sum(pos_b) < 5:
        return None, None

    # Forward positive region
    Ef_pos = Ef[pos_f]
    If_pos = If[pos_f]
    fw_idx_all = np.where(fw_mask)[0]
    fw_pos_idx = fw_idx_all[pos_f]

    # Backward region
    Eb_pos = Eb[pos_b]
    Ib_pos = Ib[pos_b]

    # Sort both
    of = np.argsort(Ef_pos)
    ob = np.argsort(Eb_pos)

    Ef_pos = Ef_pos[of]
    If_pos = If_pos[of]
    fw_pos_idx = fw_pos_idx[of]

    Eb_pos = Eb_pos[ob]
    Ib_pos = Ib_pos[ob]

    E_low = max(Ef_pos.min(), Eb_pos.min())
    E_high = min(Ef_pos.max(), Eb_pos.max())

    if E_high <= E_low:
        return None, None

    in_win = (Ef_pos >= E_low) & (Ef_pos <= E_high)
    I_back_interp = np.interp(Ef_pos[in_win], Eb_pos, Ib_pos)

    I_corr = I.copy()
    I_corr[fw_pos_idx[in_win]] = If_pos[in_win] - I_back_interp

    overlap_mask = np.zeros_like(I, bool)
    overlap_mask[fw_pos_idx[in_win]] = True

    return I_corr, overlap_mask



# =================== Q integration ===================
def integrate_Q(I, t):
    if len(I) < 2:
        return np.zeros_like(I)
    return np.cumsum((I[1:] + I[:-1]) * 0.5 * np.diff(t))



# =================== UI Layout ===================
st.title("🔬 CV Analyzer – Streamlit Version")
st.write("Upload BioLogic .MPT files and analyze CV cycles with correction, RHE conversion, and peak detection.")

uploaded = st.file_uploader("📁 Upload .MPT files", accept_multiple_files=True)

if not uploaded:
    st.stop()

# Sidebar controls
st.sidebar.header("⚙ Settings")

ref = st.sidebar.selectbox("Reference electrode", list(REF_TABLE.keys()))
pH = st.sidebar.number_input("pH", min_value=0.0, max_value=14.5, value=14.0)
line_width = st.sidebar.slider("Line width", 1, 5, 2)
font_size = st.sidebar.slider("Font size", 8, 20, 12)
do_q = st.sidebar.checkbox("Plot Q vs E")
do_oer = st.sidebar.checkbox("Subtract OER")
show_legend = st.sidebar.checkbox("Show legend", value=True)



# =================== PROCESS FILES ===================
fig, ax = plt.subplots(figsize=(10,6))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_id = 0

all_csv_data = []

for file in uploaded:
    st.subheader(f"📄 File: {file.name}")
    E, I, t, cyc = read_mpt_bytes(file)

    cycles = sorted(set(cyc))
    chosen_cycles = st.multiselect(
        f"Select cycles for {file.name}",
        options=cycles,
        default=[cycles[0]],
        key=file.name
    )

    rename_file = st.text_input(f"Rename file label ({file.name})", file.name)

    for c in chosen_cycles:
        mask = cyc == c

        E2 = convert_reference(E[mask], ref, pH)
        I2 = I[mask]
        t2 = t[mask]

        rename_cycle = st.text_input(f"Rename Cycle {c}", f"Cycle {c}", key=f"{file.name}_rename_{c}")

        # Q integration
        if do_q:
            I_plot = integrate_Q(I2, t2)
            y_label = "Q (mA·s)"
        else:
            I_plot = I2
            y_label = "Current (mA)"

        # OER subtraction
        if do_oer:
            I_corr, overlap_mask = subtract_oer(E2, I2)
        else:
            I_corr = None
            overlap_mask = None

        # Plot raw
        ax.plot(E2, I_plot, color=colors[color_id], label=f"{rename_file} – {rename_cycle} (raw)",
                linewidth=line_width, alpha=0.9)

        # Corrected region
        if I_corr is not None and overlap_mask is not None and np.any(overlap_mask):
            corrected_plot = I_corr if not do_q else integrate_Q(I_corr, t2)
            ax.plot(E2[overlap_mask], corrected_plot[overlap_mask],
                    linestyle="--", color=colors[color_id], linewidth=line_width+1,
                    label=f"{rename_file} – {rename_cycle} (corrected)")

            # Peak marking
            pos = corrected_plot[overlap_mask] > 0
            if np.any(pos):
                E_reg = E2[overlap_mask][pos]
                I_reg = corrected_plot[overlap_mask][pos]
                idxp = np.argmax(I_reg)

                ax.plot(E_reg[idxp], I_reg[idxp], "ro")
                ax.text(E_reg[idxp], I_reg[idxp],
                        f"Ox peak\n{E_reg[idxp]:.3f} V",
                        fontsize=font_size, color="red")

        color_id = (color_id + 1) % len(colors)

        # Save CSV
        df = pd.DataFrame({"E": E2, "I": I2, "t": t2, "cycle": c})
        all_csv_data.append(df)




# =================== Final plot formatting ===================
ax.set_xlabel("E vs RHE (V)", fontsize=font_size)
ax.set_ylabel(y_label, fontsize=font_size)
ax.grid(True)
ax.tick_params(labelsize=font_size)

if show_legend:
    ax.legend(fontsize=font_size)

st.pyplot(fig)



# =================== Download CSV ===================
if all_csv_data:
    full_df = pd.concat(all_csv_data, ignore_index=True)
    csv_buffer = io.StringIO()
    full_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="💾 Download all cycles CSV",
        data=csv_buffer.getvalue(),
        file_name="cv_cycles_export.csv",
        mime="text/csv"
    )


# =================== Download PNG ===================
png_buffer = io.BytesIO()
fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")

st.download_button(
    label="📥 Download plot as PNG",
    data=png_buffer.getvalue(),
    file_name="cv_plot.png",
    mime="image/png"
)
