import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

# =================== MPT reader (use your original code) ===================
def read_mpt_bytes(file):
    text = file.read().decode("latin1")
    lines = text.splitlines()

    header_index = None
    for i, line in enumerate(lines):
        if "Ewe/V" in line and "<I>/mA" in line:
            header_index = i
            break

    header = lines[header_index].split("\t")
    rows = []

    for line in lines[header_index+1:]:
        parts = line.split("\t")
        try:
            rows.append([float(x.replace(",", ".")) for x in parts])
        except:
            continue

    data = np.array(rows)

    def find(*names):
        for name in names:
            for i, h in enumerate(header):
                if name in h:
                    return i
        return None

    idx_E   = find("Ewe/V","Ewe")
    idx_I   = find("<I>/mA","I/mA")
    idx_t   = find("time/s","time")
    idx_cyc = find("cycle","Cycle","Ns")

    E = data[:, idx_E]
    I = data[:, idx_I]
    t = data[:, idx_t] if idx_t is not None else np.arange(len(E))
    cyc = data[:, idx_cyc].astype(int) if idx_cyc is not None else np.zeros(len(E), int)

    return E, I, t, cyc


# =================== Streamlit UI ===================
st.title("🔬 CV Analyzer – Streamlit Edition")

uploaded = st.file_uploader("Upload .MPT files", accept_multiple_files=True)

if uploaded:
    st.sidebar.header("Settings")
    ref = st.sidebar.selectbox("Reference electrode", 
        ["RHE","Ag/AgCl 3M KCl","Hg/HgO 1M KOH","SCE"])
    pH = st.sidebar.number_input("pH", value=14.0)
    lw = st.sidebar.slider("Line width",1,5,2)

    for file in uploaded:
        st.subheader(f"File: {file.name}")

        # Read data
        E, I, t, cyc = read_mpt_bytes(file)
        cycles = sorted(set(cyc))

        chosen_cycles = st.multiselect(
            f"Select cycles for {file.name}",
            options=cycles,
            default=[cycles[0]],
            key=file.name
        )

        # Plot for each selected cycle
        fig, ax = plt.subplots()
        for c in chosen_cycles:
            mask = (cyc == c)
            ax.plot(E[mask], I[mask], label=f"Cycle {c}", linewidth=lw)

        ax.set_xlabel("E (V)")
        ax.set_ylabel("Current (mA)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
