import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

# =================== MPT reader (use your original code) ===================
def read_mpt_bytes(uploaded_file):
    """Robust MPT reader for Streamlit file_uploader."""
    
    # Read binary content
    raw = uploaded_file.read()

    # Try decoding safely
    try:
        text = raw.decode("latin1", errors="ignore")
    except:
        raise ValueError("Cannot decode MPT file as latin1")

    lines = text.splitlines()

    # Find header line (robust search)
    header_index = None
    for i, line in enumerate(lines):
        if "Ewe/V" in line and "<I>/mA" in line:
            header_index = i
            break

    if header_index is None:
        raise ValueError("Could not find MPT header containing Ewe/V and <I>/mA")

    header = lines[header_index].split("\t")

    # Parse rows
    rows = []
    for line in lines[header_index + 1:]:
        parts = line.split("\t")
        try:
            row = [float(x.replace(",", ".")) for x in parts]
            rows.append(row)
        except:
            continue

    data = np.array(rows)

    # Helper: find column index
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
        raise ValueError("Cannot find E or I columns in MPT file")

    E = data[:, idx_E]
    I = data[:, idx_I]
    t = data[:, idx_t] if idx_t is not None else np.arange(len(E))
    cyc = data[:, idx_cyc].astype(int) if idx_cyc is not None else np.zeros(len(E), int)

    return E, I, t, cyc


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

