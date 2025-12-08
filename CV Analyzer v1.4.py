# =============================== CV Analyzer v1.3 ===============================
# NEW in v1.3:
# - Peak detection threshold:
#     • prominence
#     • height
#     • min distance
# - Cleaner peak detection (no mini peaks)
# ===============================================================================

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import find_peaks
import os


# =================== Robust MPT reader ===================
def read_mpt(path):
    with open(path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    header_index = None
    for i, line in enumerate(lines):
        if 'Ewe/V' in line and '<I>/mA' in line:
            header_index = i
            break

    if header_index is None:
        raise ValueError("Could not find Ewe/V and <I>/mA columns")

    header = lines[header_index].strip().split('\t')

    rows = []
    for line in lines[header_index+1:]:
        parts = line.strip().split('\t')
        try:
            row = [float(x.replace(',', '.')) for x in parts]
            rows.append(row)
        except:
            continue

    data = np.array(rows)

    def find(*names):
        for name in names:
            for i, h in enumerate(header):
                if name in h:
                    return i
        return None

    idx_E   = find('Ewe/V', 'Ewe')
    idx_I   = find('<I>/mA', 'I/mA')
    idx_t   = find('time/s', 'time')
    idx_cyc = find('cycle number', 'cycle', 'Cycle', 'Ns')

    E = data[:, idx_E]
    I = data[:, idx_I]
    t = data[:, idx_t] if idx_t is not None else np.arange(len(E))
    cyc = data[:, idx_cyc].astype(int) if idx_cyc is not None else np.zeros_like(E, int)

    return E, I, t, cyc


# =================== Peak detection ===================
def find_clean_peaks(E, I, min_prom=0.01, min_height=None, min_dist=20):
    """
    Returns clean anodic and cathodic peaks, using prominence threshold (most important).
    """
    # anodic peaks (maxima)
    idx_max, _ = find_peaks(
        I,
        prominence=min_prom,
        height=min_height,
        distance=min_dist
    )

    # cathodic peaks (minima)
    idx_min, _ = find_peaks(
        -I,
        prominence=min_prom,
        height=min_height,
        distance=min_dist
    )

    anodic = [(E[i], I[i]) for i in idx_max]
    cathodic = [(E[i], I[i]) for i in idx_min]

    return anodic, cathodic


# =================== Reference conversion ===================
REFS = {
    'RHE': 0.0,
    'Ag/AgCl 3M KCl': 0.205,
    'Ag/AgCl sat. KCl': 0.197,
    'Hg/HgO 1M KOH': 0.098,
    'Hg/HgO 0.1M KOH': 0.165,
    'SCE': 0.241
}

def convert_reference(E, ref_type, pH):
    if ref_type == 'RHE':
        return E
    return E + REFS.get(ref_type, 0) + 0.059 * pH


# =================== GUI ===================
class CVApp:
    def __init__(self, root):
        self.root = root
        root.title("CV Analyzer v1.3")

        self.files = []
        self.cache = {}
        self.checked = {}
        self.display_names = {}
        self.file_colors = {}

        top = tk.Frame(root)
        top.pack(fill='x')

        tk.Button(top, text="Add MPT", command=self.add_file).pack(side='left')
        tk.Button(top, text="Clear All", command=self.clear_all).pack(side='left')

        # Reference
        tk.Label(top, text="Reference:").pack(side='left')
        self.ref_var = tk.StringVar(value='Hg/HgO 1M KOH')
        ttk.Combobox(
            top, textvariable=self.ref_var, width=18,
            values=list(REFS.keys())
        ).pack(side='left')

        # pH
        tk.Label(top, text="pH:").pack(side='left')
        self.pH_var = tk.StringVar(value='14')
        tk.Entry(top, textvariable=self.pH_var, width=4).pack(side='left')

        # Peak thresholds
        tk.Label(top, text="Prominence:").pack(side='left')
        self.prom_var = tk.StringVar(value='0.01')
        tk.Entry(top, textvariable=self.prom_var, width=6).pack(side='left')

        tk.Label(top, text="Height:").pack(side='left')
        self.height_var = tk.StringVar(value='')
        tk.Entry(top, textvariable=self.height_var, width=6).pack(side='left')

        tk.Label(top, text="MinDist:").pack(side='left')
        self.dist_var = tk.StringVar(value='20')
        tk.Entry(top, textvariable=self.dist_var, width=6).pack(side='left')

        tk.Button(top, text="Plot", command=self.plot).pack(side='left')

        # Tree
        browser = tk.Frame(root)
        browser.pack(fill='both', expand=False)
        self.tree = ttk.Treeview(browser, columns=('chk', 'type'), selectmode='none')
        self.tree.heading('#0', text="File / Cycle")
        self.tree.heading('chk', text="✔")
        self.tree.heading('type', text="Type")
        self.tree.column('chk', width=40, anchor='center')
        self.tree.column('type', width=60, anchor='center')
        self.tree.pack(fill='both', expand=True)

        self.tree.bind("<Button-1>", self.on_toggle)
        self.tree.bind("<Double-1>", self.rename_item)

        # Plot
        fig = Figure(figsize=(8, 5))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(self.canvas, root)

    # ---------------- Clear ----------------
    def clear_all(self):
        self.tree.delete(*self.tree.get_children())
        self.files.clear()
        self.cache.clear()
        self.checked.clear()
        self.display_names.clear()
        self.ax.clear()
        self.canvas.draw()

    # ---------------- Add File ----------------
    def add_file(self):
        paths = filedialog.askopenfilenames(filetypes=[("MPT files", "*.mpt")])
        if not paths:
            return

        for p in paths:
            try:
                E, I, t, cyc = read_mpt(p)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                continue

            self.cache[p] = {"E": E, "I": I, "t": t, "cyc": cyc}
            file_id = self.tree.insert("", "end", text=os.path.basename(p),
                                       values=("", "file"))
            self.checked[file_id] = False

            for c in sorted(set(cyc)):
                node = self.tree.insert(file_id, "end",
                                        text=f"Cycle {c}",
                                        values=("", "cycle"))
                self.checked[node] = False

    # ---------------- Toggle checkboxes ----------------
    def on_toggle(self, event):
        col = self.tree.identify_column(event.x)
        if col != "#1":
            return
        item = self.tree.identify_row(event.y)
        if not item:
            return

        new = not self.checked[item]
        self.checked[item] = new
        vals = list(self.tree.item(item, "values"))
        vals[0] = "✔" if new else ""
        self.tree.item(item, values=vals)

        if vals[1] == "file":
            for ch in self.tree.get_children(item):
                self.checked[ch] = new
                cvals = list(self.tree.item(ch, "values"))
                cvals[0] = "✔" if new else ""
                self.tree.item(ch, values=cvals)

    # ---------------- Rename ----------------
    def rename_item(self, event):
        item = self.tree.focus()
        old = self.tree.item(item, 'text')
        new = simpledialog.askstring("Rename", f"Rename '{old}' to:")
        if new:
            self.tree.item(item, text=new)
            self.display_names[item] = new

    # ---------------- Plot ----------------
    def plot(self):
        self.ax.clear()

        try:
            pH = float(self.pH_var.get())
        except:
            pH = 14.0

        try:
            prom = float(self.prom_var.get())
        except:
            prom = 0.01

        try:
            dist = int(self.dist_var.get())
        except:
            dist = 20

        try:
            height = float(self.height_var.get()) if self.height_var.get() else None
        except:
            height = None

        for item, checked in self.checked.items():
            if not checked:
                continue
            if self.tree.item(item, 'values')[1] != 'cycle':
                continue

            parent = self.tree.parent(item)
            p = None
            for f in self.cache.keys():
                if os.path.basename(f) == self.tree.item(parent, "text"):
                    p = f
                    break
            if p is None:
                continue

            cycnum = int(self.tree.item(item, "text").split()[-1])
            E = self.cache[p]["E"]
            I = self.cache[p]["I"]
            cyc = self.cache[p]["cyc"]

            mask = cyc == cycnum
            if not np.any(mask):
                continue

            E_raw = convert_reference(E[mask], self.ref_var.get(), pH)
            I_raw = I[mask]

            # ================= PEAK detection =================
            anodic, cathodic = find_clean_peaks(
                E_raw, I_raw,
                min_prom=prom,
                min_height=height,
                min_dist=dist
            )

            # Plot raw curve
            self.ax.plot(E_raw, I_raw, linewidth=1.5)

            # Mark peaks
            for e, y in anodic:
                self.ax.plot(e, y, 'ro', markersize=7)
            for e, y in cathodic:
                self.ax.plot(e, y, 'bo', markersize=7)

        self.ax.set_xlabel("E (V vs RHE)")
        self.ax.set_ylabel("I (mA)")
        self.ax.grid(True)
        self.canvas.draw()


# =================== RUN ===================
if __name__ == '__main__':
    root = tk.Tk()
    app = CVApp(root)
    root.mainloop()
