# import streamlit as st
# import math
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # --- 1. CONFIGURATION ---
# st.set_page_config(page_title="PowerLine Analyst", layout="wide", page_icon="⚡")
# EPSILON_0 = 8.854e-12

# # Custom CSS to make it look unique
# st.markdown("""
# <style>
#     .main {
#         background-color: #f8f9fa;
#     }
#     h1 {
#         color: #1f77b4;
#         border-bottom: 2px solid #1f77b4;
#     }
#     .stMetric {
#         background-color: #ffffff;
#         padding: 10px;
#         border-radius: 5px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- 2. CORE MATH FUNCTIONS (Logic kept same, reorganized) ---

# def get_conductor_props(name):
#     # GMR, Resistance
#     props = {
#         "Drake": (0.01137, 0.0000232),
#         "Pheasant": (0.0142, 0.0000290),
#         "Rook": (0.00997, 0.0000203),
#         "Dove": (0.00957, 0.0000195)
#     }
#     return props.get(name, (0.01, 0.0001))

# def calc_gmd_gmr(gmr_strand, bundled, n, d_bundle):
#     # Returns (Ds, Dsc)
#     if not bundled:
#         return gmr_strand * 0.7788, gmr_strand
    
#     # Geometric factors for bundles
#     if n == 2:
#         ds = math.sqrt(gmr_strand * 0.7788 * d_bundle)
#         dsc = math.sqrt(gmr_strand * d_bundle)
#     elif n == 3:
#         ds = (gmr_strand * 0.7788 * d_bundle**2)**(1/3)
#         dsc = (gmr_strand * d_bundle**2)**(1/3)
#     elif n == 4:
#         ds = 1.09 * (gmr_strand * 0.7788 * d_bundle**3)**(1/4)
#         dsc = 1.09 * (gmr_strand * d_bundle**3)**(1/4)
#     else: # Generic approximation
#         ds = (gmr_strand * 0.7788 * d_bundle**(n-1))**(1/n)
#         dsc = (gmr_strand * d_bundle**(n-1))**(1/n)
#     return ds, dsc

# def calc_line_params(D_eq, Ds, Dsc):
#     L = 2e-7 * math.log(D_eq / Ds)
#     C = (math.pi * EPSILON_0) / math.log(D_eq / Dsc)
#     return L, C

# def solve_transmission_line(L, C, freq, I_load, length, V_line, pf, P_load, R_total):
#     # Derived Parameters
#     XL = 2 * math.pi * freq * L
#     XC_inv = 1 / (2 * math.pi * freq * C)
    
#     # Simple Drop Calculation
#     V_drop_L = I_load * XL * length
#     V_drop_C = I_load * XC_inv * length
    
#     # Pi Model (ABCD Parameters)
#     Z = R_total + 1j * XL
#     Y = 1j * 2 * np.pi * freq * C
    
#     # Receiving End Phasors
#     V_r_phase = V_line / np.sqrt(3) # Reference phasor
#     theta = -np.arccos(pf) # Lagging
#     I_r_phase = (P_load / (np.sqrt(3) * V_line * pf)) * np.exp(1j * theta)
    
#     # ABCD Constants
#     A = 1 + (Y * Z / 2)
#     B = Z + (Y * (Z**2) / 4)
    
#     # Sending End Calculation
#     V_s_phase = A * V_r_phase + B * I_r_phase
#     I_s_phase = (Y * V_r_phase / 2) + (A * I_r_phase)
    
#     # Performance Metrics
#     V_s_line_mag = np.abs(V_s_phase) * np.sqrt(3)
#     P_send = V_s_line_mag * np.abs(I_s_phase) * np.cos(np.angle(V_s_phase) - np.angle(I_s_phase))
    
#     efficiency = (P_load / P_send) * 100
#     regulation = ((np.abs(V_s_phase) - np.abs(V_r_phase)) / np.abs(V_r_phase)) * 100
    
#     return {
#         "L": L, "C": C, "XL": XL, "XC": XC_inv,
#         "Vs_LL": V_s_line_mag, "Is": np.abs(I_s_phase),
#         "Eff": efficiency, "Reg": regulation,
#         "V_drop_abs": abs(V_drop_L) + abs(V_drop_C)
#     }

# # --- 3. VISUALIZATION HELPERS ---
# def plot_geometry(phase_type, geometry, Dab, Dbc, Dca, n_sub, d_sub):
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.set_facecolor('#f0f2f6') # Light gray background
    
#     # Calculate centers
#     centers = []
#     if phase_type == "Single Phase":
#         centers = [(0,0), (Dab, 0)]
#     elif geometry == "Equilateral/Symmetrical":
#         centers = [(0,0), (Dab, 0), (Dab/2, Dab * np.sqrt(3)/2)]
#     else: # Flat spacing
#         centers = [(0,0), (Dab, 0), (Dab+Dbc, 0)]
        
#     # Draw bundles
#     for cx, cy in centers:
#         if n_sub == 1:
#              ax.add_patch(plt.Circle((cx, cy), 0.05, color='#E63946', zorder=5))
#         else:
#             # Draw subconductors
#             r_bundle = d_sub / (2 * np.sin(np.pi/n_sub)) if n_sub > 1 else 0
#             for k in range(n_sub):
#                 ang = 2*np.pi*k/n_sub
#                 sx = cx + r_bundle * np.cos(ang)
#                 sy = cy + r_bundle * np.sin(ang)
#                 ax.add_patch(plt.Circle((sx, sy), 0.05, color='#E63946', zorder=5))
#             # Draw dashed circle for bundle
#             if n_sub > 1:
#                 ax.add_patch(plt.Circle((cx, cy), r_bundle*1.2, fill=False, linestyle='--', color='gray', alpha=0.5))

#     ax.set_aspect('equal')
#     ax.grid(True, linestyle=':', alpha=0.6)
#     ax.set_title(f"Tower Geometry Configuration ({phase_type})")
#     ax.set_xlabel("Distance (m)")
#     return fig

# # --- 4. UI LAYOUT ---

# st.title("⚡ PowerLine Analyst")
# st.markdown("**Professional Transmission Line Parameter & Performance Calculator**")
# st.info("Configure your conductor and tower geometry below to analyze line performance.")

# # Top Level Inputs (The most important ones)
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     voltage_kv = st.number_input("Line Voltage (kV)", value=110.0, step=10.0)
#     V_base = voltage_kv * 1000
# with col2:
#     power_mw = st.number_input("Load Power (MW)", value=100.0, step=5.0)
#     P_base = power_mw * 1e6
# with col3:
#     pf = st.number_input("Power Factor", 0.5, 1.0, 0.95)
# with col4:
#     length_km = st.number_input("Line Length (km)", 1.0, 500.0, 100.0)
#     len_m = length_km * 1000

# # Expandable Configuration (Hides complexity)
# with st.expander("⚙️ Conductor & Geometry Settings", expanded=True):
#     c1, c2 = st.columns(2)
    
#     with c1:
#         st.subheader("Conductor Specs")
#         cond_type = st.selectbox("Type", ["Drake", "Pheasant", "Rook", "Dove"])
#         gmr_val, r_val = get_conductor_props(cond_type)
        
#         is_bundled = st.checkbox("Enable Bundling", value=True)
#         n_b = st.slider("Conductors per Bundle", 2, 4, 4) if is_bundled else 1
#         d_b = st.number_input("Bundle Spacing (m)", 0.1, 1.0, 0.4) if is_bundled else 0
        
#     with c2:
#         st.subheader("Tower Configuration")
#         sys_phase = st.radio("System", ["Three Phase", "Single Phase"], horizontal=True)
        
#         if sys_phase == "Single Phase":
#             d_ab = st.number_input("Distance D (m)", value=4.0)
#             d_bc, d_ca = 0, 0
#             Deq = d_ab
#             geo_type = "Flat"
#         else:
#             geo_type = st.selectbox("Arrangement", ["Flat/Rectangular", "Equilateral/Symmetrical"])
#             if geo_type == "Flat/Rectangular":
#                 c_a, c_b = st.columns(2)
#                 d_ab = c_a.number_input("Dist A-B (m)", value=4.0)
#                 d_bc = c_b.number_input("Dist B-C (m)", value=4.0)
#                 d_ca = d_ab + d_bc
#                 Deq = (d_ab * d_bc * d_ca)**(1/3)
#             else:
#                 d_ab = st.number_input("Phase Spacing (m)", value=4.0)
#                 d_bc, d_ca = d_ab, d_ab
#                 Deq = d_ab

# # --- 5. CALCULATION LOGIC ---
# if st.button("Analyze Transmission Line", type="primary", use_container_width=True):
    
#     # 1. Get Params
#     Ds, Dsc = calc_gmd_gmr(gmr_val, is_bundled, n_b, d_b)
#     L_prime, C_prime = calc_line_params(Deq, Ds, Dsc)
#     R_total = r_val * len_m
    
#     # 2. Current
#     I_load_mag = P_base / (math.sqrt(3) * V_base * pf)
    
#     # 3. Solve
#     res = solve_transmission_line(L_prime, C_prime, 50, I_load_mag, len_m, V_base, pf, P_base, R_total)
    
#     # --- 6. RESULTS DISPLAY (TABS) ---
#     tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📐 Geometry", "📝 Raw Data"])
    
#     with tab1:
#         # KPI Cards
#         kpi1, kpi2, kpi3, kpi4 = st.columns(4)
#         kpi1.metric("Efficiency", f"{res['Eff']:.2f}%", delta_color="normal")
#         kpi2.metric("Voltage Reg", f"{res['Reg']:.2f}%", delta_color="inverse")
#         kpi3.metric("Sending Voltage", f"{res['Vs_LL']/1000:.2f} kV")
#         kpi4.metric("Sending Current", f"{res['Is']:.1f} A")
        
#         st.markdown("### Parameter Summary")
#         res_col1, res_col2 = st.columns(2)
#         with res_col1:
#             st.success(f"**Inductance (L):** {res['L']*1000:.4f} mH/km")
#             st.info(f"**Reactance (XL):** {res['XL']:.4f} Ω/m")
#         with res_col2:
#             st.success(f"**Capacitance (C):** {res['C']*1e9:.4f} nF/km")
#             st.info(f"**Susceptance (B):** {1/res['XC']:.6f} S/m")
            
#     with tab2:
#         st.pyplot(plot_geometry(sys_phase, geo_type, d_ab, d_bc, d_ca, n_b, d_b))
#         st.caption("Red dots represent conductors. Dashed lines represent bundle effective radius.")

#     with tab3:
#         st.dataframe(pd.DataFrame({
#             "Metric": ["GMR (Ds)", "GMD (Deq)", "R total", "L total", "C total"],
#             "Value": [f"{Ds:.4f} m", f"{Deq:.4f} m", f"{R_total:.4f} Ω", f"{res['L']*len_m:.4f} H", f"{res['C']*len_m:.6f} F"]
#         }))


















import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PowerSys: Line Parameter Analyst",
    page_icon="⚡",
    layout="wide"
)

with st.sidebar:
    st.header("ℹ️ Project Info")
    st.info(
        "**EPT CEP Submission**\n\n"
        "**Authors:**\n"
        "- EE 22 150\n"
        "- EE 22 149\n"
        "- EE 22 167\n"
        "- EE 22 166"
    )
    st.markdown("---")

# --- CONSTANTS ---
PERMITTIVITY_0 = 8.854e-12  # F/m

# Database of conductors (Resistance Ohm/m, GMR m)
CONDUCTOR_DB = {
    "Drake":    {"r": 0.0000232, "gmr": 0.01137},
    "Pheasant": {"r": 0.0000290, "gmr": 0.0142},
    "Rook":     {"r": 0.0000203, "gmr": 0.00997},
    "Dove":     {"r": 0.0000195, "gmr": 0.00957}
}

# --- CLASS BASED LOGIC (OOP) ---
class TransmissionLine:
    def __init__(self, conductor_name, length_km, voltage_kv, freq):
        self.cond_data = CONDUCTOR_DB[conductor_name]
        self.length = length_km * 1000  # Convert to meters
        self.v_line = voltage_kv * 1000 # Convert to Volts
        self.freq = freq
        self.omega = 2 * np.pi * freq
        
    def calculate_bundle_GMR(self, n, d, is_capacitive=False):
        """
        Calculates Geometric Mean Radius for Inductance (Ds) or Capacitance (Dsc).
        If is_capacitive is True, we use r instead of r' (0.7788*r).
        """
        r_strand = self.cond_data['gmr']
        
        # For inductance, use GMR from table. For capacitance, use physical radius (approx GMR/0.7788)
        # However, to keep logic identical to your friend's code:
        # Your friend used: effective_ds (uses 0.7788*gmr) and effective_dsc (uses raw gmr)
        # Wait, checking reference...
        # Friend's Inductance: Uses gmr*0.7788 if unbundled.
        # Friend's Capacitance: Uses gmr (raw) if unbundled.
        
        # Let's map to the provided logic exactly:
        base_r = r_strand * 0.7788 if not is_capacitive else r_strand
        
        if n <= 1:
            return base_r
        
        # Bundle formulas based on friend's logic
        if n == 2:
            val = math.sqrt(base_r * d)
        elif n == 3:
            val = (base_r * (d**2))**(1/3)
        elif n == 4:
            val = 1.09 * (base_r * (d**3))**(1/4)
        else:
            val = (base_r * (d**(n-1)))**(1/n)
            
        return val

    def get_equivalent_spacing(self, phase_type, spacing_cfg):
        if phase_type == "Single Phase":
            return spacing_cfg['d_ab']
        
        # Three Phase Logic
        if spacing_cfg['type'] == "Symmetrical":
            return spacing_cfg['d_ab']
        else:
            # Unsymmetrical
            d1 = spacing_cfg['d_ab']
            d2 = spacing_cfg['d_bc']
            if spacing_cfg['geom'] == "Triangular":
                d3 = spacing_cfg['d_ca']
                return (d1 * d2 * d3)**(1/3)
            else: # Rectangular/Flat
                # Friend's logic: Dab * 2^(1/3) equivalent to (D * D * 2D)^(1/3)
                return d1 * (2**(1/3))

    def solve_circuit(self, L_per_m, C_per_m, P_load, pf):
        # 1. Impedance & Admittance per total length
        R_total = self.cond_data['r'] * self.length
        XL_total = self.omega * L_per_m * self.length
        XC_total_inv = 1 / (self.omega * C_per_m * self.length)
        
        Z = R_total + 1j * XL_total
        Y = 1j * self.omega * C_per_m * self.length
        
        # 2. Receiving End Phasors
        V_r_phase = self.v_line / np.sqrt(3)
        
        # I = P / (sqrt(3) * V * pf)
        I_mag = P_load / (np.sqrt(3) * self.v_line * pf)
        angle = -np.arccos(pf) # Lagging
        I_r_phase = I_mag * np.exp(1j * angle)
        
        # 3. ABCD Parameters (Nominal Pi Model)
        A = 1 + (Y * Z / 2)
        B = Z + (Y * (Z**2) / 4)
        # Note: Friend's code used Isp = Vrp*Y*(1 + YZ/4) + A*Irp 
        # This is the standard Pi model sending current formula.
        
        # 4. Sending End
        V_s_phase = A * V_r_phase + B * I_r_phase
        I_s_phase = (V_r_phase * Y * (1 + (Y*Z)/4)) + (A * I_r_phase)
        
        # 5. Performance
        V_s_ll_mag = np.abs(V_s_phase) * np.sqrt(3)
        I_s_mag = np.abs(I_s_phase)
        
        # Power Sending: sqrt(3) * V * I * cos(theta)
        phi_s = np.angle(V_s_phase) - np.angle(I_s_phase)
        P_send = np.sqrt(3) * V_s_ll_mag * I_s_mag * np.cos(phi_s)
        
        eff = (P_load / P_send) * 100
        reg = ((np.abs(V_s_phase) - np.abs(V_r_phase)) / np.abs(V_r_phase)) * 100
        
        return {
            "L": L_per_m, "C": C_per_m,
            "Vs": V_s_ll_mag, "Is": I_s_mag,
            "Eff": eff, "VR": reg,
            "XL": self.omega * L_per_m,
            "XC": 1/(self.omega * C_per_m)
        }

# --- VISUALIZATION ---
def plot_geometry(n_sub, d_sub, phase_mode, distances):
    fig, ax = plt.subplots(figsize=(10, 4))
    # Tech Style
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#adb5bd')
    
    conductors = []
    
    # Define Centers
    if phase_mode == "Single Phase":
        conductors = [(0,0, 'Ph A'), (distances['d_ab'], 0, 'Ph B')]
    else:
        # Assuming Flat for visualization simplicity based on friend's code logic
        d = distances['d_ab']
        conductors = [(0,0, 'A'), (d, 0, 'B'), (2*d, 0, 'C')]

    # Draw
    for cx, cy, label in conductors:
        # Draw Bundle Center
        ax.text(cx, cy - d_sub*2, label, ha='center', weight='bold')
        
        # Draw Subconductors
        if n_sub == 1:
            ax.scatter(cx, cy, s=100, c='firebrick', zorder=3)
        else:
            # Calculate grid positions
            cols = int(np.ceil(np.sqrt(n_sub)))
            rows = int(np.ceil(n_sub / cols))
            
            start_x = cx - ((cols-1) * d_sub)/2
            start_y = cy - ((rows-1) * d_sub)/2
            
            count = 0
            for r in range(rows):
                for c in range(cols):
                    if count < n_sub:
                        px = start_x + c*d_sub
                        py = start_y + r*d_sub
                        ax.scatter(px, py, s=50, c='navy', zorder=3)
                        count += 1
            
            # Draw bounding box
            rect = patches.Rectangle(
                (start_x - d_sub/2, start_y - d_sub/2),
                cols*d_sub, rows*d_sub,
                linewidth=1, edgecolor='gray', facecolor='none', linestyle=':'
            )
            ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.set_title("Conductor Arrangement View")
    return fig

# --- MAIN UI LAYOUT ---

st.markdown("## ⚡ PowerSys: Transmission Line Analyst")
st.markdown("---")

# 1. INPUTS (Dashboard Style)
with st.container():
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("🛠️ Physical Props")
        cond_type = st.selectbox("Conductor Model", list(CONDUCTOR_DB.keys()))
        line_len = st.number_input("Length (km)", 1.0, 1000.0, 100.0)
        
    with c2:
        st.subheader("🔌 Electrical Load")
        v_sys = st.number_input("System Voltage (kV)", 11.0, 765.0, 110.0)
        p_load = st.number_input("Load Power (MW)", 1.0, 2000.0, 100.0) * 1e6
        pf = st.number_input("Power Factor", 0.1, 1.0, 0.95)
        freq = st.radio("Freq (Hz)", [50, 60], horizontal=True)
        
    with c3:
        st.subheader("🧶 Bundling")
        is_bundled = st.toggle("Enable Bundling", value=True)
        if is_bundled:
            n_strands = st.slider("Sub-conductors", 2, 4, 4)
            d_bundle = st.number_input("Bundle Spacing (m)", 0.1, 1.0, 0.4)
        else:
            n_strands = 1
            d_bundle = 0.0

# 2. GEOMETRY CONFIG (Expandable)
with st.expander("📐 Tower Geometry Configuration", expanded=True):
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        sys_type = st.radio("Phase Configuration", ["Three Phase", "Single Phase"])
    
    geom_params = {'d_ab': 0.0, 'd_bc': 0.0, 'd_ca': 0.0, 'type': 'Sym', 'geom': 'Rect'}
    
    with col_g2:
        if sys_type == "Single Phase":
            geom_params['d_ab'] = st.number_input("Distance (m)", value=4.0)
        else:
            # 3-Phase Logic
            sym_mode = st.selectbox("Symmetry", ["Symmetrical", "Unsymmetrical"])
            geom_params['type'] = sym_mode
            
            if sym_mode == "Symmetrical":
                val = st.number_input("Equal Spacing D (m)", value=4.0)
                geom_params['d_ab'] = val
            else:
                geo_shape = st.radio("Shape", ["Rectangular", "Triangular"], horizontal=True)
                geom_params['geom'] = geo_shape
                
                if geo_shape == "Triangular":
                    gc1, gc2, gc3 = st.columns(3)
                    geom_params['d_ab'] = gc1.number_input("D_ab", value=4.0)
                    geom_params['d_bc'] = gc2.number_input("D_bc", value=4.0)
                    geom_params['d_ca'] = gc3.number_input("D_ca", value=4.0)
                else:
                    geom_params['d_ab'] = st.number_input("Phase Spacing (m)", value=4.0)

# 3. EXECUTION
if st.button("Run Simulation", type="primary", use_container_width=True):
    
    # Initialize Model
    model = TransmissionLine(cond_type, line_len, v_sys, freq)
    
    # Calc GMRs
    Ds = model.calculate_bundle_GMR(n_strands, d_bundle, is_capacitive=False)
    Dsc = model.calculate_bundle_GMR(n_strands, d_bundle, is_capacitive=True)
    
    # Calc GMD
    Deq = model.get_equivalent_spacing(sys_type, geom_params)
    
    # Calc L and C (Physics Core)
    L_val = 2e-7 * math.log(Deq / Ds)
    C_val = (math.pi * PERMITTIVITY_0) / math.log(Deq / Dsc)
    
    # Solve Circuit
    results = model.solve_circuit(L_val, C_val, p_load, pf)
    
    # --- OUTPUTS ---
    st.success("Calculation Complete")
    
    # Tabbed Interface for cleanliness
    tab_res, tab_vis, tab_raw = st.tabs(["📊 Performance", "👁️ Visuals", "📝 Parameters"])
    
    with tab_res:
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Efficiency", f"{results['Eff']:.2f}%")
        m2.metric("Voltage Regulation", f"{results['VR']:.2f}%", delta_color="inverse")
        m3.metric("Sending Voltage (L-L)", f"{results['Vs']/1000:.2f} kV")
        m4.metric("Sending Current", f"{results['Is']:.2f} A")
        
    with tab_vis:
        st.pyplot(plot_geometry(n_strands, d_bundle, sys_type, geom_params))
        
    with tab_raw:
        st.dataframe(pd.DataFrame({
            "Parameter": ["Inductance (L)", "Capacitance (C)", "Reactance (XL)", "Susceptance (B)", "Ds (GMR)", "Deq (GMD)"],
            "Value": [
                f"{results['L']:.6e} H/m", 
                f"{results['C']:.6e} F/m",
                f"{results['XL']:.4f} Ω/m",
                f"{1/results['XC']:.6f} S/m",
                f"{Ds:.4f} m",
                f"{Deq:.4f} m"
            ]
        }), use_container_width=True)