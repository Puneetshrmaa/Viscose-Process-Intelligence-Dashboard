from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Viscose Process Intelligence", layout="wide")
st.title("Viscose Staple Fibre - Process Intelligence Dashboard")
st.caption("Process map | Mass balance | Energy | Chemical recovery | Dosing | Opportunity register. "
           "Demo uses SYNTHETIC data - upload a plant CSV with the same columns to reuse it.")

# ---------------- data & assumptions ----------------
up = st.sidebar.file_uploader("Upload plant CSV (optional)", type="csv")
CSV = Path(__file__).parent / "data" / "viscose_plant_data.csv"
if up:
    df = pd.read_csv(up, parse_dates=["date"])
elif CSV.exists():
    df = pd.read_csv(CSV, parse_dates=["date"])
else:  # data file missing in the repo -> regenerate the synthetic dataset
    from generate_data import make_data
    df = make_data()

st.sidebar.header("Filters")
d0, d1 = st.sidebar.date_input("Date range", (df.date.min(), df.date.max()))
st.sidebar.header("Targets (editable)")
cs2_target = st.sidebar.number_input("CS2 dose target (% on alpha-cellulose)", 28.0, 36.0, 32.0, 0.5)
cs2_rec_target = st.sidebar.number_input("CS2 recovery target (%)", 50.0, 90.0, 68.0, 1.0)
naoh_rec_target = st.sidebar.number_input("NaOH recovery target (%)", 30.0, 80.0, 45.0, 1.0)
tol = st.sidebar.number_input("Mass-balance closure tolerance (+/- %)", 1.0, 10.0, 3.0, 0.5)
st.sidebar.header("Cost assumptions (Rs, illustrative)")
p_cs2 = st.sidebar.number_input("CS2 (Rs/t)", value=65000, step=1000)
p_naoh = st.sidebar.number_input("NaOH (Rs/t)", value=38000, step=1000)
p_acid = st.sidebar.number_input("H2SO4 (Rs/t)", value=9000, step=500)
p_steam = st.sidebar.number_input("Steam (Rs/t)", value=1800, step=100)
h_steam = st.sidebar.number_input("Steam enthalpy (GJ/t)", value=2.5, step=0.1)

f = df[(df.date >= pd.Timestamp(d0)) & (df.date <= pd.Timestamp(d1))].copy()
ann = 365 / max(len(f), 1)            # annualisation factor

# ---------------- derived metrics ----------------
IN = ["pulp_t", "naoh_fresh_t", "cs2_fresh_t", "h2so4_t", "znso4_t"]
OUT = ["fibre_t", "na2so4_byproduct_t", "organic_effluent_t", "gas_loss_t", "other_effluent_t"]
f["in_t"], f["out_t"] = f[IN].sum(axis=1), f[OUT].sum(axis=1)
f["closure_pct"] = f.out_t / f.in_t * 100
f["cs2_rec_pct"] = f.cs2_recovered_t / f.cs2_charged_t * 100
f["naoh_rec_pct"] = f.naoh_recovered_t / (f.naoh_fresh_t + f.naoh_recovered_t) * 100
f["steam_t_per_t"] = f.steam_t / f.fibre_t
f["power_kwh_per_t"] = f.power_mwh * 1000 / f.fibre_t
f["energy_gj_per_t"] = f.steam_t_per_t * h_steam + f.power_kwh_per_t * 0.0036
f["month"] = f.date.dt.to_period("M").astype(str)
bad_closure = f[(f.closure_pct < 100 - tol) | (f.closure_pct > 100 + tol)]

# energy baseline (regression on load, weather, CS2 dose); refit after dropping outliers
X = f[["fibre_t", "ambient_temp_c", "cs2_dose_pct"]]
y = f.steam_t_per_t
m = LinearRegression().fit(X, y)
r = y - m.predict(X)
keep = r.abs() < 1.5 * r.std()
m = LinearRegression().fit(X[keep], y[keep])
f["steam_baseline"] = m.predict(X)
f["steam_excess"] = f.steam_t_per_t - f.steam_baseline
sig = (f.steam_t_per_t - f.steam_baseline)[keep].std()
f["steam_flag"] = f.steam_excess > 2 * sig
excess_steam_t = (f.steam_excess.clip(lower=0) * f.fibre_t)[f.steam_flag].sum()

tabs = st.tabs(["1. Process map", "2. Mass balance", "3. Energy", "4. Chemicals & recovery",
                "5. Dosing assessment", "6. Opportunity register"])

# ---------------- 1. process map ----------------
with tabs[0]:
    st.subheader("End-to-end viscose process map")
    st.graphviz_chart('''digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#E3F0FB", fontsize=11];
      Pulp -> Steeping -> Pressing -> Shredding -> Ageing -> Xanthation -> Dissolving ->
      "Filtration &\\nRipening" -> Spinning -> "Draw & Cut" -> "Wash & Finish" -> Drying -> "Fibre (baling)";
      node [fillcolor="#FFF3D6"]; NaOH -> Steeping; CS2 -> Xanthation; "H2SO4 / ZnSO4" -> Spinning;
      node [fillcolor="#FDE2E2"]; Steam -> Spinning; Steam -> Drying; Power -> Spinning;
      node [fillcolor="#E4F6E4"];
      Spinning -> "CS2 recovery" [style=dashed]; "CS2 recovery" -> Xanthation [style=dashed];
      Pressing -> "Lye recovery" [style=dashed]; "Lye recovery" -> Steeping [style=dashed];
      Spinning -> "Na2SO4 plant" [style=dashed]; "Na2SO4 plant" -> "Byproduct" ; Spinning -> "H2S / CS2 vent" [color=red];
      "Wash & Finish" -> Effluent [color=red];
    }''')
    st.caption("Blue = process | Yellow = chemicals | Red = utilities/losses | Green = recovery loops (dashed)")

# ---------------- 2. mass balance ----------------
with tabs[1]:
    st.subheader("Plant-wide mass balance (chemical/solids basis, water excluded)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg closure", f"{f.closure_pct.mean():.1f} %")
    c2.metric("Days outside tolerance", len(bad_closure))
    c3.metric("Avg unaccounted", f"{(f.in_t - f.out_t).mean():.1f} t/day")
    tot = f[IN + OUT].sum()
    labels = ["Pulp", "NaOH", "CS2 (fresh)", "H2SO4", "ZnSO4", "VSF process",
              "Fibre", "Na2SO4", "Organic effluent", "Gas loss (CS2/H2S)", "Other effluent"]
    src = list(range(5)) + [5] * 5
    tgt = [5] * 5 + list(range(6, 11))
    val = list(tot[IN]) + list(tot[OUT])
    fig = go.Figure(go.Sankey(node=dict(label=labels, pad=15), link=dict(source=src, target=tgt, value=val)))
    fig.update_layout(title="Material flow (tonnes, selected period)", height=420)
    st.plotly_chart(fig)
    fig = px.line(f, x="date", y="closure_pct", title="Daily mass-balance closure (%)")
    fig.add_hrect(y0=100 - tol, y1=100 + tol, fillcolor="green", opacity=0.1, line_width=0)
    st.plotly_chart(fig)
    if len(bad_closure):
        st.warning("Closure outside tolerance - check flow meters / sampling for these days.")
        st.dataframe(bad_closure[["date", "in_t", "out_t", "closure_pct"]].round({"in_t": 2, "out_t": 2, "closure_pct": 2}))

# ---------------- 3. energy ----------------
with tabs[2]:
    st.subheader("Energy mapping & steam baseline")
    c1, c2, c3 = st.columns(3)
    c1.metric("Steam (t/t fibre)", f"{f.steam_t_per_t.mean():.1f}")
    c2.metric("Power (kWh/t fibre)", f"{f.power_kwh_per_t.mean():,.0f}")
    c3.metric("Total energy (GJ/t fibre)", f"{f.energy_gj_per_t.mean():.1f}")
    share = pd.DataFrame({"Source": ["Steam", "Power"],
                          "GJ/t": [f.steam_t_per_t.mean() * h_steam, f.power_kwh_per_t.mean() * 0.0036]})
    st.plotly_chart(px.pie(share, names="Source", values="GJ/t", title="Energy split"))
    fig = go.Figure()
    fig.add_scatter(x=f.date, y=f.steam_t_per_t, name="Actual", mode="lines")
    fig.add_scatter(x=f.date, y=f.steam_baseline, name="Regression baseline", mode="lines")
    fig.add_scatter(x=f[f.steam_flag].date, y=f[f.steam_flag].steam_t_per_t, name="Excess (>2 sigma)",
                    mode="markers", marker=dict(color="red", size=6))
    fig.update_layout(title="Specific steam consumption (t steam / t fibre)")
    st.plotly_chart(fig)
    st.info(f"Baseline model: steam/t = f(load, ambient temp, CS2 dose). Flagged excess steam in period: "
            f"{excess_steam_t:,.0f} t (~Rs {excess_steam_t * p_steam / 1e5:,.1f} lakh).")

# ---------------- 4. chemicals & recovery ----------------
with tabs[3]:
    st.subheader("Chemical consumption & recovery")
    spec = pd.DataFrame({
        "Chemical": ["Pulp", "NaOH (fresh)", "CS2 (fresh)", "H2SO4", "ZnSO4"],
        "kg/t fibre": [f.pulp_t.sum(), f.naoh_fresh_t.sum(), f.cs2_fresh_t.sum(), f.h2so4_t.sum(),
                       f.znso4_t.sum()]})
    spec["kg/t fibre"] = (spec["kg/t fibre"] / f.fibre_t.sum() * 1000).round(1)
    cost = {"NaOH (fresh)": p_naoh, "CS2 (fresh)": p_cs2, "H2SO4": p_acid}
    spec["Rs/t fibre"] = [round(r["kg/t fibre"] / 1000 * cost.get(r.Chemical, 0)) if r.Chemical in cost else None
                          for _, r in spec.iterrows()]
    st.dataframe(spec, hide_index=True)
    c1, c2 = st.columns(2)
    c1.metric("CS2 recovery", f"{f.cs2_rec_pct.mean():.1f} %", f"{f.cs2_rec_pct.mean() - cs2_rec_target:+.1f} vs target")
    c2.metric("NaOH recovery", f"{f.naoh_rec_pct.mean():.1f} %", f"{f.naoh_rec_pct.mean() - naoh_rec_target:+.1f} vs target")
    mon = f.groupby("month")[["cs2_rec_pct", "naoh_rec_pct"]].mean().reset_index()
    fig = px.line(mon, x="month", y=["cs2_rec_pct", "naoh_rec_pct"], markers=True, title="Monthly recovery (%)")
    st.plotly_chart(fig)

# ---------------- 5. dosing ----------------
with tabs[4]:
    st.subheader("Chemical dosing assessment")
    f["overdose"] = f.cs2_dose_pct > cs2_target + 1.5
    f["extra_cs2_fresh_t"] = ((f.cs2_dose_pct - cs2_target) / 100 * f.alpha_cellulose_t
                              * (1 - f.cs2_rec_pct / 100)).where(f.overdose, 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Over-dosed days (> target+1.5)", int(f.overdose.sum()))
    c2.metric("Extra fresh CS2", f"{f.extra_cs2_fresh_t.sum():.1f} t")
    c3.metric("Cost of over-dosing", f"Rs {f.extra_cs2_fresh_t.sum() * p_cs2 / 1e5:.1f} lakh")
    fig = px.scatter(f, x="cs2_dose_pct", y="tenacity_cn_tex", color="overdose",
                     title="CS2 dose vs fibre tenacity - quality plateaus, extra CS2 adds cost only")
    fig.add_vline(x=cs2_target, line_dash="dash")
    st.plotly_chart(fig)
    param = st.selectbox("Control chart parameter", ["cs2_dose_pct", "naoh_steep_pct", "h2so4_conc_gpl", "znso4_conc_gpl"])
    mu, sd = f[param].mean(), f[param].std()
    fig = px.line(f, x="date", y=param, title=f"{param}: mean +/- 3 sigma")
    for v in (mu, mu + 3 * sd, mu - 3 * sd):
        fig.add_hline(y=v, line_dash="dot")
    st.plotly_chart(fig)

# ---------------- 6. opportunity register ----------------
with tabs[5]:
    st.subheader("Opportunity register (auto-generated from the analysis)")
    cs2_gap = ((cs2_rec_target - f.cs2_rec_pct).clip(lower=0) / 100 * f.cs2_charged_t).sum()
    naoh_gap = ((naoh_rec_target - f.naoh_rec_pct).clip(lower=0) / 100 * (f.naoh_fresh_t + f.naoh_recovered_t)).sum()
    acid_x = (f.h2so4_t - f.naoh_fresh_t * 98 / 80 * 1.05).clip(lower=0).sum()
    rows = [
        ("Chemicals", "CS2 over-dosing", f"{int(f.overdose.sum())} days above target+1.5%",
         f.extra_cs2_fresh_t.sum() * p_cs2, "Dose controller / SOP limit on CS2 per alpha-cellulose", "Soft-sensor + SPC alerts"),
        ("Recovery", "CS2 recovery below target", f"{cs2_gap:.1f} t CS2 not recovered vs target",
         cs2_gap * p_cs2, "Condenser/adsorber performance check, leak survey", "Recovery-efficiency trend alarm"),
        ("Recovery", "NaOH recovery below target", f"{naoh_gap:.1f} t NaOH shortfall vs target",
         naoh_gap * p_naoh, "Dialysis / lye recovery tuning", "Monthly recovery dashboard"),
        ("Chemicals", "Spin-bath acid excess", f"{acid_x:.1f} t H2SO4 above 5% over stoichiometric",
         acid_x * p_acid, "Tighten spin-bath acid set-point", "Ratio control (acid : NaOH)"),
        ("Energy", "Steam above regression baseline", f"{excess_steam_t:,.0f} t excess steam",
         excess_steam_t * p_steam, "Steam-trap & insulation audit", "Baseline model + anomaly alert"),
        ("Data quality", "Mass-balance not closing", f"{len(bad_closure)} days outside +/-{tol:.0f}%",
         np.nan, "Flow-meter calibration, sampling audit", "Auto data-reconciliation"),
    ]
    reg = pd.DataFrame(rows, columns=["Area", "Loss point", "Evidence", "Value (Rs)", "Action", "Digital enabler"])
    reg["Est. annual value (Rs lakh)"] = (reg["Value (Rs)"] * ann / 1e5).round(1)
    reg["Priority"] = pd.cut(reg["Est. annual value (Rs lakh)"].fillna(15), [-1, 10, 30, 1e9],
                             labels=["Low", "Medium", "High"])
    reg = reg.drop(columns="Value (Rs)").sort_values("Est. annual value (Rs lakh)", ascending=False)
    st.dataframe(reg, hide_index=True)
    st.download_button("Download register (CSV)", reg.to_csv(index=False), "opportunity_register.csv")
    st.caption("Values use sidebar cost assumptions and are annualised from the selected period.")
