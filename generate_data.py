"""Generate a SYNTHETIC daily dataset for a viscose staple fibre (VSF) plant.
Replace data/viscose_plant_data.csv with real plant/historian data (same columns) to reuse the app.
Ratios are illustrative, based on typical VSF process ranges - not from any real plant."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 365
d = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n)})
i = np.arange(n)

d["ambient_temp_c"] = (25 + 8*np.sin((i-100)/365*2*np.pi) + rng.normal(0, 1.5, n)).round(1)
fibre = np.clip(rng.normal(95, 5, n), 70, 110)
fibre[rng.choice(n, 12, replace=False)] *= 0.7            # shutdown / low-load days
d["fibre_t"] = fibre.round(2)

# ---- dosing / process parameters ----
cs2_dose = rng.normal(32.0, 1.0, n)                        # % CS2 on alpha-cellulose
cs2_dose[200:240] += 3.0                                   # over-dosing episode
d["cs2_dose_pct"] = cs2_dose.round(2)
d["naoh_steep_pct"] = rng.normal(18, 0.4, n).round(2)
d["h2so4_conc_gpl"] = rng.normal(115, 4, n).round(1)       # spin-bath acid
d["znso4_conc_gpl"] = rng.normal(12, 0.6, n).round(2)

# ---- raw material & chemical flows (t/day) ----
pulp = fibre * rng.normal(1.04, 0.015, n)
alpha = pulp * 0.92
d["pulp_t"] = pulp.round(2)
d["alpha_cellulose_t"] = alpha.round(2)
naoh_fresh = fibre * rng.normal(0.55, 0.02, n)
d["naoh_fresh_t"] = naoh_fresh.round(2)
d["naoh_recovered_t"] = (fibre * rng.normal(0.50, 0.03, n)).round(2)
cs2_charged = alpha * cs2_dose / 100
cs2_rec = cs2_charged * np.clip(rng.normal(0.68, 0.03, n), 0.5, 0.85)
cs2_rec[120:150] *= 0.9                                   # recovery (condenser) degradation
d["cs2_charged_t"] = cs2_charged.round(2)
d["cs2_recovered_t"] = cs2_rec.round(2)
cs2_fresh = cs2_charged - cs2_rec
d["cs2_fresh_t"] = cs2_fresh.round(2)
excess = np.clip(0.06 + (d["h2so4_conc_gpl"]-115)/115*0.5, 0.01, None)
h2so4 = naoh_fresh * 98/80 * (1+excess)
d["h2so4_t"] = h2so4.round(2)
znso4 = fibre * rng.normal(0.025, 0.002, n)
d["znso4_t"] = znso4.round(3)

# ---- outputs (t/day) ----
na2so4 = naoh_fresh * 142/80 * rng.normal(0.97, 0.01, n)
organic = pulp * 0.11 * rng.normal(1, 0.05, n)
gas = cs2_fresh * rng.normal(0.55, 0.05, n)
inputs = pulp + naoh_fresh + cs2_fresh + h2so4 + znso4
resid = inputs - fibre - na2so4 - organic - gas
factor = rng.normal(0.90, 0.04, n)
factor[60:80] = 0.40                                      # flow-meter drift episode
d["na2so4_byproduct_t"] = na2so4.round(2)
d["organic_effluent_t"] = organic.round(2)
d["gas_loss_t"] = gas.round(2)
d["other_effluent_t"] = (resid * factor).round(2)

# ---- utilities ----
spec_steam = (17 + 0.12*(30-d["ambient_temp_c"]) + 0.06*(cs2_dose-32.0)
              + (95-fibre)*0.08 + rng.normal(0, 0.5, n))
spec_steam[280:310] += 2.5                                # steam-trap / insulation loss episode
d["steam_t"] = (fibre * spec_steam).round(1)
d["power_mwh"] = (fibre * rng.normal(1.35, 0.06, n)).round(1)
d["water_m3"] = (fibre * rng.normal(95, 6, n)).round(0)

# ---- quality: tenacity saturates ~32% CS2 dose (extra CS2 gives no benefit) ----
d["tenacity_cn_tex"] = (22 + 2.2*(1-np.exp(-np.clip(cs2_dose-27, 0, None)/2.5))
                        + rng.normal(0, 0.3, n)).round(2)

d.to_csv("data/viscose_plant_data.csv", index=False)
print(d.shape, "saved")
