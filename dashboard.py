"""
Mental Health Scheduling Dashboard
===================================
Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import date, timedelta, datetime
from io import StringIO
import sys

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Capacity Management Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Mental Health Scheduling Optimization")
st.caption("Predictive capacity management · 14-day rolling horizon")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    n_providers  = st.slider("Number of providers",   1, 6,  3)
    n_case1      = st.slider("Case 1 initials",        1, 10, 5)
    n_case2      = st.slider("Case 2 ad-hoc sessions", 1, 20, 10)
    horizon      = st.slider("Horizon (days)",          7, 28, 14)
    util_target  = st.slider("Utilisation target %",   50, 95, 80) / 100
    seed         = st.number_input("Random seed", value=42, step=1)
    run_btn      = st.button("▶ Run Optimizer", type="primary", use_container_width=True)

st.divider()

# ── Run optimizer (import and re-run with patched params) ─────────────────────
@st.cache_data(show_spinner="Running MILP optimizer…")
def run_optimizer(n_prov, n_c1, n_c2, horizon_days, util_target, seed):
    """
    Runs a simplified but faithful version of the MILP scheduler and returns
    structured DataFrames ready for visualisation.
    """
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix, csr_matrix

    np.random.seed(seed)
    SLOTS_PER_DAY  = 7
    MAX_UTIL_SLOTS = int(SLOTS_PER_DAY * util_target)
    SHADOW_MAX_N   = 7
    FOLLOWUP_GAP   = 14
    SLA_ACTIVE_START = 6
    SLA_ACTIVE_END   = 22
    START_DATE     = date(2025, 6, 10)
    PROVIDERS      = [f"P{i+1}" for i in range(n_prov)]
    DATES          = [START_DATE + timedelta(days=d) for d in range(horizon_days)]
    TOTAL_SLOTS    = n_prov * horizon_days * SLOTS_PER_DAY

    # ── SLA clock helper ──────────────────────────────────────────────────────
    def make_request_ts(day_offset):
        hour = np.random.choice([7, 9, 14, 18, 23], p=[0.2, 0.3, 0.3, 0.1, 0.1])
        minute = np.random.randint(0, 59)
        d = START_DATE + timedelta(days=day_offset)
        return datetime(d.year, d.month, d.day, hour, minute)

    def compute_sla(request_ts):
        rh = request_ts.hour
        if SLA_ACTIVE_START <= rh < SLA_ACTIVE_END:
            sla_start = request_ts
        else:
            nd = request_ts.date() + timedelta(days=1)
            sla_start = datetime(nd.year, nd.month, nd.day, SLA_ACTIVE_START)
        return sla_start, sla_start + timedelta(hours=48)

    # ── Generate demand ───────────────────────────────────────────────────────
    c1_data, c2_data = [], []
    for i in range(n_c1):
        rts = make_request_ts(np.random.randint(0, max(1, horizon_days // 3)))
        ss, sd = compute_sla(rts)
        c1_data.append({"id": f"C1-{i+1:03d}", "request_ts": rts,
                         "sla_start": ss, "sla_deadline": sd,
                         "n_followups": int(np.random.randint(3, 8))})
    for j in range(n_c2):
        rts = make_request_ts(np.random.randint(0, max(1, horizon_days // 2)))
        ss, sd = compute_sla(rts)
        c2_data.append({"id": f"C2-{j+1:03d}", "request_ts": rts,
                         "sla_start": ss, "sla_deadline": sd,
                         "max_sessions": int(np.random.randint(4, 7))})

    def slot_id(p, d, s): return p * horizon_days * SLOTS_PER_DAY + d * SLOTS_PER_DAY + s
    def date_to_idx(d):
        delta = (d - START_DATE).days
        return delta if 0 <= delta < horizon_days else None

    # ── Variable layout ───────────────────────────────────────────────────────
    X1_START = 0;            X1_SIZE = n_c1 * TOTAL_SLOTS
    X2_START = X1_SIZE;      X2_SIZE = n_c2 * TOTAL_SLOTS
    Z_START  = X2_START + X2_SIZE; Z_SIZE = n_c1 * TOTAL_SLOTS
    V_START  = Z_START + Z_SIZE;   V_SIZE = n_c1
    NVARS    = V_START + V_SIZE

    def x1_idx(c, p, d, s): return X1_START + c * TOTAL_SLOTS + slot_id(p, d, s)
    def x2_idx(j, p, d, s): return X2_START + j * TOTAL_SLOTS + slot_id(p, d, s)
    def z_idx(c, p, d, s):  return Z_START  + c * TOTAL_SLOTS + slot_id(p, d, s)
    def v_idx(c):            return V_START  + c

    # ── Objective ─────────────────────────────────────────────────────────────
    c_obj = np.zeros(NVARS)
    for c in range(n_c1):
        c_obj[v_idx(c)] = 1000
    for c in range(n_c1):
        for p in range(n_prov):
            for d in range(horizon_days):
                for s in range(SLOTS_PER_DAY):
                    c_obj[z_idx(c, p, d, s)] = 0.01

    # ── Constraints ───────────────────────────────────────────────────────────
    rows_indices, rows_data, rows_lb, rows_ub = [], [], [], []
    INF = np.inf

    def add_row(idx_list, coef_list, lb, ub):
        rows_indices.append(idx_list); rows_data.append(coef_list)
        rows_lb.append(lb); rows_ub.append(ub)

    for c in range(n_c1):
        idx = [x1_idx(c, p, d, s) for p in range(n_prov)
               for d in range(horizon_days) for s in range(SLOTS_PER_DAY)]
        add_row(idx, [1]*len(idx), 1, 1)

    for c, case in enumerate(c1_data):
        latest_d = date_to_idx(case["sla_deadline"].date())
        if latest_d is None: latest_d = horizon_days - 1
        sla_idx = [x1_idx(c, p, d, s) for p in range(n_prov)
                   for d in range(latest_d + 1) for s in range(SLOTS_PER_DAY)]
        add_row(sla_idx + [v_idx(c)], [1]*len(sla_idx) + [1], 1, INF)

    for j in range(n_c2):
        idx = [x2_idx(j, p, d, s) for p in range(n_prov)
               for d in range(horizon_days) for s in range(SLOTS_PER_DAY)]
        add_row(idx, [1]*len(idx), 1, 1)

    for p in range(n_prov):
        for d in range(horizon_days):
            for s in range(SLOTS_PER_DAY):
                idx, coef = [], []
                for c in range(n_c1):
                    idx += [x1_idx(c,p,d,s), z_idx(c,p,d,s)]; coef += [1, 1]
                for j in range(n_c2):
                    idx.append(x2_idx(j,p,d,s)); coef.append(1)
                add_row(idx, coef, 0, 1)

    for p in range(n_prov):
        for d in range(horizon_days):
            idx, coef = [], []
            for c in range(n_c1):
                for s in range(SLOTS_PER_DAY):
                    idx += [x1_idx(c,p,d,s), z_idx(c,p,d,s)]; coef += [1, 1]
            for j in range(n_c2):
                for s in range(SLOTS_PER_DAY):
                    idx.append(x2_idx(j,p,d,s)); coef.append(1)
            add_row(idx, coef, 0, MAX_UTIL_SLOTS)

    for c in range(n_c1):
        for p in range(n_prov):
            for d in range(horizon_days):
                for s in range(SLOTS_PER_DAY):
                    for k in range(1, SHADOW_MAX_N + 1):
                        fd = d + k * FOLLOWUP_GAP
                        if fd >= horizon_days: continue
                        shadow_slots = [z_idx(c, p, fd, s2) for s2 in range(SLOTS_PER_DAY)]
                        add_row(shadow_slots + [x1_idx(c,p,d,s)],
                                [1]*SLOTS_PER_DAY + [-1], 0, INF)

    for c in range(n_c1):
        for p in range(n_prov):
            assigned_on_p = [x1_idx(c,p,d0,s0)
                             for d0 in range(horizon_days) for s0 in range(SLOTS_PER_DAY)]
            for d in range(horizon_days):
                for s in range(SLOTS_PER_DAY):
                    add_row([z_idx(c,p,d,s)] + assigned_on_p,
                            [1] + [-1]*len(assigned_on_p), -INF, 0)

    nrows = len(rows_indices)
    A = lil_matrix((nrows, NVARS), dtype=np.float64)
    for r, (idxs, coefs) in enumerate(zip(rows_indices, rows_data)):
        for idx, coef in zip(idxs, coefs):
            A[r, idx] = coef
    A_csr = csr_matrix(A)

    result = milp(c=c_obj,
                  constraints=LinearConstraint(A_csr, np.array(rows_lb), np.array(rows_ub)),
                  integrality=np.ones(NVARS),
                  bounds=Bounds(np.zeros(NVARS), np.ones(NVARS)),
                  options={"disp": False, "time_limit": 60})

    if result.x is None:
        return None, None, None, None, "Solver failed — try reducing parameters."

    x = result.x

    # ── Decode into DataFrames ────────────────────────────────────────────────
    grid_rows = []
    shadow_rows = []
    util_rows = []

    for p in range(n_prov):
        for d in range(horizon_days):
            booked, shadow, free = 0, 0, 0
            for s in range(SLOTS_PER_DAY):
                c1_here  = any(x[x1_idx(c,p,d,s)] > 0.5 for c in range(n_c1))
                c2_here  = any(x[x2_idx(j,p,d,s)] > 0.5 for j in range(n_c2))
                shad_here= any(x[z_idx(c,p,d,s)]  > 0.5 for c in range(n_c1))

                if c1_here:
                    label = next(c1_data[c]["id"] for c in range(n_c1) if x[x1_idx(c,p,d,s)] > 0.5)
                    grid_rows.append({"Provider": PROVIDERS[p], "Date": DATES[d],
                                      "Slot": s+1, "Type": "Case 1 Initial",
                                      "Session": label, "Color": 1})
                    booked += 1
                elif c2_here:
                    label = next(c2_data[j]["id"] for j in range(n_c2) if x[x2_idx(j,p,d,s)] > 0.5)
                    grid_rows.append({"Provider": PROVIDERS[p], "Date": DATES[d],
                                      "Slot": s+1, "Type": "Case 2 Ad-hoc",
                                      "Session": label, "Color": 2})
                    booked += 1
                elif shad_here:
                    label = next(f"{c1_data[c]['id']} shadow" for c in range(n_c1) if x[z_idx(c,p,d,s)] > 0.5)
                    grid_rows.append({"Provider": PROVIDERS[p], "Date": DATES[d],
                                      "Slot": s+1, "Type": "Shadow Reserved",
                                      "Session": label, "Color": 3})
                    shadow += 1
                else:
                    grid_rows.append({"Provider": PROVIDERS[p], "Date": DATES[d],
                                      "Slot": s+1, "Type": "Free",
                                      "Session": "", "Color": 0})
                    free += 1

            occupied = booked + shadow
            util_rows.append({
                "Provider": PROVIDERS[p],
                "Date": DATES[d],
                "Booked": booked,
                "Shadow": shadow,
                "Free": free,
                "Utilisation": occupied / SLOTS_PER_DAY * 100,
            })

    sla_rows = []
    for c, case in enumerate(c1_data):
        violated = x[v_idx(c)] > 0.5
        assigned_date = None
        for p in range(n_prov):
            for d in range(horizon_days):
                for s in range(SLOTS_PER_DAY):
                    if x[x1_idx(c,p,d,s)] > 0.5:
                        assigned_date = DATES[d]
        sla_rows.append({
            "Case": case["id"],
            "Request": case["request_ts"].strftime("%m/%d %H:%M"),
            "SLA Deadline": case["sla_deadline"].strftime("%m/%d %H:%M"),
            "Assigned Date": str(assigned_date) if assigned_date else "—",
            "Follow-ups": case["n_followups"],
            "SLA Status": "✗ Violated" if violated else "✓ Met",
        })

    return (pd.DataFrame(grid_rows),
            pd.DataFrame(util_rows),
            pd.DataFrame(sla_rows),
            PROVIDERS,
            f"Optimal · cost={result.fun:.1f}")


# ── Auto-run on first load, re-run on button press ───────────────────────────
grid_df, util_df, sla_df, providers, status_msg = run_optimizer(
    n_providers, n_case1, n_case2, horizon, util_target, int(seed)
)

if grid_df is None:
    st.error(status_msg)
    st.stop()

# ── Status bar ────────────────────────────────────────────────────────────────
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
violations = sla_df["SLA Status"].str.contains("Violated").sum() if sla_df is not None else 0
col_s1.metric("Solver", status_msg.split("·")[0].strip(), delta=None)
col_s2.metric("Case 1 SLA violations", violations,
              delta="0 = perfect" if violations == 0 else f"{violations} breach(es)",
              delta_color="normal" if violations == 0 else "inverse")
avg_util = util_df["Utilisation"].mean()
col_s3.metric("Avg utilisation", f"{avg_util:.1f}%",
              delta=f"target {int(util_target*100)}%",
              delta_color="normal" if abs(avg_util - util_target*100) < 20 else "inverse")
col_s4.metric("Sessions scheduled", len(grid_df[grid_df["Type"] != "Free"]))

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCHEDULE GRID
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📅 Schedule Grid", "📊 Utilisation Charts", "📋 SLA Report"])

COLOR_MAP = {
    "Case 1 Initial":  "#5B8DEF",
    "Case 2 Ad-hoc":   "#F5A623",
    "Shadow Reserved": "#9B59B6",
    "Free":            "#E8EDF2",
}

with tab1:
    st.subheader("Provider schedule — 14-day horizon")
    st.caption("Click any slot for details. Shadow blocks protect Case 1 follow-up continuity.")

    provider_filter = st.multiselect("Show providers", providers, default=providers)
    filtered = grid_df[grid_df["Provider"].isin(provider_filter)]

    for prov in provider_filter:
        st.markdown(f"**{prov}**")
        prov_df = filtered[filtered["Provider"] == prov].copy()
        dates = sorted(prov_df["Date"].unique())

        # Build a pivot: rows = slots (1-7), cols = dates
        pivot = prov_df.pivot_table(index="Slot", columns="Date",
                                    values="Type", aggfunc="first")
        pivot_label = prov_df.pivot_table(index="Slot", columns="Date",
                                          values="Session", aggfunc="first")

        fig = go.Figure()

        for col_date in dates:
            for slot in range(1, 8):
                try:
                    stype = pivot.loc[slot, col_date]
                    slabel = pivot_label.loc[slot, col_date] if stype != "Free" else ""
                except KeyError:
                    stype = "Free"; slabel = ""

                color = COLOR_MAP.get(stype, "#E8EDF2")
                fig.add_trace(go.Bar(
                    name=stype,
                    x=[col_date],
                    y=[1],
                    base=slot - 1,
                    marker_color=color,
                    text=slabel[:10] if slabel else "",
                    textposition="inside",
                    hovertemplate=f"<b>{stype}</b><br>{slabel}<br>Slot {slot}<extra></extra>",
                    showlegend=(col_date == dates[0]),
                ))

        fig.update_layout(
            barmode="overlay",
            height=280,
            margin=dict(l=40, r=10, t=10, b=40),
            yaxis=dict(title="Slot", tickvals=list(range(7)),
                       ticktext=[f"S{i+1}" for i in range(7)]),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Legend
    leg_cols = st.columns(4)
    for col, (ltype, lcolor) in zip(leg_cols, COLOR_MAP.items()):
        col.markdown(
            f'<span style="background:{lcolor};padding:3px 10px;'
            f'border-radius:4px;font-size:12px;">&nbsp;</span> {ltype}',
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UTILISATION CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Provider utilisation over time")

    # Stacked bar: booked + shadow per provider per day
    fig2 = go.Figure()
    for prov in providers:
        pdata = util_df[util_df["Provider"] == prov]
        fig2.add_trace(go.Bar(
            name=f"{prov} – Booked",
            x=pdata["Date"], y=pdata["Booked"],
            marker_color="#5B8DEF", legendgroup=prov,
        ))
        fig2.add_trace(go.Bar(
            name=f"{prov} – Shadow",
            x=pdata["Date"], y=pdata["Shadow"],
            marker_color="#9B59B6", legendgroup=prov,
        ))

    # 80% cap line
    fig2.add_hline(y=int(7 * util_target), line_dash="dash",
                   line_color="red", annotation_text=f"{int(util_target*100)}% cap")

    fig2.update_layout(
        barmode="stack", height=380,
        yaxis=dict(title="Slots used", range=[0, 8]),
        xaxis=dict(title="Date"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: utilisation % by provider × day
    st.subheader("Utilisation heatmap")
    heatmap_data = util_df.pivot_table(index="Provider", columns="Date",
                                        values="Utilisation")
    fig3 = px.imshow(
        heatmap_data,
        color_continuous_scale=["#E8EDF2", "#5B8DEF", "#E74C3C"],
        zmin=0, zmax=100,
        labels=dict(color="Util %"),
        text_auto=".0f",
    )
    fig3.update_layout(height=200 + len(providers) * 40,
                       plot_bgcolor="rgba(0,0,0,0)",
                       paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    # Per-provider average utilisation gauge row
    st.subheader("Average utilisation per provider")
    gcols = st.columns(len(providers))
    for gcol, prov in zip(gcols, providers):
        avg = util_df[util_df["Provider"] == prov]["Utilisation"].mean()
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg,
            number={"suffix": "%", "font": {"size": 22}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#5B8DEF"},
                "steps": [
                    {"range": [0, 60],  "color": "#E8F4F8"},
                    {"range": [60, 80], "color": "#D4EFDF"},
                    {"range": [80, 100],"color": "#FADBD8"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": util_target * 100,
                },
            },
            title={"text": prov, "font": {"size": 14}},
        ))
        fig_g.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)")
        gcol.plotly_chart(fig_g, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SLA REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Case 1 SLA compliance report")

    def color_sla(val):
        if "Violated" in str(val):
            return "background-color: #FADBD8; color: #922B21;"
        elif "Met" in str(val):
            return "background-color: #D4EFDF; color: #1E8449;"
        return ""

    st.dataframe(
        sla_df.style.applymap(color_sla, subset=["SLA Status"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("SLA clock logic")
    st.info(
        "🕐 **SLA active window: 06:00 – 22:00 daily.**  \n"
        "Requests arriving between 22:01–05:59 have their 48-hour clock "
        "frozen until 06:00 the following morning. The table above reflects "
        "this — compare 'Request' timestamps after 22:00 with their later "
        "'SLA Deadline'."
    )

    st.subheader("Shadow capacity explanation")
    st.markdown(
        "When a **Case 1 Initial** is booked, the optimizer immediately "
        "reserves **shadow slots** at *t + 14, t + 28, … t + 14·N_max* "
        "on the **same provider**. This prevents a future day from being "
        "double-booked by ad-hoc demand.  \n\n"
        "Shadow slots are shown in **purple** on the Schedule Grid tab. "
        "Once a follow-up is confirmed, its shadow converts to a real booking."
    )
