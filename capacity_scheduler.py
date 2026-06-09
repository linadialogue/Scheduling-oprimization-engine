"""
Mental Health Scheduling Optimization Engine — Proof of Concept
================================================================
Assigns 5 Case 1 Initials + 10 Case 2 sessions to 3 full-time providers
over a 14-day horizon using scipy.optimize.milp (MILP formulation).

Key constraints enforced:
  - 80% utilization cap  → max 5 slots/day per provider (floor(7 × 0.80) = 5)
  - Case 1 continuity    → same provider for all sessions in a case
  - Shadow blocking      → t+14 slots reserved at booking time (max N=7 follow-ups)
  - 48-hour SLA window   → active 06:00–22:00; clock freezes outside those hours
  - Case 2 pooling       → any provider, no downstream blocking
  - No double-booking    → one session per provider per slot
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from datetime import date, timedelta, datetime
from dataclasses import dataclass, field
from typing import Optional
import json

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
HORIZON_DAYS   = 14
SLOTS_PER_DAY  = 7
MAX_UTIL_SLOTS = 5          # floor(7 × 0.80)
SHADOW_MAX_N   = 7          # worst-case follow-up count reserved upfront
FOLLOWUP_GAP   = 14         # days between Case 1 sessions
SLA_HOURS      = 48         # SLA window in active hours
SLA_ACTIVE_START = 6        # 06:00
SLA_ACTIVE_END   = 22       # 22:00
NUM_PROVIDERS  = 3
START_DATE     = date(2025, 6, 10)

PROVIDERS = [f"P{i+1}" for i in range(NUM_PROVIDERS)]
DATES     = [START_DATE + timedelta(days=d) for d in range(HORIZON_DAYS)]

# Weight ordering for the objective (α ≫ β ≫ γ)
W_SLA_VIOLATION = 1000
W_UTIL_DEVIATION = 1


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Case1:
    id: str
    request_ts: datetime
    n_followups_expected: int    # drawn from [3..7]
    sla_start: datetime = field(init=False)
    sla_deadline: datetime = field(init=False)

    def __post_init__(self):
        # SLA clock logic: freeze if request outside 06:00–22:00
        rh = self.request_ts.hour
        if SLA_ACTIVE_START <= rh < SLA_ACTIVE_END:
            self.sla_start = self.request_ts
        else:
            next_day = self.request_ts.date() + timedelta(days=1)
            self.sla_start = datetime(next_day.year, next_day.month, next_day.day,
                                      SLA_ACTIVE_START)
        self.sla_deadline = self.sla_start + timedelta(hours=SLA_HOURS)

    @property
    def latest_initial_date(self) -> date:
        return self.sla_deadline.date()


@dataclass
class Case2:
    id: str
    request_ts: datetime
    max_sessions: int = 5       # drawn from [4..6]
    sla_start: datetime = field(init=False)
    sla_deadline: datetime = field(init=False)

    def __post_init__(self):
        rh = self.request_ts.hour
        if SLA_ACTIVE_START <= rh < SLA_ACTIVE_END:
            self.sla_start = self.request_ts
        else:
            next_day = self.request_ts.date() + timedelta(days=1)
            self.sla_start = datetime(next_day.year, next_day.month, next_day.day,
                                      SLA_ACTIVE_START)
        self.sla_deadline = self.sla_start + timedelta(hours=SLA_HOURS)

    @property
    def latest_first_date(self) -> date:
        return self.sla_deadline.date()


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE SYNTHETIC DEMAND
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)

def make_request_ts(day_offset: int) -> datetime:
    """Random timestamp on day_offset, sometimes after 22:00 to test SLA clock."""
    hour = np.random.choice([7, 9, 14, 18, 23], p=[0.2, 0.3, 0.3, 0.1, 0.1])
    minute = np.random.randint(0, 59)
    d = START_DATE + timedelta(days=day_offset)
    return datetime(d.year, d.month, d.day, hour, minute)

case1_requests = [
    Case1(id=f"C1-{i+1:03d}",
          request_ts=make_request_ts(np.random.randint(0, 5)),
          n_followups_expected=int(np.random.randint(3, 8)))
    for i in range(5)
]

case2_requests = [
    Case2(id=f"C2-{i+1:03d}",
          request_ts=make_request_ts(np.random.randint(0, 7)),
          max_sessions=int(np.random.randint(4, 7)))
    for i in range(10)
]


# ─────────────────────────────────────────────────────────────────────────────
# SLOT INDEX HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def slot_id(p_idx: int, d_idx: int, s_idx: int) -> int:
    """Flat index into (provider × day × slot) space."""
    return p_idx * HORIZON_DAYS * SLOTS_PER_DAY + d_idx * SLOTS_PER_DAY + s_idx

TOTAL_SLOTS = NUM_PROVIDERS * HORIZON_DAYS * SLOTS_PER_DAY   # 3×14×7 = 294

def date_to_idx(d: date) -> Optional[int]:
    delta = (d - START_DATE).days
    return delta if 0 <= delta < HORIZON_DAYS else None


# ─────────────────────────────────────────────────────────────────────────────
# MILP VARIABLE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
# Variables (all binary):
#   [0 .. TOTAL_SLOTS-1]                 : x[p,d,s]  — Case 1 initials (one block per case)
# Wait — we build one variable block per decision type:
#
#   x1[c,p,d,s] : Case 1 case c assigned to provider p at (d,s)  → 5 × 3 × 14 × 7
#   x2[j,p,d,s] : Case 2 case j assigned to provider p at (d,s)  → 10 × 3 × 14 × 7
#   z[c,p,d,s]  : Shadow slot for Case 1 case c at provider p at (d,s) → 5 × 3 × 14 × 7
#   v[c]        : SLA violation for Case 1 case c                 → 5
#
# Total vars: 5×294 + 10×294 + 5×294 + 5 = 1470 + 2940 + 1470 + 5 = 5885

N_C1 = len(case1_requests)
N_C2 = len(case2_requests)

X1_START = 0
X1_SIZE  = N_C1 * TOTAL_SLOTS          # 5 × 294 = 1470
X2_START = X1_SIZE
X2_SIZE  = N_C2 * TOTAL_SLOTS          # 10 × 294 = 2940
Z_START  = X2_START + X2_SIZE
Z_SIZE   = N_C1 * TOTAL_SLOTS          # 5 × 294 = 1470
V_START  = Z_START + Z_SIZE
V_SIZE   = N_C1                        # 5
NVARS    = V_START + V_SIZE            # 5885

def x1_idx(c, p, d, s): return X1_START + c * TOTAL_SLOTS + slot_id(p, d, s)
def x2_idx(j, p, d, s): return X2_START + j * TOTAL_SLOTS + slot_id(p, d, s)
def z_idx(c, p, d, s):  return Z_START  + c * TOTAL_SLOTS + slot_id(p, d, s)
def v_idx(c):           return V_START  + c


# ─────────────────────────────────────────────────────────────────────────────
# BUILD OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────
c_obj = np.zeros(NVARS)

# Penalise SLA violations
for c in range(N_C1):
    c_obj[v_idx(c)] = W_SLA_VIOLATION

# Penalise over- or under-utilisation (soft: penalise slots used beyond cap)
# Proxy: each shadow slot beyond the floor is a mild cost
for c in range(N_C1):
    for p in range(NUM_PROVIDERS):
        for d in range(HORIZON_DAYS):
            for s in range(SLOTS_PER_DAY):
                c_obj[z_idx(c, p, d, s)] = W_UTIL_DEVIATION * 0.01


# ─────────────────────────────────────────────────────────────────────────────
# BUILD CONSTRAINTS (sparse LIL → CSR)
# ─────────────────────────────────────────────────────────────────────────────
constraints = []  # each entry: (row_vector_dict, lb, ub)

# We'll accumulate as lists of (indices, data, lb, ub)
rows_indices = []
rows_data    = []
rows_lb      = []
rows_ub      = []

def add_row(idx_list, coef_list, lb, ub):
    rows_indices.append(idx_list)
    rows_data.append(coef_list)
    rows_lb.append(lb)
    rows_ub.append(ub)

INF = np.inf


# ── C1: Each Case 1 initial assigned exactly once across all (p,d,s) ─────────
for c in range(N_C1):
    idx = [x1_idx(c, p, d, s)
           for p in range(NUM_PROVIDERS)
           for d in range(HORIZON_DAYS)
           for s in range(SLOTS_PER_DAY)]
    add_row(idx, [1]*len(idx), 1, 1)


# ── C2: SLA window — initial must be within sla_deadline (or violation = 1) ──
for c, case in enumerate(case1_requests):
    latest_d = date_to_idx(case.latest_initial_date)
    if latest_d is None:
        latest_d = HORIZON_DAYS - 1   # clamp to horizon end

    # Slots within SLA window
    sla_idx = [x1_idx(c, p, d, s)
               for p in range(NUM_PROVIDERS)
               for d in range(latest_d + 1)
               for s in range(SLOTS_PER_DAY)]
    # Σ x1 + v >= 1  →  if no in-window slot assigned, v must be 1
    add_row(sla_idx + [v_idx(c)], [1]*len(sla_idx) + [1], 1, INF)


# ── C3: Each Case 2 assigned exactly once ─────────────────────────────────────
for j in range(N_C2):
    idx = [x2_idx(j, p, d, s)
           for p in range(NUM_PROVIDERS)
           for d in range(HORIZON_DAYS)
           for s in range(SLOTS_PER_DAY)]
    add_row(idx, [1]*len(idx), 1, 1)


# ── C4: No double-booking per (provider, day, slot) ──────────────────────────
for p in range(NUM_PROVIDERS):
    for d in range(HORIZON_DAYS):
        for s in range(SLOTS_PER_DAY):
            idx, coef = [], []
            for c in range(N_C1):
                idx.append(x1_idx(c, p, d, s)); coef.append(1)
                idx.append(z_idx(c, p, d, s));  coef.append(1)
            for j in range(N_C2):
                idx.append(x2_idx(j, p, d, s)); coef.append(1)
            add_row(idx, coef, 0, 1)


# ── C5: Utilisation cap — max MAX_UTIL_SLOTS occupied per (provider, day) ────
for p in range(NUM_PROVIDERS):
    for d in range(HORIZON_DAYS):
        idx, coef = [], []
        for c in range(N_C1):
            for s in range(SLOTS_PER_DAY):
                idx.append(x1_idx(c, p, d, s)); coef.append(1)
                idx.append(z_idx(c, p, d, s));  coef.append(1)
        for j in range(N_C2):
            for s in range(SLOTS_PER_DAY):
                idx.append(x2_idx(j, p, d, s)); coef.append(1)
        add_row(idx, coef, 0, MAX_UTIL_SLOTS)


# ── C6: Shadow slots — reserve t+14k for each Case 1 initial ─────────────────
# If x1[c,p,d,s]=1 then z[c,p,d+14,*] must have ≥ 1 reserved (for k=1..N_max)
# Enforced as: Σ_s z[c,p,d+14,s] ≥ x1[c,p,d,s] for each valid future date
for c in range(N_C1):
    for p in range(NUM_PROVIDERS):
        for d in range(HORIZON_DAYS):
            for s in range(SLOTS_PER_DAY):
                for k in range(1, SHADOW_MAX_N + 1):
                    future_d = d + k * FOLLOWUP_GAP
                    if future_d >= HORIZON_DAYS:
                        continue   # beyond horizon — no constraint needed
                    shadow_slots = [z_idx(c, p, future_d, s2)
                                    for s2 in range(SLOTS_PER_DAY)]
                    # Σ z >= x1  →  Σ z - x1 >= 0
                    add_row(shadow_slots + [x1_idx(c, p, d, s)],
                            [1]*SLOTS_PER_DAY + [-1], 0, INF)


# ── C7: Case 1 continuity — shadow slots only on the assigned provider ────────
# z[c,p,d,s] ≤ Σ_{d0,s0} x1[c,p,d0,s0]  for each (c,p,d,s)
# i.e., shadows can only be on p if p is the assigned provider for case c
for c in range(N_C1):
    for p in range(NUM_PROVIDERS):
        assigned_on_p = [x1_idx(c, p, d0, s0)
                         for d0 in range(HORIZON_DAYS)
                         for s0 in range(SLOTS_PER_DAY)]
        for d in range(HORIZON_DAYS):
            for s in range(SLOTS_PER_DAY):
                # z[c,p,d,s] - Σ x1[c,p,*] ≤ 0
                add_row([z_idx(c, p, d, s)] + assigned_on_p,
                        [1] + [-1]*len(assigned_on_p), -INF, 0)


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE SPARSE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
nrows = len(rows_indices)
A = lil_matrix((nrows, NVARS), dtype=np.float64)
for r, (idxs, coefs) in enumerate(zip(rows_indices, rows_data)):
    for idx, coef in zip(idxs, coefs):
        A[r, idx] = coef
A_csr = csr_matrix(A)

lb_arr = np.array(rows_lb, dtype=np.float64)
ub_arr = np.array(rows_ub, dtype=np.float64)

lc = LinearConstraint(A_csr, lb_arr, ub_arr)

# All variables are in [0,1]; integrality = 1 for all (binary)
bounds = Bounds(lb=np.zeros(NVARS), ub=np.ones(NVARS))
integrality = np.ones(NVARS)


# ─────────────────────────────────────────────────────────────────────────────
# SOLVE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Mental Health Scheduling MILP — SciPy PoC")
print("=" * 60)
print(f"  Variables : {NVARS:,}")
print(f"  Constraints: {nrows:,}")
print(f"  Horizon   : {HORIZON_DAYS} days | Providers: {NUM_PROVIDERS}")
print(f"  Case 1    : {N_C1} initials | Case 2: {N_C2} sessions")
print()

result = milp(c=c_obj,
              constraints=lc,
              integrality=integrality,
              bounds=bounds,
              options={"disp": False, "time_limit": 60})

print(f"  Solver status : {result.message}")
print(f"  Objective cost: {result.fun:.1f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# DECODE SOLUTION
# ─────────────────────────────────────────────────────────────────────────────
if result.x is None:
    print("No feasible solution found within time limit.")
    raise SystemExit(1)

x = result.x

# ── Case 1 assignments ────────────────────────────────────────────────────────
print("─" * 60)
print("  CASE 1 INITIAL ASSIGNMENTS")
print("─" * 60)
schedule = {}   # slot_key → session label (for collision check)
shadow_map = {} # (c) → list of (date, slot)

for c, case in enumerate(case1_requests):
    assigned = False
    for p in range(NUM_PROVIDERS):
        for d in range(HORIZON_DAYS):
            for s in range(SLOTS_PER_DAY):
                if x[x1_idx(c, p, d, s)] > 0.5:
                    dt = DATES[d]
                    key = (p, d, s)
                    schedule[key] = f"C1-Initial:{case.id}"
                    sla_ok = dt <= case.latest_initial_date
                    print(f"  {case.id}  →  {PROVIDERS[p]}  |  {dt}  slot {s+1}"
                          f"  |  SLA {'✓' if sla_ok else '✗ VIOLATED'}"
                          f"  |  N_followups={case.n_followups_expected}"
                          f"  |  request={case.request_ts.strftime('%H:%M')}"
                          f"  |  sla_deadline={case.sla_deadline.strftime('%m/%d %H:%M')}")
                    assigned = True

                    # Collect shadow reservations
                    shadows = []
                    for k in range(1, SHADOW_MAX_N + 1):
                        fd = d + k * FOLLOWUP_GAP
                        if fd >= HORIZON_DAYS:
                            shadows.append((DATES[d] + timedelta(days=k*14),
                                            "beyond horizon", None))
                            continue
                        for s2 in range(SLOTS_PER_DAY):
                            if x[z_idx(c, p, fd, s2)] > 0.5:
                                skey = (p, fd, s2)
                                schedule[skey] = f"C1-Shadow:{case.id}:fu{k}"
                                shadows.append((DATES[fd], f"slot {s2+1}", s2))
                                break
                    shadow_map[c] = shadows

    if not assigned:
        print(f"  {case.id}  →  UNASSIGNED (SLA violation)")

print()

# ── Case 1 shadow capacity map ────────────────────────────────────────────────
print("─" * 60)
print("  SHADOW CAPACITY RESERVATIONS (Case 1)")
print("─" * 60)
for c, case in enumerate(case1_requests):
    shadows = shadow_map.get(c, [])
    print(f"  {case.id} ({case.n_followups_expected} expected follow-ups):")
    for k, (dt, slot_label, _) in enumerate(shadows, 1):
        marker = "← active" if k <= case.n_followups_expected else "← buffer"
        print(f"    fu{k}  {dt}  {slot_label}  {marker}")
print()

# ── Case 2 assignments ────────────────────────────────────────────────────────
print("─" * 60)
print("  CASE 2 AD-HOC ASSIGNMENTS")
print("─" * 60)
for j, case in enumerate(case2_requests):
    for p in range(NUM_PROVIDERS):
        for d in range(HORIZON_DAYS):
            for s in range(SLOTS_PER_DAY):
                if x[x2_idx(j, p, d, s)] > 0.5:
                    dt = DATES[d]
                    key = (p, d, s)
                    sla_ok = dt <= case.latest_first_date
                    schedule[key] = f"C2:{case.id}"
                    print(f"  {case.id}  →  {PROVIDERS[p]}  |  {dt}  slot {s+1}"
                          f"  |  SLA {'✓' if sla_ok else '✗ VIOLATED'}"
                          f"  |  max_sessions={case.max_sessions}")

print()

# ── Provider utilisation summary ──────────────────────────────────────────────
print("─" * 60)
print("  PROVIDER UTILISATION SUMMARY")
print("─" * 60)
print(f"  {'Provider':<10} {'Total Slots':>12} {'Available':>10} {'Util%':>8}")
print(f"  {'-'*44}")
total_available = HORIZON_DAYS * SLOTS_PER_DAY
for p in range(NUM_PROVIDERS):
    used = sum(1 for (pp, dd, ss), label in schedule.items()
               if pp == p)
    # count shadows
    shadow_count = 0
    for c in range(N_C1):
        for d in range(HORIZON_DAYS):
            for s in range(SLOTS_PER_DAY):
                if x[z_idx(c, p, d, s)] > 0.5:
                    shadow_count += 1
    total_occupied = used + shadow_count
    util_pct = total_occupied / total_available * 100
    print(f"  {PROVIDERS[p]:<10} {total_occupied:>12} {total_available:>10} {util_pct:>7.1f}%")

print()
print("─" * 60)

# ── SLA violation summary ─────────────────────────────────────────────────────
violations = sum(1 for c in range(N_C1) if x[v_idx(c)] > 0.5)
print(f"  SLA violations (Case 1): {violations} / {N_C1}")

# ── Collision check ───────────────────────────────────────────────────────────
all_slots = []
for c in range(N_C1):
    for p in range(NUM_PROVIDERS):
        for d in range(HORIZON_DAYS):
            for s in range(SLOTS_PER_DAY):
                if x[x1_idx(c, p, d, s)] > 0.5 or x[z_idx(c, p, d, s)] > 0.5:
                    all_slots.append((p, d, s))
for j in range(N_C2):
    for p in range(NUM_PROVIDERS):
        for d in range(HORIZON_DAYS):
            for s in range(SLOTS_PER_DAY):
                if x[x2_idx(j, p, d, s)] > 0.5:
                    all_slots.append((p, d, s))

duplicates = len(all_slots) - len(set(all_slots))
print(f"  Double-booking conflicts : {duplicates}  (should be 0)")
print("=" * 60)
