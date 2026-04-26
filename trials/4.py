# lost_in_space_submission.py
# Single-file deterministic solution for the Lost in Space EO scheduling track.
# Dependencies used: standard library, numpy, sgp4 only.

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sgp4.api import Satrec, jday


# ----------------------------- constants ------------------------------------
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

PROP_DT_S = 2.0                 # candidate propagation grid
ATTITUDE_DT_S = 0.05            # 20 Hz command samples, <= 50 Hz
HOLD_PAD_S = 0.40               # stable pointing pad around every shutter
OFF_NADIR_MARGIN_DEG = 5.0      # target <= 55 deg when formal limit is 60 deg
MAX_CMD_SLEW_RATE_DPS = 0.90    # conservative command slew cap
MIN_SHOT_GAP_S = 2.0
TARGET_GRID_N = 10              # 100 target centers for the 100 km x 100 km AOI
EVAL_GRID_N = 21                # scoring proxy grid
PER_TARGET_KEEP = 16             # time alternatives per target point
BEAM_WIDTH_EASY = 64
BEAM_WIDTH_MED = 52
BEAM_WIDTH_HARD = 38


# ----------------------------- time/orbit -----------------------------------
def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _gmst(dt: datetime) -> float:
    jd, fr = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute,
        dt.second + dt.microsecond * 1e-6,
    )
    T = ((jd - 2451545.0) + fr) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    ) % 86400.0
    return math.radians(gmst_sec / 240.0)


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _propagate(tle1: str, tle2: str, start_utc: str, end_utc: str, dt_s: float) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _parse_iso(start_utc)
    t1 = _parse_iso(end_utc)
    T = max(0.0, (t1 - t0).total_seconds())
    sat = Satrec.twoline2rv(tle1, tle2)
    states: List[Dict[str, Any]] = []
    n = int(math.floor(T / dt_s)) + 1
    for i in range(n + 1):
        t = min(float(i) * dt_s, T)
        utc = t0 + timedelta(seconds=t)
        jd, fr = jday(
            utc.year, utc.month, utc.day, utc.hour, utc.minute,
            utc.second + utc.microsecond * 1e-6,
        )
        err, r_km, v_kmps = sat.sgp4(jd, fr)
        if err == 0:
            states.append({
                "t": t,
                "r": np.asarray(r_km, dtype=float) * 1000.0,
                "v": np.asarray(v_kmps, dtype=float) * 1000.0,
                "gmst": _gmst(utc),
            })
        if t >= T:
            break
    return states, T


# ----------------------------- geodesy --------------------------------------
def _llh_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.array([
        (N + alt_m) * cl * co,
        (N + alt_m) * cl * so,
        (N * (1.0 - WGS84_E2) + alt_m) * sl,
    ], dtype=float)


def _ecef_rows_to_eci(target_ecef_rows: np.ndarray, gmst: float) -> np.ndarray:
    # Row-vector version of ECEF -> inertial/TEME-like reasoning frame.
    R = _rot_z(gmst)
    return target_ecef_rows @ R.T


def _ecef_to_eci(r_ecef: np.ndarray, gmst: float) -> np.ndarray:
    return _rot_z(gmst) @ r_ecef


class _LocalProjection:
    def __init__(self, lat0: float, lon0: float):
        self.lat0 = math.radians(lat0)
        self.lon0 = math.radians(lon0)
        self.cos0 = max(0.05, math.cos(self.lat0))

    def to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        return (
            (math.radians(float(lon)) - self.lon0) * self.cos0 * WGS84_A,
            (math.radians(float(lat)) - self.lat0) * WGS84_A,
        )


def _clean_polygon(poly: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    verts = list(poly)
    if len(verts) > 1 and verts[0] == verts[-1]:
        verts = verts[:-1]
    return [(float(a), float(b)) for a, b in verts]


def _point_in_poly_xy(pt: Tuple[float, float], poly_xy: Sequence[Tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly_xy)
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / max(y2 - y1, 1e-15) + x1
            if x < xin:
                inside = not inside
    return inside


def _grid_points(poly_llh: Sequence[Tuple[float, float]], proj: _LocalProjection, n: int) -> List[Tuple[float, float]]:
    verts = _clean_polygon(poly_llh)
    lats = [x[0] for x in verts]
    lons = [x[1] for x in verts]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    poly_xy = [proj.to_xy(a, b) for a, b in verts]
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        lat = lat_min + (i + 0.5) * (lat_max - lat_min) / n
        for j in range(n):
            lon = lon_min + (j + 0.5) * (lon_max - lon_min) / n
            if _point_in_poly_xy(proj.to_xy(lat, lon), poly_xy):
                pts.append((lat, lon))
    return pts


# ----------------------------- quaternion/attitude ---------------------------
def _qnorm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 0.0 or not math.isfinite(n):
        return IDENTITY_Q.copy()
    q = q / n
    if q[3] < 0.0:
        q = -q
    return q


def _mat_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        S = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-16)) * 2.0
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] >= m[2, 2]:
        S = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-16)) * 2.0
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-16)) * 2.0
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    return _qnorm(np.array([qx, qy, qz, qw], dtype=float))


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    a = _qnorm(q0)
    b = _qnorm(q1)
    d = abs(float(np.dot(a, b)))
    d = max(-1.0, min(1.0, d))
    return math.degrees(2.0 * math.acos(d))


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = _qnorm(q0)
    q1 = _qnorm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _qnorm((1.0 - u) * q0 + u * q1)
    th = math.acos(dot)
    s = math.sin(th)
    return _qnorm((math.sin((1.0 - u) * th) / s) * q0 + (math.sin(u * th) / s) * q1)


def _stare_quat_BN(r_sat: np.ndarray, r_tgt: np.ndarray, v_sat: np.ndarray) -> np.ndarray:
    z_b = r_tgt - r_sat
    z_b = z_b / max(float(np.linalg.norm(z_b)), 1e-12)
    vhat = v_sat / max(float(np.linalg.norm(v_sat)), 1e-12)
    x_b = vhat - float(np.dot(vhat, z_b)) * z_b
    if float(np.linalg.norm(x_b)) < 1e-9:
        x_b = np.array([1.0, 0.0, 0.0], dtype=float) - z_b[0] * z_b
    x_b = x_b / max(float(np.linalg.norm(x_b)), 1e-12)
    y_b = np.cross(z_b, x_b)
    y_b = y_b / max(float(np.linalg.norm(y_b)), 1e-12)
    x_b = np.cross(y_b, z_b)
    x_b = x_b / max(float(np.linalg.norm(x_b)), 1e-12)
    return _mat_to_quat_xyzw(np.column_stack([x_b, y_b, z_b]))


# ----------------------------- candidates -----------------------------------
def _approx_coverage(
    target_xy: Tuple[float, float],
    eval_xy: np.ndarray,
    sat_radius_m: float,
    off_nadir_deg: float,
    fov_deg: Sequence[float],
) -> Set[int]:
    # Fast proxy for footprint coverage. The true simulator scores exact frames;
    # this proxy is only for selecting target centers. It deliberately errs a bit
    # wide at high off-nadir where projected footprints stretch.
    altitude = max(350000.0, sat_radius_m - WGS84_A)
    hx = altitude * math.tan(math.radians(float(fov_deg[0])) * 0.5)
    hy = altitude * math.tan(math.radians(float(fov_deg[1])) * 0.5)
    c = max(0.45, math.cos(math.radians(off_nadir_deg)))
    rx = 1.10 * hx / (c ** 1.25)
    ry = 1.10 * hy / (c ** 1.10)
    dx = (eval_xy[:, 0] - target_xy[0]) / max(rx, 1.0)
    dy = (eval_xy[:, 1] - target_xy[1]) / max(ry, 1.0)
    idx = np.nonzero(dx * dx + dy * dy <= 1.0)[0]
    return set(int(i) for i in idx)


def _materialize_candidate(
    light: Dict[str, Any],
    states_by_t: Dict[float, Dict[str, Any]],
    target_llh: Sequence[Tuple[float, float]],
    target_ecef: np.ndarray,
    target_xy: Sequence[Tuple[float, float]],
    eval_xy: np.ndarray,
    fov_deg: Sequence[float],
) -> Optional[Dict[str, Any]]:
    st = states_by_t.get(float(light["t"]))
    if st is None:
        return None
    idx = int(light["idx"])
    r_tgt = _ecef_to_eci(target_ecef[idx], st["gmst"])
    q = _stare_quat_BN(st["r"], r_tgt, st["v"])
    covered = _approx_coverage(target_xy[idx], eval_xy, float(np.linalg.norm(st["r"])), float(light["off"]), fov_deg)
    if not covered:
        return None
    return {
        "t": float(light["t"]),
        "q": q,
        "covered": covered,
        "idx": idx,
        "target_llh": target_llh[idx],
        "target_xy": target_xy[idx],
        "off": float(light["off"]),
    }


def _build_candidates(
    states: Sequence[Dict[str, Any]],
    target_llh: Sequence[Tuple[float, float]],
    target_ecef: np.ndarray,
    target_xy: Sequence[Tuple[float, float]],
    eval_xy: np.ndarray,
    fov_deg: Sequence[float],
    off_nadir_limit_deg: float,
    margin_deg: float,
) -> List[Dict[str, Any]]:
    if not states or len(target_llh) == 0:
        return []
    safe_limit = max(1.0, off_nadir_limit_deg - margin_deg)
    # One best candidate per target per coarse time bin. This avoids the common
    # failure mode where all retained candidates collapse to the final second of
    # the pass and cannot be used as shutter starts.
    buckets: List[Dict[int, Dict[str, Any]]] = [dict() for _ in target_llh]
    mid_t = 0.5 * (states[0]["t"] + states[-1]["t"])

    for st in states:
        tgt_eci = _ecef_rows_to_eci(target_ecef, st["gmst"])
        los = tgt_eci - st["r"][None, :]
        los_norm = np.linalg.norm(los, axis=1)
        los_unit = los / np.maximum(los_norm[:, None], 1e-12)
        nadir = -st["r"] / max(float(np.linalg.norm(st["r"])), 1e-12)
        cosang = np.clip(los_unit @ nadir, -1.0, 1.0)
        off = np.degrees(np.arccos(cosang))
        visible = np.nonzero(off <= safe_limit)[0]
        for idx in visible:
            idx_int = int(idx)
            off_val = float(off[idx_int])
            # Low off-nadir is preferred; a tiny mid-pass bias avoids retaining
            # only edge-of-window samples when geometry is monotonic in time.
            score = -off_val - 0.002 * abs(float(st["t"]) - mid_t)
            item = {"idx": idx_int, "t": float(st["t"]), "off": off_val, "score": score}
            bin_id = int(float(st["t"]) // 30.0)
            old = buckets[idx_int].get(bin_id)
            if old is None or item["score"] > old["score"]:
                buckets[idx_int][bin_id] = item

    states_by_t = {float(st["t"]): st for st in states}
    candidates: List[Dict[str, Any]] = []
    for bucket_map in buckets:
        lights = sorted(bucket_map.values(), key=lambda x: x["score"], reverse=True)[:PER_TARGET_KEEP]
        for light in lights:
            c = _materialize_candidate(light, states_by_t, target_llh, target_ecef, target_xy, eval_xy, fov_deg)
            if c is not None:
                candidates.append(c)
    candidates.sort(key=lambda c: float(c["t"]))
    return candidates


# ----------------------------- beam selection --------------------------------
def _feasible_after(prev: Optional[Dict[str, Any]], cand: Dict[str, Any], integration_s: float) -> bool:
    if prev is None:
        available = float(cand["t"]) - HOLD_PAD_S
        angle = _quat_angle_deg(IDENTITY_Q, cand["q"])
    else:
        if float(cand["t"]) - float(prev["t"]) < MIN_SHOT_GAP_S:
            return False
        available = float(cand["t"]) - float(prev["t"]) - integration_s - 2.0 * HOLD_PAD_S
        angle = _quat_angle_deg(prev["q"], cand["q"])
    return available > 0.0 and (angle / max(available, 1e-9)) <= MAX_CMD_SLEW_RATE_DPS


def _case_params(candidates: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    if not candidates:
        return 0, BEAM_WIDTH_HARD
    offs = sorted(float(c["off"]) for c in candidates)
    med = offs[len(offs) // 2]
    if med < 22.0:
        return 32, BEAM_WIDTH_EASY
    if med < 42.0:
        return 24, BEAM_WIDTH_MED
    return 14, BEAM_WIDTH_HARD


def _score_sequence(shots: Sequence[Dict[str, Any]], cell_count: int, T: float, integration_s: float) -> float:
    if not shots:
        return 0.0
    covered: Set[int] = set()
    total_slew = 0.0
    path = 0.0
    prev_q = IDENTITY_Q
    prev_xy = None
    off_sum = 0.0
    for s in shots:
        covered |= s["covered"]
        total_slew += _quat_angle_deg(prev_q, s["q"])
        off_sum += float(s["off"])
        if prev_xy is not None:
            dx = abs(float(s["target_xy"][0]) - float(prev_xy[0])) / 100000.0
            dy = abs(float(s["target_xy"][1]) - float(prev_xy[1])) / 100000.0
            path += 0.35 * dx + 0.10 * dy
        prev_q = s["q"]
        prev_xy = s["target_xy"]
    coverage = len(covered) / max(1, cell_count)
    span = max(0.0, float(shots[-1]["t"]) + integration_s - float(shots[0]["t"]))
    avg_off = off_sum / max(1, len(shots))
    return (
        coverage
        - 0.012 * len(shots)
        - 0.00035 * total_slew
        - 0.018 * span / max(T, 1.0)
        - 0.0008 * avg_off
        - 0.010 * path
    )


def _select_shots(candidates: Sequence[Dict[str, Any]], T: float, integration_s: float, cell_count: int) -> List[Dict[str, Any]]:
    max_shots, beam_width = _case_params(candidates)
    if max_shots <= 0:
        return []
    ordered = sorted(candidates, key=lambda c: float(c["t"]))
    states: List[Tuple[float, Tuple[Dict[str, Any], ...], Set[int], Optional[Dict[str, Any]]]] = [(0.0, tuple(), set(), None)]
    best: Tuple[Dict[str, Any], ...] = tuple()
    best_score = 0.0

    for cand in ordered:
        if float(cand["t"]) + integration_s + HOLD_PAD_S > T:
            continue
        next_states = states[:]
        for _score, shots, covered, prev in states:
            if len(shots) >= max_shots:
                continue
            if prev is not None and float(cand["t"]) <= float(prev["t"]):
                continue
            if not _feasible_after(prev, cand, integration_s):
                continue
            new_cells = cand["covered"] - covered
            if not new_cells:
                continue
            overlap = len(cand["covered"] & covered) / max(1, len(cand["covered"]))
            if overlap > 0.90 and len(new_cells) < 2:
                continue
            new_shots = shots + (cand,)
            new_covered = covered | cand["covered"]
            sc = _score_sequence(new_shots, cell_count, T, integration_s)
            if sc > best_score:
                best_score = sc
                best = new_shots
            next_states.append((sc, new_shots, new_covered, cand))

        next_states.sort(key=lambda x: (x[0], len(x[2]), -len(x[1])), reverse=True)
        # Diversity pruning keeps the beam from filling with nearly identical states.
        pruned: List[Tuple[float, Tuple[Dict[str, Any], ...], Set[int], Optional[Dict[str, Any]]]] = []
        seen = set()
        for st in next_states:
            tail = tuple(round(float(s["t"]) / 6.0) for s in st[1][-4:])
            fp = (len(st[1]), tail, len(st[2]) // 3)
            if fp in seen:
                continue
            seen.add(fp)
            pruned.append(st)
            if len(pruned) >= beam_width:
                break
        states = pruned

    return list(best)


# ----------------------------- output schedule -------------------------------
def _build_schedule(shots: Sequence[Dict[str, Any]], T: float, integration_s: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    knots: List[Tuple[float, np.ndarray]] = [(0.0, IDENTITY_Q.copy())]
    shutter: List[Dict[str, Any]] = []

    for s in sorted(shots, key=lambda x: float(x["t"])):
        ts = max(0.0, float(s["t"]))
        te = min(T, ts + integration_s)
        hs = max(0.0, ts - HOLD_PAD_S)
        he = min(T, te + HOLD_PAD_S)
        q = _qnorm(s["q"])
        knots.extend([(hs, q), (ts, q), (te, q), (he, q)])
        shutter.append({"t_start": round(ts, 4), "duration": float(integration_s)})

    if knots[-1][0] < T:
        knots.append((T, knots[-1][1]))
    knots.sort(key=lambda x: x[0])

    cleaned: List[Tuple[float, np.ndarray]] = []
    for t, q in knots:
        t = max(0.0, min(float(T), float(t)))
        q = _qnorm(q)
        if cleaned and abs(t - cleaned[-1][0]) < 1e-10:
            cleaned[-1] = (t, q)
        elif cleaned and t - cleaned[-1][0] < 0.0200001:
            continue
        else:
            cleaned.append((t, q))
    if not cleaned or cleaned[0][0] != 0.0:
        cleaned.insert(0, (0.0, IDENTITY_Q.copy()))

    dense: List[Tuple[float, np.ndarray]] = []
    for i in range(len(cleaned) - 1):
        t0, q0 = cleaned[i]
        t1, q1 = cleaned[i + 1]
        if not dense:
            dense.append((t0, q0))
        dt = t1 - t0
        if dt <= 0.0:
            continue
        n = max(1, int(math.ceil(dt / ATTITUDE_DT_S)))
        while n > 1 and dt / n < 0.0200001:
            n -= 1
        for k in range(1, n + 1):
            u = k / n
            dense.append((t0 + u * dt, _slerp(q0, q1, u)))

    attitude: List[Dict[str, Any]] = []
    last_t = -1e9
    for t, q in dense:
        tr = round(float(t), 4)
        if attitude and tr - last_t < 0.0199:
            continue
        q = _qnorm(q)
        attitude.append({"t": tr, "q_BN": [float(q[0]), float(q[1]), float(q[2]), float(q[3])]})
        last_t = tr

    if not attitude:
        attitude = [{"t": 0.0, "q_BN": [0.0, 0.0, 0.0, 1.0]}]
    if attitude[0]["t"] != 0.0:
        attitude.insert(0, {"t": 0.0, "q_BN": [0.0, 0.0, 0.0, 1.0]})
    last_needed = max([0.0] + [x["t_start"] + x["duration"] for x in shutter])
    if attitude[-1]["t"] < last_needed:
        attitude.append({"t": round(last_needed, 4), "q_BN": attitude[-1]["q_BN"]})
    return attitude, shutter


def _fallback(T: float, reason: str) -> Dict[str, Any]:
    return {
        "objective": "safe_fallback",
        "attitude": [
            {"t": 0.0, "q_BN": [0.0, 0.0, 0.0, 1.0]},
            {"t": round(float(T), 4), "q_BN": [0.0, 0.0, 0.0, 1.0]},
        ],
        "shutter": [],
        "notes": reason,
        "target_hints_llh": [],
    }


# ----------------------------- required API ---------------------------------
def plan_imaging(
    tle_line1: str,
    tle_line2: str,
    aoi_polygon_llh: List[Tuple[float, float]],
    pass_start_utc: str,
    pass_end_utc: str,
    sc_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Return one pass attitude + shutter schedule."""
    states, T = _propagate(tle_line1, tle_line2, pass_start_utc, pass_end_utc, PROP_DT_S)
    if T <= 0.0 or not states:
        return _fallback(max(T, 0.0), "No propagated states available.")

    integration_s = float(sc_params.get("integration_s", 0.120))
    fov_deg = sc_params.get("fov_deg", [2.0, 2.0])
    off_limit = float(sc_params.get("off_nadir_max_deg", 60.0))

    verts = _clean_polygon(aoi_polygon_llh)
    if len(verts) < 3:
        return _fallback(T, "Invalid AOI polygon.")
    lat0 = sum(p[0] for p in verts) / len(verts)
    lon0 = sum(p[1] for p in verts) / len(verts)
    proj = _LocalProjection(lat0, lon0)

    target_llh = _grid_points(verts, proj, TARGET_GRID_N)
    eval_llh = _grid_points(verts, proj, EVAL_GRID_N)
    if not target_llh or not eval_llh:
        return _fallback(T, "AOI grid generation produced no cells.")

    target_ecef = np.array([_llh_to_ecef(a, b, 0.0) for a, b in target_llh], dtype=float)
    target_xy = [proj.to_xy(a, b) for a, b in target_llh]
    eval_xy = np.array([proj.to_xy(a, b) for a, b in eval_llh], dtype=float)

    candidates = _build_candidates(
        states, target_llh, target_ecef, target_xy, eval_xy,
        fov_deg, off_limit, OFF_NADIR_MARGIN_DEG,
    )
    if not candidates:
        # Recover hard geometries but still stay under the formal 60 deg gate.
        candidates = _build_candidates(
            states, target_llh, target_ecef, target_xy, eval_xy,
            fov_deg, off_limit, 2.0,
        )
    if not candidates:
        return _fallback(T, "No off-nadir-safe imaging candidates found.")

    shots = _select_shots(candidates, T, integration_s, len(eval_llh))
    attitude, shutter = _build_schedule(shots, T, integration_s)

    covered: Set[int] = set()
    for s in shots:
        covered |= s["covered"]
    coverage_proxy = len(covered) / max(1, len(eval_llh))
    off_values = [float(s["off"]) for s in shots]
    avg_off = sum(off_values) / max(1, len(off_values)) if off_values else 0.0
    max_off = max(off_values) if off_values else 0.0

    return {
        "objective": "coverage_first_safe_beam_search",
        "attitude": attitude,
        "shutter": shutter,
        "notes": (
            f"shots={len(shots)}, candidates={len(candidates)}, "
            f"coverage_proxy={coverage_proxy:.3f}, avg_off_nadir={avg_off:.2f}deg, "
            f"max_off_nadir={max_off:.2f}deg, slew_cap={MAX_CMD_SLEW_RATE_DPS:.2f}deg/s"
        ),
        "target_hints_llh": [
            {"lat_deg": float(s["target_llh"][0]), "lon_deg": float(s["target_llh"][1])}
            for s in sorted(shots, key=lambda x: float(x["t"]))
        ],
    }
