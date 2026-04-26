from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sgp4.api import Satrec, jday


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

SAFE_OFF_NADIR_MARGIN_DEG = 1.5
HOLD_PAD_S = 0.50
MAX_SLEW_RATE_DPS = 2.0
MIN_SHOT_SPACING_S = 3.0
TIME_GRID_S = 1.0
TARGET_GRID_N = 10
EVAL_GRID_N = 13
MAX_SHOTS = 8
MIN_NEW_CELLS = 1
BEAM_WIDTH = 72
ANGLE_BLEND_START_DEG = 18.0
ANGLE_BLEND_END_DEG = 55.0
TARGET_GRID_N_HARD = 16
EVAL_GRID_N_HARD = 21
UNIFIED_CANDIDATES_PER_TARGET = 5
UNIFIED_MIN_TIME_SEPARATION_S = 4.0
MAX_ATTITUDE_DT_S = 1.0


@dataclass
class SatState:
    t: float
    utc: datetime
    r_eci: np.ndarray
    v_eci: np.ndarray
    gmst: float


@dataclass
class ShotCandidate:
    t_start: float
    q_BN: np.ndarray
    covered_cells: Set[int]
    target_llh: Tuple[float, float]
    off_nadir_deg: float
    geometry_weight: float = 0.0
    target_xy: Tuple[float, float] = (0.0, 0.0)


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _gmst(dt: datetime) -> float:
    jd, fr = jday(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second + dt.microsecond * 1e-6,
    )
    T = ((jd - 2451545.0) + fr) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    ) % 86400.0
    if gmst_sec < 0:
        gmst_sec += 86400.0
    return math.radians(gmst_sec / 240.0)


def _llh_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    ss, cs = math.sin(lon), math.cos(lon)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.array(
        [
            (N + alt_m) * cl * cs,
            (N + alt_m) * cl * ss,
            (N * (1.0 - WGS84_E2) + alt_m) * sl,
        ],
        dtype=float,
    )


def _ecef_to_eci(r_ecef: np.ndarray, gmst: float) -> np.ndarray:
    c = math.cos(gmst)
    s = math.sin(gmst)
    return np.array(
        [
            c * r_ecef[0] - s * r_ecef[1],
            s * r_ecef[0] + c * r_ecef[1],
            r_ecef[2],
        ],
        dtype=float,
    )


def _eci_to_ecef(r_eci: np.ndarray, gmst: float) -> np.ndarray:
    c = math.cos(-gmst)
    s = math.sin(-gmst)
    return np.array(
        [
            c * r_eci[0] - s * r_eci[1],
            s * r_eci[0] + c * r_eci[1],
            r_eci[2],
        ],
        dtype=float,
    )


def _ecef_to_llh(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = float(r_ecef[0]), float(r_ecef[1]), float(r_ecef[2])
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sl = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
        alt = p / max(math.cos(lat), 1e-12) - N
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * N / max(N + alt, 1e-12)))
    sl = math.sin(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    alt = p / max(math.cos(lat), 1e-12) - N
    return math.degrees(lat), math.degrees(lon), alt


def _mat_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=float)
    return q / np.linalg.norm(q)


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    d = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    d = max(-1.0, min(1.0, d))
    return math.degrees(2.0 * math.acos(d))


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        q = (1.0 - u) * q0 + u * q1
        return q / np.linalg.norm(q)
    theta_0 = math.acos(dot)
    sin_0 = math.sin(theta_0)
    theta = theta_0 * u
    return (math.sin(theta_0 - theta) / sin_0) * q0 + (math.sin(theta) / sin_0) * q1


def _stare_quat_BN(r_sat_eci: np.ndarray, r_tgt_eci: np.ndarray, v_sat_eci: np.ndarray) -> np.ndarray:
    z_body_in_N = r_tgt_eci - r_sat_eci
    z_body_in_N = z_body_in_N / np.linalg.norm(z_body_in_N)
    vhat = v_sat_eci / np.linalg.norm(v_sat_eci)
    x_body_in_N = vhat - np.dot(vhat, z_body_in_N) * z_body_in_N
    nrm = np.linalg.norm(x_body_in_N)
    if nrm < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        x_body_in_N = fallback - np.dot(fallback, z_body_in_N) * z_body_in_N
        nrm = np.linalg.norm(x_body_in_N)
    x_body_in_N = x_body_in_N / nrm
    y_body_in_N = np.cross(z_body_in_N, x_body_in_N)
    return _mat_to_quat_xyzw(np.column_stack([x_body_in_N, y_body_in_N, z_body_in_N]))


def _quat_to_rot_BN(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _ray_ellipsoid_intersect(origin: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
    D = np.array([1.0 / WGS84_A, 1.0 / WGS84_A, 1.0 / WGS84_B], dtype=float)
    o = origin * D
    d = direction * D
    A = float(np.dot(d, d))
    B = 2.0 * float(np.dot(o, d))
    C = float(np.dot(o, o)) - 1.0
    disc = B * B - 4.0 * A * C
    if disc < 0.0 or A < 1e-18:
        return None
    sq = math.sqrt(disc)
    t1 = (-B - sq) / (2.0 * A)
    t2 = (-B + sq) / (2.0 * A)
    t = t1 if t1 >= 0.0 else t2 if t2 >= 0.0 else None
    if t is None:
        return None
    return origin + t * direction


def _project_footprint(q_BN: np.ndarray, r_eci: np.ndarray, gmst: float, fov_deg: Sequence[float]) -> Optional[List[Tuple[float, float]]]:
    Rzi = np.array(
        [
            [math.cos(-gmst), -math.sin(-gmst), 0.0],
            [math.sin(-gmst), math.cos(-gmst), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    R_BN = _quat_to_rot_BN(q_BN)
    r_ecef = Rzi @ r_eci

    fx = math.radians(float(fov_deg[0])) / 2.0
    fy = math.radians(float(fov_deg[1])) / 2.0
    tx, ty = math.tan(fx), math.tan(fy)
    rays_B = [
        np.array([+tx, +ty, 1.0], dtype=float),
        np.array([-tx, +ty, 1.0], dtype=float),
        np.array([-tx, -ty, 1.0], dtype=float),
        np.array([+tx, -ty, 1.0], dtype=float),
    ]

    corners = []
    for d_B in rays_B:
        d_B = d_B / np.linalg.norm(d_B)
        d_N = R_BN @ d_B
        d_E = Rzi @ d_N
        hit = _ray_ellipsoid_intersect(r_ecef, d_E)
        if hit is None:
            return None
        lat, lon, _ = _ecef_to_llh(hit)
        corners.append((lat, lon))
    return corners


def _off_nadir_deg(r_sat_eci: np.ndarray, r_tgt_eci: np.ndarray) -> float:
    los = r_tgt_eci - r_sat_eci
    los = los / np.linalg.norm(los)
    nadir = -r_sat_eci / np.linalg.norm(r_sat_eci)
    c = float(np.dot(los, nadir))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _point_in_convex_polygon(point_xy: Tuple[float, float], poly_xy: Sequence[Tuple[float, float]]) -> bool:
    if len(poly_xy) < 3:
        return False
    sign = 0
    px, py = point_xy
    n = len(poly_xy)
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-6:
            continue
        curr = 1 if cross > 0.0 else -1
        if sign == 0:
            sign = curr
        elif curr != sign:
            return False
    return True


class LocalProjection:
    def __init__(self, lat0_deg: float, lon0_deg: float):
        self.lat0 = math.radians(lat0_deg)
        self.lon0 = math.radians(lon0_deg)
        self.cos0 = math.cos(self.lat0)

    def to_xy(self, lat_deg: float, lon_deg: float) -> Tuple[float, float]:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        return (
            (lon - self.lon0) * self.cos0 * WGS84_A,
            (lat - self.lat0) * WGS84_A,
        )


def _grid_points(aoi_polygon_llh: Sequence[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    verts = list(aoi_polygon_llh[:-1]) if len(aoi_polygon_llh) > 1 and aoi_polygon_llh[0] == aoi_polygon_llh[-1] else list(aoi_polygon_llh)
    lats = [p[0] for p in verts]
    lons = [p[1] for p in verts]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        lat = lat_min + (i + 0.5) * (lat_max - lat_min) / n
        for j in range(n):
            lon = lon_min + (j + 0.5) * (lon_max - lon_min) / n
            pts.append((lat, lon))
    return pts


def _propagate_states(tle_line1: str, tle_line2: str, pass_start_utc: str, pass_end_utc: str, dt_s: float) -> Tuple[List[SatState], float]:
    t0 = _parse_iso(pass_start_utc)
    t1 = _parse_iso(pass_end_utc)
    T = (t1 - t0).total_seconds()
    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    states: List[SatState] = []
    n = int(math.floor(T / dt_s)) + 1
    for i in range(n):
        t = min(i * dt_s, T)
        utc = t0 + timedelta(seconds=t)
        jd, fr = jday(
            utc.year,
            utc.month,
            utc.day,
            utc.hour,
            utc.minute,
            utc.second + utc.microsecond * 1e-6,
        )
        err, r_km, v_kmps = sat.sgp4(jd, fr)
        if err != 0:
            continue
        states.append(
            SatState(
                t=t,
                utc=utc,
                r_eci=np.asarray(r_km, dtype=float) * 1000.0,
                v_eci=np.asarray(v_kmps, dtype=float) * 1000.0,
                gmst=_gmst(utc),
            )
        )
    return states, T


def _build_candidates(
    states: Sequence[SatState],
    target_points: Sequence[Tuple[float, float]],
    eval_points: Sequence[Tuple[float, float]],
    proj: LocalProjection,
    fov_deg: Sequence[float],
    off_nadir_limit_deg: float,
    safe_margin_deg: float,
    per_target_limit: int = 1,
    min_time_separation_s: float = 0.0,
) -> List[ShotCandidate]:
    eval_xy = [proj.to_xy(lat, lon) for lat, lon in eval_points]
    target_ecef = [_llh_to_ecef(lat, lon, 0.0) for lat, lon in target_points]

    candidates: List[ShotCandidate] = []
    safe_limit = off_nadir_limit_deg - safe_margin_deg

    for (target_llh, r_tgt_ecef) in zip(target_points, target_ecef):
        target_candidates: List[Tuple[float, ShotCandidate]] = []
        for st in states:
            r_tgt_eci = _ecef_to_eci(r_tgt_ecef, st.gmst)
            off_nadir = _off_nadir_deg(st.r_eci, r_tgt_eci)
            if off_nadir > safe_limit:
                continue
            q = _stare_quat_BN(st.r_eci, r_tgt_eci, st.v_eci)
            corners = _project_footprint(q, st.r_eci, st.gmst, fov_deg)
            if corners is None:
                continue
            poly_xy = [proj.to_xy(lat, lon) for lat, lon in corners]
            covered = {idx for idx, pt in enumerate(eval_xy) if _point_in_convex_polygon(pt, poly_xy)}
            if not covered:
                continue
            score = len(covered) - 0.08 * off_nadir - 0.002 * abs(st.t - states[len(states) // 2].t)
            target_candidates.append(
                (
                    score,
                    ShotCandidate(
                    t_start=st.t,
                    q_BN=q,
                    covered_cells=covered,
                    target_llh=target_llh,
                    off_nadir_deg=off_nadir,
                    geometry_weight=0.0,
                    target_xy=proj.to_xy(target_llh[0], target_llh[1]),
                    ),
                )
            )
        target_candidates.sort(key=lambda x: x[0], reverse=True)
        kept: List[ShotCandidate] = []
        for _score, cand in target_candidates:
            if any(abs(cand.t_start - prev.t_start) < min_time_separation_s for prev in kept):
                continue
            kept.append(cand)
            if len(kept) >= per_target_limit:
                break
        candidates.extend(kept)

    candidates.sort(key=lambda c: c.t_start)
    return candidates


def _feasible_after(prev_t: float, prev_q: np.ndarray, cand: ShotCandidate) -> bool:
    if prev_t < 0.0:
        return cand.t_start >= HOLD_PAD_S
    angle_deg = _quat_angle_deg(prev_q, cand.q_BN)
    required = angle_deg / MAX_SLEW_RATE_DPS
    available = cand.t_start - prev_t
    return available >= max(MIN_SHOT_SPACING_S, required + 2.0 * HOLD_PAD_S + 0.15)


def _select_shots(
    candidates: Sequence[ShotCandidate],
    T: float,
    integration_s: float,
    max_shots: int = MAX_SHOTS,
    min_new_cells: int = MIN_NEW_CELLS,
) -> List[ShotCandidate]:
    selected: List[ShotCandidate] = []
    covered: Set[int] = set()
    prev_t = -1.0
    prev_q = IDENTITY_Q.copy()

    remaining = list(candidates)
    while remaining and len(selected) < max_shots:
        best_idx = None
        best_value = 0.0
        for idx, cand in enumerate(remaining):
            if cand.t_start + integration_s + HOLD_PAD_S > T:
                continue
            if not _feasible_after(prev_t, prev_q, cand):
                continue
            new_cells = len(cand.covered_cells - covered)
            if new_cells < min_new_cells:
                continue
            angle_pen = 0.03 * _quat_angle_deg(prev_q, cand.q_BN)
            overlap = len(cand.covered_cells & covered)
            w = cand.geometry_weight
            easy_value = new_cells - 0.45 * overlap
            hard_value = 0.75 * len(cand.covered_cells) - 0.10 * cand.off_nadir_deg - 0.20 * overlap
            value = (1.0 - w) * easy_value + w * hard_value - ((1.0 - 0.35 * w) * angle_pen)
            if value > best_value:
                best_value = value
                best_idx = idx
        if best_idx is None:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered |= chosen.covered_cells
        prev_t = chosen.t_start
        prev_q = chosen.q_BN
        remaining = [c for c in remaining if c.t_start > chosen.t_start]

    if len(selected) < 3:
        covered.clear()
        selected = []
        prev_t = -1.0
        prev_q = IDENTITY_Q.copy()
        for cand in candidates:
            if cand.t_start + integration_s + HOLD_PAD_S > T:
                continue
            if not _feasible_after(prev_t, prev_q, cand):
                continue
            selected.append(cand)
            covered |= cand.covered_cells
            prev_t = cand.t_start
            prev_q = cand.q_BN
            if len(selected) >= min(10, max_shots):
                break

    return selected


def _build_attitude_schedule(shots: Sequence[ShotCandidate], T: float, integration_s: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    attitude: List[Tuple[float, np.ndarray]] = [(0.0, IDENTITY_Q.copy())]
    shutter: List[Dict[str, Any]] = []

    for shot in shots:
        hold_start = max(0.0, shot.t_start - HOLD_PAD_S)
        hold_end = min(T, shot.t_start + integration_s + HOLD_PAD_S)

        attitude.append((hold_start, shot.q_BN))
        attitude.append((shot.t_start, shot.q_BN))
        attitude.append((shot.t_start + integration_s, shot.q_BN))
        attitude.append((hold_end, shot.q_BN))
        shutter.append({"t_start": round(shot.t_start, 4), "duration": integration_s})

    if attitude[-1][0] < T:
        attitude.append((T, attitude[-1][1]))

    attitude.sort(key=lambda x: x[0])

    cleaned: List[Tuple[float, np.ndarray]] = []
    for t, q in attitude:
        t = max(0.0, min(T, t))
        if cleaned and abs(t - cleaned[-1][0]) < 1e-9:
            cleaned[-1] = (t, q)
            continue
        if cleaned and t - cleaned[-1][0] < 0.020:
            continue
        cleaned.append((t, q))

    if cleaned[0][0] > 0.0:
        cleaned.insert(0, (0.0, cleaned[0][1]))
    if shutter:
        need_end = shutter[-1]["t_start"] + integration_s
    else:
        need_end = T
    if cleaned[-1][0] < need_end:
        cleaned.append((need_end, cleaned[-1][1]))

    dense: List[Tuple[float, np.ndarray]] = []
    for i, (t0, q0) in enumerate(cleaned[:-1]):
        t1, q1 = cleaned[i + 1]
        if not dense:
            dense.append((t0, q0))
        n = max(1, int(math.ceil((t1 - t0) / MAX_ATTITUDE_DT_S)))
        for k in range(1, n + 1):
            u = k / n
            dense.append((t0 + u * (t1 - t0), _slerp_quat(q0, q1, u)))

    attitude_dicts = [{"t": round(t, 4), "q_BN": [float(x) for x in q / np.linalg.norm(q)]} for t, q in dense]
    return attitude_dicts, shutter


def _smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _geometry_weight_from_angle(off_nadir_deg: float) -> float:
    u = (off_nadir_deg - ANGLE_BLEND_START_DEG) / max(ANGLE_BLEND_END_DEG - ANGLE_BLEND_START_DEG, 1e-9)
    return _smoothstep01(u)


def _enrich_geometry_weights(candidates: Sequence[ShotCandidate], centroid_llh: Tuple[float, float]) -> List[ShotCandidate]:
    centroid = np.array(centroid_llh, dtype=float)
    if not candidates:
        return []
    max_dist = max(float(np.linalg.norm(np.array(c.target_llh, dtype=float) - centroid)) for c in candidates) + 1e-9
    enriched: List[ShotCandidate] = []
    for cand in candidates:
        base_w = _geometry_weight_from_angle(cand.off_nadir_deg)
        dist = float(np.linalg.norm(np.array(cand.target_llh, dtype=float) - centroid)) / max_dist
        edge_boost = base_w * dist
        enriched.append(
            ShotCandidate(
                t_start=cand.t_start,
                q_BN=cand.q_BN,
                covered_cells=set(cand.covered_cells),
                target_llh=cand.target_llh,
                off_nadir_deg=cand.off_nadir_deg,
                geometry_weight=min(1.0, 0.75 * base_w + 0.25 * edge_boost),
                target_xy=cand.target_xy,
            )
        )
    return enriched


def _sequence_proxy(shots: Sequence[ShotCandidate], cell_count: int, T: float, integration_s: float) -> float:
    if not shots:
        return 0.0
    covered: Set[int] = set()
    total_slew = 0.0
    strip_motion = 0.0
    prev_q = IDENTITY_Q.copy()
    prev_xy: Optional[Tuple[float, float]] = None
    for shot in shots:
        covered |= shot.covered_cells
        total_slew += _quat_angle_deg(prev_q, shot.q_BN)
        if prev_xy is not None:
            dx = abs(shot.target_xy[0] - prev_xy[0]) / 90000.0
            dy = abs(shot.target_xy[1] - prev_xy[1]) / 90000.0
            strip_motion += 0.70 * dx + 0.12 * dy
        prev_q = shot.q_BN
        prev_xy = shot.target_xy
    coverage = len(covered) / max(cell_count, 1)
    span = (shots[-1].t_start + integration_s) - shots[0].t_start
    avg_w = sum(s.geometry_weight for s in shots) / len(shots)
    if avg_w < 0.35:
        # Easy geometry → prioritize coverage
        time_cost = 0.008 * span / max(T, 1e-9)
        shot_cost = 0.004 * len(shots)
        slew_cost = 0.0003 * total_slew
    else:
        # Harder geometry → keep penalties
        time_cost = (0.012 + 0.045 * avg_w) * span / max(T, 1e-9)
        shot_cost = (0.007 + 0.012 * avg_w) * len(shots)
        slew_cost = (0.0005 + 0.0008 * avg_w) * total_slew

    strip_cost = (0.018 + 0.035 * avg_w) * strip_motion

    return coverage - time_cost - shot_cost - slew_cost - strip_cost


def _select_shots_beam(
    candidates: Sequence[ShotCandidate],
    T: float,
    integration_s: float,
    cell_count: int,
    max_shots: int = MAX_SHOTS,
    min_new_cells: int = MIN_NEW_CELLS,
    beam_width: int = BEAM_WIDTH,
) -> List[ShotCandidate]:
    # State: (score, shots, covered, prev_t, prev_q, prev_xy, total_slew, strip_motion)
    states: List[Tuple[float, List[ShotCandidate], Set[int], float, np.ndarray, Optional[Tuple[float, float]], float, float]] = [
        (0.0, [], set(), -1.0, IDENTITY_Q.copy(), None, 0.0, 0.0)
    ]
    best: List[ShotCandidate] = []
    best_proxy = 0.0

    for cand in sorted(candidates, key=lambda c: c.t_start):
        next_states = states[:]
        for _score, shots, covered, prev_t, prev_q, prev_xy, total_slew, strip_motion in states:
            if len(shots) >= max_shots:
                continue
            if cand.t_start + integration_s + HOLD_PAD_S > T:
                continue
            if not _feasible_after(prev_t, prev_q, cand):
                continue
            new_cells = cand.covered_cells - covered
            if len(new_cells) < min_new_cells:
                continue

            angle = _quat_angle_deg(prev_q, cand.q_BN)
            available = cand.t_start - prev_t if prev_t >= 0.0 else cand.t_start
            slew_rate_proxy = angle / max(available, 1e-6)
            aggressive = max(0.0, slew_rate_proxy - 0.45)
            overlap = len(cand.covered_cells & covered)
            dx_cost = 0.0
            dy_cost = 0.0
            if prev_xy is not None:
                dx_cost = abs(cand.target_xy[0] - prev_xy[0]) / 90000.0
                dy_cost = abs(cand.target_xy[1] - prev_xy[1]) / 90000.0

            w = cand.geometry_weight
            time_norm = cand.t_start / max(T, 1e-9)
            edge_time_bias = w * (0.18 - abs(time_norm - 0.55))
            marginal = (
                len(new_cells)
                - 0.18 * overlap
                - (0.045 + 0.035 * w) * angle
                - (0.60 + 1.15 * w) * aggressive
                - (0.30 + 0.65 * w) * dx_cost
                - 0.10 * dy_cost
                - (0.14 + 0.35 * w)
                + edge_time_bias
            )
            if marginal <= -3.2:
                continue

            new_shots = shots + [cand]
            new_covered = covered | cand.covered_cells
            new_strip = strip_motion + 0.70 * dx_cost + 0.12 * dy_cost
            proxy = _sequence_proxy(new_shots, cell_count, T, integration_s)
            if proxy > best_proxy:
                best_proxy = proxy
                best = new_shots
            next_states.append(
                (
                    proxy,
                    new_shots,
                    new_covered,
                    cand.t_start,
                    cand.q_BN,
                    cand.target_xy,
                    total_slew + angle,
                    new_strip,
                )
            )

        next_states.sort(key=lambda s: (s[0], len(s[2])), reverse=True)
        states = next_states[:beam_width]

    return best


def plan_imaging(
    tle_line1: str,
    tle_line2: str,
    aoi_polygon_llh: List[Tuple[float, float]],
    pass_start_utc: str,
    pass_end_utc: str,
    sc_params: Dict[str, Any],
) -> Dict[str, Any]:
    states, T = _propagate_states(tle_line1, tle_line2, pass_start_utc, pass_end_utc, TIME_GRID_S)
    integration_s = float(sc_params["integration_s"])
    off_nadir_limit_deg = float(sc_params["off_nadir_max_deg"])
    fov_deg = sc_params["fov_deg"]

    verts = list(aoi_polygon_llh[:-1]) if len(aoi_polygon_llh) > 1 and aoi_polygon_llh[0] == aoi_polygon_llh[-1] else list(aoi_polygon_llh)
    lat0 = sum(p[0] for p in verts) / len(verts)
    lon0 = sum(p[1] for p in verts) / len(verts)
    proj = LocalProjection(lat0, lon0)
    target_points = _grid_points(verts, TARGET_GRID_N) + _grid_points(verts, TARGET_GRID_N_HARD)
    eval_points = _grid_points(verts, EVAL_GRID_N_HARD)
    candidates = _build_candidates(
        states,
        target_points,
        eval_points,
        proj,
        fov_deg,
        off_nadir_limit_deg,
        safe_margin_deg=0.8,
        per_target_limit=UNIFIED_CANDIDATES_PER_TARGET,
        min_time_separation_s=UNIFIED_MIN_TIME_SEPARATION_S,
    )
    candidates = _enrich_geometry_weights(candidates, (lat0, lon0))
    avg_weight = sum(c.geometry_weight for c in candidates) / max(len(candidates), 1)

    if avg_weight < 0.35:
        max_shots = MAX_SHOTS + 4
    else:
        max_shots = int(round(MAX_SHOTS + 2 - 5.0 * avg_weight))
    min_new_cells = 1 if avg_weight < 0.70 else 2
    greedy_shots = _select_shots(
        candidates,
        T,
        integration_s,
        max_shots=max(8, max_shots),
        min_new_cells=min_new_cells,
    )
    beam_shots = _select_shots_beam(
        candidates,
        T,
        integration_s,
        cell_count=EVAL_GRID_N_HARD * EVAL_GRID_N_HARD,
        max_shots=max(5, max_shots),
        min_new_cells=min_new_cells,
    )
    shots = beam_shots
    if _sequence_proxy(greedy_shots, EVAL_GRID_N_HARD * EVAL_GRID_N_HARD, T, integration_s) > _sequence_proxy(
        beam_shots, EVAL_GRID_N_HARD * EVAL_GRID_N_HARD, T, integration_s
    ):
        shots = greedy_shots
    attitude, shutter = _build_attitude_schedule(shots, T, integration_s)

    target_hints = [{"lat_deg": float(s.target_llh[0]), "lon_deg": float(s.target_llh[1])} for s in shots]

    return {
        "objective": "angle_weighted_generalized_planner",
        "attitude": attitude,
        "shutter": shutter,
        "notes": (
            f"angle-weighted stop-and-stare; {len(shots)} shots; "
            f"avg geometry weight {avg_weight:.3f}"
        ),
        "target_hints_llh": target_hints,
    }
