# Lost in Space: EO Attitude & Imaging Scheduler

## Submission File

The final file to evaluate is:

```text
space-tech/Space Tech Aeon/optimised_submission_final.py
```

- `optimised_submission_final.py`  
  Final single-file Python submission. It exports the required `plan_imaging(...)` function and uses only the permitted dependencies: standard library, `numpy`, and `sgp4`.

## One-Line Strategy

This solution uses a **coverage-first stop-and-stare beam search**: it samples possible AOI pointings over the pass, projects each image footprint, and selects a compact mosaic of stable stare images that maximizes new AOI coverage.

## Problem Summary

The spacecraft carries a fixed imager on the body `+Z` axis. To image a ground target, the spacecraft body must slew so that `+Z` points at the target.

The planner must produce:

- a time-tagged attitude trajectory, using scalar-last body-to-inertial quaternions,
- a list of 120 ms shutter windows,
- a schedule that respects the key imaging constraints.

The automated score rewards:

- AOI coverage,
- low control effort,
- low active time,
- no smear violations.

## Core Idea

A simple nadir-tracking strategy is risky because the spacecraft continues rotating while the shutter is open. That can violate the smear constraint.

This submission instead uses a **stop-and-stare** pattern:

1. Slew between imaging opportunities.
2. Hold one inertial attitude before and during each exposure.
3. Fire the shutter only while the commanded attitude is fixed.
4. Move to the next selected target.

This keeps the commanded body rate near zero during each 120 ms integration window.

## Planner Pipeline

### 1. Propagate the Orbit

The planner uses the supplied TLE and SGP4 to sample the satellite state throughout the pass:

```python
Satrec.twoline2rv(tle_line1, tle_line2)
```

Each sampled state stores:

- time since pass start,
- ECI position,
- ECI velocity,
- GMST angle for Earth rotation.

### 2. Discretize the AOI

The AOI is converted into grid points. The code uses both a coarse and a dense grid:

```python
TARGET_GRID_N = 10
TARGET_GRID_N_HARD = 16
EVAL_GRID_N_HARD = 21
```

There are two grid roles:

- **Target grid:** possible camera aim points.
- **Evaluation grid:** approximate AOI cells used to estimate how much area a footprint covers.

The original first-order estimate is that a 100 km by 100 km AOI and a ~17 km nadir footprint imply a rough 6 by 6 raster. This planner improves on that by using a denser candidate grid and selecting only the highest-value frames instead of imaging every raster cell.

### 3. Generate Candidate Shots

For each candidate time and target point, the planner:

- converts target latitude/longitude to ECEF,
- rotates ECEF to ECI,
- computes the line of sight from spacecraft to target,
- checks the off-nadir angle,
- computes the body-to-inertial quaternion that points body `+Z` at the target,
- projects the camera footprint to the WGS-84 ellipsoid,
- counts how many AOI evaluation cells are covered.

Candidates outside the off-nadir limit are discarded.

### 4. Select a Mosaic With Beam Search

The planner uses beam search rather than a simple greedy raster.

Beam search keeps multiple partial shot sequences alive, then scores them using:

- new AOI cells covered,
- overlap with already-covered cells,
- slew angle between frames,
- time span of the selected sequence,
- geometry difficulty for high off-nadir cases.

This allows the planner to find a compact set of frames that covers most of the AOI while avoiding unnecessary images.

### 5. Build the Attitude and Shutter Schedule

For each selected shot, the output schedule contains:

- a pre-hold attitude sample,
- the same quaternion at shutter start,
- the same quaternion at shutter end,
- a post-hold sample.

This is what enforces stop-and-stare behavior.

## Constraint Handling

### Smear Constraint

Requirement:

```text
|omega_body| <= 0.05 deg/s during each 120 ms exposure
```

Planner response:

- No continuous tracking during exposure.
- The same quaternion is held across each shutter window.
- Slews are only performed between shutter windows.

### Off-Nadir Constraint

Requirement:

```text
off-nadir <= 60 deg
```

Planner response:

```python
SAFE_OFF_NADIR_MARGIN_DEG = 1.5
```

Candidate shots near or above the limit are rejected before sequence selection.

### Wheel / Control-Effort Awareness

The submission does not directly simulate a full reaction-wheel controller inside `plan_imaging`, but it reduces control risk by:

- limiting the number of selected frames,
- penalizing large slew angles,
- spacing shots apart,
- avoiding unnecessary overlap.

Important constants:

```python
HOLD_PAD_S = 0.50
MAX_SLEW_RATE_DPS = 2.0
MIN_SHOT_SPACING_S = 3.0
MAX_SHOTS = 8
BEAM_WIDTH = 72
```

## Case-3 Awareness

Case 3 has the largest score weight:

```text
case3 weight = 40%
```

It is also the hardest geometry because the AOI is near the off-nadir limit. Some regions are physically harder to image, but the projected footprint also stretches at high off-nadir.

The planner adapts using:

- denser target/evaluation grids,
- geometry weighting based on off-nadir angle,
- more selected frames when useful.

This avoids using one fixed raster pattern for all passes.

## Local Mock-Harness Result

Command:

```bash
python3 Lost-In-Space/teams_kit/test_my_submission.py \
  "space-tech/Space Tech Aeon/optimised_submission_final.py"
```

Observed local mock result:

```text
S_total = 1.2668

case1:
S_orbit = 1.2896
C       = 0.9988
frames  = 6 kept / 6 attempted

case2:
S_orbit = 1.2646
C       = 0.9760
frames  = 5 kept / 5 attempted

case3:
S_orbit = 1.2545
C       = 0.9811
frames  = 8 kept / 8 attempted

Q_smear = 1.0000 for all cases
```

## Notes on Mock vs. Basilisk

The provided mock simulator assumes ideal attitude tracking. A real Basilisk simulation may include controller lag, overshoot, and transient wheel dynamics.

This submission is therefore best described as:

> A mock-harness optimized stop-and-stare coverage planner with explicit smear and off-nadir reasoning.

The local mock score should be cited honestly as the mock-harness score, not as a guaranteed Basilisk score.

## How to Run

From the repository root:

```bash
cd /Users/saha/Downloads/hack_TM2SxAeon418-main
```

Run the provided mock tester:

```bash
python3 Lost-In-Space/teams_kit/test_my_submission.py \
  "space-tech/Space Tech Aeon/optimised_submission_final.py"
```



## Submission Checklist

- Single Python file.
- Exports `plan_imaging(...)` at module top level.
- Uses only standard library, `numpy`, and `sgp4`.
- Deterministic: no randomness, no file reads, no network calls.
- Returns `objective`, `attitude`, `shutter`, `notes`, and `target_hints_llh`.
- Shutter windows are exactly `0.120 s`.
- Attitude quaternions are scalar-last `[qx, qy, qz, qw]`.

