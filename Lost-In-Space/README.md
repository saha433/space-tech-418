# Lost in Space Track — Hackathon

[![Problem Statement](https://img.shields.io/badge/problem%20statement-read-blue)](./docs/PROBLEM_STATEMENT.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Autonomous Earth-observation scheduling on a small satellite. Teams write a Python function that plans attitude slews and shutter windows for a single LEO pass over a fixed ground target, respecting reaction-wheel momentum limits, body-rate smear constraints, and a 60° off-nadir pointing envelope. Submissions are scored automatically against three pass geometries (0°, 30°, 60° off-nadir) using a Basilisk 6-DoF simulation on the organizers' side.

## For participants

1. **Read the problem statement** → [`docs/PROBLEM_STATEMENT.md`](./docs/PROBLEM_STATEMENT.md) (PDF also available)
2. **Download the testing kit** → clone this repo or grab the latest [release](../../releases)
3. **Start from a stub** → copy `teams_kit/example_submissions/stop_and_stare.py` and edit
4. **Score locally** →
   ```bash
   cd teams_kit/
   pip install -r requirements.txt
   python test_my_submission.py my_submission.py
   ```
5. **Submit** → upload your single `.py` file via the official [Google Form](https://drive.google.com/file/d/1noVG_VyGAGF2qNE6kYSnzWupmZ9SvK9D/view?usp=sharing)

You will ultimately deliver:

- **One Python file** exporting `plan_imaging(...)` per Section 7 of the problem statement
- **A 2-minute presentation** explaining your strategy

Presentations are judged first; top 3 submissions are then verified against the real Basilisk harness.

## Repository layout

```
.
├── docs/
│   ├── PROBLEM_STATEMENT.md       # browse it on GitHub
│   ├── PROBLEM_STATEMENT.pdf      # canonical print version
│   └── PROBLEM_STATEMENT.docx     # editable source
├── teams_kit/                     # what participants use
│   ├── README.md                  # team-facing quick start
│   ├── requirements.txt
│   ├── test_my_submission.py      # single-command local scorer
│   ├── basilisk_harness/          # validator + scorer + mock sim
│   ├── configs/                   # the 3 test cases
│   └── example_submissions/       # stub + negative example + baseline
├── organizer_harness/             # what WE run to grade (requires Basilisk)
│   ├── README.md
│   ├── requirements.txt
│   ├── run_evaluation.py          # CLI grader
│   ├── calibrate_tles.py          # one-off helper for new AOIs
│   ├── basilisk_harness/          # includes basilisk_sim.py
│   ├── configs/
│   ├── example_submissions/
│   └── tests/
├── LICENSE                        # MIT
└── README.md                      # you are here
```



## Quick links

- [Problem statement (markdown)](./docs/PROBLEM_STATEMENT.md)
- [Problem statement (PDF)](./docs/PROBLEM_STATEMENT.pdf)
- [Teams' testing kit](./teams_kit/)
- [Submission Google Form](https://drive.google.com/file/d/1noVG_VyGAGF2qNE6kYSnzWupmZ9SvK9D/view?usp=sharing)
- [Organizer harness](./organizer_harness/) — the Basilisk-backed grader (optional read)

## About the harness

The organizer-side grader is open-sourced alongside the problem statement for full transparency. You can read exactly how your submission will be evaluated, what the scoring gates look like, and how frames are counted. The only thing teams cannot run locally is the Basilisk 6-DoF simulation itself (`organizer_harness/basilisk_harness/basilisk_sim.py` requires [AVS Lab's Basilisk](https://hanspeterschaub.info/basilisk)) — but the mock simulator in the teams' kit uses the identical validator, scorer, and gate logic, so local scores are honest directional indicators.

## Credits

Built by *TakeMe2Space* for *Aeon 418*, *2026*. Basilisk Astrodynamics Framework © University of Colorado AVS Lab.

Licensed under [MIT](./LICENSE).

Have a question or a bug to report? Let us know here: https://forms.gle/o7vEkvp8LeeXEPQn9
