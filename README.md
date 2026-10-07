# CASE

CASE (**C**ost-**A**ware **S**topping for **E**fficient Long-Video Agents) is a
termination controller for tool-guided long-video agents. At each causal
checkpoint, CASE decides whether to stop with the evidence collected so far or
continue the host agent's native search. The host remains responsible for which
evidence to acquire.

## Installation

```bash
python -m pip install -e .
```

The core package depends only on NumPy. Online execution uses an
OpenAI-compatible multimodal endpoint. Set `CASE_API_KEY` only when the endpoint
requires bearer authentication.

## Quick start

The repository includes a synthetic trajectory and six frozen controllers for
VideoSeek and AVP.

```python
import json
from pathlib import Path

from case_stopping import apply_policy, load_policy

root = Path(".")
trajectory = json.loads((root / "examples/example_trajectory.json").read_text())
policy = load_policy(
    root / "configs/frozen_policies.json",
    host="videoseek",
    model="qwen35",
)
decision = apply_policy([trajectory], policy)[0]
print(decision)
```

The same example is available as:

```bash
python examples/demo.py
```

## Repository layout

```text
case_stopping/
  core.py          # state features, decision-gap target, weighted Ridge
  fit.py           # probability calibration and controller fitting
  replay.py        # offline application, evaluation, operating-point search
  online.py        # online STOP/CONTINUE controller
  integrations.py  # VideoSeek / AVP integration helpers
configs/
  default.json          # default CASE hyperparameters
  videomme_split.json   # video-disjoint Video-MME split
  frozen_policies.json  # six fitted VideoSeek/AVP controllers
examples/
  demo.py
  example_trajectory.json
```

## Trajectory format

A trajectory stores the native terminal prediction and cost together with its
causal decision checkpoints. The last checkpoint represents the native terminal
state; CASE evaluates STOP/CONTINUE only at the preceding checkpoints.

Common trajectory fields are:

- `sample_id`, `video_id`, `duration_group`;
- `video_duration_sec`, `has_subtitle`;
- `native_prediction`, `native_cost`;
- `checkpoints`.

Each checkpoint contains:

- `latest_tool` and `evidence_char_count`;
- `consumed_cost`: native cost accumulated before the current stop check;
- `decision_sunk_cost`: native cost already unavoidable at that decision point;
- `probe_cost`;
- `calibrated_distribution` for frozen-policy application.

Training and calibration additionally use `gold`, per-checkpoint `probe_logits`,
`finalizer_prediction`, and `finalizer_cost`. These development-only fields are
not read by `apply_policy`.

## Fitting and selection

The fitting API mirrors the method pipeline:

```python
from case_stopping import fit_models, fit_probe_temperatures, search_operating_points
from case_stopping.fit import calibrate_trajectories

probe_temperatures = fit_probe_temperatures(train_trajectories)
train = calibrate_trajectories(train_trajectories, probe_temperatures)
models = fit_models(train)

calibration = calibrate_trajectories(calibration_trajectories, probe_temperatures)
selection = search_operating_points(calibration, models)
```

`configs/default.json` records the default cost-weight grid, decision-boundary
grid, Ridge regularization, activation rule, and calibration selection rule.
`configs/videomme_split.json` records the fixed 50/100/750 video-disjoint
Video-MME split.

## Online integration

`OnlineController` is called after the host planner proposes its next visual
action and before that action executes. A CONTINUE decision leaves the native
loop unchanged. A STOP decision invokes the same underlying VLM with a
choice-constrained native-prefix finalizer.

`case_stopping.integrations` contains thin VideoSeek and AVP hooks. GPU IDs,
service ports, local paths, and other deployment-specific values are not part of
the package.

## Data and model assets

Benchmark media and model weights are not redistributed. The repository contains
only dataset identifiers, method configuration, and the fitted CASE controller
parameters needed to load the included policies.
