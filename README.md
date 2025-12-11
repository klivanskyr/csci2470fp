TD-MPC builds on established methods in reinforcement learning and control. It lever-
ages Model Predictive Path Integral (MPPI) control [3] to generate and evaluate multi-
ple potential action sequences over a short time horizon, selecting those that maximize
expected reward. This sampling-based planning approach allows TD-MPC to handle
continuous action spaces more efficiently. It also incorporates Double-Q Learning [2]
to stabilize value estimation, reducing the overestimation bias that can occur when
learning the expected return of actions.

# How to run TDMPC Inference
## conda environment setup
```bash
conda create -n tdmpc python=3.8 -y
conda activate tdmpc
pip install -r requirements.txt
```

## Running Inference
```bash
python inferance.py --weights weights/tdmpc_step_49000.pth
```
