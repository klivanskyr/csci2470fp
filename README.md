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



5.4 Inference: The Planning LoopAt test time, the agent executes the following loop:

Observe: Get state $s_t$.

Encode: Compute $z_t = h_\theta(s_t)$.

Optimize (MPPI):Sample 512 action trajectories using Gaussian noise + the Policy Prior $\pi_\theta(z)$.Simulate forward in latent space using $d_\theta$.Compute rewards $R_\theta$ for steps $t$ to $t+H$.Compute terminal value $Q_\theta$ at step $t+H$.Compute trajectory scores (sum of rewards + terminal value).Update the mean action sequence using MPPI weights.Repeat for $N_{iter}$ iterations (e.g., 6 iterations).

Act: Execute the first action of the final mean sequence.