1. The Outer Loop (Real World Interaction)Action: The Agent interacts with the Real MuJoCo Environment.Purpose: To collect ground truth data.Result: The Replay Buffer fills up with real experiences: $(s, a, r, s')$.Note: In the very beginning, the Agent is dumb/random, so this data is just "random flailing," which is fine.

2. Train the World ModelAction: Pause the real simulation. Sample a batch from the Replay Buffer.Input: Current State + Action Taken.Target: Next State + Reward Received.Math: Minimize Regression Loss (MSE).Result: The World Model learns: "When the arm is here and you apply this torque, it moves there."

3. The Inner Loop (Agent Training / Dreaming)Action: The Agent enters the "Matrix" (the World Model).Process:Agent sees state $\rightarrow$ Picks Action.World Model sees State + Action $\rightarrow$ Predicts Next State + Reward.Update Agent: The Agent adjusts its weights to maximize the rewards it is getting in this dream.Crucial Detail: This loop happens thousands of times in the time it takes to do 1 step in the real world. This is where the speed gain comes from.

4. Repeat (Iterate)Now that the Agent is smarter (from dreaming), put it back in the Real World.It will now explore new areas (e.g., actually reaching the target).This generates new data (e.g., "what happens when I actually touch the target?").Go back to Step 2 and update the World Model with this new knowledge.






5.4 Inference: The Planning LoopAt test time, the agent executes the following loop:

Observe: Get state $s_t$.

Encode: Compute $z_t = h_\theta(s_t)$.

Optimize (MPPI):Sample 512 action trajectories using Gaussian noise + the Policy Prior $\pi_\theta(z)$.Simulate forward in latent space using $d_\theta$.Compute rewards $R_\theta$ for steps $t$ to $t+H$.Compute terminal value $Q_\theta$ at step $t+H$.Compute trajectory scores (sum of rewards + terminal value).Update the mean action sequence using MPPI weights.Repeat for $N_{iter}$ iterations (e.g., 6 iterations).

Act: Execute the first action of the final mean sequence.