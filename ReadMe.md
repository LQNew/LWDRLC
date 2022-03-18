## LWDRLC: LightWeight Deep Reinforcement Learning libraray for Continous control
LWDRLC is a deep reinforcement learning (RL) library which is inspired by some other deep RL code bases (i.e., [Spinning Up repository](https://github.com/openai/spinningup), [Stable-baselines3
](https://github.com/DLR-RM/stable-baselines3), [Fujimoto TD3 repository](https://github.com/sfujim/TD3), and [Tonic repository](https://github.com/fabiopardo/tonic)).

### :rocket: Beyond State-Of-The-Art
LWDRL provides further tricks to improve performance of state-of-the-art algorithms potentially beyond their original papers. Therefore, LWDRL enables every user to achieve professional-level performance just in a few lines of codes.

### Supported algorithms
| algorithm | continuous control | on-policy / off-policy |
|:-|:-:|:-:|
| Vanilla Policy Gradient (VPG) | :white_check_mark: | *on-policy*|
| [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) | :white_check_mark: | *on-policy* | 
| [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | :white_check_mark: | *off-policy* |
| [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/abs/1802.09477) | :white_check_mark: | *off-policy* |
| [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905) | :white_check_mark: |*off-policy* | 

## Instructions
### Recommend: Run with Docker
```bash
# python        3.6    (apt)
# pytorch       1.4.0  (pip)
# tensorflow    1.14.0 (pip)
# DMC Control Suite and MuJoCo
cd dockerfiles
docker build . -t lwdrl
```
For other dockerfiles, you can go to [RL Dockefiles](https://github.com/LQNew/Dockerfiles).

### Launch experiments
Run with the scripts `batch_off_policy_mujoco_cuda.sh` / `batch_off_policy_dmc_cuda.sh` / `batch_on_policy_mujoco_cuda.sh` / `batch_on_policy_dmc_cuda.sh`:
```bash
# eg.
bash batch_off_policy_mujoco_cuda.sh Hopper-v2 TD3 0  # env_name: Hopper-v2, algorithm: TD3, CUDA_Num : 0
```

### Plot results
```bash
# eg. Notice: `-l` denotes labels, `data/DDPG-Hopper-v2/` represents the collecting dataset, 
# and `-s` represents smoothing value.
python spinupUtils/plot.py \
    data/DDPG-Hopper-v2/ \
    -l DDPG -s 10
```

### Visualization of the environments
Run with the scripts `render_dmc.py` / `render_mujoco.py`:
```bash
# eg.
python render_dmc.py --env swimmer-swimmer6  # env_name: swimmer-swimmer6
```
### Performance on MuJoCo
Including `Ant-v2`, `HalfCheetah-v2`, `Hopper-v2`, `Humanoid-v2`, `Swimmer-v2`, `Walker2d-v2`.
<img src="images/QingLi-MuJoCo.png" width="1000" align="middle"/>
<br>

### Citation
```bash
@misc{QingLi2021lwdrl,
  author = {Qing Li},
  title = {LWDRL: LightWeight Deep Reinforcement Learning Library},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LQNew/LWDRL}}
}
```
