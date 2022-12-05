## Requirements and Installation
The repo was written using *Python 3.7* with [`conda`](https://github.com/JacopoPan/a-minimalist-guide#install-conda) on *macOS 10.15* and tested with *Python 3.8* on *macOS 12*, *Ubuntu 20.04*




### On *macOS* and *Ubuntu*
Major dependencies are [`gym`](https://gym.openai.com/docs/),  [`pybullet`](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#), 
[`stable-baselines3`](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html), and [`rllib`](https://docs.ray.io/en/master/rllib.html)

Video recording requires to have [`ffmpeg`](https://ffmpeg.org) installed, on *macOS*
```bash
$ brew install ffmpeg
```
On *Ubuntu*
```bash
$ sudo apt install ffmpeg
```

*macOS* with Apple Silicon (like the M1 Air) can only install grpc with a minimum Python version of 3.9 and these two environment variables set:
```bash
$ export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
$ export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
```

The repo is structured as a [Gym Environment](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
and can be installed with `pip install --editable`
```
$ conda create -n drones python=3.8 # or 3.9 on Apple Silicon, see the comment on grpc above
$ conda activate drones
$ pip3 install --upgrade pip
$ git clone https://github.com/utiasDSL/gym-pybullet-drones.git
$ cd gym-pybullet-drones/
$ pip3 install -e .
```
<!--
On Ubuntu and with a GPU available, optionally uncomment [line 203](https://github.com/utiasDSL/gym-pybullet-drones/blob/fab619b119e7deb6079a292a04be04d37249d08c/gym_pybullet_drones/envs/BaseAviary.py#L203) of `BaseAviary.py` to use the [`eglPlugin`](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.778da594xyte)
-->

## Examples
There are 2 basic template scripts in [`gym_pybullet_drones/`](https://github.com/utiasDSL/gym-pybullet-drones/tree/master/gym_pybullet_drones/examples): `singleagent.py` and `test_singleagent.py`

Run `singleagent.py` to train a reinforcement learning algorithm to train a drone to hover and the model is saved in [`gym_pybullet_drones/results`].
To evaluate the trained model run `test_singleagent.py`.