# Mega-Agent-2

## Please use hyper parameters from this readme. With other hyper parameters things might not work (it's RL after all)!

This is a PyTorch implementation of
* Advantage Actor Critic (A2C), a synchronous deterministic version of [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* Proximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf)
* Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation [ACKTR](https://arxiv.org/abs/1708.05144)

Also see the OpenAI posts: [A2C/ACKTR](https://blog.openai.com/baselines-acktr-a2c/) and [PPO](https://blog.openai.com/openai-baselines-ppo/) for more information.

This implementation is inspired by the OpenAI baselines for [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c), [ACKTR](https://github.com/openai/baselines/tree/master/baselines/acktr) and [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1). It uses the same hyper parameters and the model since they were well tuned for Atari games.

Please use this bibtex if you want to cite this repository in your publications:

    @misc{pytorchrl,
      author = {Kostrikov, Ilya},
      title = {PyTorch Implementations of Reinforcement Learning Algorithms},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr}},
    }

## Supported (and tested) environments (via [OpenAI Gym](https://gym.openai.com))
* [Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
* [MuJoCo](http://mujoco.org)
* [PyBullet](http://pybullet.org) (including Racecar, Minitaur and Kuka)
* [DeepMind Control Suite](https://github.com/deepmind/dm_control) (via [dm_control2gym](https://github.com/martinseilair/dm_control2gym))

I highly recommend PyBullet as a free open source alternative to MuJoCo for continuous control tasks.

All environments are operated using exactly the same Gym interface. See their documentations for a comprehensive list.

To use the DeepMind Control Suite environments, set the flag `--env-name dm.<domain_name>.<task_name>`, where `domain_name` and `task_name` are the name of a domain (e.g. `hopper`) and a task within that domain (e.g. `stand`) from the DeepMind Control Suite. Refer to their repo and their [tech report](https://arxiv.org/abs/1801.00690) for a full list of available domains and tasks. Other than setting the task, the API for interacting with the environment is exactly the same as for all the Gym environments thanks to [dm_control2gym](https://github.com/martinseilair/dm_control2gym).

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [Visdom](https://github.com/facebookresearch/visdom)
* [TensorFlow](https://www.tensorflow.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# For users behind the Great Wall
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
pip install pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Create a virtual environment
conda create -n Mega-Agent-2 python=3.6.7 -y
source activate Mega-Agent-2

# PyTorch
pip install --upgrade torch torchvision

# TensorFlow
pip install --upgrade tensorflow

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Clone code
mkdir Mega-Agent-2
cd Mega-Agent-2
git clone https://github.com/YuhangSong/Mega-Agent-2.git
cd Mega-Agent-2

# Other requirements
pip install -r requirements.txt

# Create a session to play
tmux new-session -s Mega-Agent-2
```

## Training

The code log multiple curves to help analysis the training process, run:
```
source activate Mega-Agent-2 && tensorboard --logdir ../results/ --port 6010
```
and visit ```http://localhost:6010``` for visualization with tensorboard.

### Atari

#### PPO

```bash
source activate Mega-Agent-2 && CUDA_VISIBLE_DEVICES=1 python main.py --env-name "BreakoutNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --norm-rew --vis --vis-interval 1 --log-interval 1
```

## Contributions

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also see a todo list below.

## Disclaimer

It's maybe difficult to reproduce results for Reinforcement Learning methods. See ["Deep Reinforcement Learning that Matters"](https://arxiv.org/abs/1709.06560) for more information.

### TODO
* x

## Training

Start a `Visdom` server with `python -m visdom.server`, it will serve `http://localhost:8097/` by default.

### Atari
#### A2C

```bash
python main.py --env-name "PongNoFrameskip-v4"
```

#### PPO

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01
```

#### ACKTR

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20
```

### MuJoCo

I **highly** recommend to use --add-timestep argument with some mujoco environments (for example, Reacher) despite it's not a default option with OpenAI implementations.

#### A2C

```bash
python main.py --env-name "Reacher-v2" --num-env-steps 1000000
```

#### PPO

```bash
python main.py --env-name "Reacher-v2" --algo ppo --use-gae --vis-interval 1  --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-env-steps 1000000 --use-linear-lr-decay
```

#### ACKTR

ACKTR requires some modifications to be made specifically for MuJoCo. But at the moment, I want to keep this code as unified as possible. Thus, I'm going for better ways to integrate it into the codebase.

## Enjoy

Load a pretrained model from [my Google Drive](https://drive.google.com/open?id=0Bw49qC_cgohKS3k2OWpyMWdzYkk).

Disclaimer: I might have used different hyper-parameters to train these models.

## Results
