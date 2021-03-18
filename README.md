<!-- TOC -->

- [Mega-Reward](#mega-reward)
  - [Requirements](#requirements)
  - [Training](#training)
    - [Mega](#mega)
    - [Ex-PPO](#ex-ppo)
  - [Visualization](#visualization)

<!-- /TOC -->

# Mega-Reward

## Requirements

In order to install requirements, follow:

```bash
# For users behind the Great Wall only
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
pip install pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Create a virtual environment
conda create -n Mega-Agent-2 python=3.6.7 -y
conda activate Mega-Agent-2

# PyTorch
pip install --upgrade torch torchvision

# TensorFlow
pip install --upgrade tensorflow

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..

# Clone code
mkdir Mega-Agent-2
cd Mega-Agent-2
git clone https://github.com/YuhangSong/Mega-Agent-2.git
cd Mega-Agent-2

# Other requirements
pip install -r requirements.txt
```

## Training

<!-- ### Atari

#### PPO

### Run a Batch of Games

```bash
conda activate Mega-Agent-2 && python batch_main.py --env-names Pong Breakout --cards 0 1
```

<!-- conda activate Mega-Agent-2 && python batch_main.py --env-names StarGunner Tennis UpNDown Venture VideoPinball WizardOfWor Gopher Gravitar --cards 0 1 2 3 0 1 2 3 -->

<!-- conda activate Mega-Agent-2 && python batch_main.py --env-names Pong Breakout Alien Centipede FishingDerby --cards 0 1 2 3 -->

<!-- H4n
```bash
conda activate Mega-Agent-2 && python batch_main.py --env-names IceHockey Jamesbond TimePilot Tutankham --cards 0 1 0 1
conda activate Mega-Agent-2 && python batch_main.py --env-names Freeway Frostbite MontezumaRevenge --cards 0 1
```

W4n
```bash
conda activate Mega-Agent-2 && python batch_main.py --env-names AirRaid DoubleDunk ElevatorAction Enduro JourneyEscape Kangaroo Krull KungFuMaster --cards 0 1 2 3 0 1 2 3
```

Wl0
```bash
conda activate Mega-Agent-2 && python batch_main.py --env-names MsPacman NameThisGame Phoenix Pitfall Pooyan PrivateEye Qbert Riverraid Amidar Assault Asterix Asteroids Atlantis BankHeist BattleZone BeamRider --cards 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
conda activate Mega-Agent-2 && python batch_main.py --env-names YarsRevenge Zaxxon --cards 2 3

```

T0
```bash
conda activate Mega-Agent-2 && python batch_main.py --env-names Berzerk Bowling Boxing Carnival RoadRunner Robotank Seaquest Skiing --cards 0 1 2 3 0 1 2 3
```

W5n
```bash
conda activate Mega-Agent-2 && python batch_main.py --env-names ChopperCommand --cards 0 1 2 3
```

Wx0
```bash
conda activate Mega-Agent-2 && python batch_main.py --env-names CrazyClimber DemonAttack Solaris SpaceInvaders --cards 0 1 0 1
``` -->

### Mega
```bash
conda activate Mega-Agent-2 && CUDA_VISIBLE_DEVICES=1 python main.py --env-name BreakoutNoFrameskip-v4 --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --train-with-reward in --intrinsic-reward-type latent --random-noise-frame --epsilon 5.0 --latent-control-intrinsic-reward-type G__NONE__relu__hcb__clip_G__hold_uG --hash-type hard --hard-hash-m 4 --norm-rew --latent-control-discount 0.99 --num-grid 4 --G-skip 1 --aux 23 --vis --vis-interval 1 --log-interval 1 --eval-interval 200 --save-interval 500
```

### Ex-PPO
```bash
conda activate Mega-Agent-2 && CUDA_VISIBLE_DEVICES=0 python main.py --env-name BreakoutNoFrameskip-v4 --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --train-with-reward ex --intrinsic-reward-type latent --random-noise-frame --epsilon 5.0 --latent-control-intrinsic-reward-type delta_uG__NONE__relu__sum__clip_G__hold_uG --latent-control-discount 0.99 --num-grid 4 --G-skip 1 --aux 22 --vis --vis-interval 1 --log-interval 1 --eval-interval 200 --save-interval 500
```

## Visualization

The code log multiple curves to help analysis the training process, run:
```
conda activate Mega-Agent-2 && tensorboard --logdir ../results/ --port 6010
```
and visit ```http://localhost:6010``` for visualization with tensorboard.

If your port is blocked, use natapp to forward a port.

You should see the reward of Mega increases, indicating that it has learned to play the game without access to the extrinsic reward.

<!-- ## Contributions

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also see a todo list below.

## Disclaimer

It's maybe difficult to reproduce results for Reinforcement Learning methods. See ["Deep Reinforcement Learning that Matters"](https://arxiv.org/abs/1709.06560) for more information.

### TODO
* x -->

<!-- ## Training

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

## Results -->

<!-- ## Visualization

The code log multiple curves to help analysis the training process, run:
```
conda activate Arena && tensorboard --logdir ../results/ --port 8888
```
and visit ```http://localhost:4253``` for visualization with tensorboard.

If your port is blocked, use natapp to forward a port:
```
./natapp --authtoken 710a6e3d5b6c23a5
``` -->
