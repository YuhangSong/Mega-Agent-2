<!-- TOC -->

- [Mega-Reward](#mega-reward)
  - [Requirements](#requirements)
  - [Training](#training)
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

```bash
conda activate Mega-Agent-2 && CUDA_VISIBLE_DEVICES=0 python main.py --env-name "BreakoutNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --train-with-reward in --intrinsic-reward-type latent --random-noise-frame --epsilon 5.0 --latent-control-intrinsic-reward-type delta_uG__NONE__relu__sum__clip_G__hold_uG --latent-control-discount 0.99 --num-grid 4 --G-skip 1 --aux 14 --vis --vis-interval 1 --log-interval 1 --eval-interval 200 --save-interval 500
```

## Visualization

The code log multiple curves to help analysis the training process, run:
```
conda activate Mega-Agent-2 && tensorboard --logdir ../results/ --port 6010
```
and visit ```http://localhost:6010``` for visualization with tensorboard.

If your port is blocked, use natapp to forward a port.

You should see the ex_raw increases, indicating that it has learned to play the game without access to the extrinsic reward.
