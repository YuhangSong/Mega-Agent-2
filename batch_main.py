'''build a tmux session to maintain
each python command_to_run in a window'''

import time
start_time = time.time()

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--env-names', type=str, nargs='*',)
parser.add_argument('--cards', type=str, nargs='*',)

args = parser.parse_args()

session_name = 'Mega-Agent-2-Batch'

# Mega
command_to_run = 'source activate Mega-Agent-2 && CUDA_VISIBLE_DEVICES=CARD python main.py --env-name GAME --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --train-with-reward in --intrinsic-reward-type latent --random-noise-frame --epsilon 5.0 --latent-control-intrinsic-reward-type delta_uG__NONE__relu__sum__clip_G__hold_uG --latent-control-discount 0.99 --num-grid 7 --G-skip 1 --aux 14 --vis --vis-interval 1 --log-interval 1 --eval-interval 200 --save-interval 500'

# Ex-PPO
command_to_run = 'source activate Mega-Agent-2 && CUDA_VISIBLE_DEVICES=CARD python main.py --env-name GAME --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01 --train-with-reward ex --intrinsic-reward-type latent --random-noise-frame --epsilon 5.0 --latent-control-intrinsic-reward-type delta_uG__NONE__relu__sum__clip_G__hold_uG --latent-control-discount 0.99 --num-grid 7 --G-skip 1 --aux 14 --vis --vis-interval 1 --log-interval 1 --eval-interval 200 --save-interval 500'

game_append = 'NoFrameskip-v4'

assert len(args.cards)==len(args.env_names)

import libtmux
server = libtmux.Server()

try:
    server.kill_session(session_name)
    print('Previous session {} is killed'.format(session_name))
except Exception as e:
    print('No previous session is killed')
session = server.new_session(session_name)

for i in range(len(args.cards)):
    '''build each command_to_run a window to maintain it'''
    pane = session.new_window(
        attach = False,
        window_name = '{}'.format(
            args.env_names[i],
        )
    ).list_panes()[0]

    '''send command to run command_to_run in the window's pane'''
    pane.send_keys(
        command_to_run.replace(
            'CARD',str(args.cards[i])
        ).replace(
            'GAME',(str(args.env_names[i])+game_append)
        )
    )

'''build a signal handler to kill all windows
session when the main thread is killed'''
import signal
import sys
def signal_handler(signal, frame):
    print()
    print('You pressed Ctrl+C! Now, kill the session {} after running for {:.2f} hours'.format(
        session_name,
        (time.time()-start_time)/60.0/60.0,
    ))
    session.kill_session()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while True:
    input('Session has been running for {:.2f} hours. Running {} on cards {}. Kill this thread with Ctrl+C to kill the session.'.format(
        (time.time()-start_time)/60.0/60.0,
        args.env_names,
        args.cards,
    ))
