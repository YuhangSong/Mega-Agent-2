'''build a tmux session to maintain
each python command_to_run in a window'''

import time
start_time = time.time()

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--env-names', type=str, nargs='*',)
parser.add_argument('--cards', type=str, nargs='*',)
parser.add_argument('--agent-name', type=str,)
parser.add_argument('--command', type=str,)

args = parser.parse_args()

session_name = 'Mega-Agent-2-Batch'
command_to_run = 'source activate Mega-Agent-2 && CUDA_VISIBLE_DEVICES=CARD python main.py --env-name GAME '
game_append = 'NoFrameskip-v4'

if args.agent_name is None:
    args.agent_name = input('# ACTION REQUIRED: Name the agent:')
print('# INFO: Name of the agent: {}'.format(args.agent_name))

if args.command is None:
    args.command = input('# ACTION REQUIRED: Provide command:')
print('# INFO: Command is: {}'.format(args.command))
command_to_run += args.command

assert len(args.cards)==len(args.env_names)

import libtmux
server = libtmux.Server()

try:
    server.kill_session(session_name)
    print('# INFO: Previous session {} is killed'.format(session_name))
except Exception as e:
    print('# INFO: No previous session is killed')
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
    print('# WARNING: You pressed Ctrl+C! Now, kill the session {} after running for {:.2f} hours'.format(
        session_name,
        (time.time()-start_time)/60.0/60.0,
    ))
    session.kill_session()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while True:
    input('# INFO: Up for {:.2f} hours. Running {} agents over {} on cards {}. Kill this thread with Ctrl+C to kill the session.'.format(
        (time.time()-start_time)/60.0/60.0,
        args.agent_name,
        args.env_names,
        args.cards,
    ))
