

import sys, os
from itertools import chain
from omnibelt import load_yaml, save_yaml, create_dir

from tqdm import tqdm

import omnifig as fig


@fig.Script('diplo-traj', description='Resolve a sequence of actions from an initial state')
def diplomacy_traj(A):
	'''
	Given a set of actions for all season (eg. parsed from a full order log), this script computes the game
	state after applying the actions for each season.
	
	If you have a sequence of consecutive action files (eg. parsed from a web-diplomacy or vdiplomacy order log),
	then you can compute all the states with this script. As input the directory for all actions `action-dir`
	must be provided, and as output a path to a directory `state-dir` for all states must be provided.
	For the rules (including `nodes`, `edges`, and `players`), you can either provide a path to each file
	individually or a `root` path to a directory containing to all three.
	'''
	
	silent = A.pull('silent', False, silent=True)
	
	state_dir = A.pull('state-dir')
	if not os.path.isdir(state_dir):
		create_dir(state_dir)
	assert os.path.isdir(state_dir), f'Invalid dir: {state_dir}'
	
	action_dir = A.pull('action-dir')
	assert os.path.isdir(action_dir), f'Invalid dir: {action_dir}'

	available = set(os.listdir(action_dir))
	pbar = tqdm(total=len(available)) if A.pull('pbar', True) else None
	
	state_files = os.listdir(state_dir)
	
	states = {}
	
	for fname in state_files:
		try:
			state = load_yaml(os.path.join(state_dir, fname))
			states[state['time']['turn'], state['time']['season'] + 0.5*('retreat' in state['time'])] = state
		except:
			pass
	
	if not len(states):
		print(f'Starting a new game in {state_dir}')
		A.push('save-root', state_dir, silent=True)
		state = fig.run('diplo-new', A)
		A.push('save-root', '_x_', silent=True)
		if pbar is not None:
			pbar.update(1)
	else:
		key = sorted(states.keys())[-1]
		state = states[key]
	
	turn, season = state['time']['turn'], state['time']['season']
	r = '-r' if 'retreat' in state['time'] else ''
	name = f'{turn}-{season}{r}.yaml'
	
	sname = ['', 'Spring', 'Autumn', 'Winter'][season]
	rmsg = ' (R)' if len(r) else ''
	
	while name in available:
		if pbar is not None:
			pbar.set_description(f'{turn} - {sname}{rmsg}')
		action_path = os.path.join(action_dir, name)
		
		action = load_yaml(action_path)
		
		A.push('state', state, silent=True)
		A.push('action', action, silent=True)
		
		with A.silenced():
			state = fig.run('diplo-step', A)
		
		turn, season = state['time']['turn'], state['time']['season']
		r = '-r' if 'retreat' in state['time'] else ''
		name = f'{turn}-{season}{r}.yaml'
		
		save_yaml(state, os.path.join(state_dir, name), default_flow_style=None)
		
		sname = ['', 'Spring', 'Autumn', 'Winter'][season]
		rmsg = ' (R)' if len(r) else ''
	
		if pbar is not None:
			pbar.update(1)
		
		# print(f'Completed turn {turn} - {sname}{rmsg}')
		
	if pbar is not None:
		pbar.close()
		
	print('Trajectory complete')
	
	return state