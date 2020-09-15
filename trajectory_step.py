

import sys, os
from itertools import chain
from omnibelt import load_yaml, save_yaml, create_dir

import omnifig as fig


@fig.Script('diplo-traj', description='Resolve a sequence of actions from an initial state')
def diplomacy_traj(A):
	
	silent = A.pull('silent', False, silent=True)
	
	state_dir = A.pull('state-dir')
	if not os.path.isdir(state_dir):
		create_dir(state_dir)
	assert os.path.isdir(state_dir), f'Invalid dir: {state_dir}'
	
	action_dir = A.pull('action-dir')
	assert os.path.isdir(action_dir), f'Invalid dir: {action_dir}'
	
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
		
	else:
		key = sorted(states.keys())[-1]
		state = states[key]
	
	turn, season = state['time']['turn'], state['time']['season']
	r = '-r' if 'retreat' in state['time'] else ''
	name = f'{turn}-{season}{r}.yaml'
	
	available = set(os.listdir(action_dir))
	
	while name in available:
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
		
		if not silent:
			sname = ['', 'Spring', 'Autumn', 'Winter'][season]
			rmsg = ' (retreat)' if len(r) else ''
			print(f'Completed turn {turn} - {sname}{rmsg}')
		
	print('Trajectory complete')
	
	return state