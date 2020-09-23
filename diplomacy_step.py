
import sys, os
from itertools import chain
from omnibelt import load_yaml, save_yaml

import omnifig as fig


@fig.Script('diplo-step', description='Take a Diplomacy step')
def diplomacy_step(A):
	'''
	Given the current game state (as a yaml file) and the selected actions for all players (also a yaml file),
	this updates the games state using the `pydip` adjudicator.
	
	This script requires the current state and actions (see `test/5` for examples) yaml files,
	provided as `state-path` and `action-path`. The output state should be saved at `save-path`.
	'''

	A.push('map._type', 'map', overwrite=False)
	M = A.pull('map')
	
	state = A.pull('state', None, silent=True)
	if state is None:
		state_path = A.pull('state-path')
		if not os.path.isfile(state_path):
			raise Exception(f'No state file: {state_path}')
		
		state = load_yaml(state_path)
	
	action = A.pull('action', None, silent=True)
	if action is None:
		action_path = A.pull('action-path')
		
		if action_path is None:
			action_root = A.pull('action-root', '.')
		
			if not os.path.isdir(action_root):
				raise Exception(f'No action root: {action_root}')
	
			turn, season = state['time']['turn'], state['time']['season']
			r = '-r' if 'retreat' in state['time'] else ''
			action_name = f'{turn}-{season}{r}.yaml'
			
			action_path = os.path.join(action_root, action_name)
		
		action = load_yaml(action_path)
	
	new = M.step(state, action)
	
	save_path = A.pull('save-path', None)
	if save_path is None:
		
		save_root = A.pull('save-root', None)
	
		if save_root is not None:
			turn, season = new['time']['turn'], new['time']['season']
			r = '-r' if 'retreat' in new['time'] else ''
			new_name = f'{turn}-{season}{r}.yaml'
		
			save_path = os.path.join(save_root, new_name)
	
	if save_path is not None:
		save_yaml(new, save_path, default_flow_style=None)
	
	img_fmt = A.pull('image-format', 'png', silent=True)
	frame_dir = A.pull('frame-dir', None, silent=True)
	render_actions = A.pull('render-actions', False)
	if render_actions:
		turn, season = state['time']['turn'], state['time']['season']
		r = '-r' if 'retreat' in state['time'] else ''
		name = f'{turn}-{season}{r}.yaml'
		imname = f'{name}.{img_fmt}'
		
		A.begin()
		A.push('state', state, silent=True)
		A.push('actions', action, silent=True)
		A.push('save-path', '_x_', silent=True)
		# A.push('save-path', os.path.join(frame_dir, imname) if frame_dir is not None else imname, silent=True)
		fig.run('render', A)
		
		A.abort()

	render_state = A.pull('render-state', False)
	if render_state:
		turn, season = new['time']['turn'], new['time']['season']
		r = '-r' if 'retreat' in new['time'] else ''
		new_name = f'{turn}-{season}{r}.yaml'
		imname = f'{new_name}.{img_fmt}'
		A.begin()
		A.push('state', new, silent=True)
		A.push('save-path', '_x_', silent=True)
		A.push('action', '_x_', silent=True)
		A.push('action-path', '_x_', silent=True)
		fig.run('render', A)
		
		A.abort()

	return new

if __name__ == '__main__':
	fig.entry('diplo-step')

