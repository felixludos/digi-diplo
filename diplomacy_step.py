
import sys, os
from itertools import chain
from omnibelt import load_yaml, save_yaml

import omnifig as fig


@fig.Script('diplo-step', description='Take a Diplomacy step')
def diplomacy_step(A):
	
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

	return new

if __name__ == '__main__':
	fig.entry('diplo-step')

