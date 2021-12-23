import sys, os, shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from omnibelt import create_dir
import omnifig as fig
from PIL import Image

_DEFAULT_ROOT = fig.get_current_project().get_path()


@fig.Script('create-game', description='Create a new diplomacy game')
def _create_game(A):
	root = A.pull('games-root', '<>root', None)
	if root is None:
		root = Path(_DEFAULT_ROOT) / 'games'
	root = Path(root)
	if not root.exists():
		create_dir(root)
	mdir = A.pull('assets-path', None)
	if mdir is None:
		raise FileNotFoundError('You must provide the path to the directory with the map files')
	
	name = A.pull('name', None)
	if name is None:
		num = len(list(root.glob('*'))) + 1
		name = f'game{num}'
	
	path = root / name
	# path.mkdir(exist_ok=True)
	shutil.copytree(str(mdir), str(path))
	
	print(f'Game {name} has been created (using map data in {str(mdir)})')
	return path


@fig.Script('render-state', description='Draw the state and/or orders in a game')
def _render_state(A):
	
	manager = A.pull('manager', None)
	manager.load_status()
	
	check_image = A.pull('check-image', None)
	if check_image is not None:
		path = manager.images_root / f'{check_image}.png'
		if path.exists():
			os.remove(str(path))
	
	include_actions = A.pull('include-actions', False)
	path = manager.render_latest(include_actions=include_actions)
	
	print(f'Path: {str(path)}')






