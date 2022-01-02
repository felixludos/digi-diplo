import sys, os, shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from omnibelt import create_dir, unspecified_argument
import omnifig as fig
from PIL import Image

_DEFAULT_ROOT = fig.get_current_project().get_path()


@fig.Script('create-game', description='Create a new diplomacy game')
def create(A, name=unspecified_argument, game_root=unspecified_argument, assets_path=unspecified_argument,
           silent=None):
	if silent is None:
		silent = A.pull('silent', False)
	if game_root is unspecified_argument:
		game_root = A.pull('games-root', '<>root', None)
	if game_root is None:
		game_root = Path(_DEFAULT_ROOT) / 'games'
	game_root = Path(game_root)
	if not game_root.exists():
		create_dir(game_root)
	if assets_path is unspecified_argument:
		assets_path = A.pull('assets-path', None)
	if assets_path is None:
		raise FileNotFoundError('You must provide the path to the directory with the map files')
	
	if name is unspecified_argument:
		name = A.pull('name', None)
	if name is None:
		num = len(list(game_root.glob('*'))) + 1
		name = f'game{num}'
	
	path = game_root / name
	# path.mkdir(exist_ok=True)
	shutil.copytree(str(assets_path), str(path))
	
	if not silent:
		print(f'Game {name} has been created (using map data in {str(assets_path)})')
	return path



@fig.Script('render', description='Draw the state and/or orders in a game')
def render(A, current=unspecified_argument, include_orders=None, manager=unspecified_argument,
           silent=None):
	'''
	Draws the current state (optionally including orders) and saves the resulting image to the `images/`
	directory of the game.
	
	:param A: config object
	:param current: current season (formatted as [year]-[season][retreats], such as "1-1", "1-2", "1-3", "2-1-r" etc.)
	:param include_orders: bool whether the drawing should include orders
	:param manager: manager of the current game
	:param silent: bool
	:return: path where the image was saved
	'''
	if silent is None:
		silent = A.pull('silent', False)
	
	if current is unspecified_argument:
		current = A.pull('current', None)
	
	if manager is unspecified_argument:
		manager = A.pull('manager')
	manager.load_status(current)
	
	if include_orders is None:
		include_orders = A.pull('include-orders', False)
	
	path = manager.render_latest(include_actions=include_orders)
	
	if not silent:
		result = 'orders' if include_orders else 'state'
		print(f'Saved rendered {result} to {str(path)}')
	
	return path



@fig.Script('step', description='Adjudicate the current orders to get the next state')
def step_season(A, current=unspecified_argument, manager=unspecified_argument, silent=None):
	if silent is None:
		silent = A.pull('silent', False)
	if current is unspecified_argument:
		current = A.pull('current', None)
	
	if manager is unspecified_argument:
		manager = A.pull('manager')
	manager.load_status(current)
	
	old = manager.format_date()
	
	new = manager.take_step()
	
	msg = f'Finished adjudicating {old}.\nCurrent turn: **{manager.format_date()}**'
	if not silent:
		print(msg)
	
	return new
	



