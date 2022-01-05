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
	'''
	Creates a new game, primarily by copying all the game assets (mostly the map and player info) to a newly created
	game directory where all game states, orders, and rendered maps are.
	
	:param A: config object
	:param name: of the game
	:param game_root: location to put the new game directory into (default `games/`)
	:param assets_path: path to the game variant assets (default `assets/classic`)
	:param silent: bool
	:return: game directory path
	'''
	if silent is None:
		silent = A.pull('silent', False)
	if game_root is unspecified_argument:
		game_root = A.pull('games-root', '<>root', str(Path(_DEFAULT_ROOT) / 'games'))
	game_root = Path(game_root)
	if not game_root.exists():
		create_dir(game_root)
	if assets_path is unspecified_argument:
		assets_path = A.pull('assets-path', 'assets/classic')
	assets_path = Path(assets_path)
	if not assets_path.exists():
		raise FileNotFoundError('You must provide the assets path to the directory with the map files.')
	
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
		manager = A.pull('manager', ref=True)
	manager.load_status(current)
	
	if include_orders is None:
		include_orders = A.pull('include-orders', False)
	
	path = manager.render_latest(include_actions=include_orders)
	
	if not silent:
		result = 'orders' if include_orders else 'state'
		print(f'Saved rendered {result} to {str(path)}')
	
	return path



@fig.Script('step', description='Adjudicate the current orders to get the next state')
def step_season(A, current=unspecified_argument, manager=unspecified_argument, update_state=None, silent=None):
	'''
	Adjudicates the current season based on the state and orders found by the provided manager
	(based on the game directory), and saves the new state in the `states/` directory.
	
	:param A: config object
	:param current: season code (defaults to the last one in states/)
	:param manager: game manager
	:param update_state: automatically save the new adjudicated state
	:param silent: bool
	:return: new state
	'''
	if silent is None:
		silent = A.pull('silent', False)
	if current is unspecified_argument:
		current = A.pull('current', None)
	
	if manager is unspecified_argument:
		manager = A.pull('manager', ref=True)
	manager.load_status(current)
	
	old = manager.format_date()
	
	if update_state is None:
		update_state = A.pull('update-state', True)
	
	new = manager.take_step(update_state)
	
	msg = f'Finished adjudicating {old}.\nCurrent turn: {manager.format_date()}'
	if not silent:
		print(msg)
	
	return new
	


@fig.Script('multi-step', description='Loop to adjudicate (and render) multiple seasons')
def step_season(A, current=unspecified_argument, manager=unspecified_argument, num_steps=unspecified_argument,
                allow_missing=unspecified_argument, render_state=None, render_orders=None,
                silent=None):
	'''
	
	
	:param A: config object
	:param current: season code (defaults to the last one in states/)
	:param manager: game manager
	:param num_steps: number of seasons to adjudicate (including retreats)
	:param allow_missing: adjudicate despite missing this many orders in total
	:param render_state: render the state, if it doesn't already exist (before and after)
	:param render_orders: render the orders, if it doesn't already exist
	:param silent: bool
	:return: result current state
	'''
	# if silent is None:
	# 	silent = A.pull('silent', False)
	if current is unspecified_argument:
		current = A.pull('current', None)
	
	if num_steps is unspecified_argument:
		num_steps = A.pull('num-steps', None)
	
	if allow_missing is unspecified_argument:
		allow_missing = A.pull('allow-missing', 0)
	
	if render_state is None:
		render_state = A.pull('render-state', False)
	
	if render_orders is None:
		render_orders = A.pull('render-orders', False)
	
	if manager is unspecified_argument:
		manager = A.pull('manager', ref=True)
	manager.load_status(current)
	
	new = None
	
	status = manager.get_status()
	steps = 0
	while sum(status.values()) <= allow_missing:
		if render_state:
			path = manager.find_image_path(include_actions=False)
			if path is None:
				manager.render_latest(include_actions=False)
		
		if num_steps is not None and steps == num_steps:
			break
		
		if render_orders:
			path = manager.find_image_path(include_actions=True)
			if path is None:
				manager.render_latest(include_actions=True)
		
		old = manager.format_date()
		new = manager.take_step(True)
		print(f'Adjudicated {old}')
		status = manager.get_status()
	else:
		print(f'Not adjudicating {manager.format_date()}, missing {sum(status.values())} orders.')
	
	return new
		
		
	
