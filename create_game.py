import sys, os
from itertools import chain
from omnibelt import load_yaml, save_yaml

import omnifig as fig

from src import util

def populate_map(map, players):
	
	return {name: {
			'units': [
				{'type':unit['type'],
				 # 'loc': map.convert_to_coast(unit['loc']) if unit['type'] == 'fleet' else unit['loc']
				 'loc': unit['loc']
				 }
			for unit in player['units']],
			
			'control': player['owns'].copy(),
			
			'centers': [tile for tile in player['owns'] if map.nodes[tile].get('sc', 0) > 0],
			
			'home': player['home'].copy(),
			
			'name': player.get('name', name),
		
		} for name, player in players.items()}
	
	# full = {}
	#
	# for name, player in players.items():
	# 	full[name] = {
	# 		'units': [
	# 			{'type':unit['type'], 'loc': map.convert_to_coast(unit['loc'])}
	# 		for unit in player['units']],
	#
	# 		'control': player['owns'].copy(),
	#
	# 		'centers': [tile for tile in player['owns'] if map.nodes[tile].get('sc', 0) > 0],
	#
	# 		'name': player.get('name', name),
	# 	}
	#
	# 	if 'name' in player:
	# 		full[name]['name'] = player['name']
	#
	# return full
	

@fig.Script('diplo-new', description='Create a new Diplomacy game')
def new_game(A):
	'''
	Given the map and initial player info, this creates an initial game state yaml file.
	
	This script requires the nodes and edges yaml files, as well as the players file which
	contains the starting ownership and units for every player.
	'''
	
	silent = A.pull('silent', False, silent=True)
	
	A.push('map._type', 'map', overwrite=False)
	M = A.pull('map')
	
	players = A.pull('players', None, silent=True)
	if players is None:
		players_path = util.get_map_paths(A, 'players')
		
		if not os.path.isfile(players_path):
			raise Exception('no players found')
		
		players = load_yaml(players_path)
	
	save_path = A.pull('save-path', None)
	
	if save_path is None:
		save_root = A.pull('save-root', '.')
		save_path = os.path.join(save_root, '1-1.yaml')
	
	state = {'time': {'turn': 1, 'season': 1}}
	state['players'] = populate_map(M, players)
	
	if not silent:
		for player, info in state['players'].items():
			util.print_player(info)
	
	if save_path is not None:
		save_yaml(state, save_path, default_flow_style=None)
	
	return state

if __name__ == '__main__':
	fig.entry('diplo-new')
