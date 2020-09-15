
import sys, os

from omnibelt import create_dir, save_yaml, load_yaml

import omnifig as fig

from src import util

_players_lines = {'Germany:', 'Italy:', 'Austria:', 'Turkey:', 'England:', 'Russia:', 'France:'}

def process_player(data, locs=None):
	
	info = {}
	
	if 'actions' in data:
		info['actions'] = [parse_action(line, locs=locs) for line in data['actions']]
	
	if 'retreats' in data:
		info['retreats'] = [parse_retreat(line, locs=locs) for line in data['retreats']]
	
	if 'builds' in data:
		info['builds'] = [parse_build(line, locs=locs) for line in data['builds']]
	
	return info

def parse_action(raw, locs=None):
	raw = raw[4:].replace('St. Peter', 'St Peter')
	
	unit = raw.split(' ')[0]
	
	if 'convoy' in raw and 'move' in raw:
		action = 'convoy-move'
		key = ' move to '
	elif 'convoy' in raw:
		action = 'convoy-transport'
		key = ' convoy to '
		
	elif 'support' in raw and 'hold' in raw:
		action = 'support-defend'
		key = ' support hold to '
	elif 'support' in raw:
		action = 'support'
		key = ' support move to '
	elif 'move' in raw:
		action = 'move'
		key = ' move to '
	elif 'hold' in raw:
		action = 'hold'
		key = ' hold.'
	else:
		raise Exception('raw')
	
	info = {'type': action, 'unit': unit, }
	
	if '(fail)' in raw:
		info['failed'] = True
	
	terms = raw.split(' at ')[1].split(key)
	
	loc = terms[0]
	
	rest = terms[1]
	
	info['loc'] = loc if locs is None else locs(loc)
	
	if action == 'move' or action == 'support-defend':
		info['dest'] = rest.split('.')[0]
	elif action == 'support':
		info['dest'], rest = rest.split(' from ')
		info['src'] = rest.split('.')[0]
	elif action == 'convoy-move':
		info['dest'] = rest.split(' via')[0]
	elif action == 'convoy-transport':
		info['dest'], rest = rest.split(' from ')
		info['src'] = rest.split('.')[0]
	
	if 'dest' in info:
		info['dest'] = info['dest'] if locs is None else locs(info['dest'])
	if 'src' in info:
		info['src'] = info['src'] if locs is None else locs(info['src'])
	return info

def parse_retreat(raw, locs=None):
	raw = raw[4:].replace('St. Peter', 'St Peter')
	
	unit, rest = raw.split(' at ')
	
	if 'disband' in rest:
		action = 'disband'
		idx = rest.find('disband')-1
	elif 'retreat' in rest:
		action = 'retreat'
		idx = rest.find('retreat')-1
	else:
		raise Exception(raw)
	
	loc = rest[:idx]
	if locs is not None:
		loc = locs(loc)
	
	info = {'type': action, 'unit': unit, 'loc': loc}
	
	if action == 'retreat':
		dest = rest.split(' to ')[-1][:-1]
		if locs is not None:
			dest = locs(dest)
		info['dest'] = dest

	return info

def parse_build(raw, locs=None):
	
	raw = raw.replace('St. Peter', 'St Peter')
	
	if raw.startswith('Build'):
		t = 'build'
	elif raw.startswith('Destroy'):
		t = 'destroy'
	else:
		raise Exception(raw)
	
	unit = raw.split(' ')[1]
	loc = raw.split(' at ')[-1][:-1]
	loc = loc if locs is None else locs(loc)
	
	return {'type': t, 'unit': unit, 'loc': loc}

def parse_season(raw):
	season, year = raw.split(', ')
	idx = 1 if season == 'Spring' else 2
	year = year[:4]
	num = int(year[-2:])
	
	return f'{num}-{idx}'


def separate_seasons(log):
	
	full = {}
	
	for turn, season in log.items():
		info = {}
		full[turn] = info
		for name, player in season.items():
			for t, ls in player.items():
				if t not in info:
					info[t] = {}
				info[t][name] = ls
	
	return full
	
def expand_seasons(log):
	
	full = {}
	
	keys = ['actions', 'builds', 'retreats']
	
	for turn, season in log.items():
		names = [turn, '{}-3'.format(turn.split('-')[0]), f'{turn}-r']
		for key, name in zip(keys, names):
			if key in season:
				full[name] = season[key]
				# full[name] = {'actions':season[key]}
		
	return full


@fig.Script('parse-vdip', 'Parse a vdiplomacy game log to yaml')
def parse_vdiplomacy(A):
	
	path = A.pull('log-path', '<>path')
	
	with open(path, 'r') as f:
		raw = f.read()
	
	out_dir = A.pull('out-dir', '<>out', '<>name', None)
	if out_dir is not None:
		create_dir(out_dir)
	
	lines = [line for line in raw.split('\n') if len(line) > 3]
	
	locs = None
	nodes_path = A.pull('nodes-path', None)
	if nodes_path is not None:
		locs = util.make_node_dictionary(load_yaml(nodes_path))
	
	print(f'Will parse through {len(lines)} lines')
	
	assert 'Spring' in lines[0] or 'Autumn' in lines[0], f'incomplete: {lines[0]}'
	
	full = {}
	season = None
	player = None
	data = None
	
	for line in lines:
		if 'Spring' in line or 'Autumn' in line:
			full[line] = {}
			season = full[line]
		
		elif line in _players_lines:
			season[line[:-1]] = {}
			player = season[line[:-1]]
		
		elif line == 'Diplomacy':
			player['actions'] = []
			data = player['actions']
		elif line == 'Unit-placement':
			player['builds'] = []
			data = player['builds']
		elif line == 'Retreats':
			player['retreats'] = []
			data = player['retreats']
			
		else:
			data.append(line)
	
	log = {
		parse_season(rawseason) : {player:process_player(data, locs=locs)
		                           for player, data in rawinfo.items()}
		for rawseason, rawinfo in full.items()
	}
	
	log = separate_seasons(log)
	log = expand_seasons(log)
	
	if out_dir is not None:
		print(f'Saving {len(log)} seasons to {out_dir}')
		for season, players in log.items():
			save_yaml(players, os.path.join(out_dir, f'{season}.yaml'),
			          default_flow_style=None)
	
	return log

if __name__ == '__main__':
	fig.entry('parse-vdip')

