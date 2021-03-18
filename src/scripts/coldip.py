import sys, os
import re
from difflib import get_close_matches
from tabulate import tabulate
from fuzzywuzzy import fuzz
from omnibelt import load_yaml, save_yaml, create_dir, save_json
import omnifig as fig

from ..elements import DiploMap

class ParsingFailedError(Exception):
	def __init__(self, line, reason=None, num=None):
		info =f'"{line}"' if reason is None else f'{reason}: "{line}"'
		if num is not None:
			info = f'({num}) {info}'
		super().__init__(info)

class UnknownIdentError(Exception):
	def __init__(self, ident, related=None, player=None, unit=None):
		super().__init__(f'{ident} - {related} - {player} - {unit}')

class SpecialActionError(Exception):
	pass

@fig.Component('parser')
class OrderParser:
	def __init__(self, A):
		self.loc_terms = set(A.pull('loc-terms', ['loc', 'src', 'dest']))
		self.action_filter = A.pull('term-filter', None)
		self.graph = None
		self.players = None
		self.state = None
		self.options = None
		
		aliases = None
		alias_path = A.pull('alias-path', None)
		if alias_path is not None and os.path.isfile(alias_path):
			aliases = load_yaml(alias_path)
		self.aliases = aliases
		
		self.loc_matches = []
	
	def include_info(self, graph=None, state=None):
		if graph is not None:
			self.graph = graph
			
			if self.options is None:
				self.options = self.get_region_names(graph)
				if self.aliases is not None:
					self.options.extend(self.aliases)
		
		if state is not None:
			self.state = state
			self.players = state['players']
			
	@staticmethod
	def get_region_names(data):
		options = []
		for name, info in data.items():
			if isinstance(info.get('navy-edges', []), dict):
				for coast in info['navy-edges']:
					options.append(fig.quick_run('_encode_region_name', name=name, coast=coast))
			options.append(name)
		return options
	
	def _load(self):
		raise NotImplementedError
		
	def is_neighbor(self, start, end):
		
		info = self.graph[start]
		
		if 'army-edges' in info and end in info['army-edges']:
			return True
		
		if 'fleet-edges' in info:
			if isinstance(info['fleet-edges'], dict):
				for edges in info['fleet-edges'].values():
					if end in edges:
						return True
			elif end in info['fleet-edges']:
				return True
			
		return False
		
	def __call__(self, line, player=None):
		
		action = self.split_line(line, player)
		
		# action = terms.copy()
		# for key, val in terms.items():
		# 	if key in self.loc_terms:
		# 		action[key] = self.identify(val, terms)
		
		action = self.cleanup(action)
		
		return action
		
	def cleanup(self, action):
		if self.action_filter is not None:
			for key in self.action_filter:
				if key in action:
					del action[key]
		return action
		
	def split_line(self, line, player=None):
		# assert line.startswith('A, ') or line.startswith('F, '), line
		# utype = 'army' if line.startswith('A, ') else 'fleet'
		# line = line[3:]
		
		# terms = {'unit': utype}
		terms = {}
		if player is not None:
			terms['player'] = player
		
		line = line.replace(' -> ', ' to ').replace(' supports holds ', ' support holds ')
		
		if line.startswith('Build '):
			_, loc = line.split('Build ')
			
			loc = self.identify(loc, terms)
			
			terms.update({'type': 'build', 'loc': loc})
		
		elif ' disband' in line:
			loc, _ = line.split(' disband')
			
			loc = self.identify(loc, terms)
			
			terms.update({'type': 'disband', 'loc': loc})
		
		elif ' retreats to ' in line:
			
			loc, dest = line.split(' retreats to ')
			
			loc = self.identify(loc, terms)
			dest = self.identify(dest, terms, 'dest-unit')
			
			terms.update({'type': 'retreat', 'loc': loc, 'dest': dest})
		
		elif ' supports ' in line and ' to ' in line:
			
			loc, rest = line.split(' supports ')

			loc = self.identify(loc, terms)
			
			src, dest = rest.split(' to ')
			
			src = self.identify(src, terms, 'src-unit')
			dest = self.identify(dest, terms, 'dest-unit')
			
			terms.update({'type': 'support', 'loc': loc, 'src': src, 'dest': dest})

		elif ' support holds ' in line or ' supports ' in line:
			
			word = ' support holds ' if ' support holds ' in line else ' supports '
			
			loc, dest = line.split(word)
			
			loc = self.identify(loc, terms)
			dest = self.identify(dest, terms, 'dest-unit')
			
			terms.update({'type': 'support-defend', 'loc': loc, 'dest': dest})
	
		elif ' hold' in line:
			
			loc, _ = line.split(' hold')
			
			loc = self.identify(loc, terms)
			
			terms.update({'type': 'hold', 'loc': loc})
		
		elif ' convoys ' in line:
			
			loc, rest = line.split(' convoys ')
			
			loc = self.identify(loc, terms)
			
			src, dest = rest.split(' to ')
			
			src = self.identify(src, terms, 'src-unit')
			dest = self.identify(dest, terms, 'dest-unit')

			terms.update({'type': 'convoy-transport', 'loc': loc, 'src': src, 'dest': dest})
		
		elif ' transforms into ' in line or ' transforms' in line:
			
			key = ' transforms into ' if ' transforms into ' in line else ' transforms'
			
			loc, _ = line.split(key)
			
			loc = self.identify(loc, terms)
			
			terms.update({'type': 'transform', 'loc': loc,})
		
		elif 'transform' in line or 'canal' in line:
			raise SpecialActionError(line + f' ({player})')
		
		elif ' to ' in line:
			
			loc, dest = line.split(' to ')
			
			loc = self.identify(loc, terms)
			dest = self.identify(dest, terms, 'dest-unit')
			
			action_type = 'move' if terms.get('unit', None) == 'fleet' or self.is_neighbor(loc, dest) \
				else 'convoy-move'
			
			terms.update({'type': action_type, 'loc': loc, 'dest': dest})
			
			if action_type is 'convoy-move':
				print(f'Convoying: {line} - {terms}')
		
		
		else:
			raise ParsingFailedError(line)
		
		return terms
		
	def fuzzy_match(self, query, options=None):
		matches = []
		if options is None:
			options = self.options
		for option in options:
			match = fuzz.token_sort_ratio(option, query)
			matches.append((match, option))
		return sorted(matches, reverse=True)
		
	
	def identify(self, ident, terms={}, unit_key='unit'):
		
		if ident.startswith('A, ') or ident.startswith('F, '):
			terms[unit_key] = 'army' if ident.startswith('A, ') else 'fleet'
			ident = ident[3:]
		
		if ident in self.graph:
			return ident
			
		fixed, coast = fig.quick_run('_decode_region_name', name=ident)
		if fixed in self.graph:
			return ident
		
		if ident in self.aliases:
			return self.aliases[ident]
		
		matches = self.fuzzy_match(ident)
		
		score, best = matches[0]
		
		if best in self.aliases:
			best = self.aliases[best]
		
		self.loc_matches.append([ident, best, score, ])#' | '.join(f'{s}:{n}' for s,n in matches[1:3])])
		
		# print('"{}" => "{}" : {}'.format(ident, best, ' | '.join(f'{s}:{n}' for s,n in matches[1:3])))
		
		return best
		raise UnknownIdentError(ident, related, player, unit)

world_regions = {'Europe', 'North America', 'South America', 'Asia', 'High Seas', 'Oceania', 'Africa', }

@fig.Script('parse-col-actions')
def parse_actions(A):
	
	path = A.pull('log-path', '<>path')
	
	with open(path, 'r') as f:
		lines = f.read().split('\n')
	
	action_path = A.pull('action-path', None)
	if action_path is None:
		print('WARNING: no action_path so the actions will not be saved')
	error_path = A.pull('error-path', None)
	
	graph_path = A.pull('graph-path')
	data = load_yaml(graph_path)
	
	state = None
	state_path = A.pull('state-path', None)
	if state_path is not None:
		state = load_yaml(state_path)

	if state is None:
		player_path = A.pull('players-path')
		players = load_yaml(player_path)
	else:
		players = state['players']

	parser = A.pull('parser', ref=True)
	parser.include_info(graph=data, state=state)

	
	actions = {}

	warnings = []
	errors = []

	player = None
	current = None
	for i, line in enumerate(lines):
		if len(line) > 2:

			try:
				if ' - ' in line:
					player = line.split(' - ')[0]
					assert player in players, player
					actions[player] = []
					current = actions[player]
					continue
				elif line in players:
					player = line
					actions[player] = []
					current = actions[player]
					continue
				elif line in world_regions:
					continue
				
				assert current is not None
				
				action = parser(line, player)
				
				if len(parser.loc_matches):
					warnings.extend([i+1, player, *warning, line] for warning in parser.loc_matches)
					parser.loc_matches.clear()
		
			except Exception as e:
				errors.append([repr(e), i+1, line])
				# print(f'Error {n} {e}: (line {i+1}) "{rawline}"')
				# raise
			else:
				current.append(action)
			
	if len(warnings):
		print(f'Encountered {len(warnings)} fuzzy matches')
		print(tabulate(warnings, headers=['Line #', 'Player', 'Query', 'Best Match', 'Score', 'Original Order']))
		print()
		
	if len(errors):
		print(f'Encountered {len(errors)} errors')
		print(tabulate(errors, headers=['Error Type', 'Line #', 'Original Line']))
	
		if error_path is not None:
			save_yaml(errors, error_path)
			print(f'Saved errors to {error_path}')
	
	if action_path is not None:
		save_yaml(actions, action_path)
	
	return actions


@fig.AutoModifier('col-dip')
class Col(DiploMap):
	
	def __init__(self, A, ignore_unknown=None, **kwargs):
		super().__init__(A, ignore_unknown=True, **kwargs)
		
		self.transform_types = {'army': 'fleet', 'fleet': 'army'}
	
	def _special_rules(self, state, actions, unknown, new):
		
		if 'canals' in state:
			new['canals'] = state['canals']
		
		if len(unknown):
			attacked = set()
			# for acts in actions.values():
			# 	for a in acts:
			# 		if a['type'] == 'move' or a['type'] == 'convoy-move' and 'dest' in a:
			# 			attacked.add(fig.quick_run('_decode_region_name', name=a['dest'])[0])
		
			graph_changed = False
		
			for name, acts in unknown.items():
				
				units = new['players'][name]['units']
				# units = state['players'][name]['units']
				
				for action in acts:
					typ = action['type']
					if typ == 'transform':
						utype = action['unit']
						loc = action['loc']
						dest = action.get('dest', None)
						base, coast = fig.quick_run('_decode_region_name', name=loc)
						if base not in attacked:
							for unit in units:
								if loc == unit['loc']:
									unit['type'] = self.transform_types[utype]
									if dest is not None:
										unit['loc'] = dest
									elif coast is not None:
										unit['loc'] = base
									break
									
					elif typ == 'canal':
						loc = action['loc']
						base, coast = fig.quick_run('_decode_region_name', name=loc)
						if coast is None and base not in attacked:
							
							info = self.graph[base]
							
							
							if info['type'] == 'land':
								pass
								# info['type'] = 'coast'
								# info['fleet-edges'] = [neighbor for neighbor in info['army-edges']
								#                        if self.graph[neighbor]['type'] == 'coast']
								# info['locs']['fleet'] = info['locs']['army'].copy()
								#
								# for neighbor in info['fleet-edges']:
								# 	ninfo = self.graph[neighbor]
								# 	edges = ninfo['fleet-edges']
								#
								# 	if isinstance(edges, dict):
								# 		print(f'Failed to add canal (player={name}) {action} to {neighbor}')
								# 	elif loc not in edges:
								# 		edges.append(loc)
								
							elif info['type'] == 'coast' and isinstance(info['fleet-edges'], dict):
								
								info['canal'] = True
								if 'canals' not in new:
									new['canals'] = []
								new['canals'].append(base)
								graph_changed = True
								
								cts = list(info['fleet-edges'].keys())
								
								edges = info['fleet-edges'][cts[0]]
								
								for ct in cts[1:]:
									for e in info['fleet-edges'][ct]:
										if e not in edges:
											edges.append(e)
								
								info['fleet-edges'] = edges
								
								for neighbor in edges:
									ninfo = self.graph[neighbor]
									ninfo['fleet-edges'] = [e for e in ninfo['fleet-edges']
									                        if fig.quick_run('_decode_region_name', name=e)[0] != base]
									ninfo['fleet-edges'].append(base)
								
								if 'coasts' in info:
									del info['coasts']
								
								locs = info['locs']
								if 'fleet' in locs:
									fleet = locs['fleet']
									for aname in fleet:
										fleet[aname] = fleet[aname][cts[0]]
								if 'coast-label' in locs:
									locs['coast-label'] = locs['coast-label'][cts[0]]
						
					else:
						print(f'Unknown action: {name} - {action}')
						
			if graph_changed:
				print(f'Saving updated graph to {self.graph_path}')
				if self.graph_path.endswith('.yaml'):
					save_yaml(self.graph, self.graph_path)
				else:
					save_json(self.graph, self.graph_path)