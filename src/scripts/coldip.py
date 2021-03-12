import sys, os
import re
from difflib import get_close_matches
from tabulate import tabulate
from fuzzywuzzy import fuzz
from omnibelt import load_yaml, save_yaml, create_dir
import omnifig as fig

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
		
	def is_land_neighbor(self, start, end):
		if start not in self.graph:
			return None
		return end in self.graph[start].get('army-edges', [])
		
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
		utype = 'army' if line.startswith('A, ') else 'fleet'
		# line = line[3:]
		
		terms = {'unit': utype}
		if player is not None:
			terms['player'] = player
		
		line = line.replace(' -> ', ' to ')
		
		if ' support holds ' in line:
			
			loc, dest = line.split(' support holds ')

			loc = self.identify(loc, terms)
			dest = self.identify(dest, terms, 'dest-unit')
			
			terms.update({'type': 'support-defend', 'loc': loc, 'dest': dest})
		
		elif ' supports ' in line and ' to ' in line:
			
			loc, rest = line.split(' supports ')

			loc = self.identify(loc, terms)
			
			src, dest = rest.split(' to ')
			
			src = self.identify(src, terms, 'src-unit')
			dest = self.identify(dest, terms, 'dest-unit')
			
			terms.update({'type': 'support', 'loc': loc, 'src': src, 'dest': dest})
		
		elif ' hold' in line:
			
			loc, _ = line.split(' hold')
			
			loc = self.identify(loc, terms)
			
			terms.update({'type': 'hold', 'loc': loc})
		
		elif 'transform' in line or 'canal' in line:
			raise SpecialActionError(line)
		
		elif ' to ' in line:
			
			loc, dest = line.split(' to ')
			
			loc = self.identify(loc, terms)
			dest = self.identify(dest, terms, 'dest-unit')
			
			action_type = 'move' if utype == 'fleet' or self.is_land_neighbor(loc, dest) else 'convoy-move'
			
			terms.update({'type': action_type, 'loc': loc, 'dest': dest})
		
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
			ident = ident[3:]
			terms[unit_key] = 'army' if ident.startswith('A, ') else 'fleet'
		
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

	state_path = A.pull('state-path')
	state = load_yaml(state_path)

	parser = A.pull('parser', ref=True)
	parser.include_info(graph=data, state=state)

	players = state['players']
	
	actions = {}

	warnings = []
	errors = []

	player = None
	current = None
	for i, line in enumerate(lines):
		if len(line) > 3:

			try:
				if ' - ' in line:
					player = line.split(' - ')[0]
					assert player in players, player
					actions[player] = []
					current = actions[player]
					continue
				elif not line.startswith('A, ') and not line.startswith('F, '):
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