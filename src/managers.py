import random
import sys, os
import shutil
from copy import deepcopy
from pathlib import Path
from omnibelt import unspecified_argument, load_yaml, save_yaml, load_txt
import omnifig as fig
from .parsing import parse_order, standardize_order
from .util import Versioned
from .errors import *





@fig.component('diplo-manager')
class DiplomacyManager(Versioned):
	
	@classmethod
	def init_from_config(cls, config, args = None, kwargs = None, *, silent= None):
		cls._find_root(config) # hacky
		return super().init_from_config(config, args, kwargs, silent=silent)
	
	def __init__(self, game_root, map=unspecified_argument, renderer=None, year_offset=0, **kwargs):
		game_root = self._find_root(self._my_config, root=game_root)

		graph_path = game_root / 'graph.yaml'
		player_path = game_root / 'players.yaml'
		
		states_root = game_root / 'states'
		actions_root = game_root / 'actions'
		images_root = game_root / 'images'
		
		super().__init__(**kwargs)
		
		if not states_root.exists():
			states_root.mkdir()
		if not actions_root.exists():
			actions_root.mkdir()
		if not images_root.exists():
			images_root.mkdir()
		
		self.gamemap = map
		self.renderer = renderer
		self.aliases = {alias: player for player, info in self.gamemap.player_info.items()
		                for alias in info.get('alias', [])}
		self.aliases.update({alias.lower(): player for player, info in self.gamemap.player_info.items()
		                     for alias in info.get('alias', [])})
		
		self.root = game_root
		self.graph_path = graph_path
		self.player_path = player_path
		
		self.states_root = states_root
		self.actions_root = actions_root
		self.images_root = images_root
		
		self.current_state = None
		
		self.year_offset = year_offset
		
		
	@staticmethod
	def _find_root(A, root=unspecified_argument):
		if root is unspecified_argument:
			root = A.pulls('game-root', 'game_root', default=None)
		if root is None:
			root = fig.run_script('create-game', A)
		root = Path(root)
		
		A.root.push('game_root', str(root), overwrite=False, silent=True)
		
		graph_path = root / 'graph.yaml'
		if not graph_path.exists():
			raise FileNotFoundError(str(graph_path))
		A.root.push('graph_path', str(graph_path), overwrite=False, silent=True)
		
		player_path = root / 'players.yaml'
		if not player_path.exists():
			raise FileNotFoundError(str(player_path))
		A.root.push('players_path', str(player_path), overwrite=False, silent=True)

		bg_path = root / 'bgs.yaml'
		if bg_path.exists():
			A.root.push('bgs_path', str(bg_path), overwrite=False, silent=True)
		A.root.push('regions_path', str(root / 'regions.png'), overwrite=False, silent=True)
		A.root.push('renderbase_path', str(root / 'renderbase.png'), overwrite=False, silent=True)
		A.root.push('tiles_path', str(root / 'tiles.png'), overwrite=False, silent=True)
		return root
		
		
	def load_status(self, name=None, path=None):
		self.graph = load_yaml(self.graph_path)
		self._get_base_region = {f'{base}-{coast}':base for base, node in self.graph.items()
		                         if 'fleet' in node['edges'] and isinstance(node['edges']['fleet'], dict)
		                         for coast in node['edges']['fleet']}
		self._get_base_region.update({base: base for base in self.graph})
		self._get_base_region.update({f'{base}-c': base for base, node in self.graph.items()
		                              # if 'fleet' in node['edges'] and 'army' in node['edges']})
										if 'fleet' in node['edges'] and node["type"] == "coast"})
		self._get_base_region.update({base.lower(): val for base, val in self._get_base_region.items()})
		self._get_region = {f'{base}-{coast}': f'{base}-{coast}' for base, node in self.graph.items()
		                         if 'fleet' in node['edges'] and isinstance(node['edges']['fleet'], dict)
		                         for coast in node['edges']['fleet']}
		self._get_region.update({base: base for base in self.graph})
		self._get_region.update({base.lower(): val for base, val in self._get_region.items()})
		
		self._unit_texts = {'a': 'army', 'f': 'fleet', 'army': 'army', 'fleet': 'fleet'}
		
		bad_names = [name for name in self._get_region if any(name.startswith(f'{u} ') for u in self._unit_texts)]
		if len(bad_names):
			raise BadNamesError(bad_names)
		
		if name is not None:
			if not name.endswith('.yaml'):
				name = f'{name}.yaml'
			path = self.states_root / name
		
		if path is not None and path.exists():
			self.set_state(load_yaml(path))
		else:
			self.set_state(self._find_latest_state(self.states_root))
		print(f'Loaded state: {self.time}')
		return self.time
		
		
	def set_state(self, state):
		self.state = state
		self._extract_time(self.state)
		self.actions = self._find_actions_from_state(self.state)
		
		self.units = {player: {self._get_base_region[unit['loc']]: unit for unit in info.get('units', [])}
		              for player, info in self.state['players'].items()}
		
	
	def take_step(self, update_state=False):
		new = self.gamemap.step(self.state, self._unformat_actions(self.actions))
		if update_state:
			self.set_state(new)
			self._checkpoint_state()
		return new
	
	def fix_player(self, player):
		fix = self.aliases.get(player, None)
		return self.aliases.get(player.lower(), player) if fix is None else fix
	
	
	def get_demonym(self, player):
		player = self.fix_player(player)
		return self.gamemap.player_info.get(player, {}).get('demonym', player)
	
	def check_region(self, name):
		try:
			for loc in [name.upper(), name.lower(), name.capitalize()]:
				
				try:
					base, coast = self.gamemap.decode_region_name(loc)
				except LocationError:
					pass
				else:
					break
			else:
				base, coast = self.gamemap.decode_region_name(name)
		except LocationError:
			return
		else:
			node = self.graph[base]
			
			region = {'node': deepcopy(node), 'base': base}
			if coast is not None:
				region['coast'] = coast
			units = []
			
			for player, info in self.state.get('players', {}).items():
				if base in info.get('home', []):
					region['home'] = player
				if base in info.get('control', []):
					region['control'] = player
				for unit in info.get('units', []):
					if self._to_base_region(unit['loc']) == base:
						units.append({'player': player, **unit})
			
			for player, info in self.state.get('disbands', {}).items():
				for unit in info:
					if self._to_base_region(unit['loc']) == base:
						region['disband'] = {'player': player, **unit}
						break
			
			if len(units) == 1:
				region['unit'] = units[0]
			elif len(units) > 1:
				occupant = None
				retreat = None
				for unit in units:
					ubase, _ = self.gamemap.decode_region_name(unit['loc'])
					if retreat is None:
						for src, options in self.state.get('retreats', {}).items():
							if self.gamemap.decode_region_name(src)[0] == ubase:
								retreat = {'options': options, **unit}
								break
						else:
							occupant = unit
					else:
						occupant = unit
				assert occupant is not None
				region['unit'] = occupant
				region['retreat'] = retreat
			
			return region
		
	
	def _extract_time(self, state):
		year = int(state['time']['turn'])
		season = int(state['time']['season'])
		retreat = state['time'].get('retreat', False)
		retreat = '-r' if retreat else ''
		self.year, self.season, self.retreat = year, season, retreat
		self.time = self._season_date(year, season, retreat)
	
	
	def _season_date(self, year, season, retreat):
		return f'{year}-{season}{retreat}'
	
	def format_date(self):
		rmsg = ' (retreats)' if self.retreat else ''
		season = {1:'Spring', 2: 'Fall', 3: 'Winter'}
		
		year = str(self.year + self.year_offset) if self.year_offset is not None else f'Year {self.year}'
		
		return f'{year} {season.get(self.season, self.season)}{rmsg}'
	
	def _get_action_path(self):
		return self.actions_root / f'{self.time}.yaml'

	
	def _get_state_path(self):
		return self.states_root / f'{self.time}.yaml'

	
	def _find_latest_state(self, state_root, persistent=True):
		state_paths = list(state_root.glob('*-*.yaml'))
		if not len(state_paths):
			state = self.generate_initial_state()
			if persistent:
				save_yaml(state, state_root / '1-1.yaml')
			return state
		
		times = {}
		for path in state_paths:
			year, season, *other = path.stem.split('-')
			times[int(year), int(season), len(other) and other[0] == 'r'] = path
		latest = times[max(times.keys())]
		return load_yaml(latest)
	
	
	def _find_actions_from_state(self, state, persistent=True):
		actions_path = self._get_action_path()
		if actions_path.exists():
			actions = load_yaml(actions_path)
			actions = {player: {action['loc']: action for action in acts} for player, acts in actions.items()}
			return actions
		
		actions = self.generate_actions_from_state(state)
		if persistent:
			save_yaml(actions, actions_path)
		return actions
	
	
	def generate_actions_from_state(self, state):
		return {player: {} for player in state['players']}
	
	
	def generate_initial_state(self):
		return self.gamemap.generate_initial_state()
	
	def get_status(self):
		if self.season == 3:
			reqs = {player: abs(delta) for player, delta in self.state.get('adjustments', {}).items()}
		elif self.retreat:
			reqs = {player: len(retreats) for player ,retreats in self.state.get('retreats', {}).items()}
		else:
			reqs = {player: len(self.units[player]) for player in self.actions}
		missing = {player: num - len(self.actions.get(player, {}))
		           for player, num in reqs.items()}
		
		return {player: num for player, num in missing.items() if num > 0}
	
	
	def get_missing(self):
		if self.season == 3:
			reqs = {player: (delta//abs(delta)) * (abs(delta)-len(self.actions.get(player, {})))
			        for player, delta in self.state.get('adjustments', {}).items()
			        if delta != 0 and abs(delta)-len(self.actions.get(player, {}))}
		elif self.retreat:
			reqs = {}
			for player, units in self.state.get('disbands', {}).items():
				for unit in units:
					base = self._to_base_region(unit['loc'])
					if base not in self.actions.get(player, {}):
						if player not in reqs:
							reqs[player] = {}
						if 'disbands' not in reqs[player]:
							reqs[player]['disbands'] = []
						reqs[player]['disbands'].append(unit['loc'])
			
			for player, retreats in self.state.get('retreats', {}).items():
				for loc in retreats:
					base = self._to_base_region(loc)
					if base not in self.actions.get(player, {}):
						if player not in reqs:
							reqs[player] = {}
						if 'retreats' not in reqs[player]:
							reqs[player]['retreats'] = []
						reqs[player]['retreats'].append(loc.split('-c')[0])
			
		else:
			reqs = {player: [unit['loc'] for base, unit in units.items() if base not in self.actions.get(player, {})]
			        for player, units in self.units.items()}
			reqs = {player: sorted(locs) for player, locs in reqs.items() if len(locs)}
		
		return reqs
	
	
	
	def _unformat_actions(self, actions):
		return {player: list(actions.values()) for player, actions in actions.items()}
	
	
	def _checkpoint_actions(self):
		return save_yaml(self._unformat_actions(self.actions), self._get_action_path())
	
	
	def _checkpoint_state(self):
		return save_yaml(self.state, self._get_state_path())


	def verify_action(self, player, action):
		player = self.fix_player(player)

		if action.get('type') == 'build':
			if action['loc'] not in self.state['players'][player]['centers']:
				raise InvalidActionError(f'Build location {action["loc"]} is not a supply center')
			# check if location is a home center
			if action['loc'] not in self.state['players'][player]['home']:
				raise InvalidActionError(f'Build location {action["loc"]} is not a home center')
			if action.get('unit') == 'fleet' and self._region_type(action['loc']) != 'coast':
				raise ParsingFailedError(f'Fleet must be built on a coast (not {action["loc"]})')

		if action.get('unit', None) == 'fleet':
			if action.get('type') == 'convoy-transport' and self._region_type(action['loc']) != 'sea':
				raise InvalidActionError(f'To convoy a fleet must be on the sea (not {action["loc"]})')
			if action.get('type') == 'convoy-move':
				raise InvalidActionError(f'Fleets cannot move by convoy.')
			if action['type'] == 'move' and self._region_type(action['loc']) == 'land':
				raise InvalidActionError(f'Fleets cannot move to land regions.')
			for key in ['loc', 'dest', 'src']:
				if key in action:
					loc = action[key]
					coasts = self._extract_coasts(loc)
					if not self._is_coast(loc) and coasts is not None:
						raise ParsingFailedError(f'Coast required for {loc.upper()}: '
						                         f'{" or ".join(self._join_coast(loc.upper(), c) for c in coasts)}')

		if action.get('unit') == 'army':
			if action['type'] == 'convoy-transport':
				raise InvalidActionError(f'Armies cannot convoy.')
			if action['type'] in ['move', 'convoy-move'] and self._region_type(action['loc']) == 'sea':
				raise InvalidActionError(f'Armies cannot move to sea regions.')

		if action.get('type') == 'disband':
			if not any(action.get('loc') == unit['loc'] for unit in self.state['players'][player]['units']):
				raise InvalidActionError(f'Disband location {action["loc"]} has no a unit')

		return True
	
	
	def record_actions(self, actions, persistent=True):
		errors = {}
		for player, acts in actions.items():
			player = self.fix_player(player)
			ers = []
			for action in acts:
				try:
					self.record_action(player, action, persistent=False)
				except ParsingFailedError as e:
					ers.append((action, e))
			if len(ers):
				errors[player] = ers
		if persistent:
			self._checkpoint_actions()
		if len(errors):
			return errors
	
	
	def record_action(self, player, terms, persistent=True):
		player = self.fix_player(player)
		if isinstance(terms, str):
			terms = self.parse_action(player, terms)
		
		loc = self._to_base_region(terms['loc'])
		self.verify_action(player, terms)
		if player not in self.actions:
			self.actions[player] = {}
			
		if loc in self.actions[player]:
			print('WARNING: replacing existing order: "{}"'.format(
				self.format_action(player, self.actions[player][loc])))
		self.actions[player][loc] = terms
		if persistent:
			self._checkpoint_actions()
		return terms
		
	def remove_action(self, player, loc, persistent=True):
		player = self.fix_player(player)
		base = self._to_base_region(loc)
		
		action = None
		if player in self.actions and base in self.actions[player]:
			action = self.actions[player][base]
			del self.actions[player][base]
		if persistent:
			self._checkpoint_actions()
		return action
		
	def format_action(self, player, terms):
		player = self.fix_player(player)
		
		unit = 'A' if terms.get('unit') == 'army' else 'F'
		sunit = 'A' if terms.get('src-unit') == 'army' else 'F'
		
		if terms['type'] == 'move':
			return '**{loc}** *to* **{dest}**'.format(punit=unit, **terms)
			# return 'Move {punit} **{loc}** *to* **{dest}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'support':
			return '**{loc}** *supports* **{src}** *to* **{dest}**'.format(punit=unit, src_unit=sunit, **terms)
			# return '{punit} **{loc}** *supports* {src_unit} **{src}** to **{dest}**'.format(punit=unit, src_unit=sunit,
			#                                                                                 **terms)
		
		if terms['type'] == 'support-defend':
			return '**{loc}** *support holds* **{dest}**'.format(punit=unit, **terms)
			# return '{punit} **{loc}** *support holds* **{dest}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'disband':
			return '**{loc}** *disbands*'.format(punit=unit, **terms)
			# return '*Disband* {punit} **{loc}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'retreat':
			return '*Retreat* **{loc}** *to* **{dest}**'.format(punit=unit, **terms)
			# return '*Retreat* {punit} **{loc}** to **{dest}**'.format(punit=unit, **terms)

		if terms['type'] == 'build':
			return '*Build* **{punit}** *in* **{loc}**'.format(punit=unit, **terms)
			# return '*Build* {punit} in **{loc}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'hold':
			return '**{loc}** *holds*'.format(punit=unit, **terms)
			# return '*Hold* {punit} **{loc}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'convoy-transport':
			return '**{loc}** *convoys* **{src}** *to* **{dest}**'.format(src_unit=sunit, punit=unit, **terms)
			# return '{punit} **{loc}** *convoys* {src_unit} **{src}** to **{dest}**'.format(src_unit=sunit,
			#                                                                                punit=unit, **terms)
		
		if terms['type'] == 'convoy-move':
			return '**{loc}** *to* **{dest}** (by convoy)'.format(punit=unit, **terms)
			# return 'Move {punit} **{loc}** *to* **{dest}** (by convoy)'.format(punit=unit, **terms)
		
		raise Exception(f'Unknown action: {terms}')
		

	
	def format_all_actions(self, include_defaults=True):
		
		status = self.get_status()
		
		full = {}
		
		for player, actions in self.actions.items():
			
			orders = [self.format_action(player, action) for _, action in sorted(actions.items(),
			                                                                             key= lambda x: x[0])]
			if status.get(player, 0) and include_defaults:
				if self.season == 3:
					delta = self.state.get('adjustments', {}).get(player, 0)
					if delta > 0:
						orders.append(f'- Deferred {status.get(player,0)} build/s -')
					elif delta < 0:
						orders.append(f'- **Must still disband {status.get(player,0)} unit/s** -')
						
				elif self.retreat:
					orders.append(f'- **Must still retreat {status[player]} unit/s** -')
				else:
					orders.extend(
						'{} (by default)'.format(self.format_action(player,
							{'type': 'hold', 'loc': unit['loc'], 'unit': unit['type']}))
						for loc, unit in sorted(self.units[player].items(), key= lambda x: x[0])
						if loc not in actions
					)
			full[player] = orders
			
		return full


	def format_state(self, player=None):
		
		if player is None:
			return {player: self.format_state(player) for player in self.state['players']}
		else:
			player = self.fix_player(player)
		
		info = self.state['players'][player]
		
		lines = ['Centers: ' + ', '.join(info['centers'])]
		
		for unit in info['units']:
			loc, typ = unit['loc'], unit['type']
			lines.append(f'*{typ.capitalize()}* in **{loc}**')
		
		return lines

	
	@staticmethod
	def action_format():
		return {
			'move': '[X] to [Y]',
			'support move': '[X] supports [Y] to [Z]',
			'support hold': '[X] support holds [Y]',
			'convoy': '[X] convoys [Y] to [Z]',
			'hold': '[X] holds',
			'retreat': '[X] retreats to [Y]',
			'disband': '[X] disbands',
			'build': 'Build [T] in [X]',
		}

	
	def sample_action(self, player, n=1):
		player = self.fix_player(player)
		actions = []
		
		if self.retreat:
			if player in self.state.get('retreats', {}):
				retreats = self.state['retreats'][player]
				for loc, options in retreats.items():
					unit = self._find_unit(loc, player)
					actions.append(self.record_action(player, {'type': 'retreat', 'loc': loc, 'unit': unit,
					                                           'dest': random.choice(options)}))
		elif self.season == 3:
			delta = self.state.get('adjustments', {}).get(player)
			if delta is None or delta == 0:
				return
			
			if delta < 0:
				locs = random.sample(list(self.units[player]), k=abs(delta))
				for loc in locs:
					unit = self._find_unit(loc, player)
					action = {'type': 'disband', 'loc': loc, 'unit': unit}
					actions.append(self.record_action(player, action, persistent=False))
			else:
				empty = [home for home in self.state['players'][player]['home'] if self._find_unit(home) is None]
				delta = min(len(empty), delta)
				locs = random.sample(empty, k=min(len(empty), delta))
				for loc in locs:
					unit = random.choice([utype for utype in ['army', 'fleet']
					                      if utype in self.graph[self._to_base_region(loc)]['edges']])
					action = {'type': 'build', 'loc': loc, 'unit': unit}
					actions.append(self.record_action(player, action, persistent=False))
		else:
			weights = {
				'move': 3,
				'hold': 1,
				'support': 3,
				'support-defend': 3,
			}
			
			typs, wts = zip(*weights.items())
			
			locs = [u['loc'] for loc, u in self.units[player].items()
			                      if loc not in self.actions[player]]
			if n > 0:
				locs = random.sample(locs, k=n)
			else:
				n = len(locs)
			print(f'Generating {len(locs)} action/s for {player}.')
			
			typs = random.choices(typs, weights=wts, k=len(locs))
			for loc, typ in zip(locs, typs):
				unit = self._find_unit(loc, player)
				
				neighbors = self._get_neighbors(loc, unit).copy()
				occupied = dict(self._find_loc_unit(loc) for loc in neighbors)
				if None in occupied:
					del occupied[None]
				occupied = {loc: (utype, [neighbor for neighbor in self._get_neighbors(loc, utype)
					                      if neighbor in neighbors])
				            for loc, utype in occupied.items() if utype is not None}
				occupied = {loc: (utype, dests) for loc, (utype, dests) in occupied.items() if len(dests)}
				
				if len(occupied) == 0 and 'support' in typ:
					typ = 'move'

				action = {'type': typ, 'loc': loc, 'unit': unit}
				if typ == 'move':
					dest = random.choice(neighbors)
					action['dest'] = dest
				elif typ == 'support':
					src = random.choice(list(occupied))
					src_unit, dests = occupied[src]
					dest = random.choice(dests)
					action.update({'dest': dest, 'src-unit':src_unit, 'src': src})
				elif typ == 'support-defend':
					dest = random.choice(list(occupied))
					action['dest'] = dest
				actions.append(self.record_action(player, action, persistent=False))
			
		self._checkpoint_actions()
		return actions

	
	def _to_base_region(self, name):
		if name not in self._get_base_region:
			raise LocationError(name)
		return self._get_base_region[name]
	
	
	def _to_region_name(self, name):
		if name not in self._get_region:
			raise LocationError(name)
		return self._get_region[name]

	def region_full_name(self, code: str):
		if code not in self._get_region:
			raise LocationError(code)
		code = self._get_region[code]
		base = self._get_base_region[code]

		info = self.graph[base]
		name = info['name']

		if code != base:
			coast = code.split('-')[-1]
			if len(coast) == 2:
				coast = {'nc': 'North Coast', 'sc': 'South Coast', 'ec': 'East Coast', 'wc': 'West Coast'}[coast]
				name += f' ({coast})'
		return name
	
	def _has_coasts(self, loc):
		loc = self._to_region_name(loc)
		return loc in self.graph and 'fleet' in self.graph[loc]['edges'] \
		       and isinstance(self.graph[loc]['edges']['fleet'], dict)

	def _region_type(self, loc): # returns 'coast', 'land', 'sea', or None
		loc = self._to_region_name(loc)
		if loc in self.graph:
			return self.graph[loc]['type']
		else:
			return None
	
	def _extract_coasts(self, loc):
		loc = self._to_region_name(loc)
		if self._has_coasts(loc):
			return list(self.graph[loc]['edges']['fleet'].keys())
	
	
	def _is_coast(self, loc):
		return self._to_region_name(loc) != self._to_base_region(loc)
	
	
	def _extract_coast(self, loc):
		if self._is_coast(loc):
			return loc.split('-')[-1]
		
	
	def _join_coast(self, loc, coast):
		return f'{loc}-{coast}'
	
	
	def _is_neighbor(self, src, unit, dest):
		base = self._to_base_region(src)
		if self._is_coast(src):
			options = self.graph[base]['edges']['fleet'][self._extract_coast(src)]
		else:
			options = self.graph.get(base, {}).get('edges', {}).get(unit, [])
		return dest in options
	
	
	def _parse_location(self, loc):#, player=None, unit=None):
		# if unit is None:
		# 	for text, utype in self._unit_texts.items():
		# 		if loc.startswith(f'{text} '):
		# 			unit = utype
		
		loc = self._to_region_name(loc)
		# base = self._to_base_region(loc)
		
		# if unit is None and player is not None:
		# 	unit = self.units[player].get(base, {}).get('type', None)
		# if unit == 'fleet' and self._has_coasts(base) and not self._is_coast(loc):
		# 	raise MissingCoastError(loc)
		return loc#, unit
	
	
	def _get_neighbors(self, loc, unit=None):
		base = self._to_base_region(loc)
		if unit is None:
			unit = self._find_unit(loc)
		assert unit is not None
		edges = self.graph[base]['edges'][unit]
		if isinstance(edges, dict):
			coast = self._extract_coast(loc)
			if coast is None:
				raise MissingCoastError(loc)
			return edges[coast]
		return edges
	
	
	def _find_unit(self, loc, player=None):
		return self._find_loc_unit(loc, player)[1]
	
	
	def _find_loc_unit(self, loc, player=None):
		base = self._to_base_region(loc)
		if player is None:
			for player, units in self.units.items():
				if base in units:
					break
			else:
				return None, None
		unit = self.units[player].get(base, {})#.get('type', None)
		loc, unit = unit.get('loc'), unit.get('type')
		if unit == 'fleet' and self._has_coasts(base) and not self._is_coast(loc):
			raise MissingCoastError(loc)
		return loc, unit

	def find_unit(self, loc, player=None):
		loc, unit = self._find_loc_unit(loc, player)
		return {'loc': loc, 'unit': unit} if unit is not None else None
	
	
	def _parse_unit(self, unit):
		utype = self._unit_texts.get(unit, unit)
		if utype not in {'army', 'fleet'}:
			raise UnknownUnitTypeError(unit)
		return utype

	def player_order_context(self, player):
		if 'adjustments' in self.state:
			delta = self.state['adjustments'].get(player, 0)
			if delta == 0:
				return None
			elif delta < 0:
				return 'lose'
			else:
				return 'gain'
		if 'retreats' in self.state:
			return 'retreat'
		return 'action'

	def compute_scores(self) -> list[tuple[str, int]]:
		data = {player: len(self.state['players'][player]['centers']) for player in self.state['players']}
		scores = sorted(data.items(), key=lambda x: x[1], reverse=True)
		return scores

	
	def parse_action(self, player, text, terms=None):
		player = self.fix_player(player)

		line = text.lower()

		context = self.player_order_context(player)
		if context is None:
			raise InvalidActionError(f'You have no actions available.')

		options = parse_order(line, context)
		if options is None:
			raise ParsingFailedError(f'Could not parse action: "{text}"')

		if len(options) > 1:
			keys = f', '.join(f'`{key}`' for key in options)
			raise AmbiguousOrderError(f'Can\'t decide between: {keys}')

		order_type = list(options.keys())[0]
		terms = options[order_type]
		terms = standardize_order(terms)

		select_keys = {'loc': 'loc', 'dest': 'dest', 'target': 'src', 'unit': 'unit', 'tunit': 'src-unit'}
		terms = {select_keys[k]: v for k, v in terms.items() if k in select_keys}
		for loc_key in ['loc', 'dest', 'src']:
			if loc_key in terms:
				terms[loc_key] = self._parse_location(terms[loc_key])
		if terms.get('unit') is None:
			terms['unit'] = self._find_unit(terms['loc'], player)
		if (order_type == 'move' and 'dest' in terms
				and not self._is_neighbor(terms['loc'], terms['unit'], terms['dest'])):
			order_type = 'convoy-move'
		if 'src-unit' in terms and terms['src-unit'] is None:
			terms['src-unit'] = self._find_unit(terms['src'], player)
		order_types = {'support_hold': 'support-defend', 'support_move': 'support', 'convoy': 'convoy-transport', }
		terms['type'] = order_types.get(order_type,order_type)
		return terms
	
	
	def find_image_path(self, include_actions=False):
		path = self._generate_image_path(include_actions)
		if path.exists():
			return path
	
	
	def _generate_image_path(self, include_actions=False):
		name = f'{self.time}-actions.png' if include_actions else f'{self.time}.png'
		return self.images_root / name
	
	
	def render_latest(self, path=None, include_actions=False):
		
		if path is None:
			path = self._generate_image_path(include_actions)
		print(f'Rendering {path.stem}')
		
		self.renderer(self.state, self.actions if include_actions else None, savepath=path)
		return path
	
	
	







