import random
import sys, os
import shutil
from pathlib import Path
from omnibelt import unspecified_argument, load_yaml, save_yaml, load_txt
import omnifig as fig


_DEFAULT_ROOT = str(Path(__file__).parents[0])


class ParsingFailedError(Exception):
	pass


class UnknownUnitTypeError(ParsingFailedError):
	pass


class MissingCoastError(ParsingFailedError):
	pass


class NoUnitFoundError(ParsingFailedError):
	pass
	
	
class LocationError(ParsingFailedError):
	pass

		
class BadGraphError(Exception):
	pass
	
	
class BadNamesError(BadGraphError):
	def __init__(self, names):
		super().__init__('These region names are ambiguous: {}'.format(', '.join(names)))


class DiplomacyRenderer:
	pass



@fig.Script('create-game')
def _create_game(A):
	root = A.pull('base-root', '<>root', None)
	if root is None:
		root = Path(_DEFAULT_ROOT) / 'data'
	root = Path(root)
	mdir = A.pull('map-path', None)
	if mdir is None:
		raise FileNotFoundError('You must provide the path to the directory with the map data (--map-path)')
	
	name = A.pull('name', None)
	if name is None:
		num = len(list(root.glob('*'))) + 1
		name = f'game{num}'
	
	path = root / name
	# path.mkdir(exist_ok=True)
	shutil.copytree(str(mdir), str(path))
	
	print(f'Game {name} has been created (using map data in {str(mdir)})')
	return path



@fig.Component('diplo-manager')
class DiplomacyManager(fig.Configurable):
	def __init__(self, A, gamemap=unspecified_argument, renderer=unspecified_argument,
	             game_root=unspecified_argument, **kwargs):
		game_root = self._find_root(A, root=game_root)

		graph_path = game_root / 'graph.yaml'
		player_path = game_root / 'players.yaml'
		
		states_root = game_root / 'states'
		actions_root = game_root / 'actions'
		images_root = game_root / 'images'
		
		if gamemap is unspecified_argument:
			gamemap = A.pull('map')
		
		if renderer is unspecified_argument:
			renderer = A.pull('renderer', None)
		
		super().__init__(A, **kwargs)
		
		if not states_root.exists():
			states_root.mkdir()
		if not actions_root.exists():
			actions_root.mkdir()
		if not images_root.exists():
			images_root.mkdir()
		
		self.gamemap = gamemap
		self.renderer = renderer
		
		self.root = game_root
		self.graph_path = graph_path
		self.player_path = player_path
		
		self.states_root = states_root
		self.actions_root = actions_root
		self.images_root = images_root
		
		self.current_state = None
		
		
	@staticmethod
	def _find_root(A, root=unspecified_argument):
		if root is unspecified_argument:
			root = A.pull('game-root', '<>game_root', None)
		if root is None:
			root = fig.run('create-game', A)
		root = Path(root)
		
		graph_path = root / 'graph.yaml'
		if not graph_path.exists():
			raise FileNotFoundError(str(graph_path))
		A.push('graph-path', str(graph_path), overwrite=False, silent=True)
		
		player_path = root / 'players.yaml'
		if not player_path.exists():
			raise FileNotFoundError(str(player_path))
		A.push('players-path', str(player_path), overwrite=False, silent=True)

		A.push('region-path', str(root / 'regions.png'), overwrite=False, silent=True)
		A.push('renderbase-path', str(root / 'renderbase.png'), overwrite=False, silent=True)
		A.push('tiles-path', str(root / 'tiles.png'), overwrite=False, silent=True)
		return root
		
		
	def load_status(self):
		self.graph = load_yaml(self.graph_path)
		self._get_base_region = {f'{base}-{coast}':base for base, node in self.graph.items()
		                         if 'fleet' in node['edges'] and isinstance(node['edges']['fleet'], dict)
		                         for coast in node['edges']['fleet']}
		self._get_base_region.update({base: base for base in self.graph})
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
		
		self.player_info = load_yaml(self.player_path)
		
		self.set_state(self._find_latest_state(self.states_root))
		print(f'Loaded state: {self.time}')
		
		
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
	
	
	def _extract_time(self, state):
		year = int(state['time']['turn'])
		season = int(state['time']['season'])
		retreat = state['time'].get('retreat', False)
		retreat = '-r' if retreat else ''
		self.year, self.season, self.retreat = year, season, retreat
		self.time = self._season_date(year, season, retreat)
	
	
	def _season_date(self, year, season, retreat):
		return f'{year}-{season}{retreat}'
	
	
	def _get_action_path(self):
		return self.actions_root / f'{self.time}.yaml'

	
	def _get_state_path(self):
		return self.states_root / f'{self.time}.yaml'

	
	def _find_latest_state(self, state_root, persistent=True):
		state_paths = list(state_root.glob('*'))
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
		players = {}
		for name, info in self.player_info.items():
			player = {}
			player['control'] = info['territory'].copy()
			player['centers'] = [loc for loc in info['territory'] if self.graph.get(loc, {}).get('sc', 0) > 0]
			player['units'] = [{'loc': loc, 'type': typ} for typ in ['army', 'fleet'] for loc in info.get(typ, [])]
			players[name] = player
		return {'players': players, 'time': {'turn': 1, 'season': 1}}
	
	
	def get_status(self):
		missing = {player: len(self.units[player]) - len(self.actions.get(player, {}))
		           for player in self.actions}
		return {player: num for player, num in missing.items() if num > 0}
	
	
	def _unformat_actions(self, actions):
		return {player: list(actions.values()) for player, actions in actions.items()}
	
	
	def _checkpoint_actions(self):
		return save_yaml(self._unformat_actions(self.actions), self._get_action_path())
	
	
	def _checkpoint_state(self):
		return save_yaml(self.state, self._get_state_path())


	def verify_action(self, player, action):
		return True
	
	
	def record_actions(self, actions, persistent=True):
		errors = {}
		for player, acts in actions.items():
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
		if isinstance(terms, str):
			terms = self.parse_action(player, terms)
		
		loc = self._to_base_region(terms['loc'])
		# self.verify_action(player, terms)
		if player not in self.actions:
			self.actions[player] = {}
			
		if loc in self.actions[player]:
			print('WARNING: replacing existing order: "{}"'.format(
				self.format_action(player, self.actions[player][loc])))
		self.actions[player][loc] = terms
		if persistent:
			self._checkpoint_actions()
		return terms
		
		
	def format_action(self, player, terms):
		
		unit = 'A' if terms['unit'] == 'army' else 'F'
		sunit = 'A' if terms.get('src-unit') == 'army' else 'F'
		
		if terms['type'] == 'move':
			return '*Move* **{loc}** *to* **{dest}**'.format(punit=unit, **terms)
			# return 'Move {punit} **{loc}** *to* **{dest}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'support':
			return '**{loc}** *supports* **{src}** *to* **{dest}**'.format(punit=unit, src_unit=sunit, **terms)
			# return '{punit} **{loc}** *supports* {src_unit} **{src}** to **{dest}**'.format(punit=unit, src_unit=sunit,
			#                                                                                 **terms)
		
		if terms['type'] == 'support-defend':
			return '**{loc}** *support holds* **{dest}**'.format(punit=unit, **terms)
			# return '{punit} **{loc}** *support holds* **{dest}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'disband':
			return '*Disband* **{loc}**'.format(punit=unit, **terms)
			# return '*Disband* {punit} **{loc}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'retreat':
			return '*Retreat* **{loc}** *to* **{dest}**'.format(punit=unit, **terms)
			# return '*Retreat* {punit} **{loc}** to **{dest}**'.format(punit=unit, **terms)

		if terms['type'] == 'build':
			return '*Build **{punit}** *in* **{loc}**'.format(punit=unit, **terms)
			# return '*Build* {punit} in **{loc}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'hold':
			return '*Hold* **{loc}**'.format(punit=unit, **terms)
			# return '*Hold* {punit} **{loc}**'.format(punit=unit, **terms)
		
		if terms['type'] == 'convoy':
			return '**{loc}** *convoys* **{src}** *to* **{dest}**'.format(src_unit=sunit, punit=unit, **terms)
			# return '{punit} **{loc}** *convoys* {src_unit} **{src}** to **{dest}**'.format(src_unit=sunit,
			#                                                                                punit=unit, **terms)
		
		if terms['type'] == 'convoy-move':
			return '*Move* **{loc}** *to* **{dest}** (by convoy)'.format(punit=unit, **terms)
			# return 'Move {punit} **{loc}** *to* **{dest}** (by convoy)'.format(punit=unit, **terms)
		
		raise NotImplementedError

	
	def sample_action(self, player, n=1):
		
		actions = []
		
		if self.retreat:
			if player not in self.state.get('retreats', {}):
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
				empty = [home for home in self.state['players']['home'] if self._find_unit(home) is None]
				delta = min(len(empty), delta)
				locs = random.sample(empty, k=min(len(empty), delta))
				for loc in locs:
					unit = random.choice([utype for utype in ['army', 'fleet']
					                      if utype in self.graph[self._to_base_region(loc)]['edges']])
					action = {'type': 'build', 'loc': loc, 'unit': unit}
					actions.append(self.record_action(player, action, persistent=False))
		else:
			if n < 0:
				n = max(1, len(self.units[player]) - len(self.actions[player]))
			print(f'Generating {n} action/s for {player}.')
			
			weights = {
				'move': 3,
				'hold': 1,
				'support': 3,
				'support-defend': 3,
			}
			
			typs, wts = zip(*weights.items())
			
			locs = random.sample([u['loc'] for loc, u in self.units[player].items()
			                      if loc not in self.actions[player]], k=n)
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
	
	
	def _has_coasts(self, loc):
		loc = self._to_region_name(loc)
		return loc in self.graph and 'fleet' in self.graph[loc]['edges'] \
		       and isinstance(self.graph[loc]['edges']['fleet'], dict)
	
	
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
				raise NoUnitFoundError(loc)
		unit = self.units[player].get(base, {})#.get('type', None)
		loc, unit = unit.get('loc'), unit.get('type')
		if unit == 'fleet' and self._has_coasts(base) and not self._is_coast(loc):
			raise MissingCoastError(loc)
		return loc, unit
	
	
	def _parse_unit(self, unit):
		utype = self._unit_texts.get(unit, unit)
		if utype not in {'army', 'fleet'}:
			raise UnknownUnitTypeError(unit)
		return utype
	
	
	def parse_action(self, player, text, terms=None):
		if terms is None:
			terms = {}
		
		# terms['player'] = player
		line = text.lower()
		
		if line.startswith('build '):
			_, info = line.split('build ')
			unit, loc = info.split(' in ')
			loc = self._parse_location(loc)
			unit = self._parse_unit(unit)
			terms.update({'type': 'build', 'loc': loc, 'unit': unit})
		
		elif ' disband' in line:
			loc, _ = line.split(' disband')
			loc = self._parse_location(loc)
			unit = self._find_unit(player, loc)
			if unit is None:
				raise NoUnitFoundError(loc)
			terms.update({'type': 'disband', 'loc': loc, 'unit': unit})
		
		elif ' retreats to ' in line:
			loc, dest = line.split(' retreats to ')
			loc = self._parse_location(loc)
			unit = self._find_unit(player, loc)
			if unit is None:
				raise NoUnitFoundError(loc)
			dest = self._parse_location(dest)
			terms.update({'type': 'retreat', 'loc': loc, 'unit': unit, 'dest': dest})
		
		elif ' supports ' in line and ' to ' in line:
			loc, rest = line.split(' supports ')
			loc = self._parse_location(loc)
			unit = self._find_unit(player, loc)
			if unit is None:
				raise NoUnitFoundError(loc)
			src, dest = rest.split(' to ')
			src = self._parse_location(src)
			src_unit = self._find_unit(src)
			dest = self._parse_location(dest)
			terms.update({'type': 'support', 'loc': loc, 'unit': unit,
			              'src': src, 'src-unit': src_unit, 'dest': dest})
		
		elif ' support holds ' in line or ' supports ' in line:
			keyword = ' support holds ' if ' support holds ' in line else ' supports '
			loc, dest = line.split(keyword)
			loc = self._parse_location(loc)
			unit = self._find_unit(player, loc)
			if unit is None:
				raise NoUnitFoundError(loc)
			dest = self._parse_location(dest)
			terms.update({'type': 'support-defend', 'loc': loc, 'unit': unit, 'dest': dest})
		
		elif ' hold' in line:
			loc, _ = line.split(' hold')
			loc = self._parse_location(loc)
			unit = self._find_unit(player, loc)
			if unit is None:
				raise NoUnitFoundError(loc)
			terms.update({'type': 'hold', 'loc': loc, 'unit': unit})
		
		elif ' convoys ' in line:
			loc, rest = line.split(' convoys ')
			loc = self._parse_location(loc)
			unit = self._find_unit(player, loc)
			if unit is None:
				raise NoUnitFoundError(loc)
			src, dest = rest.split(' to ')
			src = self._parse_location(src)
			src_unit = self._find_unit(src)
			dest = self._parse_location(dest)
			terms.update({'type': 'convoy-transport', 'loc': loc, 'unit': unit,
			              'src': src, 'src-unit': src_unit, 'dest': dest})
		
		elif ' to ' in line:
			loc, dest = line.split(' to ')
			loc = self._parse_location(loc)
			unit = self._find_unit(loc, player)
			if unit is None:
				raise NoUnitFoundError(loc)
			dest = self._parse_location(dest)
			action_type = 'move' if unit == 'fleet' or self._is_neighbor(loc, unit, dest) else 'convoy-move'
			terms.update({'type': action_type, 'loc': loc, 'dest': dest, 'unit': unit})
			if action_type == 'convoy-move':
				print(f'Convoying: {line} - {terms}')
		
		else:
			raise ParsingFailedError(line)
		
		return terms
		
	
	def render_latest(self, include_actions=False):
		return self.renderer.render(self.state, self.actions if include_actions else None)
	
	







