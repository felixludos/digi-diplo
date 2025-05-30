
import sys, os
from itertools import chain
from omnibelt import load_yaml, save_yaml, unspecified_argument, save_json, load_json

import omnifig as fig

from tabulate import tabulate

import pydip
from pydip.map import Map
from pydip.player import Player, command, Unit, UnitTypes
# from pydip.unit import Unit
from pydip.turn import resolve_turn, resolve_retreats
from pydip.turn.adjustment import resolve_adjustment__validated, calculate_adjustments
from pydip.map.map import OwnershipMap, SupplyCenterMap

from . import util
from copy import deepcopy



@fig.component('diplo-map')
class DiploMap(util.Versioned):
	def __init__(self, graph_path, players_path=None,
	             ignore_unknown=None, fix_types=True, **kwargs):
		
		if ignore_unknown is None:
			ignore_unknown = type(self) != DiploMap
		
		super().__init__(**kwargs)
		
		self.graph_path = graph_path
		self.player_path = players_path

		self.auto_fix_types = fix_types
		self.nodes, self.edges = self._load_graph_info(graph_path)
		self.player_info = self._load_player_info(players_path)
		self.get_ID_from_name = util.make_node_dictionary(self.nodes)
		
		self.dmap = self._create_dip_map(self.nodes, self.edges)
		
		self.ignore_unknown = ignore_unknown
	
	
	def generate_initial_state(self):
		players = {}
		for name, info in self.player_info.items():
			player = {}
			player['control'] = info['territory'].copy()
			player['centers'] = [loc for loc in info['territory'] if self.graph.get(loc, {}).get('sc', 0) > 0]
			player['home'] = player['centers'].copy()
			player['units'] = [{'loc': loc, 'type': typ} for typ in ['army', 'fleet'] for loc in info.get(typ, [])]
			player['units'].extend(unit.copy() for unit in info.get('units', []))
			players[name] = player
		return {'players': players, 'time': {'turn': 1, 'season': 1}}
	
	
	def get_supply_centers(self):
		return [node for node in self.nodes if 'sc' in node and node['sc'] > 0]


	@staticmethod
	def encode_region_name(name=None, coast=unspecified_argument, node_type=None, unit_type=None):
		args = dict(name=name, unit_type=unit_type, node_type=node_type)
		if coast is not unspecified_argument:
			args['coast'] = coast
		return fig.quick_run('_encode_region_name', **args)
	
	
	@staticmethod
	def decode_region_name(name=None, allow_dir=None):
		return fig.quick_run('_decode_region_name', name=name, allow_dir=allow_dir)
	
	
	def get_node(self, name):
		return self.graph[name]
	
	@staticmethod
	def _load_player_info(player_path):
		return load_yaml(player_path)
	
	def _fix_node_type(self, node):
		edges = node['edges']
		if 'army' in edges and 'fleet' in edges:
			return 'coast'
		if 'army' in edges:
			return 'land'
		return 'sea'
	
	def _load_graph_info(self, graph_path):
		
		graph = load_yaml(graph_path) if graph_path.endswith('.yaml') else load_json(graph_path)
		for key, info in graph.items():
			try:
				if self.auto_fix_types:
					info['type'] = self._fix_node_type(info)
					
				if 'army' in info['edges']:
					info['army-edges'] = info['edges']['army']
				if 'fleet' in info['edges']:
					info['fleet-edges'] = info['edges']['fleet']
			except:
				print('Error processing the following node in the graph:')
				print(key, info)
				raise
		coasts = {}
		
		edges = {'army': {name: info['army-edges'] for name, info in graph.items() if 'army-edges' in info}}
		
		fleet = {}
		edges['fleet'] = fleet
		
		for name, info in graph.items():
			if 'fleet-edges' in info:
				es = info['fleet-edges']
				if isinstance(es, dict):
					info['coasts'] = []
					for coast, ces in es.items():
						coast = self.encode_region_name(name=name, coast=coast)
						coasts[coast] = {'name': coast, 'type': 'coast', 'coast-of': name, 'dir': coast}
						info['coasts'].append(coast)
						fleet[coast] = ces
				else:
					if info['type'] == 'coast':
						coast = self.encode_region_name(name=name, node_type='coast', unit_type='fleet')
						coasts[coast] = {'name': coast, 'type': 'coast', 'coast-of': name}
						info['coasts'] = [coast]
					fleet[name] = es
		
		nodes = graph
		
		self.graph = graph
		self.coasts = coasts
		
		return nodes, edges

	def _load_map_info(self, nodes_path, edges_path):
		
		nodes = load_yaml(nodes_path)
		
		coasts = util.separate_coasts(nodes)
		
		self.coasts = coasts
		
		for ID, coast in coasts.items():
			origin = coast['coast-of']
			if 'coasts' not in nodes[origin]:
				nodes[origin]['coasts'] = []
			nodes[origin]['coasts'].append(ID)
		
		
		for ID, node in nodes.items():
			node['ID'] = ID
		
		edges = load_yaml(edges_path)
		
		return nodes, edges

	@classmethod
	def _create_dip_map(cls, nodes, edges):
		
		# descriptors
		
		descriptors = []
		
		for ID, node in nodes.items():
			desc = {'name': ID}
			# if node['type'] != 'sea':
			# 	desc['coasts'] = node.get('coasts', [])
			
			desc['coasts'] = []
			if node['type'] == 'sea':
				del desc['coasts']
			elif 'fleet' in node['edges']:
				if isinstance(node['edges']['fleet'], dict):
					desc['coasts'] = [cls.encode_region_name(name=ID, unit_type='fleet', coast=coast)
					                  for coast in node['edges']['fleet']]
				else:
					desc['coasts'] = [cls.encode_region_name(name=ID, unit_type='fleet', coast=True)]
			descriptors.append(desc)
			
		# adjacencies
		
		adjacencies = set()
		for ID, node in nodes.items():
			for utype, edges in node['edges'].items():
				start = cls.encode_region_name(name=ID, unit_type=utype,
				                               node_type=nodes[ID]['type'] if ID in nodes else None)
				if isinstance(edges, dict):
					for coast, edges in edges.items():
						start = cls.encode_region_name(name=ID, unit_type=utype, coast=coast,
						                               node_type=nodes[ID]['type'] if ID in nodes else None)
						for neighbor in edges:
							end = cls.encode_region_name(name=neighbor, unit_type=utype,
							                             node_type=nodes[neighbor]['type'] if neighbor in nodes else None)
							if (end, start) not in adjacencies:
								adjacencies.add((start, end))
				else:
					for neighbor in edges:
						end = cls.encode_region_name(name=neighbor, unit_type=utype,
						                             node_type=nodes[neighbor]['type'] if neighbor in nodes else None)
						if (end, start) not in adjacencies:
							adjacencies.add((start, end))

		adjacencies = list(adjacencies)
		return Map(descriptors, adjacencies)

	def fix_loc(self, loc, utype):
		if utype in util.UNIT_ENUMS:
			utype = util.UNIT_ENUMS[utype]
		return self.encode_region_name(name=loc, unit_type=utype,
		                               node_type=self.nodes[loc]['type'] if loc in self.nodes else None)
		return util.fix_loc(loc, utype, self.nodes[loc]['type']) if loc in self.nodes else loc

	def prep_players(self, players):
		
		full = {}
		unit_info = []
		unit_locs = {} # player -> base -> {loc, type}
		
		for name, player in players.items():
			
			tiles = {loc: None for loc in player.get('control', [])}
			
			units = {unit['loc']: unit['type']
			              for unit in player.get('units', [])}
			
			locs = {}
			unit_locs[name] = locs
			
			for loc, unit in units.items():
				base = self.decode_region_name(name=loc)[0]
				locs[base] = {'loc': loc, 'unit': unit}
			
			for loc in units:
				if loc in self.coasts:
					cst = self.coasts[loc]['coast-of']
					if cst in tiles:
						del tiles[cst]
			
			tiles.update(units)
			
			unit_info.extend([loc, name, utype] for loc, utype in units.items())
			
			config = [{'territory_name':self.fix_loc(loc, utype),
                        'unit_type': None if utype is None else util.UNIT_TYPES[utype],
		            } for loc, utype in tiles.items()]
			
			try:
				full[name] = Player(name=name, game_map=self.dmap,
			                    starting_configuration=config)
			except AssertionError:
				print(name)
				raise
		
		self.unit_locs = unit_locs
		
		units = {}
		for loc, player, utype in unit_info:
			fixed = self.fix_loc(loc, utype)
			base = self.uncoastify(loc, True)
			unit = full[player].find_unit(fixed)
			
			for loc in {loc, fixed, base}:
				if loc in units:
					unit = None
				units[loc] = unit
		units = {loc:unit for loc, unit in units.items() if unit is not None}
			
		self.units = units
		self.players = full
		return self.players
	
	def get_unit(self, loc, utype=None, player=None):
		
		if loc in self.units:
			return self.units[loc]
		
		if utype is None and loc in self.nodes and self.nodes[loc]['type'] != 'coast':
			utype = 'army' if self.nodes[loc]['type'] == 'land' else 'fleet'
		
		loc = self.fix_loc(loc, utype)
		
		if player is not None:
			player = self.players.get(player, player)
			try:
				return player.find_unit(loc)
			except:
				try:
					return player.find_unit(self.fix_loc(loc, 'fleet'))
				except:
					pass
				print(player.name, loc, utype)
				raise
		
		if utype in util.UNIT_TYPES:
			utype = util.UNIT_TYPES[utype]
		
		return Unit(utype, loc)
		
	def _find_dest(self, src, utype, dest):
		
		if utype in util.UNIT_ENUMS:
			utype = util.UNIT_ENUMS[utype]
		
		sbase, scoast = self.decode_region_name(name=src)
		dbase, dcoast = self.decode_region_name(name=dest)
		
		edges = self.graph[sbase]['edges'][utype]
		if isinstance(edges, dict):
			edges = edges[scoast]
		
		for e in edges:
			b, c = self.decode_region_name(name=e)
			if b == dbase:
				return self.encode_region_name(name=b, coast=c,
				                               node_type=self.graph[b]['type'], unit_type=utype)
		
		return self.encode_region_name(name=dbase, coast=dcoast, unit_type=utype)
		raise Exception(f'Cant find an edge: src={src}, utype={utype}, dest={dest}')
		
	
	def _fill_missing_actions(self, full):
		existing = {name: {a['loc']: a for a in actions} for name, actions in full.items()}
		
		for name, player in self.players.items():
			for unit in player.units:
				utype = util.UNIT_ENUMS[unit.unit_type]
				loc = self.uncoastify(unit.position)
				base = self.uncoastify(loc, True)
				
				if name not in existing:
					full[name] = []
				actions = existing.get(name, {})
				if loc not in actions and base not in actions:
					full[name].append({'loc':loc, 'type':'hold', 'unit':utype})
		
		return full
	
	def _process_action(self, name, action):
		player = self.players[name]
		unit = self.get_unit(action['loc'], action.get('unit', None))
		if action['type'] == 'move':
			dest = self.fix_loc(action['dest'], unit.unit_type)
			return command.MoveCommand(player, unit, dest)
		elif action['type'] == 'hold':
			return command.HoldCommand(player, unit)
		elif 'support' in action['type']:
			if 'defend' in action['type']:
				sup_unit = self.get_unit(action['dest'])
				dest = sup_unit.position
			else:
				sup_unit = self.get_unit(action['src'], utype=action.get('src-unit', None))
				dest = self._find_dest(action['src'], sup_unit.unit_type, action['dest'])
			try:
				return command.SupportCommand(player, unit, sup_unit, dest)
			except:
				print(player, unit, sup_unit, dest)
				raise
		elif action['type'] == 'convoy-move':
			dest = self.fix_loc(action['dest'], unit.unit_type)
			return command.ConvoyMoveCommand(player, unit, dest)
		elif action['type'] == 'convoy-transport':
			transport = self.get_unit(action['src'], action.get('unit', None), player=name)
			dest = self.fix_loc(action['dest'], transport.unit_type)
			return command.ConvoyTransportCommand(player, unit, transport, dest)
		
		raise Exception(f'unknown: {action}')
	
	def process_actions(self, full, ignore_unknown=False):
		
		assert hasattr(self, 'players'), 'players have not been loaded'
		
		full = self._fill_missing_actions(full)
		
		cmds = []
		unknown = {}
		
		for name, actions in full.items():
			for action in actions:
				try:
					cmds.append(self._process_action(name, action))
				except:
					print('FAILED ORDER:', name, action)
					if ignore_unknown:
						cmds.append(self._process_action(name, {'loc': action['loc'], 'unit': action['unit'],
						                                        'type': 'hold'}))
						if name not in unknown:
							unknown[name] = []
						unknown[name].append(action)
					else:
						raise
		
		return cmds, unknown
	

	def _process_retreat(self, name, action, retreats):
		player = self.players[name]
		unit = self.get_unit(action['loc'], action.get('unit', None), player=name)
		
		if action['type'] == 'disband':
			if unit in retreats[name]:
				return command.RetreatDisbandCommand(retreats, player, unit)
		elif action['type'] == 'retreat':
			dest = self.fix_loc(action['dest'], unit.unit_type)
			return command.RetreatMoveCommand(retreats, player, unit, dest)

		raise Exception(f'unknown: {action}')
		
		
	def process_retreats(self, full, retreats, ignore_unknown=False):
		
		assert hasattr(self, 'players'), 'players have not been loaded'
		
		cmds = []
		unknown = {}
		done = set()
		for name, actions in full.items():
			for action in actions:
				try:
					retreat = self._process_retreat(name, action, retreats)
					cmds.append(retreat)
					done.add(retreat.unit)
				except:
					print('FAILED ORDER:', name, action)
					
					# if ignore_unknown:
					# 	cmds.append(self._process_action(name, {'loc': action['loc'], 'unit': action['unit'],
					# 	                                        'type': 'hold'}))
					# 	if name not in unknown:
					# 		unknown[name] = []
					# 	unknown[name].append(action)
					raise
		
		for name, rs in retreats.items():
			for unit, options in rs.items():
				if options is not None and unit not in done:
					cmds.append(command.RetreatDisbandCommand(retreats, self.players[name], unit))
					
		return cmds, unknown
	

	def _process_build(self, name, action, ownership):
		player = self.players[name]
		unit = self.get_unit(action['loc'], action.get('unit', None))
		
		if action['type'] == 'build':
			return command.AdjustmentCreateCommand(ownership, player, unit)
		elif action['type'] == 'destroy' or action['type'] == 'disband':
			return command.AdjustmentDisbandCommand(player, unit)
			
		raise Exception(f'unknown: {action}')
		
		
	def process_builds(self, full, ownership, ignore_unknown=False):
		
		assert hasattr(self, 'players'), 'players have not been loaded'
		
		cmds = []
		unknown = {}
		
		for name, actions in full.items():
			for action in actions:
				try:
					cmds.append(self._process_build(name, action, ownership))
				except:
					print(name, action)
					raise
		
		return cmds, unknown
		
	def _compute_retreat_map(self, retreats, players, disbands):
		
		rmap = {
			player: {self.get_unit(unit['loc'], utype=unit['type'], player=player): None
			         for unit in info['units']}
		for player, info in players.items()}
		
		for player, info in retreats.items():
			if player in rmap:
				rmap[player].update({self.get_unit(loc, player=player): set(options)
				                     for loc, options in info.items()})
		
		for player, info in disbands.items():
			if player not in rmap:
				rmap[player] = {}
			for unit in info:
				unit = self.get_unit(unit['loc'], utype=unit['type'])
				rmap[player][unit] = set()
				
				self.players[player].units.append(unit)
		
		return rmap
		
	def _default_retreats(self, actions, disbands):
		for name, ds in disbands.items():
			if name not in actions:
				actions[name] = []
			acts = actions[name]
			locs = {a['loc'] for a in acts}
			for d in ds:
				if d['loc'] not in locs:
					acts.append({'loc': d['loc'], 'type': 'disband', 'unit': d['type']})
		
	def uncoastify(self, loc, including_dirs=False):
		return self.coasts[loc]['coast-of'] if loc in self.coasts and \
			('dir' not in self.coasts[loc] or including_dirs) else loc
		
	def fix_actions(self, state, actions):
		
		missing = []
		
		for name, acts in actions.items():
			for a in acts:
				if 'unit' not in a:
					try:
						base, coast = self.decode_region_name(name=a['loc'])
						a['unit'] = self.unit_locs[name][base]['unit']
					except:
						missing.append([name, str(a)])
		
		if len(missing):
			print(tabulate(missing, headers=['Player', 'Action']))
			raise Exception('missing units')
		
		return actions
		
	def step(self, state, actions, ignore_unknown=None):
		if ignore_unknown is None:
			ignore_unknown = self.ignore_unknown
		
		self.prep_players(state['players'])
		
		actions = self.fix_actions(state, actions)
		
		turn, season = state['time']['turn'], state['time']['season']
		retreat = 'retreat' in state['time']
		
		new = {'players':{player: {'units':[], 'control':[], 'centers':[], 'home':info.get('home', [])}
		                  for player, info in state['players'].items()}}
		
		
		players = new['players']

		retreats_needed = False
		adjustments_needed = False

		control = {}
		
		for name, info in players.items():
			old = state['players'][name]
			if 'name' in old:
				info['name'] = old['name']
			# info['centers'] = old.get('centers', []).copy()
			for ctrl in old['control']:
				control[ctrl] = name
		
		if season < 3 and not retreat:
			commands, unknown = self.process_actions(actions, ignore_unknown=ignore_unknown)
			
			# resolve
			resolution = resolve_turn(self.dmap, commands)
			
			retreats = {}
			disbands = {}
			
			for player, units in resolution.items():
				for unit, sol in units.items():
					if sol is not None:
						if not len(sol):
							if player not in disbands:
								disbands[player] = []
							disbands[player].append({'loc':self.uncoastify(unit.position),
					                                 'type': util.UNIT_ENUMS[unit.unit_type]})
							continue
						retreats_needed = True
						if player not in retreats:
							retreats[player] = {}
						retreats[player][unit.position] = list(sol)
					players[player]['units'].append({'loc':self.uncoastify(unit.position),
					                                 'type': util.UNIT_ENUMS[unit.unit_type]})
					
					base = self.uncoastify(unit.position, True)
					if (season == 2 or 'sc' not in self.nodes[base] or self.nodes[base]['sc'] == 0) \
							and self.nodes[base]['type'] != 'sea':
						control[base] = player
			
			if len(retreats):
				new['retreats'] = retreats
			
			if len(disbands):
				new['disbands'] = disbands
			
		elif retreat:
			
			retreat_map = self._compute_retreat_map(state.get('retreats', {}), state['players'],
			                                        state.get('disbands', {}))
			
			self._default_retreats(actions, state.get('disbands', {}))
			
			commands, unknown = self.process_retreats(actions, retreat_map, ignore_unknown=ignore_unknown)
			
			resolution = resolve_retreats(retreat_map, commands)
			
			for player, units in resolution.items():
				for unit in units:
					players[player]['units'].append({'loc': self.uncoastify(unit.position),
					                                 'type': util.UNIT_ENUMS[unit.unit_type]})
					
					base = self.uncoastify(unit.position, True)
					if (season == 2 or 'sc' not in self.nodes[base] or self.nodes[base]['sc'] == 0) \
							and self.nodes[base]['type'] != 'sea':
						control[base] = player
			
		elif season == 3:
			
			smap = SupplyCenterMap(self.dmap, {ID for ID, node in self.nodes.items()
			                                         if 'sc' in node and node['sc'] > 0})
			owned = {name: set(player.get('centers', [])) for name, player in state['players'].items()}
			home = {name: set(player.get('home', [])) for name, player in state['players'].items()}
			
			omap = OwnershipMap(smap, owned, home)
			
			units = {name: set(self.get_unit(unit['loc'], unit['type'], player=name)
			                   for unit in info['units'])
			         for name, info in state['players'].items()}
			
			omap, counts = calculate_adjustments(omap, units)
			
			commands, unknown = self.process_builds(actions, omap, ignore_unknown=ignore_unknown)
			
			resolution = resolve_adjustment__validated(omap, counts, units, commands)
			
			for player, units in resolution.items():
				for unit in units:
					players[player]['units'].append({'loc': self.uncoastify(unit.position),
					                                 'type': util.UNIT_ENUMS[unit.unit_type]})
		
		else:
			raise Exception(f'unknown: {turn} {season} {retreat}')
		
		for loc, player in control.items():
			players[player]['control'].append(loc)
		
		if not retreats_needed and season == 2:
			
			forces = {player: len(info.get('units', [])) for player, info in players.items()}
			
			# compute adjustments
			centers = {loc: player for loc, player in control.items()
			           if 'sc' in self.nodes[loc] and self.nodes[loc]['sc'] >= 1}
			
			for loc, player in centers.items():
				if loc not in players[player]['centers']:
					players[player]['centers'].append(loc)
			
			new['adjustments'] = {player: len(players[player]['centers']) - score
			                      for player, score in forces.items()}
			
			adjustments_needed = any(new['adjustments'].values())
	
		else:
			for name, info in players.items():
				info['centers'] = state['players'][name].get('centers', []).copy()
	
		# update time
		
		if retreats_needed:
			new['time'] = {'turn': turn, 'season': season, 'retreat': True}
			
		elif adjustments_needed:
			new['time'] = {'turn': turn, 'season': 3}
			
		elif season == 1:
			new['time'] = {'turn': turn, 'season': 2}
			
		else:
			new['time'] = {'turn': turn+1, 'season': 1}
	
		new = self.special_rules(state, actions, unknown, new)

		return new

	def special_rules(self, state, actions, unknown, new):
		out = self._special_rules(state, actions, unknown, new)
		if out is None:
			out = new
		return out

	def _special_rules(self, state, actions, unknown, new):
		return new

# @fig.AutoComponent('state')
# class DiploState:
# 	def __init__(self, data=None, data_path=None):
# 		if data is None:
# 			assert data_path is not None, 'No data specified'
# 			data = load_yaml(data_path)
# 		if data is None:
# 			data = {}


@fig.modifier('dash-coasts')
class DashCoast(DiploMap):
	
	@classmethod
	def encode_region_name(cls, name=None, coast=None, node_type=None, unit_type=None):
		assert name is not None
		
		if coast is None:
			name, coast = cls.decode_region_name(name)
		
		if coast is not None and coast in {'nc', 'sc', 'wc', 'ec', 'NC', 'SC', 'WC', 'EC'}:
			return f'{name}-{coast}'
		elif coast:
			return f'{name}-c'
		
		if unit_type == 'fleet' and node_type == 'coast':
			return f'{name}-c'
		
		return name
	
	_coast_decoder = {'-nc': 'nc', '-sc': 'sc', '-wc': 'wc', '-ec': 'ec',
	                  '-NC': 'nc', '-SC': 'sc', '-WC': 'wc', '-EC': 'ec'}
	
	@classmethod
	def decode_region_name(cls, name=None, allow_dir=None):
		if len(name) < 3:
			return name, None
		
		end = name[-3:]
		if end in cls._coast_decoder:
			if allow_dir:
				return name, cls._coast_decoder[end]
			return name[:-3], cls._coast_decoder[end]
		
		if name.endswith('-c') or name.endswith('-C'):
			return name[:-2], True
		return name, None



		

		