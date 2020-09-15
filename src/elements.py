
import sys, os
from itertools import chain
from omnibelt import load_yaml, save_yaml

import omnifig as fig

import pydip
from pydip.map import Map
from pydip.player import Player, command, Unit, UnitTypes
# from pydip.unit import Unit
from pydip.turn import resolve_turn, resolve_retreats
from pydip.turn.adjustment import resolve_adjustment__validated, calculate_adjustments
from pydip.map.map import OwnershipMap, SupplyCenterMap

from . import util

@fig.Component('map')
class DiploMap:
	def __init__(self, A):
		
		nodes_path = A.pull('nodes-path', None)
		edges_path = A.pull('edges-path', None)
		pos_path = A.pull('pos-path', None)
		
		name = A.pull('name', None)
		
		if nodes_path is None:
			nodes_path = 'nodes.yaml' if name is None else f'{name}_nodes.yaml'
		if edges_path is None:
			edges_path = 'edges.yaml' if name is None else f'{name}_edges.yaml'
		if pos_path is None:
			pos_path = 'pos.yaml' if name is None else f'{name}_pos.yaml'
		
		root = A.pull('root', None)
		if root is not None:
			nodes_path = os.path.join(root, nodes_path)
			edges_path = os.path.join(root, edges_path)
			pos_path = os.path.join(root, pos_path)
		
		self.nodes, self.edges = self._load_map_info(nodes_path, edges_path)
		self.get_ID_from_name = util.make_node_dictionary(self.nodes)
		
		self.pos = None if pos_path is None or not os.path.isfile(pos_path) else load_yaml(pos_path)
		
		self.dmap = self._create_dip_map(self.nodes, self.edges)
	
	def get_supply_centers(self):
		return [node for node in self.nodes if 'sc' in node]

	def _load_map_info(self, nodes_path, edges_path):
		
		nodes = load_yaml(nodes_path)
		
		coasts = util.separate_coasts(nodes)
		
		self.coasts = coasts
		
		# for node in nodes.values():
		# 	if 'coasts' in node:
		# 		del node['coasts']
		
		for ID, coast in coasts.items():
			origin = coast['coast-of']
			if 'coasts' not in nodes[origin]:
				nodes[origin]['coasts'] = []
			nodes[origin]['coasts'].append(ID)
		
		
		for ID, node in nodes.items():
			node['ID'] = ID
		
		edges = load_yaml(edges_path)
		
		return nodes, edges

	@staticmethod
	def _create_dip_map(nodes, edges):
		
		# descriptors
		
		descriptors = []
		
		for ID, node in nodes.items():
			desc = {'name': ID}
			if node['type'] != 'sea':
				desc['coasts'] = node.get('coasts', [])
			descriptors.append(desc)
			
		# adjacencies
		
		all_edges = util.list_edges(edges, nodes)
		adjacencies = set()

		for e in all_edges:
			
			start, end = e['start'], e['end']
			
			start = util.fix_loc(start, e['type'], nodes[start]['type']) if start in nodes else start
			end = util.fix_loc(end, e['type'], nodes[end]['type']) if end in nodes else end
			
			if (end, start) not in adjacencies:
				adjacencies.add((start, end))
			# adjacencies.add((start, end))

		adjacencies = list(adjacencies)
		
		
		return Map(descriptors, adjacencies)

	def fix_loc(self, loc, utype):
		return util.fix_loc(loc, utype, self.nodes[loc]['type']) if loc in self.nodes else loc

	def load_players(self, players):
		
		full = {}
		unit_info = []
		
		for name, player in players.items():
			
			tiles = {loc: None for loc in player.get('control', [])}
			
			units = {unit['loc']: unit['type']
			              for unit in player.get('units', [])}
			
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
			
			full[name] = Player(name=name, game_map=self.dmap,
		                    starting_configuration=config)
		
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
			return player.find_unit(loc)
		
		if utype in util.UNIT_TYPES:
			utype = util.UNIT_TYPES[utype]
		
		return Unit(utype, loc)
		
	def _find_dest(self, src, utype, dest):
		
		# start = self.nodes[src]
		# if 'dirs' in start:
		# 	pass
		
		end = self.nodes[dest]
		if 'dirs' in end and utype in {'fleet', UnitTypes.FLEET}:
			edges = self.edges['fleet'][src]
			for e in edges:
				if e in self.coasts:
					node = self.coasts[e]
					if 'coast-of' in node:
						return e
		
		return self.fix_loc(dest, utype)
	
	def process_actions(self, full):
		
		assert hasattr(self, 'players'), 'players have not been loaded'
		
		cmds = []
		
		for name, actions in full.items():
			
			player = self.players[name]
			
			for action in actions:
				unit = self.get_unit(action['loc'], action.get('unit', None))
				
				if action['type'] == 'move':
					dest = self.fix_loc(action['dest'], unit.unit_type)
					cmds.append(command.MoveCommand(player, unit, dest))
				elif action['type'] == 'hold':
					cmds.append(command.HoldCommand(player, unit))
				elif 'support' in action['type']:
					if 'defend' in action['type']:
						sup_unit = self.get_unit(action['dest'])
						dest = sup_unit.position
					else:
						sup_unit = self.get_unit(action['src'])
						dest = self._find_dest(action['src'], sup_unit.unit_type, action['dest'])
					cmds.append(command.SupportCommand(player, unit, sup_unit, dest))
				elif action['type'] == 'convoy-move':
					dest = self.fix_loc(action['dest'], unit.unit_type)
					cmds.append(command.ConvoyMoveCommand(player, unit, dest))
				elif action['type'] == 'convoy-transport':
					transport = self.get_unit(action['src'], action['unit'], player=name)
					dest = self.fix_loc(action['dest'], transport.unit_type)
					cmds.append(command.ConvoyTransportCommand(player, unit, transport, dest))
				else:
					raise Exception(f'unknown: {action}')
		
		return cmds
		
	def process_retreats(self, full, retreats):
		
		assert hasattr(self, 'players'), 'players have not been loaded'
		
		cmds = []
		
		for name, actions in full.items():
			
			player = self.players[name]
			
			for action in actions:
				unit = self.get_unit(action['loc'], action['unit'], player=name)
				
				if action['type'] == 'disband':
					if unit in retreats[name]:
						cmds.append(command.RetreatDisbandCommand(retreats, player, unit))
				elif action['type'] == 'retreat':
					dest = self.fix_loc(action['dest'], unit.unit_type)
					cmds.append(command.RetreatMoveCommand(retreats, player, unit, dest))
				else:
					raise Exception(f'unknown: {action}')
		
		return cmds
	
	def process_builds(self, full, ownership):
		
		assert hasattr(self, 'players'), 'players have not been loaded'
		
		cmds = []
		
		for name, actions in full.items():
			
			player = self.players[name]
			
			for action in actions:
				unit = self.get_unit(action['loc'], action.get('unit', None))
					
				if action['type'] == 'build':
					cmds.append(command.AdjustmentCreateCommand(ownership, player, unit))
				elif action['type'] == 'destroy':
					cmds.append(command.AdjustmentDisbandCommand(player, unit))
				else:
					raise Exception(f'unknown: {action}')
		
		return cmds
		
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
		
	# def _compute_ownership_map(self, players):
	#
	# 	raise NotImplementedError
		
	def uncoastify(self, loc, including_dirs=False):
		return self.coasts[loc]['coast-of'] if loc in self.coasts and \
			('dir' not in self.coasts[loc] or including_dirs) else loc
		
	def step(self, state, actions):
		
		self.load_players(state['players'])
		
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
			actions = self.process_actions(actions)
			
			# resolve
			resolution = resolve_turn(self.dmap, actions)
			
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
			
			retreat_map = self._compute_retreat_map(state['retreats'], state['players'], state.get('disbands', {}))
			
			actions = self.process_retreats(actions, retreat_map)
			
			resolution = resolve_retreats(retreat_map, actions)
			
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
			
			actions = self.process_builds(actions, omap)
			
			resolution = resolve_adjustment__validated(omap, counts, units, actions)
			
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

		return new

# @fig.AutoComponent('state')
# class DiploState:
# 	def __init__(self, data=None, data_path=None):
# 		if data is None:
# 			assert data_path is not None, 'No data specified'
# 			data = load_yaml(data_path)
# 		if data is None:
# 			data = {}
		
		
		