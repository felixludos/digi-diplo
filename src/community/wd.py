from omnibelt import load_yaml, save_yaml
import omnifig as fig

import numpy as np
import matplotlib.pyplot as plt

from ..colors import hex_to_rgb
from ..elements import DiploMap, DashCoast
from ..render import DiplomacyRenderer, DefaultRenderer
from ..managers import DiplomacyManager, NoUnitFoundError

_wd_version = (1,1)

@fig.Component('wd-manager')
class WD_Manager(DiplomacyManager):
	__version__ = (1,2)
	def format_action(self, player, terms):
		unit = 'A' if terms.get('unit') == 'army' else 'F'
		
		if terms['type'] == 'core':
			return '**{loc}** *core*'.format(punit=unit, **terms)
		else:
			return super().format_action(player, terms)

	def check_region(self, loc):
		if not isinstance(loc, str):
			return
		out = super().check_region(loc.upper())
		if out is not None:
			base = out['base']
			for player, info in self.state.get('players', {}).items():
				if base == info.get('capital', None):
					out['capital'] = player
					break
		return out

	def action_format(self):
		return {'core': '[X] core', **super().action_format()}
	
	
	def parse_action(self, player, text, terms=None):
		if terms is None:
			terms = {}
		
		# terms['player'] = player
		line = text.lower()
		
		if ' core' in line:
			loc, _ = line.split(' core')
			loc = self._parse_location(loc)
			unit = self._find_unit(loc, player)
			if unit is None:
				raise NoUnitFoundError(loc)
			terms.update({'type': 'core', 'loc': loc, 'unit': unit})
			return terms
		else:
			return super().parse_action(player, text, terms=None)



@fig.Component('wd-renderer')
class WD_Rendering(DefaultRenderer):
	__version__ = (1,2)
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		self.year_offset = A.pull('year-offset', 0)
		self.year_props = A.pull('year-props', None)
		self.season_titles = A.pull('season-titles', {})
		self.season_props = A.pull('season-props', None)
		
		self.core_props = A.pull('core-props', {})
		self._known_action_drawers['core'] = self._draw_core
	
	def _render(self, state, actions=None, **kwargs):
		self.capitals = {info['capital']: name for name, info in state.get('players', {}).items()}
		return super()._render(state, actions=actions, **kwargs)
	
	def _render_env(self, state, actions=None, **kwargs):
		out = super()._render_env(state, actions=actions, **kwargs)
		
		date = state.get('time', {})
		season = date.get('season', None)
		if str(season) in self.season_titles and self.season_props is not None:
			plt.text(s=self.season_titles[str(season)], **self.season_props)

		year = date.get('turn', None)
		if self.year_props is not None:
			plt.text(s=str(year + self.year_offset), **self.year_props)
		
		plt.text(x=10, y=2220, s='Adjudicated and rendered automatically using Digi-diplo '
		           '(GitHub: felixludos/digi-diplo)', color='w',
		         fontsize=2, fontfamily='monospace') # PLEASE DO NOT REMOVE
	
		return out
	
	def _get_unit_pos(self, loc, retreat=False):
		return super()._get_unit_pos(loc, retreat=retreat)[::-1]
	
	
	def _get_label_pos(self, loc, coast=None):
		return super()._get_label_pos(loc, coast=coast)[::-1]


	def _get_sc_pos(self, loc):
		pos = super()._get_sc_pos(loc)
		if pos is not None:
			return pos[::-1]
		
		
	def _load_overlay(self):
		return self._load_img(self.overlay_path, rgba=True)


	def _draw_labels(self, state, actions=None):
		pass
	
	
	def _draw_core(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		plt.plot(x, y, **self.core_props)



@fig.Component('wd-hills-rendering')
class WD_Hills_Rendering(WD_Rendering):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		self.unit_bg = A.pull('unit-bg', {})
		self.unit_labels = {'army': 'A', 'fleet': 'F'}
		self.sc_dot_props = A.pull('sc-dot-props', {})
		self.unit_delta = A.pull('unit-delta', None)
	
	def _draw_hold(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		plt.plot(x, y, **self.hold_props)
	
	def _draw_build(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		plt.plot(x, y, **self.build_props)
		self._draw_unit(player, action['loc'], action['unit'])
	
	def _draw_unit(self, player, loc, utype, retreat=False):
		color = self.players[player].get('color', 'w')
		pos = self._get_unit_pos(loc, retreat=retreat)
		
		# if 'bbox' in self.unit_props:
		# 	self.unit_props['bbox']['facecolor'] = color
		
		
		plt.plot(*pos, color=color, **self.unit_bg)
		x, y = pos
		if self.unit_delta is not None:
			x, y = x+self.unit_delta[0], y+self.unit_delta[1]
		pos = x, y
		return plt.text(*pos, s=self.unit_labels.get(utype), **self.unit_props)
	
	def _draw_sc(self, loc, owner=None):
		# color = self.players.get(owner, {}).get('color')
		
		pos = self._get_sc_pos(loc)
		if pos is not None:
			plt.plot(*pos, **self.sc_dot_props)
			# x, y = pos
			# self.base[y.astype(int), x.astype(int)] = 0
			plt.plot(*pos, **self.sc_props)
			
			if loc in self.capitals:
				return plt.plot(*pos, **self.capital_props)
			
	
	def _draw_home(self, player, loc):
		color = self.players.get(player, {}).get('color')
		pos = self._get_sc_pos(loc)
		if pos is not None:
			return plt.plot(*pos, markerfacecolor=color, **self.home_props)


@fig.Component('wd-pixel-rendering')
class WD_Pixel_Rendering(WD_Rendering):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		# self.arrow_hop = A.pull('arrow-hop', 0.)
		self.neutral_color = self._format_color(A.pull('neutral-color', 'w'))
		self.pattern_bases = self._format_pattern(A.pull('patterns', {}, silent=True))
		self.pattern_colors = self._format_pattern_colors(A.pull('pattern-colors', {}))
		self.unit_zorder = A.pull('unit-zorder', 100)
		self.sc_zorder = A.pull('sc-zorder', 100)
	
	def _format_pattern(self, patterns):
		return {key: np.array(val) for key, val in patterns.items()}
	
	def _format_color(self, color):
		return hex_to_rgb(color)
	
	def _format_pattern_colors(self, pattern_colors):
		return {key: {i: self._format_color(c) for i,c in enumerate(val)}
		        for key, val in pattern_colors.items()}
	
	
	def _draw_pixel_pattern(self, pattern, cy, cx, colors, base):
		h, w = pattern.shape
		
		X, Y = np.mgrid[:h, :w]
		X -= int(h/2 - 0.5)
		Y -= int(w/2 - 0.5)
		X, Y, C = zip(*[(x, y, colors[v])
		               for x,y,v in zip(X.reshape(-1), Y.reshape(-1), pattern.reshape(-1))
		               if v in colors and colors[v] is not None])
		C = [self._format_color(c) for c in C]
		X, Y, C = np.array(X), np.array(Y), np.array(C)
		
		base[(X + cx).astype(int), (Y + cy).astype(int), :3] = C
		if base.shape[-1] == 4:
			base[(X + cx).astype(int), (Y + cy).astype(int), -1] = 255
	
	
	def _prep_assets(self, state, actions=None, **kwargs):
		
		out = super()._prep_assets(state, actions=actions, **kwargs)
		
		self.unit_overlay = self.overlay.copy() * 0
		self.sc_overlay = self.overlay.copy() * 0
		
		return out
	
	def _finalize_render(self, state, actions=None, **kwargs):
		
		if self.unit_overlay is not None:
			plt.imshow(self.unit_overlay, zorder=self.unit_zorder)

		if self.sc_overlay is not None:
			plt.imshow(self.sc_overlay, zorder=self.sc_zorder)
			
		out = super()._finalize_render(state, actions=actions, **kwargs)
		return out
		
	
	def _draw_hold(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		plt.plot(x, y, **self.hold_props)

	def _draw_build(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		plt.plot(x, y, **self.build_props)
		self._draw_unit(player, action['loc'], action['unit'])
	
	def _draw_unit(self, player, loc, utype, retreat=False):
		color = self.players[player].get('color', self.neutral_color)
		pos = self._get_unit_pos(loc, retreat=retreat)
		
		colors = self.pattern_colors.get(utype, {}).copy()
		colors[1] = color
		
		for x,y in zip(*pos):
			self._draw_pixel_pattern(self.pattern_bases[utype], x, y, colors, self.unit_overlay)
		
	def _draw_sc(self, loc, owner=None, home=None):
		pos = self._get_sc_pos(loc)
		
		key = 'capital' if loc in self.capitals else 'sc'
		
		colors = self.pattern_colors.get(key, {0:[0,0,0], 1:self.neutral_color}).copy()
		if home is not None:
			color = self.players.get(home, {}).get('color')
			if color is not None:
				colors[1] = color
			
		for x,y in zip(*pos):
			self._draw_pixel_pattern(self.pattern_bases[key], x, y, colors, self.sc_overlay)
			
	def _draw_home(self, player, loc):
		return self._draw_sc(loc, home=player)
	

# @fig.AutoModifier('wd-unit-arrows')
# class UnitArrows(WD_Rendering):
	
	# def _draw_shortest_arrow(self, start, end, arrow_props={}, use_annotation=False):
	# 	x, y = start
	# 	x, y = x.reshape(1, -1), y.reshape(1, -1)
	# 	ex, ey = end
	# 	ex, ey = ex.reshape(-1, 1), ey.reshape(-1, 1)
	# 	dx, dy = ex - x, ey - y
	# 	x, y = ex - dx, ey - dy
	# 	x, y = x.reshape(-1), y.reshape(-1)
	# 	dx, dy = dx.reshape(-1), dy.reshape(-1)
	# 	D = dx ** 2 + dy ** 2
	# 	idx = np.argmin(D)
	#
	# 	x, y = x[idx], y[idx]
	# 	dx, dy = dx[idx], dy[idx]
	# 	if use_annotation:
	# 		return plt.annotate('', xytext=(x + dx, y + dy), xy=(x, y), **arrow_props)
	# 	else:
	# 		dx, dy = dx * self.arrow_ratio, dy * self.arrow_ratio
	# 		return plt.arrow(x, y, dx, dy, **arrow_props)


@fig.Component('wd-map')
class WD_Map(DashCoast, DiploMap):
	__version__ = (1,2)
	def generate_initial_state(self):
		state = super().generate_initial_state()
		
		for player, info in state['players'].items():
			if 'capital' in self.player_info.get(player, {}):
				info['capital'] = self.player_info[player]['capital']
		
		return state
		

	def step(self, state, actions, **kwargs):
		
		dests = {action['dest'] for _, acts in actions.items() for action in acts
		         if 'dest' in action and action.get('type') == 'convoy-move'}
		
		capcores = {}
		cores = {}
		for player, acts in actions.items():
			for action in acts:
				if action['loc'] not in dests and action['type'] == 'core':
					action['type'] = 'hold'
					if action['loc'] in state['players'][player].get('home', {}):
						if player in capcores:
							del capcores[player]
						else:
							capcores[player] = action['loc']
					else:
						cores[action['loc']] = player
		
		past = state.get('cores', {})
		partial, done = {}, {}
		
		for player, loc in capcores.items():
			if loc in past:
				done[loc] = player
			else:
				partial[loc] = player
		for loc, player in cores.items():
			if loc in past:
				done[loc] = player
			else:
				partial[loc] = player
		new = super().step(state, actions, **kwargs)
		
		if state["time"].get("retreat", False):
			new["cores"] = past
		
		for player, acts in actions.items():
			for action in acts:
				if action['loc'] in cores or action['loc'] in capcores:
					action['type'] = 'core'
		
		for player, info in new['players'].items():
			if 'capital' in state['players'][player]:
				info['capital'] = state['players'][player]['capital']
		
		if len(partial):
			new['cores'] = partial
		
		for loc, player in done.items():
			if player in capcores and capcores[player] == loc:
				new['players'][player]['capital'] = loc
			else:
				new['players'][player]['home'].append(loc)
		
		return new
	
	
	# def _special_rules(self, state, actions, unknown, new):
	# 	pass

