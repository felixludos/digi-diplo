from pathlib import Path
from tqdm import tqdm
from omnibelt import save_yaml, load_yaml
import omnifig as fig

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
from matplotlib.figure import figaspect
import matplotlib.patheffects as path_effects

from .util import Versioned
from .colors import hex_to_rgb


class DiplomacyRenderer(Versioned):
	def __init__(self, map=None, pbar=True, quiet=False, **kwargs):
		self.map = map
		self._use_pbar = pbar
		self._pbar = None
		self._quiet = quiet
		
	def __call__(self, state, action=None, savepath=None, processargs={}, **kwargs):
		img = self._render(state, actions=action, **kwargs)
		out = self._process(img, savepath=savepath, **processargs)
		return out
	
	def _render(self, state, actions=None, **kwargs):
		self.state, self.actions = state, actions
		self._prep_assets(state, actions=actions, **kwargs)
		
		self._render_env(state, actions=actions, **kwargs)
		
		self._render_players(state, actions=actions, **kwargs)
		
		img = self._finalize_render(state, actions=actions, **kwargs)
		return img
	
	
	def _prep_assets(self, state, actions=None, **kwargs):
		if self._use_pbar:
			if self._pbar is not None:
				self._pbar.close()
				print()
			self._pbar = None
	
	
	def _finalize_render(self, state, actions=None, **kwargs):
		pass
	
	
	def _render_env(self, state, actions=None, **kwargs):
		if not self._quiet:
			print('Rendering environment')
		
		self._draw_bg(state, actions=actions)
		
		self._draw_labels(state, actions=actions)
		
		self._draw_scs(state, actions=actions)
	
	def _draw_bg(self, state, actions=None):
		raise NotImplementedError
	
	def _draw_labels(self, state, actions=None):
		raise NotImplementedError
	
	def _draw_scs(self, state, actions=None):
		raise NotImplementedError
	
	
	def _render_players(self, state, actions=None, **kwargs):
		if not self._quiet:
			print('Rendering players')
		itr = state['players'].items()
		if not self._quiet and self._use_pbar:
			itr = tqdm(itr, total=len(state['players']))
		
		for player, info in itr:
			if not self._quiet and self._use_pbar:
				itr.set_description(f'Rendering: {player}')
			self._draw_player(player, state, actions=actions, **kwargs)
	
	
	def _draw_player(self, player, state, actions=None, **kwargs):
		raise NotImplementedError
	
	
class MatplotlibRenderer(DiplomacyRenderer):
	def __init__(self, view=False, mlp_backend=None, img_scale=1, **kwargs):
		if mlp_backend is None:
			mlp_backend = 'qt5agg' if view else 'agg'
		if mlp_backend is not None:
			plt.switch_backend(mlp_backend)
			
		super().__init__(**kwargs)
		
		self.img_scale = img_scale
		self._view = view
	
	
	def _process(self, img, savepath=None, **kwargs):
		if savepath is not None:
			H, W, _ = self.base.shape
			w, h = figaspect(H / W)
			w, h = self.img_scale * w, self.img_scale * h
			plt.savefig(savepath, dpi=W / w)
			if not self._view:
				plt.close(plt.gcf())


def replace_dashes(data):
	if isinstance(data, list):
		return [replace_dashes(v) for v in data]
	elif isinstance(data, dict):
		return {k.replace('-', '_'): replace_dashes(v) for k, v in data.items()}
	return data


@fig.component('default-renderer')
class DefaultRenderer(MatplotlibRenderer):
	def __init__(self, renderbase_path, skip_control=False, show_labels=True,
	             label_props=None, coast_label_props=None, unit_shapes=None, unit_props=None,
	             sc_props=None, capital_props=None, home_props=None, retreat_props=None,
	             retreat_arrow=None, disband_props=None, arrow_ratio=None, build_props=None,
	             hold_props=None, move_arrow=None, support_props=None, support_arrow=None,
	             support_dot=None, support_defend_props=None, convoy_props=None, convoy_arrow=None,
	             convoy_dot=None, retreat_action_props=None, graph_path=None, regions_path=None,
	             bgs_path=None, players_path=None,
	             **kwargs):
		if label_props is None:
			label_props = {}
		label_props = replace_dashes(label_props)
		if coast_label_props is None:
			coast_label_props = {}
		coast_label_props = replace_dashes(coast_label_props)
		if unit_shapes is None:
			unit_shapes = {'army': 'o', 'fleet': 'v'}
		unit_shapes = replace_dashes(unit_shapes)
		if unit_props is None:
			unit_props = {}
		unit_props = replace_dashes(unit_props)
		if sc_props is None:
			sc_props = {}
		sc_props = replace_dashes(sc_props)
		if capital_props is None:
			capital_props = {}
		capital_props = replace_dashes(capital_props)
		if home_props is None:
			home_props = {}
		home_props = replace_dashes(home_props)
		if retreat_props is None:
			retreat_props = {}
		retreat_props = replace_dashes(retreat_props)
		if retreat_arrow is None:
			retreat_arrow = {}
		retreat_arrow = replace_dashes(retreat_arrow)
		if disband_props is None:
			disband_props = {}
		disband_props = replace_dashes(disband_props)
		if arrow_ratio is None:
			arrow_ratio = 0.9
		if build_props is None:
			build_props = {}
		build_props = replace_dashes(build_props)
		if hold_props is None:
			hold_props = {}
		hold_props = replace_dashes(hold_props)
		if move_arrow is None:
			move_arrow = {}
		move_arrow = replace_dashes(move_arrow)
		if support_props is None:
			support_props = {}
		support_props = replace_dashes(support_props)
		if support_arrow is None:
			support_arrow = {}
		support_arrow = replace_dashes(support_arrow)
		if support_dot is None:
			support_dot = {}
		support_dot = replace_dashes(support_dot)
		if support_defend_props is None:
			support_defend_props = {}
		support_defend_props = replace_dashes(support_defend_props)
		if convoy_props is None:
			convoy_props = {}
		convoy_props = replace_dashes(convoy_props)
		if convoy_arrow is None:
			convoy_arrow = {}
		convoy_arrow = replace_dashes(convoy_arrow)
		if retreat_action_props is None:
			retreat_action_props = {}
		retreat_action_props = replace_dashes(retreat_action_props)
		super().__init__(**kwargs)
		
		self.base_path = Path(renderbase_path)
		assert self.base_path.exists(), 'No render base found"'
		
		self.base_root = self.base_path.parents[0]
		
		self.overlay_path = self.base_root / 'overlay.png'
		
		self.skip_control = skip_control
		
		self.show_labels = show_labels
		
		self.label_props = label_props
		if self.label_props.get('bbox', {}).get('facecolor', '') is None:
			self.label_props['bbox']['facecolor'] = 'none'
		if self.label_props.get('bbox', {}).get('edgecolor', '') is None:
			self.label_props['bbox']['edgecolor'] = 'none'
		self.coast_label_props = coast_label_props
		if self.coast_label_props.get('bbox', {}).get('facecolor', '') is None:
			self.coast_label_props['bbox']['facecolor'] = 'none'
		if self.coast_label_props.get('bbox', {}).get('edgecolor', '') is None:
			self.coast_label_props['bbox']['edgecolor'] = 'none'
		self.unit_shapes = unit_shapes
		self.unit_props = unit_props
		self.sc_props = sc_props
		self.capital_props = capital_props
		self.home_props = home_props
		self.retreat_props = retreat_props
		if self.retreat_props.get('mfc', '') is None:
			self.retreat_props['mfc'] = 'none'
		self.retreat_arrow = retreat_arrow
		self.disband_props = disband_props
		self.arrow_ratio = arrow_ratio
		self.build_props = build_props
		if self.build_props.get('mfc', '') is None:
			self.build_props['mfc'] = 'none'
		self.hold_props = hold_props
		if self.hold_props.get('mfc', '') is None:
			self.hold_props['mfc'] = 'none'
		self.move_arrow = move_arrow
		self.support_props = support_props
		self.support_arrow = support_arrow
		self.support_dot = support_dot
		self.support_defend_props = support_defend_props
		self.convoy_props = convoy_props
		self.convoy_arrow = convoy_arrow
		self.convoy_dot = convoy_dot
		self.retreat_action_props = retreat_action_props
		
		self.graph_path = graph_path
		self.regions_path = regions_path
		self.bg_path = bgs_path
		self.players_path = players_path
		self.players = load_yaml(self.players_path)
		self.capitals = {info['capital']: name for name, info in self.players.items()
		                 if 'capital' in info}
		
		# self.config = self.my_config
		
		self._known_action_drawers = {
			'build': self._draw_build,
			'retreat': self._draw_retreat,
			'disband': self._draw_disband, # disband due to retreat
			'destroy': self._draw_destroy, # disband during winter
			'hold': self._draw_hold,
			'move': self._draw_move,
			'support': self._draw_support,
			'support-defend': self._draw_support_defend,
			'convoy-move': self._draw_move,
			'convoy-transport': self._draw_convoy,
		}
	
	def _prep_assets(self, state, actions=None, **kwargs):
		super()._prep_assets(state, actions=actions, **kwargs)
		
		self.base = self._load_renderbase()
		self.lbls = self._load_img(self.regions_path)
		
		if self.overlay_path.exists():
			self.overlay = self._load_overlay()
		
		H, W, _ = self.base.shape
		w, h = figaspect(H / W)
		w, h = self.img_scale * w, self.img_scale * h
		
		self.figax = plt.subplots(figsize=(w, h))
		
	def _finalize_render(self, state, actions=None, **kwargs):
		
		plt.imshow(self.base, zorder=-1)
		
		plt.axis('off')
		plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
		                    wspace=0, hspace=0)
		
		if self.overlay is not None:
			# sel = overlay[...,-1] > 0
			# self.base[sel] = overlay[sel,:3]
			plt.imshow(self.overlay)
		
	
	def _load_img(self, path, rgb=False, rgba=False):
		img = Image.open(str(path))
		if rgb:
			img = img.convert("RGB")
		if rgba:
			img = img.convert('RGBA')
		return np.array(img)
	
	def _save_img(self, img, path):
		return Image.fromarray(img).save(str(path))
	
	def _load_renderbase(self):
		return self._load_img(self.base_path, rgb=True)

	def _load_overlay(self):
		return self._load_img(self.overlay_path, rgba=True)
	
	def _get_unit_pos(self, loc, retreat=False):
		base, coast = self.map.decode_region_name(loc)
		node = self.map.nodes[base]
		
		if retreat:
			pos = node['locs']['retreat'] if coast is None or 'coast-unit' not in node['locs'] \
				else node['locs']['coast-retreat'][coast]
		else:
			pos = node['locs']['unit'] if coast is None or 'coast-unit' not in node['locs'] \
				else node['locs']['coast-unit'][coast]
		
		if len(pos) and isinstance(pos[0], (float, int)):
			pos = [pos]
		
		pos = list(map(np.array, zip(*pos)))
		return pos#[::-1]
	
	def _get_label_pos(self, loc, coast=None):
		if coast is None:
			base, coast = self.map.decode_region_name(loc)
		else:
			base, coast = loc, coast
		node = self.map.nodes[base]
		pos = node['locs']['label'] if coast is None or 'coast-label' not in node['locs'] \
			else node['locs']['coast-label'][coast]
		pos = list(map(np.array, zip(*pos)))
		return pos#[::-1]
	
	def _get_sc_pos(self, loc):
		pos = self.map.nodes[loc]['locs'].get('sc')
		if pos is not None:
			pos = list(map(np.array, zip(*pos)))
			return pos#[::-1]
	
	def _draw_shortest_arrow(self, start, end, arrow_props={}, use_annotation=False):
		x, y = start
		x, y = x.reshape(1, -1), y.reshape(1, -1)
		ex, ey = end
		ex, ey = ex.reshape(-1, 1), ey.reshape(-1, 1)
		dx, dy = ex - x, ey - y
		x, y = ex - dx, ey - dy
		x, y = x.reshape(-1), y.reshape(-1)
		dx, dy = dx.reshape(-1), dy.reshape(-1)
		D = dx ** 2 + dy ** 2
		idx = np.argmin(D)
		
		x, y = x[idx], y[idx]
		dx, dy = dx[idx], dy[idx]
		if use_annotation:
			return plt.annotate('', xytext=(x + dx, y + dy), xy=(x, y), **arrow_props)
		else:
			dx, dy = dx * self.arrow_ratio, dy * self.arrow_ratio
			return plt.arrow(x, y, dx, dy, **arrow_props)
	
	
	def _draw_bg(self, state, actions=None):
		pass
	
	
	def _draw_labels(self, state, actions=None):
		for name, node in self.map.nodes.items():
			
			self._draw_label(name)
			
			if 'fleet' in node.get('edges', {}) and isinstance(node['edges']['fleet'], dict):
				for coast in node['edges']['fleet']:
					self._draw_label(name, coast)
					
	
	def _draw_label(self, loc, coast=None):
		if self.show_labels:
			pos = self._get_label_pos(loc, coast=coast)
			if coast is None:
				plt.text(*pos, s=loc.upper(), **self.label_props)
			else:
				# name = self.map.encode_region_name(loc, coast=coast)
				name = coast
				plt.text(*pos, s=name.upper(), **self.coast_label_props)
			
	
	def _draw_scs(self, state, actions=None):
		
		owners = {center: player for player, info in state['players'].items() for center in info.get('centers')}
		centers = [name for name, node in self.map.nodes.items() if node.get('sc', 0) > 0]
		
		for center in centers:
			self._draw_sc(center, owners.get(center, None))
	
	def _draw_sc(self, loc, owner=None):
		
		# color = self.players.get(owner, {}).get('color')
		
		pos = self._get_sc_pos(loc)
		if pos is not None:
			if loc in self.capitals:
				return plt.plot(*pos, **self.capital_props)
			else:
				return plt.plot(*pos, **self.sc_props)
	
	def _draw_player(self, player, state, actions=None, **kwargs):
		info = state['players'][player]
		
		if not self.skip_control:
			for loc in info.get('control', []):
				self._draw_territory(player, loc)
		for home in info.get('home', []):
			self._draw_home(player, home)
		
		retreats = {self.map.decode_region_name(loc, allow_dir=True)[0]:
			            [self.map.decode_region_name(option, allow_dir=True)[0] for option in options]
		            for loc, options in state.get('retreats', {}).get(player, {}).items()}
		for loc, options in retreats.items():
			self._draw_retreats(player, loc, options)
		
		disbands = {self.map.decode_region_name(u['loc'], allow_dir=True)[0]: u['type']
		            for u in state.get('disbands', {}).get(player, [])}
		for loc, utype in disbands.items():
			self._draw_destroy(player, loc, utype)
		
		for unit in info.get('units', []):
			self._draw_unit(player, unit['loc'], unit['type'],
			                retreat=unit['loc'] in retreats or unit['loc'] in disbands)
		
		if actions is not None:
			for action in actions.get(player, {}).values():
				self._draw_action(player, action)
	
	def _draw_territory(self, player, loc):
		color = hex_to_rgb(self.players[player]['color'])
		ID = self.map.nodes[loc]['ID']
		self.base[self.lbls == ID] = color
	
	def _draw_home(self, player, loc):
		color = self.players.get(player, {}).get('color')
		pos = self._get_sc_pos(loc)
		if pos is not None:
			return plt.plot(*pos, color=color, **self.home_props)
	
	def _draw_unit(self, player, loc, utype, retreat=False):
		shape = self.unit_shapes.get(utype)
		color = self.players[player].get('color', 'w')
		
		pos = self._get_unit_pos(loc, retreat=retreat)
		return plt.plot(*pos, marker=shape, mfc=color, **self.unit_props)
	
	def _draw_retreats(self, player, loc, options):
		X, Y = self._get_unit_pos(loc, retreat=True)
		mk = plt.plot(X, Y, **self.retreat_props)
		rets = [self._draw_shortest_arrow((X.copy(), Y.copy()), self._get_label_pos(option), self.retreat_arrow)
		        for option in options]
		return [mk, *rets]
	
	def _draw_retreat(self, player, action):
		self._draw_shortest_arrow(self._get_unit_pos(action['loc']), self._get_label_pos(action['dest']),
		                          self.retreat_action_props)
	
	def _draw_disband(self, player, loc, utype=None):
		if isinstance(loc, dict):
			utype = loc.get('unit')
			loc = loc['loc']
		self._draw_unit(player, loc, utype, retreat=True)
		return self._draw_destroy(player, loc, retreat=True)
	
	def _draw_action(self, player, action):
		fn = self._known_action_drawers.get(action.get('type'))
		if fn is None:
			return self._draw_unknown_action(player, action)
		return fn(player, action)
	
	def _draw_build(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		shape = self.unit_shapes.get(action.get('unit'))
		plt.plot(x, y, marker=shape, **self.build_props)
	
	def _draw_destroy(self, player, loc, retreat=False):
		if isinstance(loc, dict):
			loc = loc['loc']
		x, y = self._get_unit_pos(loc, retreat=retreat)
		return plt.plot(x, y, **self.disband_props)
	
	def _draw_hold(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		shape = self.unit_shapes.get(action.get('unit'))
		plt.plot(x, y, marker=shape, **self.hold_props)
	
	def _draw_move(self, player, action):
		self._draw_shortest_arrow(self._get_unit_pos(action['loc']),
		                          self._get_label_pos(action['dest']),
		                          self.move_arrow)

	def _draw_support(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		
		x1, y1 = self._get_unit_pos(action['src'])
		x2, y2 = self._get_label_pos(action['dest'])
		
		lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
		
		self._draw_shortest_arrow((x1, y1), (x2, y2), self.support_arrow)
		self._draw_shortest_arrow((x, y), (lx, ly), self.support_props, use_annotation=True)
		if self.support_dot is not None:
			plt.plot([lx], [ly], **self.support_dot)
		
	def _draw_support_defend(self, player, action):
		self._draw_shortest_arrow(self._get_unit_pos(action['loc']), self._get_unit_pos(action['dest']),
		                          self.support_defend_props, use_annotation=True)
		
	def _draw_convoy(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		x1, y1 = self._get_unit_pos(action['src'])
		x2, y2 = self._get_label_pos(action['dest'])
		
		lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
		
		self._draw_shortest_arrow((x1, y1), (x2, y2), self.convoy_arrow)
		self._draw_shortest_arrow((x, y), (lx, ly), self.convoy_props, use_annotation=True)
		if self.convoy_dot is not None:
			plt.plot([lx], [ly], **self.convoy_dot)
		
	
	def _draw_unknown_action(self, player, action):
		raise NotImplementedError
	
	
	
	

