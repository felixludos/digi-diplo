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

from .colors import hex_to_rgb


class DiplomacyRenderer(fig.Configurable):
	def __init__(self, A, **kwargs):

		self.map = A.pull('map', None)

		self._use_pbar = A.pull('pbar', False)
		self._pbar = None
		self._quiet = A.pull('quiet', False)
		
	def __call__(self, state, action=None, savepath=None, processargs={}, **kwargs):
		img = self._render(state, actions=action, **kwargs)
		out = self._process(img, savepath=savepath, **processargs)
		return out
	
	def _render(self, state, actions=None, **kwargs):
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
			itr.set_description(f'Rendering: {player}')
			self._draw_player(player, state, actions=actions, **kwargs)
	
	
	def _draw_player(self, player, state, actions=None, **kwargs):
		raise NotImplementedError
	
	
class MatplotlibRenderer(DiplomacyRenderer):
	def __init__(self, A, **kwargs):
		view = A.pull('view', False)
		mlp_backend = A.pull('mlp-backend', 'qt5agg' if view else 'agg')
		if mlp_backend is not None:
			plt.switch_backend(mlp_backend)
			
		super().__init__(A, **kwargs)
		
		self.img_scale = A.pull('img-scale', 1)
	
	
	def _process(self, img, savepath=None, **kwargs):
		if savepath is not None:
			H, W, _ = self.base.shape
			w, h = figaspect(H / W)
			w, h = self.img_scale * w, self.img_scale * h
			plt.savefig(savepath, dpi=W / w)


@fig.Component('default-renderer')
class DefaultRenderer(MatplotlibRenderer):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
		self.base_path = Path(A.pull('renderbase-path'))
		assert self.base_path.exists(), 'No render base found"'
		
		self.base_root = self.base_path.parents[0]
		
		self.overlay_path = self.base_root / 'overlay.png'
		
		self.skip_control = A.pull('skip-control', False)
		
		self.unit_shapes = A.pull('unit-shapes', {'army': 'o', 'fleet': 'v'})
		self.unit_props = A.pull('unit-props', {})
		self.sc_props = A.pull('sc-props', {})
		self.capital_props = A.pull('capital-props', {})
		self.home_props = A.pull('home-props', {})
		self.retreat_props = A.pull('retreat-props', {})
		self.retreat_arrow = A.pull('retreat-arrow', {})
		self.disband_props = A.pull('disband-props', {})
		self.arrow_ratio = A.pull('arrow-ratio', 0.9)
		self.build_props = A.pull('build-props', {})
		self.hold_props = A.pull('hold-props', {})
		if self.hold_props.get('mfc', '') is None:
			self.hold_props['mfc'] = 'none'
		self.move_arrow = A.pull('move-arrow', {})
		self.support_props = A.pull('support-props', {})
		self.support_arrow = A.pull('support-arrow', {})
		self.support_dot = A.pull('support-dot', None)
		self.support_defend_props = A.pull('support-defend-props', {})
		self.convoy_props = A.pull('convoy-props', {})
		self.convoy_arrow = A.pull('convoy-arrow', {})
		self.convoy_dot = A.pull('convoy-dot', None)
		self.retreat_action_props = A.pull('retreat-action-arrow', {})
		
		self.graph_path = A.pull('graph-path', None)
		self.regions_path = A.pull('regions-path', None)
		self.bg_path = A.pull('bgs-path', None)
		self.players_path = A.pull('players-path', None)
		self.players = load_yaml(self.players_path)
		self.capitals = {info['capital']: name for name, info in self.players.items()
		                 if 'capital' in info}
		
		self.config = A
		
		self._known_action_drawers = {
			'build': self._draw_build,
			'retreat': self._draw_retreat,
			'disband': self._draw_destroy,
			'destroy': self._draw_destroy,
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
		
		H, W, _ = self.base.shape
		w, h = figaspect(H / W)
		w, h = self.img_scale * w, self.img_scale * h
		
		self.figax = plt.subplots(figsize=(w, h))
		
	def _finalize_render(self, state, actions=None, **kwargs):
		
		plt.imshow(self.base, zorder=-1)
		
		plt.axis('off')
		plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
		                    wspace=0, hspace=0)
		
		if self.overlay_path.exists():
			overlay = self._load_img(self.overlay_path)
			# sel = overlay[...,-1] > 0
			# self.base[sel] = overlay[sel,:3]
			plt.imshow(overlay)
		
	
	def _load_img(self, path, rgb=False):
		img = Image.open(str(path))
		if rgb:
			img = img.convert("RGB")
		return np.array(img)
	
	def _save_img(self, img, path):
		return Image.fromarray(img).save(str(path))
	
	def _load_renderbase(self):
		return self._load_img(self.base_path, rgb=True)
	
	
	def _get_unit_pos(self, loc, retreat=False):
		base, coast = self.map.decode_region_name(loc)
		node = self.map.nodes[base]
		
		if retreat:
			pos = node['locs']['retreat'] if coast is None or 'coast-unit' not in node['locs'] \
				else [node['locs']['coast-retreat'][coast]]
		else:
			pos = node['locs']['unit'] if coast is None or 'coast-unit' not in node['locs'] \
				else [node['locs']['coast-unit'][coast]]
		
		pos = list(map(np.array, zip(*pos)))
		return pos[::-1]
	
	def _get_label_pos(self, loc):
		base, coast = self.map.decode_region_name(loc)
		node = self.map.nodes[base]
		pos = node['locs']['label']
		pos = list(map(np.array, zip(*pos)))
		return pos[::-1]
	
	def _draw_shortest_arrow(self, start, end, arrow_props={}, use_annotation=False):
		x, y = start
		x, y = x.reshape(1, -1), y.reshape(1, -1)
		dx, dy = end
		dx, dy = dx.reshape(-1, 1) - x, dy.reshape(-1, 1) - y
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
		pass
	
	def _draw_scs(self, state, actions=None):
		
		owners = {center: player for player, info in state['players'].items() for center in info.get('centers')}
		centers = [name for name, node in self.map.nodes.items() if node.get('sc', 0) > 0]
		
		for center in centers:
			self._draw_sc(center, owners.get(center, None))
	
	def _draw_sc(self, loc, owner=None):
		
		# color = self.players.get(owner, {}).get('color')
		
		coords = self.map.nodes[loc]['locs'].get('sc')
		if coords is not None:
			for pos in coords:
				if loc in self.capitals:
					return plt.plot(*pos[::-1], **self.capital_props)
				else:
					return plt.plot(*pos[::-1], **self.sc_props)
	
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
		
		disbands = {self.map.decode_region_name(u['loc'], allow_dir=True)[0]
		            for u in state.get('disbands', {}).get(player, [])}
		for loc in disbands:
			self._draw_disband(player, loc)
		
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
		coords = self.map.nodes[loc]['locs'].get('sc')
		if coords is not None:
			for pos in coords:
				y, x = pos
				return plt.plot(x, y, color=color, **self.home_props)
	
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
	
	def _draw_disband(self, player, loc):
		x, y = self._get_unit_pos(loc, retreat=True)
		return plt.plot(x, y, **self.disband_props)
	
	
	def _draw_action(self, player, action):
		fn = self._known_action_drawers.get(action.get('type'))
		if fn is None:
			return self._draw_unknown_action(player, action)
		return fn(player, action)
	
	def _draw_build(self, player, action):
		x, y = self._get_unit_pos(action['loc'])
		shape = self.unit_shapes.get(action.get('unit'))
		plt.plot(x, y, marker=shape, **self.build_props)
	
	def _draw_destroy(self, player, action):
		return self._draw_disband(player, action['loc'])
	
	def _draw_retreat(self, player, action):
		self._draw_shortest_arrow(self._get_unit_pos(action['loc']), self._get_label_pos(action['dest']),
		                          self.retreat_action_props)
	
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
	
	
	
	
