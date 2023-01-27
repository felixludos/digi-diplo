

from omnibelt import load_yaml, save_yaml
import omnifig as fig

import numpy as np
import matplotlib.pyplot as plt

from ..colors import hex_to_rgb
from ..elements import DiploMap, DashCoast
from ..render import DiplomacyRenderer, DefaultRenderer
from ..managers import DiplomacyManager, NoUnitFoundError


@fig.component('balkans-renderer')
class Balkans_Rendering(DefaultRenderer):
	__version__ = (1, 0)
	
	def __init__(self, year_offset=0, year_props=None, season_titles=None, season_props=None, **kwargs):
		if season_titles is None:
			season_titles = {}
		super().__init__(**kwargs)
		self.year_offset = year_offset
		self.year_props = year_props
		self.season_titles = season_titles
		self.season_props = season_props
		
	# 	self.core_props = A.pull('core-props', {})
	# 	self._known_action_drawers['core'] = self._draw_core
	#
	# def _render(self, state, actions=None, **kwargs):
	# 	self.capitals = {info['capital']: name for name, info in state.get('players', {}).items()}
	# 	return super()._render(state, actions=actions, **kwargs)
	
	def _render_env(self, state, actions=None, **kwargs):
		out = super()._render_env(state, actions=actions, **kwargs)
		
		date = state.get('time', {})
		season = date.get('season', None)
		if str(season) in self.season_titles and self.season_props is not None:
			plt.text(s=self.season_titles[str(season)], **self.season_props)
		
		year = date.get('turn', None)
		if self.year_props is not None:
			plt.text(s=str(year + self.year_offset), **self.year_props)
		
		plt.text(x=10, y=4405, s='Adjudicated and rendered automatically using Digi-diplo '
		                         '(GitHub: felixludos/digi-diplo)', color='w',
		         fontsize=2, fontfamily='monospace')  # PLEASE DO NOT REMOVE
		
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



