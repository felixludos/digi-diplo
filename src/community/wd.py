from omnibelt import load_yaml, save_yaml
import omnifig as fig

from ..elements import DiploMap
from ..render import DiplomacyRenderer, DefaultRenderer


@fig.Component('wd-renderer')
class WD_Rendering(DefaultRenderer):
	
	def _get_unit_pos(self, loc, retreat=False):
		return super()._get_unit_pos(loc, retreat=False)[::-1]
	
	def _get_label_pos(self, loc):
		return super()._get_label_pos(loc)[::-1]

	def _get_sc_pos(self, loc):
		pos = super()._get_sc_pos(loc)
		if pos is not None:
			return pos[::-1]
		
	def _load_overlay(self):
		return self._load_img(self.overlay_path, rgba=True)

	def _draw_labels(self, state, actions=None):
		pass



class WD_Map(DiploMap):
	
	
	
	pass

