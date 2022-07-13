from omnibelt import load_yaml, save_yaml
import omnifig as fig

import numpy as np
import matplotlib.pyplot as plt

from .wd import WD_Pixel_Rendering



@fig.Component('ad-pixel-rendering')
class AD_Pixel_Rendering(WD_Pixel_Rendering):
	# def _get_unit_pos(self, loc, retreat=False):
	# 	return super(AD_Pixel_Rendering, self)._get_unit_pos(loc, retreat)#[::-1]
	
	def _render_env(self, state, actions=None, wmx=10, wmy=2550, **kwargs):
		return super(AD_Pixel_Rendering, self)._render_env(state, actions, wmx=wmx, wmy=wmy, **kwargs)

	def _get_label_pos(self, loc, coast=None):
		if coast is None:
			base, coast = self.map.decode_region_name(loc)
		else:
			base, coast = loc, coast
		node = self.map.nodes[base]
		
		if coast is None:# or 'coast-label' not in node['locs']:
			pos = node['locs']['unit']
			# pos = node['locs'].get('label', node['locs']['unit'])
		else:
			pos = node['locs']['coast-unit'][coast]
			# pos = node['locs']['coast-label'][coast]
		# pos = node['locs']['label'] if coast is None or 'coast-label' not in node['locs'] \
		# 	else node['locs']['coast-label'][coast]
		pos = list(map(np.array, zip(*pos)))
		return pos[::-1]
	
	
	# def _get_sc_pos(self, loc):
	# 	return super(AD_Pixel_Rendering, self)._get_sc_pos(loc)#[::-1]
	
	# def _draw_shortest_arrow(self, start, end, arrow_props={}, use_annotation=False):
	# 	return super()._draw_shortest_arrow(start, end, arrow_props, use_annotation)
	
	# def _draw_support_defend(self, player, action):
	# 	self._draw_shortest_arrow(self._get_unit_pos(action['loc']), self._get_unit_pos(action['dest']),
	# 	                          self.support_defend_props, use_annotation=True)
	
	# def _draw_support(self, player, action):
	# 	x, y = self._get_unit_pos(action['loc'])
	#
	# 	x1, y1 = self._get_unit_pos(action['src'])
	# 	x2, y2 = self._get_label_pos(action['dest'])
	#
	# 	lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
	#
	# 	self._draw_shortest_arrow((x1, y1), (x2, y2), self.support_arrow)
	# 	self._draw_shortest_arrow((x, y), (lx, ly), self.support_props, use_annotation=True)
	# 	if self.support_dot is not None:
	# 		plt.plot([lx], [ly], **self.support_dot)
	
	# def _draw_convoy(self, player, action):
	# 	x, y = self._get_unit_pos(action['loc'])
	# 	x1, y1 = self._get_unit_pos(action['src'])
	# 	x2, y2 = self._get_label_pos(action['dest'])
	#
	# 	lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
	#
	# 	self._draw_shortest_arrow((x1, y1), (x2, y2), self.convoy_arrow)
	# 	self._draw_shortest_arrow((x, y), (ly, lx), self.convoy_props, use_annotation=True)
	# 	if self.convoy_dot is not None:
	# 		plt.plot([lx], [ly], **self.convoy_dot)










