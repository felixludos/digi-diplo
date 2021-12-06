from pathlib import Path
from omnibelt import save_yaml
import omnifig as fig

import numpy as np
from PIL import Image


@fig.Component('map-renderer')
class DiplomacyRenderer(fig.Configurable):
	def __init__(self, A, **kwargs):
		
		self.base_path = Path(A.pull('renderbase-path'))
		assert self.base_path.exists(), 'No render base found"'
		

		self.regions_path = A.pull('regions-path', None)
		self.bg_path = A.pull('bgs-path', None)
		self.players_path = A.pull('players-path', None)
	
	
	def __call__(self, state, action=None, savepath=None, **kwargs):
		img = self._render(state, action=action, **kwargs)
		if savepath is not None:
			self._save_img(img, savepath)
		return img
	
	
	def _load_img(self, path):
		return np.array(Image.open(str(path)))
	
	
	def _save_img(self, img, path):
		return Image.fromarray(img).save(str(path))
	
	
	def _load_renderbase(self):
		return self._load_img(self.base_path)
	
	
	def _render(self, state, action=None):
		base = self._load_renderbase()
		
		# TODO
		
		return base
	
	
	pass

