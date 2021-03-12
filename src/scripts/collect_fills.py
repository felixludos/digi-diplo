

import sys, os

import omnifig as fig

import numpy as np
from omnibelt import load_yaml, save_yaml
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

from PIL import Image

from ..colors import fill_region
from .. import util

@fig.Script('collect-fills', 'Collects points to fill on map for control')
def _collect_adj(A):
	'''
	Collects the coordinates on an image to flood fill with colors in rendering.

	Click on every location that should be flood-filled when rendering a game state based on the owner of the territory
	Press space to move to the next territory
	All selected coordinates are saved automatically to the pos yaml file when the window is closed.
	'''
	mlp_backend = A.pull('mlp_backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	nodes_path, pos_path = util.get_map_paths(A, 'nodes', 'pos')

	image_path = A.pull('image-path')
	
	nodes = load_yaml(nodes_path)
	
	pos = load_yaml(pos_path) if os.path.isfile(pos_path) else {}
	
	img = np.array(Image.open(image_path).convert("RGB"))
	
	fill_val = A.pull('fill-val', [0,0,0])
	fill_val = np.array(fill_val)
	threshold = A.pull('threshold', 0.001)
	
	todo = list(nodes)
	
	neutrals = A.pull('include-neutrals', True)
	
	if neutrals:
		todo.extend(['neutral-lands', 'neutral-seas'])
	
	done = []
	
	current = None
	fill = None
	
	def _next_prompt():
		
		nonlocal current, fill
		
		if current is None:
			if not len(todo):
				plt.title('Done!')
				plt.draw()
				return
			fill = None
			current = todo.pop()
		
		if fill is None:
			if current not in pos:
				pos[current] = {}
			if 'fill' not in pos[current]:
				pos[current]['fill'] = []
			fill = pos[current]['fill'].copy()
		
		if fill_val is not None:
			while len(fill):
				x, y = fill.pop()
				x, y = int(x), int(y)
				fill_region(img, (y,x), val=fill_val, threshold=threshold)
				
				plt.plot([x], [y], color='r', ls='', markersize=12,
				         markeredgewidth=3, marker='o', markeredgecolor='k', zorder=5)
			
		name = nodes.get(current, {}).get('name', current)
		plt.cla()
		plt.title(name)
		plt.imshow(img)
		plt.draw()
		
	def onclick(event):
		
		nonlocal current, fill
		
		btn = event.button  # 1 is left, 3 is right
		try:
			xy = [float(event.xdata), float(event.ydata)]
		except:
			return
		
		if btn == 1:
			
			print(f'{current}: {xy}')
			
			if fill is not None:
				fill.append(xy)
				pos[current]['fill'].append(xy)
			
			_next_prompt()
		
		else:  # invalid button
			print(f'unknown button: {btn}')
			return
	
	def onkey(event=None):
		
		nonlocal current, fill
		
		key = None if event is None else event.key
		
		if key is None or key == ' ':
			
			if not len(todo):
				plt.title('Done')
				plt.draw()
				return
			
			if current is not None:
				
				done.append(current)
				current = None
		
			_next_prompt()
		
		
		elif key == 'backspace':
			
			if current is not None:
				todo.append(current)
			
			if len(done):
				current = done.pop()
				fill = []
			
	
	fg, ax = plt.subplots(figsize=(12, 8))
	
	plt.imshow(img)
	plt.axis('off')
	
	plt.title('test')
	
	plt.tight_layout()
	
	bid = fg.canvas.mpl_connect('key_press_event', onkey)
	cid = fg.canvas.mpl_connect('button_press_event', onclick)
	
	_next_prompt()
	
	plt.show(block=True)

	print(f'All fill locs collected, saved to: {pos_path}')
	save_yaml(pos, pos_path, default_flow_style=None)
	
	return pos
	
if __name__ == '__main__':
	fig.entry('collect-fills')


