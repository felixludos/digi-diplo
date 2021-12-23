
import sys, os
from pathlib import Path

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from ..colors import process_color, hex_to_rgb

from sklearn.neighbors import NearestNeighbors
import networkx as nx

import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.segmentation import *
from skimage.segmentation import watershed, expand_labels
from skimage.morphology import closing, square
from skimage.color import label2rgb
from PIL import Image
import matplotlib.patches as mpatches
from itertools import chain

from scipy import misc, signal

from omnibelt import unspecified_argument
from omnibelt import save_yaml, load_yaml
import omnifig as fig

from ..colors import fill_region, lighter

@fig.Script('map-identities')
def map_identities(A):
	mlp_backend = A.pull('mlp_backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	outpath = A.pull('data-path')
	outpath = Path(outpath)
	
	snapshot = None
	if outpath.exists() and A.pull('resume', True):
		print(f'Resuming {str(outpath)}')
		snapshot = load_yaml(outpath)
	
	im = fig.run('_load-binary', A)
	expanded = get_expanded_from_binary(im)

	N = expanded.max()
	print(f'Found {N} enclosed regions')
	
	H, W = im.shape # must be a binary image (black=boundary)
	
	borders = get_borders_from_expanded(expanded)
	
	clean = expanded.copy().clip(max=1)
	clean[borders] = 0
	clean = np.stack(3*[clean],2)*255
	
	regions = regionprops(expanded)
	
	idents = np.arange(len(regions))
	if snapshot is not None:
		idents = np.array(snapshot['idents'])
	
	current = A.pull('skip', None)
	if current is None and snapshot is not None:
		current = snapshot['last']-1
	
	def _next_prompt():
		
		nonlocal current, idents
		
		if current is None:
			current = 0
		else:
			current += 1
		
		if current == len(regions):
			plt.title('Done!')
			plt.draw()
			return
		
		img = clean.copy()
		img[expanded==current+1] = [255,0,0]
		
		plt.cla()
		plt.title(f'{current+1}/{len(regions)}')
		plt.imshow(img)
		
		cy, cx = regions[current].centroid
		
		r = 500
		plt.xlim(cx-r, cx+r)
		plt.ylim(cy+r, cy-r)
		plt.draw()
	
	def onclick(event):
		
		nonlocal current
		
		btn = event.button  # 1 is left, 3 is right
		try:
			yx = [float(event.ydata), float(event.xdata)]
		except:
			return
		
		if btn == 3:
			
			yx = tuple(map(int, yx))
			
			val = expanded[yx]
			
			print(f'{current}: {val}')
			
			idents[current] = val-1
			
			expanded[expanded == current + 1] = val
			
			_next_prompt()
		
		else:  # invalid button
			print(f'unknown button: {btn}')
			return
	
	def onkey(event=None):
		
		nonlocal H,W, current
		
		key = None if event is None else event.key
		
		if key == 'up':
			plt.xlim(0,W)
			plt.ylim(H,0)
			plt.draw()
			return
		elif key == 'left':
			current -= 2
		
		_next_prompt()
	
	fsize = A.pull('fsize', 8)
	fg, ax = plt.subplots(figsize=(fsize, fsize * (H / W)))
	
	plt.imshow(clean)
	plt.axis('off')
	
	plt.title('test')
	
	plt.subplots_adjust(0, 0, 1, 0.93)

	
	bid = fg.canvas.mpl_connect('key_press_event', onkey)
	cid = fg.canvas.mpl_connect('button_press_event', onclick)
	
	_next_prompt()

	plt.show(block=True)
	
	idents = idents.tolist()

	print(idents)
	print(f'All region idents collected, saved to: {outpath}')
	
	save_yaml({'idents':idents, 'last':current}, outpath, default_flow_style=None)
	
	
	return idents


def fix_binary(im, fix_diagonals=True):
	if fix_diagonals:
		# im = np.logical_not(im).astype(int)
		
		kernel = np.array([[1, -1], [-1, 1]])
		diags = np.abs(signal.convolve2d(im, kernel, boundary='fill')) > 1
		
		im += diags[1:, :-1].astype(np.uint8)
		im += diags[:-1, :-1].astype(np.uint8)
		im = im.astype(bool).astype(np.uint8)
	im = np.logical_not(im)
	
	
	return im

@fig.Script('_load-binary')
def load_binary(A, path=None, fix_diagonals=None):
	
	if path is None:
		path = A.pull('img-path')
	
	im = Image.open(path)
	im = np.array(im)

	if fix_diagonals is None:
		fix_diagonals = A.pull('fix-diagonals', False)

	return fix_binary(im, fix_diagonals=fix_diagonals)


def get_expanded_from_binary(binary):
	label_image = label(binary)
	
	expanded = expand_labels(label_image, distance=20)
	return expanded

def get_borders_from_expanded(expanded):
	return find_boundaries(expanded, connectivity=1, mode='thick', background=0)

def merge_idents(idents):
	if not isinstance(idents, np.ndarray):
		idents = np.array(idents)
	step = idents.copy()
	
	old = None
	while old is None or not ((old - step) == 0).all():
		old = step
		step = idents[step]
		
	return step



@fig.Script('view-idents')
def _view_idents(A):
	mlp_backend = A.pull('mlp_backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	path = A.pull('data-path')
	data = load_yaml(path)
	
	idents = np.array(data['idents'])
	
	step = merge_idents(idents)
	
	exists = np.bincount(step, minlength=len(idents))
	key = (exists > 0).cumsum()  # - 1
	
	codes = key[step]
	
	groups = {}
	for i, x in enumerate(codes):
		if x not in groups:
			groups[x] = set()
		groups[x].add(i)
	
	fields = [[] for _ in range(len(groups))]
	for i, field in enumerate(fields):
		field.extend(groups[i+1])
		
	if 'fields' not in data:
		data['fields'] = fields
		save_yaml(data, path)
	
	N = len(fields)
	print(f'Map contains {N} fields (with {len(step)} regions)')
	
	im = fig.run('_load-binary', A)
	H, W = im.shape
	
	expanded = get_expanded_from_binary(im)
	borders = get_borders_from_expanded(expanded)
	
	clean = expanded.copy().clip(max=1)
	clean[borders] = 0
	clean = np.stack(3 * [clean], 2) * 255
	
	current = A.pull('field', 0)
	
	regions = regionprops(expanded)
	
	def _next_prompt():
		
		nonlocal current
		
		if current < 0:
			current = 0
		elif current >= N:
			current = len(fields)
		
		img = clean.copy()
		for tile in fields[current]:
			img[expanded == tile + 1] = [255, 0, 0]
			
		plt.cla()
		plt.imshow(img)
		plt.axis('off')
		plt.title(f'{current + 1}/{len(fields)}')
		
		cy, cx = regions[current].centroid
		
		r = 500
		plt.xlim(cx - r, cx + r)
		plt.ylim(cy + r, cy - r)
		plt.draw()
	
	def onkey(event=None):
		nonlocal current
		
		key = None if event is None else event.key
		
		if key == 'right':
			current += 1
		elif key == 'left':
			current -= 1
		
		else:
			print(f'unknown key: {key}')
		
		_next_prompt()
	
	fsize = A.pull('fsize', 8)
	fg, ax = plt.subplots(figsize=(fsize, fsize * (H / W)))
	
	plt.imshow(clean)
	plt.axis('off')
	
	plt.title('test')
	
	plt.subplots_adjust(0, 0, 1, 0.9)
	
	bid = fg.canvas.mpl_connect('key_press_event', onkey)
	
	_next_prompt()
	
	plt.show(block=True)


@fig.Script('match-names')
def _match_names(A):
	mlp_backend = A.pull('mlp_backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	path = A.pull('data-path')
	data = load_yaml(path)
	
	if 'fields' not in data:
	
		idents = np.array(data['idents'])
		
		step = merge_idents(idents)
		
		exists = np.bincount(step, minlength=len(idents))
		key = (exists > 0).cumsum()  # - 1
		
		codes = key[step]
		
		groups = {}
		for i, x in enumerate(codes):
			if x not in groups:
				groups[x] = set()
			groups[x].add(i)
		
		fields = [[] for _ in range(len(groups))]
		for i, field in enumerate(fields):
			field.extend(groups[i+1])
			
		if 'fields' not in data:
			data['fields'] = fields
			save_yaml(data, path)
	
	fields = data['fields']
	
	N = len(fields)
	
	im = fig.run('_load-binary', A)
	H, W = im.shape
	
	expanded = get_expanded_from_binary(im)
	borders = get_borders_from_expanded(expanded)
	
	clean = expanded.copy().clip(max=1)
	clean[borders] = 0
	clean = np.stack(3 * [clean], 2) * 255
	
	X, Y = np.mgrid[0:H,0:W]
	
	if 'nodes' in data:
		order = data['nodes']
	else:
		order = ['UNKNOWN']*len(fields)
		data['nodes'] = order
	
	txt = []
	options = []
	with open(A.pull('names-path'), 'r') as f:
		options = f.read().split('\n')
	done = []
	
	print(f'Map contains {N} fields and {len(options)} node names')
	
	top = []
	shift = 0

	nones = 0
	def assign(field, sel=None):
		nonlocal nones #current
		if sel == 0:
			sel = 1
		if sel is None:
			sel = 1
		pick = top[sel-1]
		out = None
		if pick.startswith('>'):
			pick = pick[1:]
		elif pick in options:
			options.remove(pick)
			done.append(pick)
		if pick == 'none':
			pick = f'none{nones}'
			nones += 1
		if pick in order:
			idx = order.index(pick)
			order[idx] = 'UNKNOWN'
			out = idx
		order[field] = pick
		extra = '' if out is None else f' (replacing {out})'
		print(f'{field}: {pick}{extra}')
		return out
	
	def update():
		nonlocal top, shift
		
		if len(txt):
			search = ''.join(txt).lower()
			
			matches = [name for name in options if name.lower().startswith(search)]
			
			if len(matches) < 9:
				matches.extend('>' + name for name in done if name.lower().startswith(search))
			
			idx = max(0, min(shift, len(matches)-9))
			
			top = matches[idx:idx+9]
		
		else:
			idx = max(0, min(shift, len(options) - 9))
			top = options[idx:idx+9]
		
		plt.title(' '.join(txt))
		patches = [mpatches.Patch(color=f'C{i}', label=f'{i+1}: {t}') for i,t in enumerate(top)]
		plt.legend(handles=patches,fontsize=16)
		plt.xticks([])
		plt.yticks([])
		plt.xlabel(f'{current + 1}/{len(fields)}: {order[current]}')
		plt.subplots_adjust(0, 0.07, 1, 0.9)
		plt.draw()
	
	current = A.pull('field', None)
	
	if current is None:
		for i, o in enumerate(order):
			if o == 'UNKNOWN':
				current = i
				print(f'Skipping to {current}')
				break
		
		if current is None:
			print('All done!')
			current = 0
	
	for n in order:
		if n != 'UNKNOWN':
			if n in options:
				options.remove(n)
			elif n.startswith('none'):
				options.remove('none')
			done.append(n)
		if n.startswith('none'):
			nones += 1
	
	
	def _next_prompt(force=None):
		
		nonlocal current, txt, shift
		
		txt.clear()
		shift = 0
		
		if force is None:
			old = current
			current = None
			for i in range(len(order)):
				i = (i+old)%len(order)
				if order[i] == 'UNKNOWN':
					current = i
					break
		
		if current is None:
			print('DONE!')
			current = 0
		
		if current < 0:
			current = 0
		elif current >= N:
			current = len(fields)
		
		xs = []
		ys = []
		
		img = clean.copy()
		for tile in fields[current]:
			sel = expanded == tile + 1
			img[sel] = [255, 0, 0]
			xs.extend(X[sel])
			ys.extend(Y[sel])
			
		plt.cla()
		plt.imshow(img)
		# plt.axis('off')

		cx, cy = np.mean(ys), np.mean(xs)
		# cy, cx = np.mean(ys), np.mean(xs)
		
		r = 500
		plt.xlim(cx - r, cx + r)
		plt.ylim(cy + r, cy - r)
		
		update()
		
	def onkey(event=None):
		nonlocal current, txt, options, shift
		
		key = None if event is None else event.key
		

		if key == '=':
			plt.xlim(0,W)
			plt.ylim(H,0)
			plt.draw()
	
	
		elif key == 'enter':
			_next_prompt(assign(current))
		
		elif key in '123456789':
			# pick = options[sel]
			# del options[sel]
			# order[current] = pick
			_next_prompt(assign(current, int(key)))

		elif key == 'right':
			_next_prompt()
		elif key == 'left':
			current -= 1
			_next_prompt(current)
		
		
		elif key == 'backspace':
			if len(txt):
				txt.pop()
			update()
		
		elif key == 'up':
			shift = max(0, shift-1)
			update()
		elif key == 'down':
			shift += 1
			update()
		
		else:
			if key == '-':
				key = 'q'
			txt.append(str(key))
			plt.title(' '.join(txt))
			plt.draw()
			update()
			# print(f'unknown key: {key}')
		
		
	
	fsize = A.pull('fsize', 8)
	fg, ax = plt.subplots(figsize=(fsize, fsize * (H / W)))
	
	plt.imshow(clean)
	plt.axis('off')
	
	# plt.title('test')
	# plt.subplots_adjust(0, 0, 1, 0.9)
	
	bid = fg.canvas.mpl_connect('key_press_event', onkey)
	
	_next_prompt(current)
	
	plt.show(block=True)
	
	print(order)
	print(f'All region idents collected, saved to: {path}')
	
	save_yaml(data, path, default_flow_style=None)
	
	return order

def _format_color(color):
	if isinstance(color, (list,tuple)) and isinstance(color[0], int):
		return [c/255 for c in color]
	return color
	
@fig.Script('categorize-fields')
def categorize_fields(A):
	mlp_backend = A.pull('mlp_backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	path = A.pull('data-path')
	data = load_yaml(path)
	
	if 'fields' not in data:
	
		idents = np.array(data['idents'])
		
		step = merge_idents(idents)
		
		exists = np.bincount(step, minlength=len(idents))
		key = (exists > 0).cumsum()  # - 1
		
		codes = key[step]
		
		groups = {}
		for i, x in enumerate(codes):
			if x not in groups:
				groups[x] = set()
			groups[x].add(i)
		
		fields = [[] for _ in range(len(groups))]
		for i, field in enumerate(fields):
			field.extend(groups[i+1])
			
		if 'fields' not in data:
			data['fields'] = fields
			save_yaml(data, path)
	
	fields = data['fields']
	
	N = len(fields)
	
	options = A.pull('cats', '<>categories')
	cat_colors = A.pull('cat-colors', None)
	
	fpath = A.pull('fields-img', None)
	if fpath is None:
		raise NotImplementedError
		im = fig.run('_load-binary', A)
	else:
		im = Image.open(fpath)
		im = np.array(im)
	
	H, W = im.shape
	
	borders = get_borders_from_expanded(im)
	
	im[borders == 1] = 0
	
	clean = im.copy().clip(max=1)
	clean = np.stack(3 * [clean], 2) * 255
	
	regions = regionprops(im)
	
	nodes = data['nodes'] if 'nodes' in data else None
	fields = data['fields'] if 'fields' in data else None
	
	if 'cats' in data:
		cats = data['cats']
	else:
		cats = [None]*len(fields)
		data['cats'] = cats
	
	N = len(cats)
	print(f'Map contains {N} fields with categories: {list(options.values())}')
	
	cat_colors_names = None
	if cat_colors is not None:
		cat_colors_names = {options[k]:cat_colors[k] for k in options}
	
	current = A.pull('field', None)
	if current is None:
		for i, o in enumerate(cats):
			if o is None:
				current = i
				print(f'Skipping to {current}')
				break
			elif cat_colors is not None and o in cat_colors_names:
					clean[im == i + 1] = cat_colors_names[o]
		if current is None:
			print('All done!')
			current = 0
	
	
	def _next_prompt(force=None):
		nonlocal current
		
		if force is None:
			old = current
			current = None
			for i in range(N):
				i = (i+old)%N
				if cats[i] is None:
					current = i
					break
		
		if current is None:
			print('DONE!')
			current = 0
		
		if current < 0:
			current = 0
		elif current >= N:
			current = len(cats)
		
		img = clean.copy()
		sel = im == current + 1
		img[sel] = [255, 0, 0]
		
		plt.cla()
		plt.imshow(img)
		# plt.axis('off')

		region = regions[current]

		y1,x1,y2,x2 = region.bbox
		x,y = x1,y1
		h = y2-y1
		w = x2-x1
		cy, cx = region.centroid
		
		r = 1000
		if h < r:
			h = r
			y = cy-r/2
		if w < r:
			w = r
			x = cx-r/2
		
		plt.xlim(x, x+w)
		plt.ylim(y+h, y)
		
		if nodes is not None:
			plt.title(f'{nodes[current]}')
		patches = [mpatches.Patch(color=f'C{i%10}' if cat_colors is None
											else _format_color(cat_colors.get(k,f'C{i%10}')),
		                          label=f'{k}: {v}') for i,(k,v) in enumerate(options.items())]
		plt.legend(handles=patches,fontsize=16)
		plt.xticks([])
		plt.xlabel(f'{current+1}/{len(cats)}: {cats[current]}')
		plt.yticks([])
		plt.subplots_adjust(0, 0.05, 1, 0.95)
		plt.draw()
	
	def onkey(event=None):
		nonlocal current, options
		
		key = None if event is None else event.key
		
		if key == '=':
			plt.xlim(0,W)
			plt.ylim(H,0)
			plt.draw()
	
		if key in options:
			cats[current] = options[key]
			if cat_colors is not None and key in cat_colors:
				clean[im==current+1] = cat_colors[key]
			_next_prompt()
	
		elif key == 'right':
			_next_prompt()
		elif key == 'left':
			current -= 1
			_next_prompt(current)
		
	
	fsize = A.pull('fsize', 8)
	fg, ax = plt.subplots(figsize=(fsize, fsize * (H / W)))
	
	plt.imshow(clean)
	plt.axis('off')
	
	# plt.title('test')
	# plt.subplots_adjust(0, 0, 1, 0.9)
	
	bid = fg.canvas.mpl_connect('key_press_event', onkey)
	
	_next_prompt(current)
	
	plt.show(block=True)
	
	print(cats)
	print(f'All region idents collected, saved to: {path}')
	
	save_yaml(data, path, default_flow_style=None)
	
	return cats
	
	

def extend_slice(slic, num=1):
	new = []
	for s in slic:
		new.append(slice(max(0,s.start-num), s.stop+num))
	return tuple(new)

def generate_edges(idx, sub):
	
	bd = find_boundaries((sub == idx), mode='outer')
	breg = regionprops(bd.astype(int))[0]
	
	coords = breg.coords
	
	clf = NearestNeighbors(n_neighbors=2).fit(coords)
	G = clf.kneighbors_graph()
	T = nx.from_scipy_sparse_matrix(G)
	
	neighbors = []
	pieces = list(nx.connected_components(T))
	for piece in pieces:
		
		order = list(nx.dfs_preorder_nodes(T, min(piece)))
		
		for i in sub[bd][order].tolist():
			if i not in neighbors:
				neighbors.append(i)
	return neighbors

def boundary_midpoint(idx, sub, neighbors, offset=None):

	bd = find_boundaries((sub == idx), mode='outer')
	breg = regionprops(bd.astype(int))[0]
	
	coords = breg.coords
	
	clf = NearestNeighbors(n_neighbors=2).fit(coords)
	G = clf.kneighbors_graph()
	T = nx.from_scipy_sparse_matrix(G)
	
	valids = []
	
	pieces = list(nx.connected_components(T))
	for piece in pieces:
		
		order = np.array(list(nx.dfs_preorder_nodes(T, min(piece))), dtype='int')
		
		fixed = sub[bd][order]
		
		sel = fixed == neighbors[0]
		for n in neighbors[1:]:
			sel += fixed == n
		valids.append((sel, order))
		
	
	best, order = max(valids, key=lambda x: x[0].sum())
	options = order[best]
	x,y = coords[options[len(options)//2]]
	
	if offset is not None:
		x += offset[0]
		y += offset[1]
	return x,y


def coords_order(coords):
	
	clf = NearestNeighbors(n_neighbors=2).fit(coords)
	G = clf.kneighbors_graph()
	T = nx.from_scipy_sparse_matrix(G)
	
	orders = [np.array(list(nx.dfs_preorder_nodes(T, min(piece))), dtype='int')
	          for piece in nx.connected_components(T)]
	return orders

@fig.Script('edge-finder')
def edge_finder(A):
	mlp_backend = A.pull('mlp_backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	datapath = A.pull('data-path')
	data = load_yaml(datapath)
	
	names = data['nodes'] if 'nodes' in data else None
	if names is None:
		raise Exception('missing node names, using script: "match-names"')
	name_idx = {n:i for i,n in enumerate(names)}
	fields = data['fields'] if 'fields' in data else None
	if fields is None:
		
		if 'idents' not in data:
			raise Exception('missing idents, using script: "map-identities"')
		idents = np.array(data['idents'])
		
		step = merge_idents(idents)
		
		exists = np.bincount(step, minlength=len(idents))
		key = (exists > 0).cumsum()  # - 1
		
		codes = key[step]
		
		groups = {}
		for i, x in enumerate(codes):
			if x not in groups:
				groups[x] = set()
			groups[x].add(i)
		
		fields = [[] for _ in range(len(groups))]
		for i, field in enumerate(fields):
			field.extend(groups[i + 1])
	cats = data['cats'] if 'cats' in data else None
	if cats is None:
		raise Exception('missing field regions, using script: "categorize-fields"')
	
	# env_type = {'desert':'land', 'land':'land', 'island':'land', 'mountains':'land',
	#             'sea':'water', 'lake':'water', 'river':'water', 'background':'background'}
	
	N = len(fields)
	
	# options = A.pull('cats', '<>categories')
	# cat_colors = A.pull('cat-colors', None)
	
	binary = fig.run('_load-binary', A)
	rim = get_expanded_from_binary(binary)
		
	fpath = A.pull('fields-img', None)
	if fpath is None:
		raise NotImplementedError
	else:
		im = Image.open(fpath)
		im = np.array(im)
	
	regions = regionprops(im)
	subregions = regionprops(rim)
	
	ordered_edges = A.pull('ordered-edges', True)
	key = 'ordered-edges' if ordered_edges else 'unordered-edges'
	edges = data[key] if key in data else None
	
	if edges is None:
		if ordered_edges:
			edges = [generate_edges(idx, im[extend_slice(region.slice)]-1)
			         for idx, region in tqdm(enumerate(regions), total=len(regions), desc='Generate Ordered Edges')]
		else:
		
			pairs = set(zip(im[..., :-1].reshape(-1), im[..., 1:].reshape(-1)))
			pairs.update(zip(im[:-1].reshape(-1), im[1:].reshape(-1)))
			raw_edges = {}
			for x, y in pairs:
				if x != y:
					if x not in raw_edges:
						raw_edges[x] = set()
					raw_edges[x].add(y)
					if y not in raw_edges:
						raw_edges[y] = set()
					raw_edges[y].add(x)
			edges = [[] for _ in range(len(raw_edges))]
			for k, vs in raw_edges.items():
				k = k - 1
				if len(edges[k]):
					print(k, 'failed')
				edges[k].extend(v.item()-1 for v in vs)
			
		data[key] = edges
		save_yaml(data, datapath)
	
	H, W = im.shape

	# borders = get_borders_from_expanded(im)
	# clean = im.copy().clip(max=1)
	# clean[borders == 1] = 0
	# clean = np.stack(3 * [clean], 2) * 255
	
	N = len(cats)
	print(f'Map contains {N} fields')
	
	outpath = A.pull('nodes-path', 'out.yaml')
	outpath = Path(outpath)
	
	if outpath.exists():
		assert False, str(outpath)
		# nodes = load_yaml(outpath)
	else:
		nodes = {}
		
	rivers = []
	fg_types = {'land', 'sea', 'coast', 'island'}
	if A.pull('use-deserts', True):
		fg_types.add('desert')
	
	# filter edges
	fg = []
	bg = set()
	for idx in range(N):
		cat = cats[idx]
		if cat in fg_types:
			fg.append(idx)
		else:
			bg.add(idx)
		if cat == 'river':
			rivers.append(idx)
			
	print(f'Found {len(fg)} fields and {len(bg)} background tiles ({len(rivers)} rivers)')

	# skip_islands = set(name_idx[n] for n in A.pull('skip-islands', []))
	multi_nodes = {name_idx[n]:[name_idx[s] for s in subs] for n,subs in A.pull('multi-nodes', {}).items()}
	sub_nodes = {}
	for n, subs in multi_nodes.items():
		sub_nodes.update({s:n for s in subs})
	
	islands = {}
	for idx in tqdm(fg, desc='Filtering edges'):
		name = names[idx]
		cat = cats[idx]
		es = edges[idx]
		
		node = {'idx': idx, 'name': name}
		
		if idx in sub_nodes:
			es = edges[sub_nodes[idx]].copy()
		
		if cat == 'island':
			node['island'] = es[0]
			islands[idx] = es[0]
			es = edges[es[0]].copy()
			es.remove(idx)
		
		node['edges'] = es
		
		if cat == 'desert':
			cat = 'land'
		node['env'] = cat
		
		nodes[idx] = node
	
	print(f'Found {len(islands)} islands')
	
	for node in nodes.values():
		for e in multi_nodes:
			if e in node['edges']:
				ind = node['edges'].index(e)
				for n in multi_nodes[e]:
					node['edges'].insert(ind,n)
				node['edges'].remove(e)
				
		if node['env'] == 'land':
			for e in node['edges']:
				if cats[e] == 'sea' or cats[e] == 'river':
					node['env'] = 'coast'
		
		node['edges'] = [e for e in node['edges'] if e in nodes]
	
	coasts = set(i for i, n in nodes.items() if n['env'] == 'coast')
	# seas = set(i for i, n in nodes.items() if n['env'] == 'sea')
	
	errors = [n['name'] for n in nodes.values() if len(n['edges']) == 0]
	if len(errors):
		raise Exception(f'Nodes without edges: {len(errors)}')
	
	split_nodes = {name_idx[n] for n in A.pull('split-nodes', [])}
	
	def get_region(idx):
		if idx in split_nodes:
			subs = [subregions[i] for i in fields[idx]]
			ind = max((s.area, i) for i, s in enumerate(subs))[1]
			return subs[ind]
		return regions[idx]
	def get_neighboring_regions(idx, neighbors):
		
		regs = []
		
		cx = get_region(idx).centroid
		
		for e in neighbors:
			if e in split_nodes:
				subs = [subregions[i] for i in fields[e]]
				cxs = np.array([reg.centroid for reg in subs])
				nearest = ((cxs-cx)**2).sum(1).argmin()
				regs.append(subs[nearest])
			else:
				regs.append(regions[e])
		
		return regs
	
	# def order_edges(idx, es):
	#
	# 	region = get_region(idx)
	# 	cx = np.array(region.centroid)
	#
	# 	neighbors = get_neighboring_regions(idx, es)
	#
	# 	cxs = np.array([reg.centroid for reg in neighbors])
	# 	diff = cxs-cx
	# 	angles = np.arctan2(diff[:,0], -diff[:,1])
	#
	# 	return [e for a,e in sorted(zip(angles,es))]
	

	skip_coasts = set(name_idx[n] for n in A.pull('skip-coasts', []))
	
	_coast_names = ['NC', 'EC', 'SC', 'WC']
	_coast_dirs = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]]).T
	
	# find coasts
	for coast in coasts:# tqdm(coasts, desc='Processing coasts'):
		
		node = nodes[coast]
		cname = node['name']
		
		# ces = order_edges(coast, node['edges'])
		ces = node['edges']
		if coast not in skip_coasts:
			ecats = [nodes[e]['env'] for e in ces]
			try:
				start = ecats.index('coast')
			except ValueError:
				pass
			else:
				groups = [[]]
				current = groups[0]
				for i in range(len(ces)):
					ind = (start+i)%len(ces)
					e = ces[ind]
					cat = ecats[ind]
					if cat == 'sea':
						current.append(e)
					elif len(current):
						groups.append([])
						current = groups[-1]
					
				groups = [g for g in groups if len(g)]
				
				if len(groups) > 1:

					if len(groups) > 4:
						raise Exception('not setup to deal with more than 4 coasts per field')
					dirs = np.stack([np.mean([reg.centroid for reg in get_neighboring_regions(coast, g)],0)
					                 for g in groups]) - np.array(get_region(coast).centroid)
					dirs = dirs/np.linalg.norm(dirs,axis=1,keepdims=True)
					
					resps = {}
					
					matches = dirs @ _coast_dirs
					options = matches.copy()
					inds = np.arange(len(_coast_names)).tolist()
					for i in reversed(np.argsort(matches.max(1))):
						pick = np.argmax(options[i])
						ind = inds[pick]
						resps[_coast_names[ind]] = groups[i]
						del inds[pick]
						options = matches[:,inds]
						
					pg = {c:[nodes[e]['name'] for e in g] for c,g in resps.items()}
					print(f'Found coast: {cname} {pg}')
					node['coasts'] = resps
		
	land_types = {'land', 'coast', 'desert'}
	if A.pull('army-islands', True):
		land_types.add('island')
	
	sea_types = {'sea', 'ocean', 'island', 'coast'}
	
	cat_decoder = A.pull('cat-decoder')
	
	# edge processing
	for node in tqdm(sorted(nodes.values(), key=lambda n: 'coasts' not in n),
	                 total=len(nodes), desc='Processing edge types'):
		idx = node['idx']
		cat = node['env']
		
		typ = cat_decoder[cat]
		node['type'] = typ
		
		node['pos'] = np.array(get_region(idx).centroid).tolist()
		
		if cat in land_types:
			node['army-edges'] = [n for n in node['edges'] if nodes[n]['env'] in land_types]
		
		if typ == 'coast' and 'coasts' in node:

			naive = [n for n in node['edges'] if nodes[n]['env'] in sea_types]
			
			# for n in naive:
			# 	assert 'coasts' not in nodes[n], f'adjacent coasts not supported yet: {node} vs {nodes[n]}'
			
			coasts = node['coasts']
			node['dirs'] = list(coasts.keys())
			
			# coast pos
			cpos = {wind: np.array(boundary_midpoint(idx, im[extend_slice(get_region(idx).slice)]-1, seas,
			                                         get_region(idx).bbox[:2])).tolist()
			        for wind, seas in coasts.items()}
			node['coast-pos'] = cpos
			
			cedges = {}
			for cdir, seas in coasts.items():
				nedges = []
				cedges[cdir] = nedges
				for e in naive:
					if e in seas: # coast to sea
						nedges.append(e)
					elif 'coasts' in nodes[e]: # coast to specific coast
						pick = None
						for s in seas:
							if pick is None:
								for odir, oseas in nodes[e]['coasts'].items():
									if s in oseas:
										nedges.append([e,odir])
										pick = odir
										break
					else: # coast to general coast
						for s in seas:
							if e in nodes[s]['edges']:
								nedges.append(e)
								break
		

			node['fleet-edges'] = cedges
		
		elif typ in sea_types:
			
			naive = [n for n in node['edges'] if nodes[n]['env'] in sea_types]
			
			nedges = []
			for e in naive:
				if 'coasts' in nodes[e]:
					assert 'fleet-edges' in nodes[e], f'must work on {nodes[e]} before this node {node}'
					pick = None
					for cdir, es in nodes[e]['fleet-edges'].items():
						if idx in es:
							nedges.append([e,cdir])
							pick = cdir
							break
					# assert pick is not None, f'no shared coast edge found: {node}'
				else:
					nedges.append(e)
			
			node['fleet-edges'] = nedges
			
		
		# for naval-edges, check all neighbors if they have coasts, if so look at responsibility and shared coasts to identify all edges
		
		# if cat == 'sea':
		# 	node['fleet-edges'] = order_edges(idx, [n['idx'] for n in node['edges'] if n['env'] in {'sea', 'coast'}])
	
	
	# boundary_midpoint(idx, im[extend_slice(get_region(idx).slice)]-1, seas)
	
	save_yaml(nodes, outpath)
	
	print('Saved node information (including positions and edges)')
	
	return nodes


_coast_decoder = {' (NC)': 'NC', ' (SC)': 'SC', ' (WC)': 'WC', ' (EC)': 'EC'}

@fig.Script('_decode_region_name')
def decode(A, name=None, allow_dir=None):
	if name is None:
		name = A.pull('name', silent=True)
	
	if len(name) < 5:
		return name, None
	
	end = name[-5:]
	if end in _coast_decoder:
		if allow_dir is None:
			allow_dir = A.pull('allow_dir', False, silent=True)
		if allow_dir:
			return name, _coast_decoder[end]
		return name[:-5], _coast_decoder[end]
	
	if name.endswith(' (Coast)'):
		return name[:-8], True
	
	return name, None
	
@fig.Script('_encode_region_name')
def encode(A, name=None, coast=unspecified_argument, node_type=None, unit_type=None):
	if name is None:
		name = A.pull('name', silent=True)
	
	if coast is unspecified_argument:
		coast = A.pull('coast', None, silent=True)
	
	if coast is not None and coast in {'NC', 'SC', 'WC', 'EC'}:
		return f'{name} ({coast})'
	elif coast:
		return f'{name} (Coast)'
	
	if node_type is None:
		node_type = A.pull('node_type', None, silent=True)
	
	if unit_type is None:
		unit_type = A.pull('unit_type', None, silent=True)
	
	if node_type == 'coast' and unit_type == 'fleet':
		return f'{name} (Coast)'
	
	return name
	
	
@fig.Script('map-attribute')
def map_attribute(A):
	mlp_backend = A.pull('mlp_backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	attr_key = A.pull('attr-key')
	
	path = A.pull('path')
	data = load_yaml(path)
	
	N = len(data)
	
	options = A.pull('cats', '<>categories')
	cat_colors = A.pull('cat-colors', None)
	
	fpath = A.pull('fields-img', None)
	if fpath is None:
		raise NotImplementedError
		im = fig.run('_load-binary', A)
	else:
		im = Image.open(fpath)
		im = np.array(im)
	
	H, W = im.shape
	
	borders = get_borders_from_expanded(im)
	
	im[borders == 1] = 0
	
	clean = im.copy().clip(max=1)
	clean = np.stack(3 * [clean], 2) * 255
	
	regions = regionprops(im)
	
	cats = {ID: info[attr_key] for ID, info in data.items() if attr_key in info}
	
	print(f'Found {len(cats)} existing categorizations for {attr_key}')
	
	todo = list(data.keys())
	
	flts = A.pull('filters', None)
	
	if flts is not None:
		for attr, valid in flts.items():
			if isinstance(valid, (list, tuple)):
				valid = set(valid)
			todo = [ID for ID in todo if (attr in data[ID] and data[ID][attr] in valid)]

	N = len(todo)
	print(f'Categorizing {N} nodes')
	
	cat_colors_names = None
	if cat_colors is not None:
		cat_colors_names = {options[k]: cat_colors[k] for k in options}
	
	for ID, cat in cats.items():
		if cat_colors is not None and cat in cat_colors:
			clean[im == data[ID]['idx'] + 1] = cat_colors_names[cat]
	
	current = 0
	for i, ID in enumerate(todo):
		if ID not in cats:
			current = i
			break
	
	def _next_prompt(force=None):
		nonlocal current
		
		if force is None:
			old = current
			current = None
			for i in range(N):
				i = (i + old) % N
				if todo[i] not in cats:
					current = i
					break
		
		if current is None:
			print('DONE!')
			current = 0
		
		if current < 0:
			current = 0
		elif current >= N:
			current = len(cats)
		
		ID = todo[current]
		info = data[ID]
		
		name = info.get('name', ID)
		idx = info['idx']
		
		img = clean.copy()
		sel = im == idx + 1
		img[sel] = [255, 0, 0]
		
		plt.cla()
		plt.imshow(img)
		# plt.axis('off')
		
		region = regions[idx]
		
		y1, x1, y2, x2 = region.bbox
		x, y = x1, y1
		h = y2 - y1
		w = x2 - x1
		cy, cx = region.centroid
		
		r = 1000
		if h < r:
			h = r
			y = cy - r / 2
		if w < r:
			w = r
			x = cx - r / 2
		
		plt.xlim(x, x + w)
		plt.ylim(y + h, y)
		
		plt.title(f'{name} ({current + 1}/{len(todo)})')
		patches = [mpatches.Patch(color=f'C{i % 10}' if cat_colors is None
		else _format_color(cat_colors.get(k, f'C{i % 10}')),
		                          label=f'{k}: {v}') for i, (k, v) in enumerate(options.items())]
		plt.legend(handles=patches, fontsize=16)
		if ID in cats:
			plt.xlabel(f'{attr_key}: {cats[ID]}')
		plt.xticks([])
		plt.yticks([])
		plt.subplots_adjust(0, 0.05, 1, 0.93)
		plt.draw()
	
	def onkey(event=None):
		nonlocal current, options
		
		key = None if event is None else event.key
		
		if key == '=':
			plt.xlim(0, W)
			plt.ylim(H, 0)
			plt.draw()
		
		if key in options:
			ID = todo[current]
			cats[ID] = options[key]
			if cat_colors is not None and key in cat_colors:
				clean[im == data[ID]['idx'] + 1] = cat_colors[key]
			_next_prompt()
		
		elif key == 'right':
			_next_prompt()
		elif key == 'left':
			current -= 1
			_next_prompt(current)
	
	fsize = A.pull('fsize', 8)
	fg, ax = plt.subplots(figsize=(fsize, fsize * (H / W)))
	
	plt.imshow(clean)
	plt.axis('off')
	
	# plt.title('test')
	# plt.subplots_adjust(0, 0, 1, 0.9)
	
	bid = fg.canvas.mpl_connect('key_press_event', onkey)
	
	_next_prompt(current)
	
	plt.show(block=True)
	
	print(cats)
	print(f'All fields categorized, saved to: {path}')
	
	for ID, cat in cats.items():
		data[ID][attr_key] = cat
	
	save_yaml(data, path, default_flow_style=None)
	
	return cats

