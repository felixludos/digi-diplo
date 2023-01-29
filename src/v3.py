from typing import Dict, Optional, Union
import sys, os
from tabulate import tabulate
from tqdm import tqdm
from collections import Counter
from pathlib import Path
import numpy as np
import networkx as nx
from networkx.algorithms.coloring import greedy_color
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import expand_labels, find_boundaries
from skimage.measure import label, regionprops
from PIL import Image
from scipy.optimize import linear_sum_assignment

from omnibelt import load_yaml, save_yaml, load_txt, save_txt
import omnifig as fig

from .colors import hex_to_rgb, rgb_to_hex, process_color



def coords_order(coords):
	clf = NearestNeighbors(n_neighbors=2).fit(coords)
	G = clf.kneighbors_graph()
	T = nx.from_scipy_sparse_matrix(G)
	orders = [np.array(list(nx.dfs_preorder_nodes(T, min(piece))), dtype='int')
			  for piece in nx.connected_components(T)]
	return orders



def extract_neighbors(im, grow=100, pbar=None):
	fixed = expand_labels(im, grow)
	
	neighbors = {}
	
	nodes = set(fixed.reshape(-1).tolist())
	itr = iter(nodes)
	# itr = range(im.max()+1)
	if pbar is not None:
		itr = pbar(itr, total=len(nodes))
	for idx in itr:
		if idx != 0:
			sel = fixed == idx
			reg = regionprops(find_boundaries(sel, mode='outer').astype(int))[0]
			ords = coords_order(reg.coords)
			nss = [set(fixed[tuple(reg.coords[o].T)]) for o in ords]
			neighbors[idx] = {n for ns in nss for n in ns if n != 0}
		
	g = nx.from_dict_of_lists(neighbors)
	return g



def fill_diagonals(img):
	H, W = img.shape
	kernel = np.array([[0, 1, 0, 1, 0],
					   [1, 1, 1, 1, 1],
					   [0, 1, 1, 1, 0],
					   [1, 1, 1, 1, 1],
					   [0, 1, 0, 1, 0]])
	out = cov_transpose(img, kernel, stride=3, padding=1)
	out = np.logical_not(out)
	out = out.astype(bool).astype(int)
	return out



def cov_transpose(img, kernel, stride, padding):
	# Ensure type is bool
	img = img.astype('bool')
	kernel = kernel.astype('bool')
	
	# Extension
	out = np.repeat(img, stride, axis=0)  # Duplicate rows
	out = np.repeat(out, stride, axis=1)  # Duplicate columns
	
	# Padding
	out = np.pad(out, (padding, padding), 'constant', constant_values=(0, 0))
	
	# Window
	view = np.lib.stride_tricks.sliding_window_view(out, window_shape=kernel.shape,
													writeable=True)  # stride = 1 by default
	view = view[::stride, ::stride]
	
	# Mask - only apply pattern if center == 1
	middle = int(np.floor(kernel.shape[0] / 2))  # center of kernel
	mask = view[:, :, middle, middle] == 1
	
	# Apply kernel
	view[mask] = view[mask] | kernel
	
	# Cut off padding
	out = out[1:-1, 1:-1]
	
	return out



DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
				  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
def color_map(im: Union[nx.Graph, np.ndarray], g = None,
			  colors: Optional[Dict[int, str]] = None, strategy = 'largest_first',
			  grow = 100, pbar = None) -> np.ndarray:
	if colors is None:
		colors = DEFAULT_COLORS
	
	if g is None:
		g = extract_neighbors(im, grow=grow, pbar=pbar)
	
	coloring = greedy_color(g, strategy)
	
	cmap = np.stack([im]*3, -1).astype(np.uint8) * 0
	for idx, cidx in coloring.items():
		c = colors[cidx % len(colors)]
		if isinstance(c, str):
			c = hex_to_rgb(c)
		cmap[im==idx] = c
	
	return cmap
	

def index_map(rgb, lbls, locs=None, fontsize=3):#figsize=None, scale=1.):
	H, W, _ = rgb.shape
	# figsize, scale = None, 1.
	# if figsize is None:
	scale = 1.
	aw, ah = figaspect(H / W)
	aw, ah = scale * aw, scale * ah
	figsize = aw, ah
	
	fg = plt.figure(figsize=figsize, dpi=H/aw)
	
	plt.imshow(rgb)
	plt.axis('off')
	plt.subplots_adjust(0, 0, 1, 1)
	
	infos = regionprops(lbls)
	for idx, info in tqdm(enumerate(infos), total=len(infos)):
		y,x = info.centroid if locs is None else locs[idx+1]
		plt.text(x,y, str(idx+1), va='center', ha='center', fontsize=fontsize,
				 bbox=dict(
					 facecolor='1', ec=None, ls='-', lw=0,
					 # edgecolor='0',
					 alpha=0.6, pad=0)
				 )

	# data = np.frombuffer(fg.canvas.tostring_rgb(), dtype=np.uint8)
	# data = data.reshape(fg.canvas.get_width_height()[::-1] + (3,))
	# return data
	
	return fig_to_rgba(fg)
	

def label_tiles(rgb, border_color='#000000'):
	bcolor = hex_to_rgb(border_color) if isinstance(border_color, str) else border_color
	bcolor = np.array(bcolor).reshape(1, 1, -1)
	
	border = np.abs(rgb - bcolor).sum(-1) == 0
	lbls = label(fill_diagonals(border))[1::3, 1::3]
	
	return lbls


def generate_tiles(rgb, path=None, g=None, border_color='#000000', colors=None, strategy='largest_first',
				   pbar=None, grow=100):
	
	lbls = label_tiles(rgb, border_color=border_color)
	
	if path is not None:
		path = Path(path) #/ 'tiles.png'
		Image.fromarray(lbls.astype(np.int16)).save(path)
	
	return color_map(lbls, g=g, colors=colors, pbar=pbar, grow=grow, strategy=strategy)


def fig_to_rgba(fg):
	fg.canvas.draw()
	w, h = fg.canvas.get_width_height()
	buf = np.fromstring(fg.canvas.tostring_argb(), dtype=np.uint8)
	buf.shape = (h, w, 4)
	buf = np.roll(buf, 3, axis=2)
	return buf

	
class ArgumentError(Exception):
	def __init__(self, key, msg=None):
		if msg is None:
			msg = key
		else:
			msg = f'{key} - {msg}'
		super().__init__(msg)


@fig.script('tile-img')
def tile_img(A):
	plt.switch_backend('agg')
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {str(root)}')
	
	rgb_path = A.pull('rgb-path', None)
	if rgb_path is None:
		raise ArgumentError('rgb-path', 'Path to rgb image of blank map is required.')
	rgb_path = Path(rgb_path)
	if not rgb_path.exists():
		if root is not None and (root / rgb_path).exists():
			rgb_path = root / rgb_path
		else:
			raise ArgumentError('rgb-path', f'Path to rgb image invalid: {str(rgb_path)}')
	
	fontsize = A.pull('fontsize', 3)
	
	rgb = np.asarray(Image.open(rgb_path).convert('RGBA'))
	rgb = rgb[...,:3]
	
	border_color = A.pull('border-color', '#000000')

	lbls = label_tiles(rgb, border_color=border_color)
	
	tiles_path = root / A.pull('tiles-name', "tiles.png")
	Image.fromarray(lbls.astype(np.int16)).save(tiles_path)
	print(f'Saved tiles image to {str(tiles_path)}')
	
	ind_path = root / A.pull('ind-name', 'ind-tiles.png')
	ind_img = index_map(rgb, lbls, fontsize=fontsize)
	Image.fromarray(ind_img).save(ind_path)
	print(f'Saved tiles index image to {str(ind_path)}')
	
	viz_path = root / A.pull('vis-name', 'vis-tiles.png')
	viz = color_map(lbls, colors=None, pbar=tqdm, grow=100, strategy='largest_first')
	Image.fromarray(viz).save(viz_path)
	print(f'Saved tiles visualization to {str(viz_path)}')
	
	return lbls


def assign_dots(lbls, dots, cats=None):
	infos = regionprops(label(dots))
	coords = np.array([info.centroid for info in infos])#.astype(int)
	ys, xs = coords.astype(int).T
	picks = lbls[ys, xs]
	# print(picks)
	locs = [{'loc': [y,x], 'id': i+1, 'lbl': pick}
			for i, (pick, (y,x)) in enumerate(zip(picks.tolist(), coords.tolist()))]
	if cats is not None:
		for loc, cat in zip(locs, cats[ys, xs].tolist()):
			loc['cat'] = cat
	return locs

def auto_sea(options):
	options = list(options)
	dirs = np.array(options)
	affins = dirs #/ np.linalg.norm(dirs,2,-1).reshape(-1,1)
	op_id, typ_id = linear_sum_assignment(-affins)
	return dict(zip(map(tuple,dirs[op_id].tolist()),np.array(['bg', 'land', 'sea'])[typ_id].tolist()))


@fig.script('link-tiles')
def tiles_to_regions(A):
	plt.switch_backend('agg')
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {str(root)}')
	
	rgb_path = A.pull('rgb-path', None)
	if rgb_path is None:
		raise ArgumentError('rgb-path', 'Path to rgb image of blank map is required.')
	rgb_path = Path(rgb_path)
	if not rgb_path.exists():
		if root is not None and (root / rgb_path).exists():
			rgb_path = root / rgb_path
		else:
			raise ArgumentError('rgb-path', f'Path to rgb image invalid: {str(rgb_path)}')
	
	fontsize = A.pull('fontsize', 3)
	
	rgb = Image.open(rgb_path).convert('RGBA')
	rgb = np.asarray(rgb)[..., :3]
	
	dot_path = A.pull('dot-path', None)
	if dot_path is None:
		raise ArgumentError('dot-path', 'Path to rgb image of dots is required.')
	dot_path = Path(dot_path)
	if not dot_path.exists():
		if root is not None and (root / dot_path).exists():
			dot_path = root / dot_path
		else:
			raise ArgumentError('dot-path', f'Path to rgb image invalid: {str(dot_path)}')
	
	dots = Image.open(dot_path).convert('RGBA')
	dots = np.asarray(dots)#[..., :3]
	
	tile_path = A.pull('tile-path', 'tiles.png')
	if tile_path is not None:
		tile_path = Path(tile_path)
		if not tile_path.exists():
			if root is not None and (root / tile_path).exists():
				tile_path = root / tile_path
			else:
				raise ArgumentError('tile-path', f'Path to tile image invalid: {str(tile_path)}')
		
		# lbls = Image.open(tile_path)#.convert('RGBA')
		lbls = np.array(Image.open(tile_path))
	else:
		assert False, 'missing tiles.png'
		border_color = A.pull('border-color', '#000000')
		lbls = label_tiles(rgb, border_color=border_color)
	
	num_tiles = lbls.max()
	
	def extract_color(sel):
		c = Counter(map(tuple, sel.tolist()))
		c, _ = c.most_common(1)[0]
		return c
	
	dot_rgb = dots[..., :3]
	dot = dots[..., -1] > 0
	
	bases = assign_dots(lbls, dot, dot_rgb)

	allocs = Counter([base['lbl'] for base in bases])
	
	bad = []
	for tile, num in allocs.most_common():
		if num > 1:
			print(f'Map ERROR: tile {tile} was assigned to {num} dots (check the tiles and dots)')
			bad.append(tile)
			
	if A.pull('strict-alloc', True) and len(bad):
		print('Please fix these warnings before continuing.')
		return
	
	tile_colors = [extract_color(rgb[lbls == idx]) for idx in tqdm(range(1, num_tiles + 1),
	                                                               desc='Collecting tile colors')]
	
	for base in bases:
		# y, x = base['loc']
		base['color'] = tile_colors[base['lbl']-1]
		base['lbl'] = [base['lbl']]
	
	cats = {tuple(base['cat']) for base in bases}
	catnames = auto_sea(cats)
	catcolors = {cat: color for color, cat in catnames.items()}
	for base in bases:
		base['cat'] = catnames[tuple(base['cat'])]
	print(tabulate(catcolors.items(), headers=['Name', 'Color']))
	
	tile_info = regionprops(lbls)
	tile_coords = np.array([info.centroid for info in tile_info])
	
	done = {lbl for base in bases for lbl in base['lbl']}
	todo = np.array([x for x in range(1, num_tiles + 1) if x not in done])
	
	todo_coords = tile_coords[todo - 1]
	ys, xs = todo_coords.astype(int).T
	
	# todo_colors = list(map(tuple, rgb[ys, xs].tolist()))
	todo_colors = [tile_colors[i-1] for i in todo]
	
	color_bases = {}
	color_locs = {}
	for i, base in enumerate(bases):
		c = tuple(base['color'])
		if c not in color_bases:
			color_bases[c] = []
			color_locs[c] = []
		color_bases[c].append(base['id'])
		color_locs[c].append(base['loc'])
	
	color_bases = {c: np.array(ids) for c, ids in color_bases.items()}
	color_locs = {c: np.array(yx) for c, yx in color_locs.items()}
	
	bgs = []
	for tile_id, tile_color, tile_loc in zip(todo.tolist(), todo_colors, todo_coords):
		
		if tile_color in color_bases:
			
			base_options = color_bases[tile_color]
			base_locs = color_locs[tile_color]
			
			base_id = base_options[np.sum((base_locs - tile_loc.reshape(1, -1)) ** 2, -1).argmin()]
			bases[base_id - 1]['lbl'].append(tile_id)
		
		else:
			bgs.append(tile_id)
	
	bases.append({'id': len(bases) + 1, 'cat': 'bg', 'lbl': bgs, })
	
	assert len({base['id'] for base in bases}) == len(bases), 'overlapping region IDs'
	
	regions = {int(base['id']): {'type': str(base['cat']), 'id': int(base['id']), 'tiles': list(map(int,base['lbl']))}
	           for base in bases}
	save_yaml(regions, root/'regions.yaml')
	
	regimg = lbls * 0
	for base in tqdm(bases, desc='Assembling regions'):
		for tile_id in base['lbl']:
			regimg[lbls == tile_id] = base['id']
	
	regions_path = root / A.pull('regions-name', "regions.png")
	Image.fromarray(regimg.astype(np.int16)).save(regions_path)
	print(f'Saved regions image to {str(regions_path)}')
	
	catimg = rgb * 0
	for base in tqdm(bases, desc='Drawing region types'):
		catimg[regimg == base['id']] = catcolors.get(base['cat'], [128, 128, 128])
		# for tile_id in base['lbl']:
		# 	catimg[lbls == tile_id] = catcolors[base['cat']]
	# for bg in tqdm(bgs):
	# 	catimg[lbls == bg] = catcolors.get('bg', [255, 0, 0])
	
	H, W, _ = rgb.shape
	scale = 1
	aw, ah = figaspect(H / W)
	aw, ah = scale * aw, scale * ah
	figsize = aw, ah
	fg = plt.figure(figsize=figsize, dpi=H / aw)
	
	plt.imshow(catimg)
	
	# for base in tqdm(bases):
	for base in tqdm(bases):
		if 'loc' in base and base.get('cat') != 'bg':
			y, x = base['loc']
			for tile_id in base['lbl']:
				y2, x2 = tile_coords[tile_id - 1]
				dx, dy = x2 - x, y2 - y
				#     plt.annotate('', xytext=(x + dx, y + dy), xy=(x, y))
				plt.arrow(x, y, dx, dy, shape='full')
			plt.text(x, y, str(base['id']), va='center', ha='center', fontsize=3,
					 bbox=dict(
						 facecolor='1', ec=None, ls='-', lw=0,
						 # edgecolor='0',
						 alpha=0.6, pad=0)
					 )
	
	plt.axis('off')
	plt.subplots_adjust(0,0,1,1)
	
	viz_path = root / A.pull('vis-name', 'vis-regions.png')
	viz = fig_to_rgba(fg)
	Image.fromarray(viz).save(viz_path)
	print(f'Saved regions visualization to {str(viz_path)}')
	
	return regions
	


# def link_tiles(rgb, dots, lbls=None, catnames=None, color_threshold=0, border_color='#000000'):
# @fig.Script('link-tiles')
def old_tiles_to_regions(A):
	plt.switch_backend('agg')
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {str(root)}')
	
	rgb_path = A.pull('rgb-path', None)
	if rgb_path is None:
		raise ArgumentError('rgb-path', 'Path to rgb image of blank map is required.')
	rgb_path = Path(rgb_path)
	if not rgb_path.exists():
		if root is not None and (root/rgb_path).exists():
			rgb_path = root / rgb_path
		else:
			raise ArgumentError('rgb-path', f'Path to rgb image invalid: {str(rgb_path)}')
	
	fontsize = A.pull('fontsize', 3)
	
	rgb = Image.open(rgb_path).convert('RGBA')
	rgb = np.asarray(rgb)[..., :3]
	
	dot_path = A.pull('dot-path', None)
	if dot_path is None:
		raise ArgumentError('dot-path', 'Path to rgb image of dots is required.')
	dot_path = Path(dot_path)
	if not dot_path.exists():
		if root is not None and (root / dot_path).exists():
			dot_path = root / dot_path
		else:
			raise ArgumentError('dot-path', f'Path to rgb image invalid: {str(dot_path)}')
	
	dots = Image.open(dot_path).convert('RGBA')
	dots = np.asarray(dots)[..., :3]
	
	tile_path = A.pull('tile-path', 'tiles.png')
	if tile_path is not None:
		tile_path = Path(tile_path)
		if not tile_path.exists():
			if root is not None and (root / tile_path).exists():
				tile_path = root / tile_path
			else:
				raise ArgumentError('tile-path', f'Path to tile image invalid: {str(tile_path)}')

		# lbls = Image.open(tile_path)#.convert('RGBA')
		lbls = np.array(Image.open(tile_path))
	else:
		border_color = A.pull('border-color', '#000000')
		lbls = label_tiles(rgb, border_color=border_color)
	
	color_threshold = A.pull('color-threshold', 0)
	
	catnames = A.pulls('categories', 'cats', default=None)
	
	if catnames is None:
		ccs = set(map(tuple, dots.reshape(-1, 3 if dots.shape[-1] == 3 else 1).tolist()))
		ccs.discard((0, 0, 0))
		ccs.discard((0,))
		
		catnames = {c: f'cat{i}' for i, c in enumerate(ccs)}
	else:
		catnames = {tuple(hex_to_rgb(k)): v for k,v in catnames.items()}
	
	cathex = {process_color(k):v for k,v in catnames.items()}
	print(f'Using categories: {cathex}')
	
	dlbls = label(dots.astype(np.uint32).sum(-1))
	
	def extract_color(sel):
		c = Counter(map(tuple, sel.tolist()))
		c, _ = c.most_common(1)[0]
		return c
	def extract_id(sel):
		n = Counter(sel.tolist())
		n, _ = n.most_common(1)[0]
		return n
	
	dinfo = regionprops(dlbls)
	dlocs = []
	dcols = []
	dbases = []
	dot_cats = {}
	for idx, info in tqdm(enumerate(dinfo), total=len(dinfo), desc='Processing Dots'):
		sel = dlbls == idx+1
		dot_cats[idx+1] = catnames.get(extract_color(dots[sel]))
		dlocs.append(info.centroid)
		dcols.append(extract_color(rgb[sel]))
		dbases.append(extract_id(lbls[sel]))
	dlocs = np.array(dlocs)
	dcols = np.array(dcols)
	dids = np.arange(1,len(dlocs)+1)
	dbases = np.array(dbases) - 1 # one-based ID -> zero-based ID
	
	dot_inds = np.arange(1,len(dinfo)+1)
	tile_inds = dbases + 1
	
	refs = dict(zip(tile_inds.tolist(), dot_inds.tolist()))
	
	tinfo = regionprops(lbls)
	tlocs = np.array([info.centroid for info in tinfo])
	tcols = np.array([extract_color(rgb[lbls == idx + 1]) for idx, info in enumerate(tinfo)])
	
	todo = np.ones(len(tinfo)).astype(bool)
	todo[dbases] = 0
	
	inds = np.arange(len(tlocs))[todo] + 1
	locs = tlocs[todo]
	cols = tcols[todo]
	for idx, loc, c in tqdm(zip(inds.tolist(), locs, cols), total=len(inds), desc='Linking Tiles'):
		c = c.reshape(-1, 3)
		diffs = np.abs(dcols - c).sum(-1)
		sel = diffs <= color_threshold
		
		if sel.any():
			loc = loc.reshape(-1, 2)
			
			dot_options = dids[sel]
			dot_locs = dlocs[sel]
			
			dists = (np.abs(dot_locs - loc) ** 2).sum(-1)
			refs[idx] = dot_options[dists.argmin()].item()
		
	regs = {}
	for tile_idx, reg_idx in refs.items():
		if reg_idx not in regs:
			regs[reg_idx] = {'tiles': [], 'cat': dot_cats[reg_idx]}
		regs[reg_idx]['tiles'].append(tile_idx)
	
	save_yaml(regs, root/'regions.yaml')
	
	# plot outputs and save regions image
	
	reg_img = lbls * 0
	for idx, info in tqdm(regs.items(), total=len(regs), desc='Create Regions'):
		sel = sum(lbls == t for t in info['tiles']).astype(bool)
		reg_img[sel] = idx
	
	regions_path = root / A.pull('regions-name', "regions.png")
	Image.fromarray(reg_img.astype(np.int16)).save(regions_path)
	print(f'Saved regions image to {str(regions_path)}')
	
	ind_path = root / A.pull('ind-name', 'ind-regions.png')
	ind_img = index_map(rgb, reg_img, {i + 1: tuple(xy) for i, xy in enumerate(dlocs.tolist())},
						fontsize=fontsize)
	Image.fromarray(ind_img).save(ind_path)
	print(f'Saved regions index image to {str(ind_path)}')
	
	viz_path = root / A.pull('vis-name', 'vis-regions.png')
	viz = color_map(reg_img, colors=None, pbar=tqdm, grow=100, strategy='largest_first')
	Image.fromarray(viz).save(viz_path)
	print(f'Saved regions visualization to {str(viz_path)}')
	
	return regs


@fig.script('view-labels')
def view_labels(A):
	# plt.switch_backend('agg')
	plt.switch_backend(A.pull('matplotlib-backend', 'tkagg'))
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {str(root)}')
	
	rgb_path = A.pull('rgb-path', None)
	if rgb_path is None:
		raise ArgumentError('rgb-path', 'Path to rgb image of blank map is required.')
	rgb_path = Path(rgb_path)
	if not rgb_path.exists():
		if root is not None and (root / rgb_path).exists():
			rgb_path = root / rgb_path
		else:
			raise ArgumentError('rgb-path', f'Path to rgb image invalid: {str(rgb_path)}')
	
	rgb = Image.open(rgb_path).convert('RGBA')
	rgb = np.asarray(rgb)[..., :3]
	
	label_path = A.pull('label-path', None)
	if label_path is None:
		raise ArgumentError('label-path', 'Path to label image is required.')
	label_path = Path(label_path)
	if not label_path.exists():
		if root is not None and (root / label_path).exists():
			label_path = root / label_path
		else:
			raise ArgumentError('label-path', f'Path to label image invalid: {str(label_path)}')
	
	lbls = np.array(Image.open(label_path))
	
	infos = {}
	
	info_path = A.pull('info-path', 'regions.yaml' if 'regions' in str(label_path) else None)
	if info_path is not None:
		info_path = Path(info_path)
		if not info_path.exists():
			if root is not None and (root / info_path).exists():
				info_path = root / info_path
			# else:
			# 	raise ArgumentError('info-path', f'Info path is invalid: {str(info_path)}')
		
		if info_path.exists():
			infos = load_yaml(info_path)

	H, W, _ = rgb.shape
	if (H > 3000 or W > 3000) and A.pull('auto-scale', True):
		rgb = rgb[::2, ::2]
		lbls = lbls[::2, ::2]
	
	def highlight(rgb, mask, opacity=0.2):
		alpha = np.zeros_like(mask).astype(np.uint8)
		
		img = rgb.astype(np.uint8)
		
		alpha[mask != 0] = 255
		alpha[mask == 0] = int(255 * opacity)
		
		H, W, C = img.shape
		
		if C == 4:
			img[..., -1] = alpha
		else:
			img = np.concatenate([img, alpha.reshape(H, W, 1)], -1)
		return img
	
	opacity = A.pull('opacity', 0.2)

	scale = A.pull('img-scale', 1)
	aw, ah = figaspect(H / W)
	aw, ah = scale * aw, scale * ah
	figsize = aw, ah
	fg = plt.figure(figsize=figsize, )#dpi=H / aw)
	
	max_id = lbls.max()
	
	valid = set(lbls.reshape(-1).tolist())
	valid.discard(0)
	total = len(valid)
	
	ind = min(max(1,A.pull('start', 1)),max_id)
	
	def _draw_region():
		if ind in valid:
			plt.clf()
			
			info = infos.get(ind, {})
			
			name = info.get('name', 'LABEL')
			typ = ' ({})'.format(info['type']) if 'type' in info else ''
			
			ID = ind
			
			title = f'{name}{typ} {ID}/{max_id} (total: {total})'
			
			plt.title(title)
			plt.imshow(highlight(rgb, lbls == ind, opacity=opacity))
			plt.imshow(lbls % 1000, alpha=0., zorder=10)
		else:
			plt.title(f'ID {ind} not found')
		
		plt.xlabel('Press right arrow for next region, left arrow to go back (on the keyboard)')
		plt.xticks([])
		plt.yticks([])
		# plt.axis('off')
		plt.tight_layout()
		# plt.subplots_adjust(0, 0, 1, 1)
		plt.draw()
	
	def _onkey(event=None):
		nonlocal ind
		key = None if event is None else event.key
		if key == 'left':
			# print('Backward')
			ind = max(min(valid),ind-1)
			_draw_region()
		if key =='right' or key =='enter':
			# print('Forward')
			ind = min(max(valid),ind+1)
			_draw_region()
	
	fg.canvas.mpl_connect('key_press_event', _onkey)
	
	_draw_region()
	plt.show()
	
	print('Done')



@fig.script('link-names')
def link_names(A):
	plt.switch_backend('agg')
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {str(root)}')
	
	region_info_path = A.pull('region-info-path', 'regions.yaml')
	if region_info_path is None:
		raise ArgumentError('region-info-path', 'Path to regions.yaml is required.')
	region_info_path = Path(region_info_path)
	if not region_info_path.exists():
		if root is not None and (root / region_info_path).exists():
			region_info_path = root / region_info_path
		else:
			raise ArgumentError('region-path', f'Path to regions.yaml invalid: {str(region_info_path)}')

	regions = load_yaml(region_info_path)
	
	names = None
	name_path = A.pull('names-path', 'names.txt')
	if name_path is not None:
		name_path = Path(name_path)
		if not name_path.exists():
			if root is not None and (root / name_path).exists():
				name_path = root / name_path
			else:
				raise ArgumentError('name-path', f'Path to tile image invalid: {str(name_path)}')
		
		names = load_txt(name_path).split('\n')
	
	if len(regions) != len(names):
		print(f'WARNING: Found {len(regions)} regions, but {len(names)} '
			  f'names were provided (should be the same number).')
		
		if len(regions) > len(names):
			print('Not enough names.')
			raise Exception(f'not enough names provided: found {len(names)} (should be {len(regions)})')
		
		else:
			print(f'Will use only the first {len(regions)} names (ignoring the last {len(names)-len(regions)})')
			names = names[:len(regions)]
	
	for num, region in regions.items():
		region['name'] = names[num-1]
		
	save_yaml(regions, region_info_path)
	print(f'Saved regions with linked names to {str(region_info_path)}')
	return regions



@fig.script('extract-graph')
def extract_graph(A):
	plt.switch_backend('agg')
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {str(root)}')
	
	reg_img_path = A.pull('region-path', 'regions.png')
	if reg_img_path is None:
		raise ArgumentError('region-path', 'Path to region image is required.')
	reg_img_path = Path(reg_img_path)
	if not reg_img_path.exists():
		if root is not None and (root / reg_img_path).exists():
			reg_img_path = root / reg_img_path
		else:
			raise ArgumentError('region-path', f'Path to region image invalid: {str(reg_img_path)}')
	
	lbls = np.array(Image.open(reg_img_path))
	
	region_info_path = A.pull('region-info-path', 'regions.yaml')
	if region_info_path is None:
		raise ArgumentError('region-info-path', 'Path to regions.yaml is required.')
	region_info_path = Path(region_info_path)
	if not region_info_path.exists():
		if root is not None and (root / region_info_path).exists():
			region_info_path = root / region_info_path
		else:
			raise ArgumentError('region-info-path', f'Path to regions.yaml invalid: {str(region_info_path)}')

	regions = load_yaml(region_info_path)
	for ID, reg in regions.items():
		if 'name' not in reg:
			reg['name'] = f'region{str(ID).zfill(4)}'
	
	fixed = expand_labels(lbls, 10000)
	
	# regs = regionprops(fixed)
	
	ntx = {}
	
	for idx, node in tqdm(regions.items(), total=len(regions), desc='Generating neighbors'):
		if idx not in ntx:
			reg = regionprops(find_boundaries(fixed == idx, mode='outer').astype(int))
			if len(reg) == 0:
				print(f'WARNING: Region does not exist on the map: {idx} {node.get("name", "")}')
				ntx[idx] = []
			else:
				reg = reg[0]
				ords = coords_order(reg.coords)
				nss = [set(fixed[tuple(reg.coords[o].T)]) for o in ords]
				
				ans = set()
				nts = []
				for i, ns in enumerate(nss):
					ns = {n for n in ns if n not in ans and n in regions}
					ans.update(ns)
					if len(ns):
						nts.append(ns)
				ntx[idx] = nts
	
	ntx = {idx: {n for ns in nss for n in ns if regions[n].get('type') != 'bg'}
	       for idx, nss in ntx.items() if regions[idx].get('type') != 'bg'}
	
	for idx, region in regions.items():
		if region['type'] == 'land':
			if any(True for n in ntx[idx] if regions[n]['type'] == 'sea'):
				region['env'] = 'coast'
			else:
				region['env'] = 'land'
		elif region['type'] == 'sea':
			region['env'] = 'sea'
		else:
			if region['type'] != 'bg':
				print(f'WARNING: {region["name"]} is not a land, sea, or bg type region.')
			region['env'] = 'bg'
	
	# for idx, ns in ntx.items():
	# 	pass
	#
	# # TODO: include coastal regions in "fleet" edges, and vice versa
	#
	strict_neighbors = A.pull('strict-neighbors', True)
	
	neighbors = {}
	for idx, ns in ntx.items():
		neighbors[idx] = {}
		
		army = [n for n in ns if regions[n].get('env') in {'land', 'coast'}]
		fleet = [n for n in ns if regions[n].get('env') in {'sea', 'coast'}]
		
		if regions[idx].get('env') in {'land', 'coast'}:
			neighbors[idx]['army'] = army
		if regions[idx].get('env') in {'sea', 'coast'}:
			neighbors[idx]['fleet'] = fleet
		
		# land = [n for n in ns if regions[n].get('type') == 'land']
		# sea = [n for n in ns if regions[n].get('type') == 'sea']
		# if strict_neighbors:
		# 	assert len(land) + len(sea) == len(ns), regions[idx]['name']
		#
		# if len(land):
		# 	neighbors[idx]['army'] = land
		# if len(sea):
		# 	neighbors[idx]['fleet'] = sea
		#
		# if len(land) and regions[idx].get('env') == 'coast':
		# 	if 'fleet' not in neighbors[idx]:
		# 		neighbors[idx]['fleet'] = []
		# 	neighbors[idx]['fleet'].extend([n for n in land if regions[n].get('env') == 'coast'])
		# if regions[idx].get('env') == 'sea':
		# 	if 'fleet' not in neighbors[idx]:
		# 		neighbors[idx]['fleet'] = []
		# 	neighbors[idx]['fleet'].extend([n for n in ns if regions[n].get('env') == 'coast'])
	
	# ntx = {regions[idx]['name']: [regions[n]['name'] for n in ns if regions[n].get('type') != 'bg']
	#        for idx, ns in ntx.items()}
	
	graph = {regions[idx]['name']: {'edges':{etype:[regions[n]['name'] for n in ns]
	                                         for etype, ns in nns.items()}, **regions[idx]}
	         for idx, nns in neighbors.items()}
	
	for node in graph.values():
		if 'id' in node:
			node['ID'] = node['id']
			del node['id']
	
	graph_path = root / 'rough-graph.yaml'
	save_yaml(graph, graph_path)
	print(f'Saved rough graph to {str(graph_path)}')
	
	return graph
	
	edges = {}
	for idx in tqdm(regions, total=len(ntx), desc='Reformating neighbors'):
		node = regions[idx]
		ns = ntx[idx]
		es = {}
		
		seas = {n for n in ns if nodeIDs[n]['type'] == 'sea'}
		if node['type'] == 'sea':
			es['fleet'] = set(ns)
		else:
			es['army'] = set(ns - seas)
			if len(seas):
				node['type'] = 'coast'
				coasts = []
				while len(seas):
					sea = seas.pop()
					
					for coast in coasts:
						if coast.intersection(ntx[sea]):
							coast.add(sea)
							break
					else:
						coasts.append({sea})
				
				if len(coasts) > 1:
					joined = [coasts[0]]
					for coast in coasts[1:]:
						seeds = {n for s in coast for n in ntx[s] if nodeIDs[n]['type'] == 'sea'}
						for sel in joined:
							if seeds.intersection(sel):
								sel.update(coast)
								break
						else:
							joined.append(coast)
					coasts = joined
				
				lands = [{l for c in coast for l in ntx[c] if l in ns and nodeIDs[l]['type'] != 'sea'} for coast in
						 coasts]
				fleet = {i: {*f, *a} for i, (f, a) in enumerate(zip(coasts, lands))}
				es['fleet'] = fleet[0] if len(fleet) == 1 else fleet
		edges[idx] = es
	
	return ntx



@fig.script('include-location')
def include_coordinates(A):
	plt.switch_backend('agg')
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {str(root)}')
	
	reg_img_path = A.pull('reg-img-path', 'regions.png')
	if reg_img_path is not None:
		reg_img_path = Path(reg_img_path)
		if not reg_img_path.exists():
			if root is not None and (root / reg_img_path).exists():
				reg_img_path = root / reg_img_path
			else:
				raise ArgumentError('reg-img-path', f'Path to region image invalid: {str(reg_img_path)}')
	
	lbls = np.array(Image.open(reg_img_path))
	lbls = expand_labels(lbls, 10000)
	
	locs_path = A.pull('loc-path', None)
	if locs_path is not None:
		locs_path = Path(locs_path)
		if not locs_path.exists():
			if root is not None and (root / locs_path).exists():
				locs_path = root / locs_path
			else:
				raise ArgumentError('loc-path', f'Path to region image invalid: {str(locs_path)}')
	
	graph_path = A.pull('graph-path', 'graph.yaml')
	if graph_path is None:
		raise ArgumentError('graph-path', 'Path to graph.yaml is required.')
	graph_path = Path(graph_path)
	if not graph_path.exists():
		if root is not None and (root / graph_path).exists():
			graph_path = root / graph_path
		else:
			raise ArgumentError('graph-path', f'Path to graph invalid: {str(graph_path)}')

	out_path = A.pull('out-path', 'graph.yaml')
	
	reverse_coordinates = A.pull('reverse-coordinates', False)

	graph = load_yaml(graph_path)

	loc_name = A.pull('loc-name')
	overwrite = A.pull('replace', False)
	
	if overwrite:
		for node in graph.values():
			if 'locs' in node and loc_name in node['locs']:
				del node['locs'][loc_name]
	
	if locs_path is not None:
		dots = np.array(Image.open(locs_path).convert('RGBA')).astype(np.uint32)
		dots = dots[...,-1] > 0
		
		picks = assign_dots(lbls, dots)
		# {'loc': [y,x], 'id': i+1, 'lbl': pick}
		
		nodeIDs = {node['ID']: node for node in graph.values()}
		
		for pick in tqdm(picks, desc=f'Adding {loc_name} locations'):
			ID = pick['lbl']
			if ID in nodeIDs:
				node = nodeIDs[ID]
				if 'locs' not in node:
					node['locs'] = {}
				locs = node['locs']
				if loc_name not in locs:
					locs[loc_name] = []
				y, x = pick['loc']
				locs[loc_name].append([y,x] if reverse_coordinates else [x,y])
	
	count = Counter()
	
	for name, node in graph.items():
		if name not in count:
			count[name] = 0
		count[name] += len(node.get('locs', {}).get(loc_name, []))
	

	singles = [[name, num] for name, num in count.most_common() if num == 1]
	print(f'These {len(singles)} regions have a single assigned "{loc_name}" location:')
	print(', '.join(name for name, num in singles))
	
	multis = [[name, num] for name, num in count.most_common() if num > 1]
	print(f'These {len(multis)} regions are assigned more than one "{loc_name}" location:')
	print(', '.join(f'{name} ({num})' for name, num in multis))
	# for name, num in multis:
	# 	print(f'{num:>3} - {name}')
	
	missing = [[name, num] for name, num in count.most_common() if num == 0]
	print(f'These {len(missing)} regions do not have an assigned "{loc_name}" location:')
	print(', '.join(name for name, num in missing))
	# for name, num in missing:
	# 	print(name)
		
	out_path = root / out_path
	save_yaml(graph, out_path)
	print(f'Saved updated graph to {str(out_path)}')
	
	return graph


@fig.component('default-region-name-parser')
class Region_Name_Splitter(fig.Configurable):
	@staticmethod
	def split(name):
		terms = name.split('-')
		if len(terms) == 1:
			base = terms[0]
			coast = None
		else:
			base = '-'.join(terms[:-1])
			coast = terms[-1]
		return base, coast

	@staticmethod
	def join(name, coast=None):
		if coast is None:
			return name
		else:
			return f'{name}-{coast}'
		


@fig.script('check-graph', description='Checks the consistency of the graph.')
def include_coordinates(A):
	plt.switch_backend('agg')
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {root}')

	A.push('parser._type', 'default-region-name-parser', overwrite=False, silent=True)
	parser = A.pull('parser')
	
	graph_path = A.pull('graph-path', 'graph.yaml')
	if graph_path is None:
		raise ArgumentError('graph-path', 'Path to graph.yaml is required.')
	graph_path = Path(graph_path)
	if not graph_path.exists():
		if root is not None and (root / graph_path).exists():
			graph_path = root / graph_path
		else:
			raise ArgumentError('graph-path', f'Path to graph invalid: {graph_path}')
	
	auto_fix = A.pull('auto-fix', False)
	
	graph = load_yaml(graph_path)
	
	missing = []
	inconsistent = []
	
	def _check_edge(start, typ, end):
		try:
			ebase, ecoast = parser.split(end)
			if ebase not in graph:
				missing.append([start, typ, end])
				return
			
			options = graph[ebase]['edges'].get(typ, None)
			if options is None \
				or (isinstance(options, list) and start not in options)\
				or (isinstance(options, dict) and start not in options.get(ecoast, [])):
				inconsistent.append([start, typ, end])
				
				if auto_fix:
					if options is None:
						graph[ebase]['edges'][typ] = [] if ecoast is None else {ecoast: []}
					if ecoast is None:
						graph[ebase]['edges'][typ].append(start)
					else:
						graph[ebase]['edges'][typ][ecoast].append(start)
				return
		except:
			print(f'Error encountered: {start} ({typ}) -> {end}')
			raise
	
	for name, node in tqdm(graph.items(), desc='Checking edges'):
		for typ, eds in node.get('edges', {}).items():
			if isinstance(eds, list):
				for e in eds:
					_check_edge(name, typ, e)
			elif isinstance(eds, dict):
				for coast, es in eds.items():
					start = parser.join(name, coast)
					for e in es:
						_check_edge(start, typ, e)
			
	print(f'Found {len(missing)} edges to regions that don\'t exist')
	for start, typ, end in missing:
		print(f' - {end} in {start} ({typ})')
	
	print(f'Found {len(inconsistent)} inconsistent edges (A->B is missing, but B->A exists)')
	for start, typ, end in inconsistent:
		print(f' - {end} -> {start} ({typ})')
	
	if auto_fix and len(inconsistent):
		print('Fixing inconsistent edges (note that missing edges are not fixed)')
		save_yaml(graph, graph_path)
	
	return graph
	

@fig.script('set-supply-centers', description='Set supply centers in the graph.')
def include_coordinates(A):
	plt.switch_backend('agg')
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {str(root)}')
	
	reg_img_path = A.pull('reg-img-path', 'regions.png')
	if reg_img_path is not None:
		reg_img_path = Path(reg_img_path)
		if not reg_img_path.exists():
			if root is not None and (root / reg_img_path).exists():
				reg_img_path = root / reg_img_path
			else:
				raise ArgumentError('reg-img-path', f'Path to region image invalid: {str(reg_img_path)}')
	
	lbls = np.array(Image.open(reg_img_path))
	lbls = expand_labels(lbls, 10000)
	
	locs_path = A.pull('loc-path', None)
	if locs_path is None:
		raise ArgumentError('locs-path', 'Path to SC location image is required.')
	if locs_path is not None:
		locs_path = Path(locs_path)
		if not locs_path.exists():
			if root is not None and (root / locs_path).exists():
				locs_path = root / locs_path
			else:
				raise ArgumentError('loc-path', f'Path to SC location image invalid: {str(locs_path)}')
	
	graph_path = A.pull('graph-path', 'graph.yaml')
	if graph_path is None:
		raise ArgumentError('graph-path', 'Path to graph.yaml is required.')
	graph_path = Path(graph_path)
	if not graph_path.exists():
		if root is not None and (root / graph_path).exists():
			graph_path = root / graph_path
		else:
			raise ArgumentError('graph-path', f'Path to graph invalid: {str(graph_path)}')
	
	out_path = A.pull('out-path', 'graph.yaml')
	
	remove_existing = A.pull('remove-existing', False)
	sc_value = A.pull('sc-value', 1)
	
	graph = load_yaml(graph_path)
	
	if remove_existing:
		print('Removing existing supply centers...')
		for node in graph.values():
			if 'sc' in node:
				del node['sc']
	
	if locs_path is not None:
		dots = np.array(Image.open(locs_path).convert('RGBA')).astype(np.uint32)
		dots = dots[..., -1] > 0
		
		picks = assign_dots(lbls, dots)
		nodeIDs = {node['ID']: node for node in graph.values()}
		for pick in tqdm(picks, desc=f'Adding supply centers'):
			ID = pick['lbl']
			if ID in nodeIDs:
				node = nodeIDs[ID]
				node['sc'] = sc_value
				
	singles = [name for name, node in graph.items() if 'sc' in node]
	print(f'These {len(singles)} regions are set as supply centers with value {sc_value}:')
	print(', '.join(name for name in singles))
	
	out_path = root / out_path
	save_yaml(graph, out_path)
	print(f'Saved updated graph to {out_path}')
	
	return graph





