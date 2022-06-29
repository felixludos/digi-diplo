from typing import Dict, Optional, Union
import sys, os
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
			         facecolor='1', ec=None, ls='',
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


@fig.Script('tile-img')
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

	viz_path = root / A.pull('vis-name', 'vis-tiles.png')
	viz = color_map(lbls, colors=None, pbar=tqdm, grow=100, strategy='largest_first')
	Image.fromarray(viz).save(viz_path)
	print(f'Saved tiles visualization to {str(viz_path)}')
	
	ind_path = root / A.pull('ind-name', 'ind-tiles.png')
	ind_img = index_map(rgb, lbls, fontsize=fontsize)
	Image.fromarray(ind_img).save(ind_path)
	print(f'Saved tiles index image to {str(ind_path)}')
	
	return lbls



# def link_tiles(rgb, dots, lbls=None, catnames=None, color_threshold=0, border_color='#000000'):
@fig.Script('link-tiles')
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
	
	catnames = A.pull('categories', '<>cats', None)
	
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

	viz_path = root / A.pull('vis-name', 'vis-regions.png')
	viz = color_map(reg_img, colors=None, pbar=tqdm, grow=100, strategy='largest_first')
	Image.fromarray(viz).save(viz_path)
	print(f'Saved regions visualization to {str(viz_path)}')
	
	ind_path = root / A.pull('ind-name', 'ind-regions.png')
	ind_img = index_map(rgb, reg_img, {i+1: tuple(xy) for i, xy in enumerate(dlocs.tolist())},
	                    fontsize=fontsize)
	Image.fromarray(ind_img).save(ind_path)
	print(f'Saved regions index image to {str(ind_path)}')
	
	return regs



@fig.Script('link-names')
def link_names(A):
	plt.switch_backend('agg')
	
	root = A.pull('root', '.')
	if root is None:
		raise ArgumentError('root', 'Must not be None.')
	root = Path(root)
	print(f'Will save output to {str(root)}')
	
	region_path = A.pull('region-path', None)
	if region_path is None:
		raise ArgumentError('region-path', 'Path to rgb image of blank map is required.')
	region_path = Path(region_path)
	if not region_path.exists():
		if root is not None and (root / region_path).exists():
			region_path = root / region_path
		else:
			raise ArgumentError('region-path', f'Path to rgb image invalid: {str(region_path)}')

	regions = load_yaml(region_path)
	
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
			raise Exception(f'not enough names provided: found {len(names)} (should be {len(regions)}')
		
		else:
			print(f'Will use only the first {len(regions)} names (ignoring the last {len(names)-len(regions)})')
			names = names[:len(regions)]
	
	for num, region in regions.items():
		region['name'] = names[num-1]
		
	save_yaml(regions, region_path)
	print(f'Saved regions with linked names to {str(region_path)}')
	return regions



# @fig.Script('extract-graph')
# def extract_graph(A):
#
#
#
# 	pass

