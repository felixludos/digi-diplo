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
	
	# tile_colors = [extract_color(rgb[lbls == idx]) for idx in tqdm(range(1, num_tiles + 1),
	# 																desc='Collecting tile colors')]
	tile_colors = [(0, 62, 123), (79, 79, 79), (46, 44, 183), (79, 79, 79), (0, 62, 123), (0, 62, 123), (0, 62, 123),
	               (0, 62, 123), (79, 79, 79), (0, 62, 123), (0, 62, 123), (0, 62, 123), (214, 203, 158), (46, 44, 183),
	               (214, 203, 158), (214, 203, 158), (0, 62, 123), (0, 62, 123), (254, 48, 48), (215, 255, 214),
	               (214, 203, 158), (0, 62, 123), (0, 62, 123), (214, 203, 158), (214, 203, 158), (0, 62, 123),
	               (0, 62, 123), (0, 62, 123), (0, 62, 123), (0, 62, 123), (215, 255, 214), (214, 203, 158),
	               (0, 62, 123), (0, 62, 123), (214, 203, 158), (214, 203, 158), (214, 203, 158), (214, 203, 158),
	               (239, 239, 239), (0, 62, 123), (239, 239, 239), (239, 239, 239), (0, 62, 123), (0, 62, 123),
	               (214, 203, 158), (214, 203, 158), (0, 62, 123), (0, 62, 123), (0, 62, 123), (215, 255, 214),
	               (239, 239, 239), (239, 239, 239), (0, 62, 123), (214, 203, 158), (214, 203, 158), (254, 48, 48),
	               (215, 255, 214), (214, 203, 158), (0, 62, 123), (214, 203, 158), (0, 73, 142), (181, 41, 50),
	               (0, 62, 123), (0, 62, 123), (0, 62, 123), (0, 62, 123), (0, 62, 123), (181, 41, 50), (0, 50, 96),
	               (214, 203, 158), (239, 239, 239), (239, 239, 239), (214, 203, 158), (239, 239, 239), (0, 62, 123),
	               (46, 44, 183), (0, 62, 123), (0, 50, 96), (239, 239, 239), (46, 44, 183), (46, 44, 183),
	               (239, 239, 239), (254, 48, 48), (0, 62, 123), (214, 203, 158), (215, 255, 214), (0, 62, 123),
	               (46, 44, 183), (46, 44, 183), (181, 41, 50), (0, 62, 123), (214, 203, 158), (214, 203, 158),
	               (193, 46, 117), (0, 62, 123), (239, 239, 239), (0, 62, 123), (181, 41, 50), (0, 50, 96),
	               (214, 203, 158), (254, 48, 48), (181, 41, 50), (254, 197, 190), (0, 62, 123), (0, 62, 123),
	               (0, 50, 96), (0, 62, 123), (254, 48, 48), (254, 48, 48), (239, 239, 239), (239, 239, 239),
	               (0, 46, 140), (0, 73, 142), (0, 46, 140), (214, 203, 158), (0, 46, 140), (239, 239, 239),
	               (214, 203, 158), (214, 203, 158), (0, 62, 123), (181, 41, 50), (214, 203, 158), (158, 158, 158),
	               (0, 62, 123), (158, 158, 158), (214, 203, 158), (0, 62, 123), (0, 62, 123), (0, 62, 123),
	               (158, 158, 158), (0, 62, 123), (254, 48, 48), (158, 158, 158), (254, 197, 190), (0, 62, 123),
	               (214, 203, 158), (181, 41, 50), (214, 203, 158), (158, 158, 158), (0, 62, 123), (226, 174, 111),
	               (0, 62, 123), (158, 158, 158), (254, 48, 48), (214, 203, 158), (46, 44, 183), (0, 73, 142),
	               (158, 158, 158), (0, 62, 123), (46, 44, 183), (46, 44, 183), (46, 44, 183), (0, 62, 123),
	               (226, 174, 111), (0, 62, 123), (254, 197, 190), (0, 62, 123), (254, 48, 48), (0, 62, 123),
	               (0, 62, 123), (254, 197, 190), (158, 158, 158), (214, 203, 158), (0, 62, 123), (214, 203, 158),
	               (0, 62, 123), (0, 62, 123), (214, 203, 158), (0, 62, 123), (214, 203, 158), (0, 62, 123),
	               (214, 203, 158), (214, 203, 158), (158, 158, 158), (214, 203, 158), (0, 62, 123), (0, 62, 123),
	               (214, 203, 158), (46, 44, 183), (214, 203, 158), (0, 62, 123), (17, 97, 191), (214, 203, 158),
	               (214, 203, 158), (0, 62, 123), (214, 203, 158), (181, 41, 50), (214, 203, 158), (214, 203, 158),
	               (17, 97, 191), (255, 137, 237), (17, 97, 191), (214, 203, 158), (214, 203, 158), (214, 203, 158),
	               (214, 203, 158), (214, 203, 158), (17, 97, 191), (17, 97, 191), (214, 203, 158), (214, 203, 158),
	               (214, 203, 158), (239, 239, 239), (205, 165, 50), (181, 41, 50), (205, 165, 50), (46, 44, 183),
	               (214, 203, 158), (0, 62, 123), (239, 239, 239), (46, 44, 183), (46, 44, 183), (17, 97, 191),
	               (46, 44, 183), (214, 203, 158), (2, 132, 0), (205, 165, 50), (0, 62, 123), (214, 203, 158),
	               (214, 203, 158), (205, 165, 50), (17, 97, 191), (2, 132, 0), (0, 62, 123), (239, 239, 239),
	               (0, 62, 123), (214, 203, 158), (0, 62, 123), (2, 132, 0), (214, 203, 158), (0, 62, 123),
	               (0, 62, 123), (2, 132, 0), (17, 97, 191), (0, 62, 123), (214, 203, 158), (205, 165, 50),
	               (214, 203, 158), (239, 239, 239), (181, 41, 50), (0, 62, 123), (239, 239, 239), (181, 41, 50),
	               (0, 46, 140), (0, 62, 123), (17, 97, 191), (214, 203, 158), (0, 62, 123), (214, 203, 158),
	               (205, 165, 50), (0, 62, 123), (2, 132, 0), (2, 132, 0), (0, 62, 123), (226, 174, 0), (0, 62, 123),
	               (214, 203, 158), (214, 203, 158), (0, 62, 123), (0, 62, 123), (0, 62, 123), (0, 62, 123),
	               (214, 203, 158), (214, 203, 158), (214, 203, 158), (0, 46, 140), (214, 203, 158), (214, 203, 158),
	               (46, 44, 183), (17, 97, 191), (0, 62, 123), (2, 132, 0), (255, 137, 237), (214, 203, 158),
	               (214, 203, 158), (214, 203, 158), (214, 203, 158), (0, 62, 123), (0, 62, 123), (0, 62, 123),
	               (0, 62, 123), (181, 41, 50), (0, 62, 123), (254, 48, 48), (67, 178, 193), (214, 203, 158),
	               (0, 62, 123), (188, 73, 73), (0, 62, 123), (0, 46, 140), (226, 174, 0), (214, 203, 158),
	               (214, 203, 158), (214, 203, 158), (0, 62, 123), (0, 62, 123), (0, 62, 123), (214, 203, 158),
	               (214, 203, 158), (67, 178, 193), (2, 132, 0), (255, 137, 237), (0, 46, 140), (0, 46, 140),
	               (0, 62, 123), (2, 132, 0), (0, 62, 123), (0, 62, 123), (0, 46, 140), (0, 62, 123), (0, 62, 123),
	               (67, 178, 193), (0, 62, 123), (214, 203, 158), (214, 203, 158), (0, 46, 140), (0, 50, 96),
	               (214, 203, 158), (206, 182, 0), (188, 73, 73), (214, 203, 158), (0, 62, 123), (0, 62, 123),
	               (214, 203, 158), (0, 62, 123), (0, 62, 123), (214, 203, 158), (0, 62, 123), (0, 46, 140),
	               (162, 188, 122), (0, 62, 123), (0, 62, 123), (67, 178, 193), (0, 62, 123), (214, 203, 158),
	               (226, 174, 0), (255, 137, 237), (0, 62, 123), (214, 203, 158), (67, 178, 193), (0, 62, 123),
	               (0, 153, 84), (214, 203, 158), (188, 73, 73), (255, 137, 237), (0, 62, 123), (0, 62, 123),
	               (0, 62, 123), (0, 62, 123), (0, 62, 123), (162, 188, 122), (214, 203, 158), (0, 62, 123),
	               (206, 182, 0), (0, 62, 123), (2, 132, 0), (0, 62, 123), (214, 203, 158), (0, 62, 123),
	               (214, 203, 158), (0, 46, 140), (214, 203, 158), (214, 203, 158), (0, 50, 96), (162, 188, 122),
	               (0, 62, 123), (214, 203, 158), (0, 62, 123), (0, 62, 123), (0, 62, 123), (214, 203, 158),
	               (214, 203, 158), (17, 97, 191), (17, 97, 191), (0, 50, 96), (255, 137, 237), (0, 62, 123),
	               (0, 46, 140), (0, 46, 140), (214, 203, 158), (254, 48, 48), (0, 46, 140), (226, 174, 0),
	               (206, 182, 0), (0, 46, 140), (0, 62, 123), (0, 46, 140), (214, 203, 158), (0, 50, 96), (0, 62, 123),
	               (0, 46, 140), (0, 50, 96), (0, 50, 96), (17, 97, 191), (17, 97, 191), (17, 97, 191), (255, 137, 237),
	               (214, 203, 158), (254, 48, 48), (162, 188, 122), (67, 178, 193), (226, 174, 0), (255, 137, 237),
	               (0, 50, 96), (0, 50, 96), (0, 62, 123), (0, 62, 123), (162, 188, 122), (214, 203, 158),
	               (17, 97, 191), (0, 62, 123), (214, 203, 158), (0, 62, 123), (214, 203, 158), (226, 174, 0),
	               (0, 46, 140), (0, 62, 123), (214, 203, 158), (0, 50, 96), (0, 62, 123), (0, 62, 123),
	               (214, 203, 158), (206, 182, 0), (0, 62, 123), (214, 203, 158), (2, 132, 0), (214, 203, 158),
	               (2, 132, 0), (214, 203, 158), (206, 182, 0), (0, 46, 140), (214, 203, 158), (206, 182, 0),
	               (65, 160, 160), (0, 46, 140), (214, 203, 158), (0, 50, 96), (254, 48, 48), (0, 62, 123),
	               (254, 48, 48), (214, 203, 158), (0, 62, 123), (0, 62, 123), (254, 48, 48), (162, 188, 122),
	               (0, 62, 123), (226, 174, 0), (0, 62, 123), (2, 132, 0), (206, 182, 0), (214, 203, 158),
	               (142, 54, 133), (254, 48, 48), (214, 203, 158), (254, 48, 48), (0, 62, 123), (214, 203, 158),
	               (2, 132, 0), (0, 62, 123), (0, 62, 123), (0, 62, 123), (214, 203, 158), (0, 62, 123),
	               (214, 203, 158), (0, 62, 123), (0, 62, 123), (0, 62, 123), (214, 203, 158), (65, 160, 160),
	               (142, 54, 133), (0, 62, 123), (142, 54, 133), (214, 203, 158), (206, 182, 0), (214, 203, 158),
	               (0, 50, 96), (214, 203, 158), (0, 46, 140), (214, 203, 158), (142, 54, 133), (0, 62, 123),
	               (0, 62, 123), (0, 46, 140), (255, 137, 237), (0, 62, 123), (0, 62, 123), (0, 46, 140), (0, 62, 123),
	               (0, 50, 96), (0, 62, 123), (214, 203, 158), (0, 62, 123), (214, 203, 158), (214, 203, 158),
	               (214, 203, 158), (0, 46, 140), (65, 160, 160), (0, 50, 96), (214, 203, 158), (254, 48, 48),
	               (214, 203, 158), (142, 54, 133), (214, 203, 158), (255, 137, 237), (0, 46, 140), (0, 62, 123),
	               (0, 62, 123), (142, 54, 133), (214, 203, 158), (0, 62, 123), (214, 203, 158), (0, 62, 123),
	               (214, 203, 158), (214, 203, 158), (254, 48, 48), (68, 163, 144), (214, 203, 158), (65, 160, 160),
	               (0, 62, 123), (214, 203, 158), (0, 50, 96), (0, 46, 140), (0, 62, 123), (214, 203, 158),
	               (214, 203, 158), (214, 203, 158), (0, 62, 123), (214, 203, 158), (65, 160, 160), (0, 62, 123),
	               (214, 203, 158), (214, 203, 158), (214, 203, 158), (214, 203, 158), (214, 203, 158), (214, 203, 158),
	               (0, 50, 96), (254, 48, 48), (0, 62, 123), (0, 62, 123), (0, 62, 123), (214, 203, 158),
	               (226, 174, 111), (0, 62, 123), (0, 62, 123), (65, 160, 160), (214, 203, 158), (214, 203, 158),
	               (0, 62, 123), (0, 62, 123), (17, 97, 191), (214, 203, 158), (0, 62, 123), (0, 50, 96), (0, 62, 123),
	               (0, 62, 123), (214, 203, 158), (74, 191, 78), (214, 203, 158), (0, 46, 140), (0, 62, 123),
	               (17, 97, 191), (214, 203, 158), (0, 46, 140), (2, 132, 0), (214, 203, 158), (74, 191, 78),
	               (214, 203, 158), (214, 203, 158), (0, 62, 123), (188, 73, 73), (0, 62, 123), (214, 203, 158),
	               (214, 203, 158), (0, 62, 123), (0, 62, 123), (0, 50, 96), (214, 203, 158), (214, 203, 158),
	               (0, 50, 96), (0, 62, 123), (214, 203, 158), (0, 62, 123), (0, 62, 123), (214, 203, 158),
	               (214, 203, 158), (0, 62, 123), (0, 62, 123), (214, 203, 158), (17, 97, 191), (0, 73, 142),
	               (214, 203, 158), (0, 62, 123), (0, 46, 140), (0, 50, 96), (214, 203, 158), (254, 48, 48),
	               (0, 46, 140), (74, 191, 78), (214, 203, 158), (198, 111, 94), (188, 73, 73), (0, 62, 123),
	               (17, 97, 191), (0, 62, 123), (0, 62, 123), (214, 203, 158), (0, 46, 140), (0, 46, 140), (0, 62, 123),
	               (46, 44, 183), (0, 46, 140), (0, 73, 142), (0, 46, 140), (0, 62, 123), (74, 191, 78),
	               (214, 203, 158), (74, 191, 78), (2, 132, 0), (0, 50, 96), (0, 46, 140), (17, 97, 191), (0, 46, 140),
	               (0, 50, 96), (214, 203, 158), (74, 191, 78), (0, 50, 96), (0, 46, 140), (198, 111, 94),
	               (214, 203, 158), (214, 203, 158), (0, 50, 96), (0, 62, 123), (0, 46, 140), (46, 44, 183),
	               (0, 46, 140), (214, 203, 158), (0, 46, 140), (17, 97, 191), (46, 44, 183), (74, 191, 78),
	               (0, 46, 140), (0, 46, 140), (214, 203, 158), (0, 62, 123), (214, 203, 158), (0, 62, 123),
	               (0, 62, 123), (198, 111, 94), (214, 203, 158), (0, 62, 123), (0, 62, 123), (214, 203, 158),
	               (254, 48, 48), (0, 62, 123), (214, 203, 158), (0, 62, 123), (214, 203, 158), (2, 132, 0),
	               (214, 203, 158), (0, 62, 123), (0, 62, 123), (0, 62, 123), (0, 62, 123), (0, 73, 142),
	               (226, 174, 111), (214, 203, 158), (0, 62, 123), (17, 97, 191), (0, 62, 123), (214, 203, 158),
	               (226, 174, 111), (214, 203, 158), (0, 46, 140), (0, 62, 123), (0, 46, 140), (198, 111, 94),
	               (0, 62, 123), (226, 174, 111), (74, 191, 78), (0, 62, 123), (214, 203, 158), (0, 62, 123),
	               (0, 62, 123), (74, 191, 78), (0, 62, 123), (214, 203, 158), (214, 203, 158), (158, 158, 158),
	               (0, 62, 123), (0, 62, 123), (0, 62, 123), (0, 62, 123), (214, 203, 158), (0, 62, 123), (0, 62, 123),
	               (214, 203, 158), (214, 203, 158), (0, 46, 140), (0, 46, 140), (0, 62, 123), (0, 46, 140),
	               (214, 203, 158), (46, 44, 183), (0, 46, 140), (214, 203, 158), (0, 62, 123), (0, 62, 123),
	               (0, 62, 123), (151, 232, 237), (0, 62, 123), (226, 174, 111), (188, 73, 73), (158, 158, 158),
	               (0, 46, 140), (0, 62, 123), (214, 203, 158), (226, 174, 111), (0, 62, 123), (151, 232, 237),
	               (92, 237, 66), (74, 191, 78), (0, 62, 123), (214, 203, 158), (0, 62, 123), (188, 73, 73),
	               (0, 62, 123), (0, 62, 123), (188, 73, 73), (254, 48, 48), (79, 79, 79), (0, 46, 140), (17, 97, 191),
	               (151, 232, 237), (214, 203, 158), (0, 46, 140), (74, 191, 78), (188, 73, 73), (214, 203, 158),
	               (214, 203, 158), (214, 203, 158), (0, 62, 123), (0, 62, 123), (151, 232, 237), (214, 203, 158),
	               (214, 203, 158), (151, 232, 237), (214, 203, 158), (0, 46, 140), (0, 62, 123), (0, 62, 123),
	               (17, 97, 191), (214, 203, 158), (79, 79, 79), (151, 232, 237), (0, 62, 123), (254, 48, 48),
	               (0, 62, 123), (0, 62, 123), (254, 48, 48), (254, 48, 48), (0, 62, 123), (0, 62, 123), (0, 62, 123),
	               (0, 62, 123), (0, 62, 123), (151, 232, 237), (254, 48, 48), (0, 46, 140), (0, 46, 140), (0, 46, 140),
	               (214, 203, 158), (254, 48, 48), (193, 46, 117), (214, 203, 158), (0, 62, 123), (0, 62, 123),
	               (0, 62, 123), (0, 62, 123), (0, 46, 140), (214, 203, 158), (0, 62, 123), (0, 46, 140), (0, 62, 123),
	               (0, 62, 123), (0, 62, 123), (0, 46, 140), (254, 48, 48), (0, 46, 140), (0, 62, 123)]
	
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
	
	for base in bases:
		# y, x = base['loc']
		base['color'] = tile_colors[base['lbl']-1]
		base['lbl'] = [base['lbl']]
	
	cats = {tuple(base['cat']) for base in bases}
	catnames = auto_sea(cats)
	catcolors = {cat: color for color, cat in catnames.items()}
	for base in bases:
		base['cat'] = catnames[tuple(base['cat'])]
	
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
		catimg[regimg == base['id']] = catcolors[base['cat']]
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


@fig.Script('view-regions')
def view_regions(A):
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

	H, W, _ = rgb.shape
	scale = A.pull('img-scale', 1)
	aw, ah = figaspect(H / W)
	aw, ah = scale * aw, scale * ah
	figsize = aw, ah
	fg = plt.figure(figsize=figsize, )#dpi=H / aw)
	
	max_id = max(reg['id'] for reg in regions.values())
	
	ind = min(max(1,A.pull('start', 1)),max_id)
	
	def _draw_region():
		if ind in regions:
			current = regions[ind]
			plt.clf()
			plt.title('Region {id}/{total} ({type})'.format(total=len(regions), **current))
			plt.imshow(highlight(rgb, lbls == ind, opacity=opacity))
		else:
			plt.title(f'Region {ind} not found')
		
		plt.xlabel('Press right arrow for next region, left arrow to go back (on the keyboard)')
		plt.xticks([])
		plt.yticks([])
		# plt.axis('off')
		plt.tight_layout()
		# plt.subplots_adjust(0, 0, 1, 1)
	
	def _onkey(event=None):
		nonlocal ind
		key = None if event is None else event.key
		if key == 'left':
			print('Backward')
			ind = max(1,ind-1)
			_draw_region()
		if key =='right' or key =='enter':
			print('Forward')
			ind = min(max_id,ind+1)
			_draw_region()
	
	fg.canvas.mpl_connect('key_press_event', _onkey)
	
	_draw_region()
	plt.show()
	
	print('Done')



@fig.Script('link-names')
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
			raise Exception(f'not enough names provided: found {len(names)} (should be {len(regions)}')
		
		else:
			print(f'Will use only the first {len(regions)} names (ignoring the last {len(names)-len(regions)})')
			names = names[:len(regions)]
	
	for num, region in regions.items():
		region['name'] = names[num-1]
		
	save_yaml(regions, region_info_path)
	print(f'Saved regions with linked names to {str(region_info_path)}')
	return regions



@fig.Script('extract-graph')
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
	
	regs = regionprops(fixed)
	
	neighbors = {}
	
	for idx, node in tqdm(regions.items(), total=len(regions), desc='Generating neighbors'):
		if idx not in neighbors:
			reg = regionprops(find_boundaries(fixed == idx, mode='outer').astype(int))[0]
			ords = coords_order(reg.coords)
			nss = [set(fixed[tuple(reg.coords[o].T)]) for o in ords]
			
			ans = set()
			nts = []
			for i, ns in enumerate(nss):
				ns = {n for n in ns if n not in ans and n in regions}
				ans.update(ns)
				if len(ns):
					nts.append(ns)
			neighbors[idx] = nts
	
	ntx = {idx: {n for ns in nss for n in ns if regions[n].get('type') != 'bg'}
	       for idx, nss in neighbors.items() if regions[idx].get('type') != 'bg'}
	
	strict_neighbors = A.pull('strict-neighbors', True)
	
	neighbors = {}
	for idx, ns in ntx.items():
		neighbors[idx] = {}
		land = [n for n in ns if regions[n].get('type') == 'land']
		sea = [n for n in ns if regions[n].get('type') == 'sea']
		if strict_neighbors:
			assert len(land) + len(sea) == len(ns), regions[idx]['name']
		if len(land):
			neighbors[idx]['land'] = land
		if len(sea):
			neighbors[idx]['sea'] = sea
	
	# ntx = {regions[idx]['name']: [regions[n]['name'] for n in ns if regions[n].get('type') != 'bg']
	#        for idx, ns in ntx.items()}
	
	graph = {regions[idx]['name']: {'edges':{etype:{regions[n]['name'] for n in ns}
	                                         for etype, ns in nns.items()}, **regions[idx]}
	         for idx, nns in neighbors.items()}
	
	graph_path = root / 'rough-graph.yaml'
	print(f'Saved rough graph to {str(graph_path)}')
	save_yaml(graph, graph_path)
	
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



@fig.Script('add-coordinates')
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
	
	locs_path = A.pull('locs-path', )
	if locs_path is not None:
		locs_path = Path(locs_path)
		if not locs_path.exists():
			if root is not None and (root / locs_path).exists():
				locs_path = root / locs_path
			else:
				raise ArgumentError('locs-path', f'Path to region image invalid: {str(locs_path)}')
	
	locs = np.array(Image.open(locs_path).convert('RGBA')).astype(np.uint32)
	locs = locs[...,-1] > 0
	
	loc_lbls = label(locs)
	loc_info = regionprops(loc_lbls)
	
	graph_path = A.pull('graph-path', None)
	if graph_path is None:
		raise ArgumentError('graph-path', 'Path to graph.yaml is required.')
	graph_path = Path(graph_path)
	if not graph_path.exists():
		if root is not None and (root / graph_path).exists():
			graph_path = root / graph_path
		else:
			raise ArgumentError('graph-path', f'Path to graph invalid: {str(graph_path)}')

	graph = load_yaml(graph_path)
	
	
	













