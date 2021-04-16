import sys, os
from itertools import chain
from omnibelt import load_yaml, save_yaml, create_dir
from tqdm import tqdm
from copy import deepcopy
import omnifig as fig

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import ArrowStyle
from matplotlib.figure import figaspect
import matplotlib.patheffects as path_effects

import numpy as np
from scipy.spatial import distance_matrix
from PIL import ImageColor, Image

from src.colors import lighter, dimmer, fill_region

from .automap import get_borders_from_expanded
from ..colors import hex_to_rgb, process_color

import pydip

_SEASONS = ['', 'Spring', 'Autumn', 'Winter']


# @fig.Script('render', description='Render a Diplomacy state')
def render_diplo_state(A):
	save_path = A.pull('render-path', '<>save-path', None)
	view = A.pull('view', save_path is None)
	
	mlp_backend = A.pull('mlp-backend', 'qt5agg' if view else 'agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	A.push('map._type', 'map', overwrite=False)
	M = A.pull('map')
	
	image_path = A.pull('image-path')
	
	state = A.pull('state', None, silent=True)
	if state is None:
		state_path = A.pull('state-path')
		if not os.path.isfile(state_path):
			raise Exception(f'No state file: {state_path}')
		
		state = load_yaml(state_path)
	
	M.load_players(state['players'])
	
	# action edges
	
	action = A.pull('action', None, silent=True)
	if action is None:
		action_path = A.pull('action-path', None)
		if action_path is not None:
			action = load_yaml(action_path)
	
	# prev_state = A.pull('prev-state', None, silent=True)
	# if prev_state is None:
	# 	prev_state_path = A.pull('prev-state-path', None)
	# 	if prev_state_path is not None:
	# 		prev_state = load_yaml(prev_state_path)
	
	# control
	
	colors = A.pull('colors', {})
	threshold = A.pull('threshold', 0.1)
	
	# color_full = A.pull('color-full', False)
	color_by_unit = A.pull('color-by-unit', True)
	
	control = {}
	
	redraw_all = A.pull('full-redraw', True)
	if redraw_all:
		
		for loc, node in M.nodes.items():
			control[loc] = 'sea' if node['type'] == 'sea' else 'neutral'
			pass
	
	units = {}
	
	for player, info in state['players'].items():
		for loc in info['control']:
			control[M.uncoastify(loc, True)] = player
		for unit in info['units']:
			ID = M.uncoastify(unit['loc'], True)
			if M.nodes[ID]['type'] != 'sea':
				units[ID] = player
	
	scale = A.pull('scale', 1)
	
	if not os.path.isfile(image_path):
		print(f'No image found: {image_path}')
		return None
	
	img = np.array(Image.open(image_path).convert("RGB"))
	
	H, W, _ = img.shape
	
	w, h = figaspect(H / W)
	w, h = scale * w, scale * h
	
	figax = plt.subplots(figsize=(w, h))
	
	tiles = control.copy()
	if color_by_unit:
		tiles.update(units)
	
	pbar = A.pull('pbar', True)
	
	itr = tiles.items()
	if pbar:
		itr = tqdm(itr, total=len(tiles))
	
	if redraw_all:
		
		if 'neutral-lands' in M.pos and 'fill' in M.pos['neutral-lands'] and 'impassable' in colors:
			for x, y in M.pos['neutral-lands']['fill']:
				fill_region(img, (int(y), int(x)), val=list(ImageColor.getrgb(colors['impassable'])),
				            threshold=threshold)
	
		if 'neutral-seas' in M.pos and 'fill' in M.pos['neutral-seas'] and 'sea' in colors:
			for x, y in M.pos['neutral-seas']['fill']:
				fill_region(img, (int(y), int(x)), val=list(ImageColor.getrgb(colors['sea'])),
				            threshold=threshold)
			
	
	for loc, owner in itr:
		if pbar:
			itr.set_description(f'Filling in {loc}')
		pts = M.pos.get(loc, {})
		if owner in colors and 'fill' in pts:
			for x, y in pts['fill']:
				fill_region(img, (int(y), int(x)), val=list(ImageColor.getrgb(colors[owner])), threshold=threshold)
	
	plt.imshow(img)
	
	plt.axis('off')
	plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
	
	include_title = A.pull('include-title', True)
	if include_title:
		
		A.push('title', {}, overwrite=False)
		title_color = A.pull('title.color', 'k')
		title_font = A.pull('title.font', 'sans-serif')
		title_weight = A.pull('title.weight', 'normal')
		title_size = A.pull('title.size', 20)
		
		tx, ty = A.pull('title.x', 10), A.pull('title.y', 10)
		
		name = A.pull('title.name', None)
		
		turn, season = state['time']['turn'], state['time']['season']
		retreat = 'retreat' in state['time']
		
		offset = A.pull('title.year', None)
		year = f'Turn {turn}' if offset is None else offset + turn
		season = _SEASONS[season]
		
		suffix = ' (retreats)' if retreat else ''
		
		if action is not None:
			suffix += ': Actions'
		
		title = f'{year} {season}{suffix}' if name is None else f'{name} ({year}, {season}){suffix}'
		
		plt.text(tx,ty, title, fontsize=title_size, fontfamily=title_font, fontweight=title_weight,
		         va='top', ha='left',
		         color=title_color, zorder=10)
	
	A.push('marker', {}, overwrite=False)
	A.push('retreat', {}, overwrite=False)
	ms = A.pull('marker.size', 12)
	mew = A.pull('marker.edge', 3)
	
	rmk = A.pull('retreat.style', 'o')
	dmk = A.pull('retreat.disband', 'x')
	rms = A.pull('retreat.size', 16)
	rew = A.pull('retreat.edge', 1)
	rmc = A.pull('retreat.color', 'r')
	
	A.push('font', {}, overwrite=False)
	A.push('text', {}, overwrite=False)
	font_size = A.pull('font.size', 6)
	pad = A.pull('text.pad', 2.)
	font_weight = A.pull('text.weight', 'normal')
	
	clear_bg = A.pull('text.clear-bg', False)
	clear_edge = A.pull('text.clear-edge', False)
	
	lighten = A.pull('lighten', 0.2)
	darken = A.pull('darken', None)
	
	unit_colors = colors.copy()
	if lighten is not None:
		unit_colors = {player: lighter(color, lighten) for player, color in unit_colors.items()}
	if darken is not None:
		unit_colors = {player: dimmer(color, darken) for player, color in unit_colors.items()}
	
	default_facecolor = A.pull('default-color', 'w')
	default_edgecolor = A.pull('default-edge', 'k')
	
	A.push('center', {}, overwrite=False)
	sc_ms = A.pull('center.size', 10)
	sc_mew = A.pull('center.edge-width', 1)
	center_marker = A.pull('center.marker', '*')
	
	A.push('arrow', {}, overwrite=False)
	arrow_ratio = A.pull('arrow.ratio', 0.8)
	arrow_width = A.pull('arrow.width', 5)
	arrow_head = A.pull('arrow.head-width', 20)
	arrow_ec = A.pull('arrow.edge-color', default_edgecolor)
	
	for ID, node in M.nodes.items():
		
		pos = M.pos.get(ID, {})
		
		if 'text' in pos:
			x, y = pos['text']
			
			plt.text(x, y, s=ID.upper(), size=font_size,
			         ha='center', va='center',
			         bbox=dict(facecolor='none' if clear_bg else default_facecolor,
			                   edgecolor='none' if clear_edge else default_edgecolor, pad=pad),
			         fontweight=font_weight,
			         zorder=3, )
		else:
			print(f'WARNING: missing text pos for: {ID}')
		
		if 'sc' in node and node['sc'] > 0:
			if 'center' in pos:
				x, y = pos['center']
				
				color = colors.get(control.get(ID, None), default_facecolor)
				
				plt.plot([x], [y], marker=center_marker, ms=sc_ms, mfc=color,
				         mec=default_edgecolor, mew=sc_mew, ls='')
			else:
				print(f'WARNING: missing center pos for: {ID}')
	
	for ID, node in M.coasts.items():
		if 'dir' in node:
			
			D = node['dir']
			
			pos = M.pos.get(ID, {})
			
			if 'text' in pos:
				x, y = pos['text']
				
				plt.text(x, y, s=D.upper(), size=font_size,
				         ha='center', va='center',
				         bbox=dict(facecolor='none' if clear_bg else default_facecolor,
				                   edgecolor='none' if clear_edge else default_edgecolor, pad=pad),
				         fontweight=font_weight,
				         zorder=3, )
			else:
				print(f'WARNING: missing text pos for: {ID}')
	
	# color
	
	shapes = A.pull('unit-shapes', {'army': 'o', 'fleet': 'v'})
	
	for player, info in state['players'].items():
		disbands = {u['loc'] for u in state.get('disbands', {}).get(player, {})}
		
		color = unit_colors.get(player, 'w')
		
		for unit in info.get('units', []):
			
			utype = unit['type']
			
			shape = shapes[utype]
			
			loc = M.fix_loc(unit['loc'], utype)
			
			if loc in state.get('retreats', {}).get(player, {}):
				pass
			elif loc in disbands or unit['loc'] in disbands:
				pass
			
			elif loc in M.pos and 'base' in M.pos[loc]:
				
				x, y = M.pos[loc]['base']
				
				plt.plot([x], [y], marker=shape, ms=ms, mfc=color,
				         mec=default_edgecolor, mew=mew, ls='', zorder=4)
			
			
			else:
				print(f'WARNING: missing base pos for: {loc}')
	
	for player, info in state.get('retreats', {}).items():
		
		color = unit_colors.get(player, 'w')
		
		for loc, opts in info.items():
			
			uloc = M.uncoastify(loc)
			utype = [unit['type'] for unit in state['players'][player]['units'] if unit['loc'] == uloc][0]
			
			x, y = M.pos[loc]['retreat']
			
			plt.plot([x], [y], marker=shapes[utype], ms=ms, mfc=color,
			         mec=default_edgecolor, mew=mew, ls='', zorder=7)
			
			if action is None:
			
				plt.plot([x], [y], marker=rmk, ms=rms, mfc='none',
				         mec=rmc, mew=rew, ls='', zorder=8)
				
					
				for opt in opts:
					dx, dy = M.pos[M.uncoastify(opt)]['text']
					dx, dy = dx - x, dy - y
					dx, dy = dx * arrow_ratio, dy * arrow_ratio
					
					plt.arrow(x, y, dx, dy, width=arrow_width,  # color=move_color,
					          head_width=arrow_head,
					          length_includes_head=True,
					
					          fc=rmc, ec=arrow_ec,
					          # head_starts_at_zero=True,
					          zorder=8)

	for player, units in state.get('disbands', {}).items():
		
		color = unit_colors.get(player, 'w')
		
		for unit in units:
			utype = unit['type']
			loc = M.fix_loc(unit['loc'], utype)
			
			shape = shapes[utype]
			
			x, y = M.pos[loc]['retreat']
			
			plt.plot([x], [y], marker=shape, ms=ms, mfc=color,
			         mec=default_edgecolor, mew=mew, ls='', zorder=5)
			

			if action is None:
				plt.plot([x], [y], marker=dmk, ms=rms, mfc='none',
				         mec=rmc, mew=rew, ls='', zorder=6)
	
	# draw actions
	
	A.push('move', {}, overwrite=False)
	move_color = A.pull('move.color', 'w')
	
	A.push('hold', {}, overwrite=False)
	hold_color = A.pull('hold.color', 'w')
	# hmk = A.pull('hold.style', 'o')
	hms = A.pull('hold.size', 16)
	hew = A.pull('hold.edge', 1)
	
	A.push('support', {}, overwrite=False)
	sup_color = A.pull('support.color', 'm')
	A.push('sup-def', {}, overwrite=False)
	sup_ratio = A.pull('support.ratio', 0.9)
	supdef_color = A.pull('support.def-color', '<>support.color', 'm')
	sup_ls = A.pull('support.line-style', '--')
	sup_mk = A.pull('support.style', 'o')
	sup_ms = A.pull('support.size', 5)
	sup_lw = A.pull('support.line-width', 1)
	
	A.push('convoy', {}, overwrite=False)
	conv_color = A.pull('convoy.color', 'c')
	conv_move = A.pull('convoy.include-move', False)

	A.push('build', {}, overwrite=False)
	build_color = A.pull('build.color', 'c')
	build_style = A.pull('build.style', '+')
	
	if action is not None:
		for player, actions in action.items():
			
			color = unit_colors.get(player, 'w')
			
			for a in actions:
				
				utype = a['unit']
				
				x, y = M.pos[M.fix_loc(a['loc'], utype)]['base']
				
				if a['type'] == 'hold':
					
					hmk = shapes[utype]
					
					plt.plot([x], [y], marker=hmk, ms=hms, mfc='none',
					         mec=hold_color, mew=hew, ls='', zorder=6)
				
				elif a['type'] == 'move' or a['type'] == 'convoy-move':
					dx, dy = M.pos[M.uncoastify(a['dest'])]['text']
					dx, dy = dx - x, dy - y
					dx, dy = dx * arrow_ratio, dy * arrow_ratio
					
					plt.arrow(x, y, dx, dy, width=arrow_width,  # color=move_color,
					          head_width=arrow_head,
					          length_includes_head=True,
					
					          fc=conv_color if conv_move and 'convoy' in a['type'] else move_color, ec=arrow_ec,
					          # head_starts_at_zero=True,
					          zorder=8)
				
				elif a['type'] == 'support':
					
					sup = M.get_unit(a['src'])
					
					x1, y1 = M.pos[sup.position]['base']
					x2, y2 = M.pos[M.uncoastify(a['dest'])]['text']
					
					lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
					
					arrowprops = dict(  # arrowstyle='simple',
						arrowstyle='wedge',
						# color=sup_color,
						capstyle='round',
						# linewidth=10, mutation_scale=150,
						linewidth=1,  # headwidth=arrow_head,
						facecolor=sup_color, edgecolor=arrow_ec,
						connectionstyle="arc3,rad=0.2",
					)
					plt.annotate('', xytext=(lx, ly), xy=(x, y),
					             # xycoords='data',
					             # textcoords='data',
					             # lw=2,
					             zorder=7,
					             arrowprops=arrowprops)
					
					# plt.plot([x,lx],[y,ly], lw=sup_lw, color=sup_color,
					#          ls=sup_ls,
					#          # fc=sup_color, ec=arrow_ec,
					#          path_effects=[pe.Stroke(linewidth=2, foreground=arrow_ec), pe.Normal()],
					#                        # head_starts_at_zero=True,
					#          zorder=6)
					
					dx, dy = arrow_ratio * (x2 - x1), arrow_ratio * (y2 - y1)
					# dx, dy = (x2-x1), (y2-y1)
					
					plt.arrow(x1, y1, dx, dy, width=arrow_width,  # color=sup_color,
					          head_width=arrow_head, ls='--',
					          length_includes_head=True,
					          fc=sup_color, ec=arrow_ec,
					          zorder=6)
					
					plt.plot([lx], [ly], marker=sup_mk, ms=sup_ms, color=sup_color, mec=arrow_ec, zorder=9)
				
				elif a['type'] == 'support-defend':
					
					sup = M.get_unit(a['dest'])
					
					dx, dy = M.pos[sup.position]['base']
					dx, dy = dx - x, dy - y
					dx, dy = dx * sup_ratio, dy * sup_ratio
					
					arrowprops = dict(  # arrowstyle='simple',
						arrowstyle='wedge',
						# color=sup_color,
						capstyle='round',
						linewidth=1,  # headwidth=arrow_head,
						facecolor=supdef_color, edgecolor=arrow_ec,
						connectionstyle="arc3,rad=0.2",
					)
					plt.annotate('', xytext=(x + dx, y + dy), xy=(x, y),
					             textcoords='data', xycoords='data',
					             zorder=6,
					             arrowprops=arrowprops)
				
				# plt.arrow(x, y, dx, dy, width=arrow_width, color=sup_color,
				#           arrowprops=dict(arrowstyle="-["),
				#           head_width=arrow_head,
				#           length_includes_head=True,
				#           # head_starts_at_zero=True,
				#           zorder=6)
				
				# plt.plot([x,x+dx], [y, y+dy], color=sup_color,
				#           arrowprops=dict(arrowstyle="-["),
				#           ls='-',
				#           zorder=6)
				
				elif a['type'] == 'convoy-transport':
					
					transport = M.get_unit(a['src'])
					
					x1, y1 = M.pos[transport.position]['base']
					x2, y2 = M.pos[M.uncoastify(a['dest'])]['text']
					
					lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
					
					arrowprops = dict(  # arrowstyle='simple',
						arrowstyle='wedge',
						# color=sup_color,
						capstyle='round',
						# linewidth=10, mutation_scale=150,
						linewidth=1,  # headwidth=arrow_head,
						facecolor=conv_color, edgecolor=arrow_ec,
						connectionstyle="arc3,rad=0.2",
					)
					plt.annotate('', xytext=(lx, ly), xy=(x, y),
					             # xycoords='data',
					             # textcoords='data',
					             # lw=2,
					             zorder=7,
					             arrowprops=arrowprops)
					
					# plt.plot([x,lx],[y,ly], lw=sup_lw, color=sup_color,
					#          ls=sup_ls,
					#          # fc=sup_color, ec=arrow_ec,
					#          path_effects=[pe.Stroke(linewidth=2, foreground=arrow_ec), pe.Normal()],
					#                        # head_starts_at_zero=True,
					#          zorder=6)
					
					dx, dy = arrow_ratio * (x2 - x1), arrow_ratio * (y2 - y1)
					# dx, dy = (x2-x1), (y2-y1)
					
					plt.arrow(x1, y1, dx, dy, width=arrow_width,  # color=sup_color,
					          head_width=arrow_head, ls='--',
					          length_includes_head=True,
					          fc=conv_color, ec=arrow_ec,
					          zorder=6)
					
					plt.plot([lx], [ly], marker=sup_mk, ms=sup_ms, color=conv_color, mec=arrow_ec, zorder=9)
				
				elif a['type'] == 'retreat':
					
					utype = a['unit']

					loc = M.fix_loc(a['loc'], utype)
					x, y = M.pos[loc]['retreat']
					
					# plt.plot([x], [y], marker=shapes[utype], ms=ms, mfc=color,
					#          mec=default_edgecolor, mew=mew, ls='', zorder=7)
					
					# plt.plot([x], [y], marker=rmk, ms=rms, mfc='none',
					#          mec=rmc, mew=rew, ls='', zorder=8)
					
					# for opt in opts:
					dx, dy = M.pos[M.uncoastify(a['dest'])]['text']
					dx, dy = dx - x, dy - y
					dx, dy = dx * arrow_ratio, dy * arrow_ratio
					
					plt.arrow(x, y, dx, dy, width=arrow_width,  # color=move_color,
					          head_width=arrow_head,
					          length_includes_head=True,
					
					          fc=rmc, ec=arrow_ec,
					          # head_starts_at_zero=True,
					          zorder=8)
					
				elif a['type'] == 'disband':
					
					utype = a['unit']
					loc = M.fix_loc(a['loc'], utype)
					
					x, y = M.pos[loc]['retreat']
					
					plt.plot([x], [y], marker=dmk, ms=rms, mfc='none',
					         mec=rmc, mew=rew, ls='', zorder=8)
					
				elif a['type'] == 'build':
					
					utype = a['unit']
					
					plt.plot([x], [y], marker=shapes[utype], ms=ms, mfc=color,
					         mec=default_edgecolor, mew=mew, ls='', zorder=7)
	
					plt.plot([x], [y], marker=build_style, ms=rms, mfc='none',
					         mec=build_color, mew=rew, ls='', zorder=8)
					pass
				
				elif a['type'] == 'destroy':
					
					# plt.plot([x], [y], marker=shapes[utype], ms=ms, mfc=color,
					#          mec=default_edgecolor, mew=mew, ls='', zorder=7)
	
					plt.plot([x], [y], marker=dmk, ms=rms, mfc='none',
					         mec=rmc, mew=rew, ls='', zorder=8)
					pass
		
	if view:
		plt.show(block=True)
	
	if save_path is not None:
		plt.savefig(save_path, dpi=W / w)

# @fig.Script('render-traj', description='Render all frames in a trajectory')
def render_traj(A):
	'''
	Given a set of states and actions, this script visualizes every game state.
	
	Visualizes the game state given a directory of states `state-dir` and actions `action-dir`.
	You can also provide a directory to save the rendered frames as `frame-dir`.
	'''
	silent = A.pull('silent', False, silent=True)
	
	root = A.pull('root', None)
	
	state_dir = A.pull('state-dir', None)
	if state_dir is None:
		assert root is not None, 'no info provided'
		state_dir = os.path.join(root, 'states')
	
	action_dir = A.pull('action-dir', None)
	if action_dir is None:
		assert root is not None, 'no info provided'
		action_dir = os.path.join(root, 'actions')
	
	assert os.path.isdir(state_dir), f'invalid path: {state_dir}'
	assert os.path.isdir(action_dir), f'invalid path: {action_dir}'
	
	state_files = []
	for fname in os.listdir(state_dir):

		terms = fname.split('.')
		name, ext = '.'.join(terms[:-1]), terms[-1]
		if ext == 'yaml':
			turn, season, *other = name.split('-')
			turn, season = int(turn), int(season)
			if len(other):
				season += 0.5
			
			state_files.append((fname, (turn,season)))
	state_files = [name for name, val in sorted(state_files, key=lambda x: x[1])]
	
	action_files = set(os.listdir(action_dir))
	
	frame_dir = A.pull('frame-dir', 'frames' if root is None else os.path.join(root, 'frames'))
	create_dir(frame_dir)
	
	frame_files = set(os.listdir(frame_dir))
	
	img_fmt = A.pull('image-format', 'png')
	
	include_actions = A.pull('include-actions', True)
	
	A.push('view', False)
	pbar = A.pull('pbar', True)
	A.push('pbar', False, silent=True)
	
	if pbar:
		state_files = tqdm(state_files)
	
	for fname in state_files:
		
		terms = fname.split('.')
		name, ext = '.'.join(terms[:-1]), terms[-1]
		
		if pbar:
			state_files.set_description(name)
		
		imname = f'{name}.{img_fmt}'
		
		A.push('save-path', os.path.join(frame_dir, imname), silent=pbar)
		A.push('state-path', os.path.join(state_dir, fname), silent=pbar)
		
		A.push('action-path', None, silent=pbar)
		if fname in action_files and imname not in frame_files:
			with A.silenced():
				fig.run('render', A)

		if include_actions:
			imname = f'{name}-actions.{img_fmt}'
			if fname in action_files and imname not in frame_files:
				A.push('action-path', os.path.join(action_dir, fname), silent=pbar)
				
				A.push('save-path', os.path.join(frame_dir, imname), silent=pbar)
				with A.silenced():
					fig.run('render', A)
			
		if not pbar:
			print(f'Rendered: {name}')
		
		
	
	print('All rendering complete')

def load_image(path):
	im = Image.open(str(path))
	im = np.array(im)
	return im

@fig.Component('map-artist')
class MapArtist(fig.Configurable):
	def __init__(self, A, **kwargs):
		
		textprops = A.pull('text-props', {})
		text_border = None
		if 'border' in textprops:
			text_border = textprops['border']
			del textprops['border']
			print('Using text border')
		
		skip_retreats = A.pull('skip-retreats', False)
		
		unit_props = A.pull('unit-props', {'army': {'marker':'o'}, 'fleet': {'marker':'v'}})
		color_key = A.pull('unit-color-key', 'mfc')
		default_color = A.pull('unit-default-color', 'w')
		auto_color = A.pull('unit-auto-color', True)
		unit_colors = A.pull('unit-colors', {})
		lighten_units = A.pull('lighten-units', 0.1)
		
		support_wedge = A.pull('support-wedge', {})
		support_arrow = A.pull('support-arrow', {})
		support_dot = A.pull('support-dot', {})
		move_arrow = A.pull('move-arrow', {})
		
		transform_props = A.pull('transform-props', {})
		if 'mfc' in transform_props and transform_props['mfc'] == None:
			transform_props['mfc'] = 'None'

		
		action_props = A.pull('action-props', {})
		for vals in action_props.values():
			for k, v in vals.items():
				if v is None and k in {'color', 'c', 'mfc', 'mec'}:
					vals[k] = str(v)
		
		action_aliases = A.pull('action-pos-aliases', {})
		
		sc_props = A.pull('sc-props', {})
		capital_props = A.pull('capital-props', {})
		home_props = A.pull('home-props', {})
		
		color_map = A.pull('color-map', {'background': [0, 0, 0],
		                 'land': [44, 160, 44],
		                 'coast': [44, 160, 44],
		                 'sea': [31, 119, 180], })
		
		include_owned = A.pull('include-owned', True)
		color_water = A.pull('color-water', False)
		
		arrow_ratio = A.pull('arrow-ratio', 0.8)
		retreat_arrow = A.pull('retreat-arrow', {})
		
		super().__init__(A, **kwargs)
		self.graph = None
		self.state = None
		self.players = None
		self.bgs = None
		self.actions = None
		
		self.skip_retreats = skip_retreats
		
		self.include_owned = include_owned
		self.color_water = color_water
		
		self.color_map = color_map
		
		self.textprops = textprops
		self.textborder = text_border
		
		self.move_arrow = move_arrow
		self.support_wedge = support_wedge
		self.support_arrow = support_arrow
		self.support_dot = support_dot
		
		self.transform_props = transform_props
		
		self.unit_props = unit_props
		self.unit_color_key = color_key
		self.unit_auto_color = auto_color
		self.unit_colors = unit_colors
		self.unit_default_color = default_color
		self.unit_lighten_factor = lighten_units
		
		self.owned = None
		self.units = None
		self.num_units = None
		
		self.action_props = action_props
		self.action_aliases = action_aliases
		
		self.sc_props = sc_props
		self.capital_props = capital_props
		self.home_props = home_props
		
		self.arrow_ratio = arrow_ratio
		self.retreat_arrow = retreat_arrow
		
		self.capitals = None
		self.node_img = None
		self.target_locs = None
		
	def _target_locs(self, graph):
		
		if self.target_locs is None:
		
			targets = {}
			
			for name, info in graph.items():
			
				locs = info['locs']
				
				if 'label' in locs:
					targets[name] = np.array(locs['label']).reshape(-1,2)
				
				if 'coast-label' in locs and isinstance(locs['coast-label'], dict):
					for coast, pos in locs['coast-label'].items():
						target = fig.quick_run('_encode_region_name', name=name, coast=coast)
						targets[target] = np.array(pos).reshape(-1,2)
			
			
			self.target_locs = targets
		
	def _format_actions(self, units, actions):
		
		unit_locs = {}
		for us in units.values():
			unit_locs.update({u['loc']:u for u in us})
		
		for name, us in units.items():
			if name in actions:
				acts = {a['loc']:a for a in actions[name]}
				
				for u in us:
					if u['loc'] in acts:
						cmd = acts[u['loc']]
						if cmd['type'] == 'support-defend':
							if cmd['dest'] in unit_locs:
								cmd['dest-unit'] = unit_locs[cmd['dest']]
							else:
								raise Exception
						elif cmd['type'] in {'support', 'convoy-transport'}:
							if cmd['src'] in unit_locs:
								cmd['src-unit'] = unit_locs[cmd['src']]
							else:
								raise Exception
						u['command'] = cmd
						del acts[u['loc']]
					# else:
					# 	print(u, acts)
					# 	raise Exception
				if len(acts):
					print(acts)
					raise Exception
				
		
	def include_graph(self, graph=None, state=None, players=None, bgs=None, actions=None):
		if graph is not None:
			self.graph = graph
		if state is not None:
			self.state = state
		if players is not None:
			self.players = players
			
			capitals = {player['capital']: name for name, player in players.items()}
			
			self.capitals = capitals
			
		if bgs is not None:
			self.bgs = bgs
		if actions is not None:
			self.actions = actions

		unit_state = None
		if state is None:
			unit_state = players
		else:
			unit_state = state['players']
		
		num_units = 0
		units = None
		owned = None
		if unit_state is not None:
			
			units = {}
			owned = {}
			
			for pname, info in unit_state.items():
				if 'color' in players[pname]:
					
					units[pname] = info.get('units', [])
					num_units += len(units[pname])
					
					ownership = info.get('owns', None)
					if ownership is None:
						ownership = info.get('control', None)
					
					if ownership is not None:
						for loc in ownership:
							owned[loc] = players[pname]['color']
				
				else:
					
					print(f'WARNING: no color for player: {pname}')
			
			if state is not None:
				if 'retreats' in state:
					self._target_locs(self.graph)
					for name, us in units.items():
						if name in state['retreats']:
							r = state['retreats'][name]
							options = {
								fig.quick_run('_decode_region_name', name=start)[0]:
									[fig.quick_run('_decode_region_name', name=end)[0] for end in ends]
								for start, ends in r.items()
							}
							
							for u in us:
								loc, coast = fig.quick_run('_decode_region_name', name=u['loc'])
								if loc in options:
									u['action'] = 'retreat'
									u['retreat-options'] = options[loc]
					
				if 'disbands' in state:
					for name, us in units.items():
						if name in state['disbands']:
							for u in state['disbands'][name]:
								u['action'] = 'disband'
								us.append(u)
							
			
		self.num_units = num_units
		self.owned = owned
		self.units = units
		
		if self.actions is not None:
			self.skip_retreats = True
			self._target_locs(self.graph)
			self._format_actions(units, actions)
		
	def include_image(self, img=None):
		self.node_img = img
		
	def fill_region(self, img, loc, color=None):
		
		if loc in self.bgs:
			
			info = self.bgs[loc]
			
			if color is None and self.color_water and self.include_owned and 'island' in info \
						and self.owned is not None:
				color = self.owned.get(info['island'], None)
				
		else:
			assert loc in self.graph
			info = self.graph[loc]
			
			if color is None and self.include_owned and self.owned is not None:
				color = self.owned.get(loc, None)
			
		if color is None:
			ev = info.get('env', None)
			color = self.color_map.get(ev, None)
			if color is None:
				typ = info.get('type', None)
				color = self.color_map.get(typ, None)
		
		idx = info['idx'] + 1
		
		if color is not None:
			# img[self.node_img == idx] = (np.random.rand(3)*255).astype(int)
			img[self.node_img == idx] = hex_to_rgb(process_color(color))
		
		return img
		
	def draw_label(self, loc, include_coasts=True):
		
		locs = self.graph[loc]['locs']
		pos = locs['water'].get('label', None) if 'water' in locs else locs.get('label', None)
		if pos is None:
			print(f'WARNING: no label found for {loc}')
			return
		pos = np.array(pos)
		if len(pos.shape) == 1:
			pos = [pos]
		
		out = []
		for x,y in pos:
		
			txt = plt.text(x,y, loc, **self.textprops)
			out.append(txt)
			if self.textborder is not None:
				txt.set_path_effects([path_effects.Stroke(**self.textborder),
				                      path_effects.Normal()])
			
			if include_coasts:
				coasts = locs.get('coast-label', None)
				if isinstance(coasts, dict):
					out = [out]
					for coast, pos in coasts.items():
						pos = np.array(pos)#[..., ::-1]
						if len(pos.shape) == 1:
							pos = [pos]
						for x,y in pos:
							txt = plt.text(x,y, coast.upper(), **self.textprops)
							if self.textborder is not None:
								txt.set_path_effects([path_effects.Stroke(**self.textborder),
								                      path_effects.Normal()])
							out.append(txt)
	
		if len(out) == 1:
			return out[0]
		return out
		
	def draw_sc(self, loc, val=1):
		
		pos = self.graph[loc]['locs'].get('sc', None)
		if pos is None:
			print(f'WARNING: {loc} doesnt have an sc pos')
			return
		pos = np.array(pos)
		if len(pos.shape) == 2:
			pos = pos.T
			
		out = []
		
		if loc in self.capitals:
			out.append(plt.plot(*pos, **self.capital_props))
		else:
			out.append(plt.plot(*pos, **self.sc_props))
		
		for name, info in self.state['players'].items():
			if loc in info['home'] and loc in info['centers']:
				out.append(plt.plot(*pos, mfc=self.players[name]['color'], **self.home_props))
		
	def draw_all_unit_locs(self, loc):
		
		info = self.graph[loc].get('locs', None)
		if info is None:
			return
		
		if 'army' in info:
			self.draw_unit({'type': 'army', 'loc': loc})
			self.draw_unit({'type': 'army', 'loc': loc, 'action': 'retreat'})
			
		if 'fleet' in info:
			for action, fleet in info['fleet'].items():
				if not isinstance(fleet, dict):
					fleet = {None: fleet}
				for k, v in fleet.items():
					uloc = fig.quick_run('_encode_region_name', name=loc, coast=k)
					self.draw_unit({'type': 'fleet', 'loc': uloc, 'action': action})
		
		
	def _auto_color(self, unit):
		
		if 'player' in unit:
		
			player = unit['player']
			
			color = self.players[player]['color']
			if self.unit_lighten_factor is not None and self.unit_lighten_factor > 0:
				color = lighter(color, self.unit_lighten_factor)
			self.unit_colors[player] = color
			return color
		
		return self.unit_default_color
		
	def draw_units(self, pbar=None):
		if self.units is None or len(self.units) == 0:
			return
		itr = None
		if pbar is not None:
			itr = pbar(total=self.num_units, desc=f'Drawing {self.num_units} units')
		
		for pname, us in self.units.items():
			for u in us:
				u['player'] = pname
				self.draw_unit(u)
				if itr is not None:
					itr.update(1)
		
	def draw_unit(self, unit):
		
		loc, coast = fig.quick_run('_decode_region_name', name=unit['loc'])
		typ = unit['type']
		player = unit.get('player', None)
		
		locs = self.graph[loc].get('locs', None)
		if locs is None:
			print(f'WARNING: no locations found for {unit}')
			return
			
		props = self.unit_props.get(typ, {}).copy()
		
		if player in self.unit_colors:
			color = self.unit_colors[player]
		elif self.unit_auto_color:
			color = self._auto_color(unit)
		else:
			color = self.unit_default_color
			
		if self.unit_color_key is not None:
			props[self.unit_color_key] = color
			
		action = unit.get('action', 'occupy')
		# area = data[loc]['area']
		
		if action == 'retreat' and 'zorder' in props:
			props['zorder'] += 3
		
		pos = locs.get(typ, locs).get(action, None)
		if pos is None and action in self.action_aliases:
			pos = locs.get(typ, locs)
			pos = pos.get(self.action_aliases[action], None)
		
		if pos is None:
			return
			
		if isinstance(pos, dict):
			assert coast is not None, f'{loc} {coast}, {pos}'
			pos = pos[coast]
		pos = np.array(pos)
		if len(pos.shape) == 2:
			pos = pos.T
	
		out = plt.plot(*pos, **props)
	
		if action in self.action_props:
			out = [out, plt.plot(*pos, **self.action_props[action])]
		
		options = unit.get('retreat-options', [])
		if not self.skip_retreats:
			for dest in options:

				target = self.graph[dest]['locs'].get('label', None)
				if target is not None:
					target = np.array(target).reshape(-1,2)
					if len(pos.shape) == 2:
						pos = pos.T
					pos = pos.reshape(-1,2)
					
					x,y, tx, ty = self._best_arrow_coords(pos, target)
					dx, dy = tx - x, ty - y
					dx, dy = dx * self.arrow_ratio, dy * self.arrow_ratio
					
					plt.arrow(x, y, dx, dy, **self.retreat_arrow)
					# plt.annotate('', xy=(x, y), xytext=(dx, dy), **self.retreat_arrow)
		
		if 'command' in unit:
			cmd = unit['command']
			
			if len(pos.shape) == 2:
				pos = pos.T
			pos = pos.reshape(-1, 2)
			
			atype = cmd['type']
			
			if atype == 'move' or atype == 'convoy-move' or atype == 'retreat':
				target = self.target_locs[cmd['dest']]
				x, y, tx, ty = self._best_arrow_coords(pos, target)
				dx, dy = tx - x, ty - y
				dx, dy = dx * self.arrow_ratio, dy * self.arrow_ratio

				arrow = self.move_arrow.copy()
				if 'convoy' in atype:
					arrow['facecolor'] = 'C0'
				
				plt.arrow(x, y, dx, dy, **arrow)
				
			elif atype == 'support' or atype == 'convoy-transport':
				src = cmd['src-unit']
				sbase, scoast = fig.quick_run('_decode_region_name', name=src['loc'])
				spos = self.graph[sbase]['locs'][src['type']]['occupy']
				if isinstance(spos, dict):
					spos = spos[scoast]
				srcpos = np.array(spos).reshape(-1,2)
				
				dest = cmd['dest']
				destpos = self.target_locs[dest]
				
				x1, y1, x2, y2 = self._best_arrow_coords(srcpos, destpos)
				
				lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
				x,y, _, _ = self._best_arrow_coords(pos, destpos)
				
				wedge = deepcopy(self.support_wedge)
				arrow = self.support_arrow.copy()
				dot = self.support_dot.copy()
				
				style = ArrowStyle("wedge", shrink_factor=0.5, tail_width=.05)
				wedge['arrowprops']['arrowstyle'] = style
				
				if 'convoy' in atype:
					wedge['arrowprops']['facecolor'] = 'c'
					arrow['facecolor'] = 'c'
					dot['color'] = 'c'
				
				dx, dy = self.arrow_ratio * (x2 - x1), self.arrow_ratio * (y2 - y1)
				
				plt.annotate('', xytext=(lx, ly), xy=(x, y), **wedge)
				plt.arrow(x1, y1, dx, dy, **arrow)
				plt.plot([lx], [ly], **dot)
				
			elif atype == 'support-defend':
				dest = cmd['dest-unit']
				sbase, scoast = fig.quick_run('_decode_region_name', name=dest['loc'])
				spos = self.graph[sbase]['locs'][dest['type']]['occupy']
				if isinstance(spos, dict):
					spos = spos[scoast]
				destpos = np.array(spos).reshape(-1, 2)
				

				wedge = deepcopy(self.support_wedge)
				style = ArrowStyle("wedge", shrink_factor=0.5, tail_width=.05)
				wedge['arrowprops']['arrowstyle'] = style
				
				x, y, tx, ty = self._best_arrow_coords(pos, destpos)
				plt.annotate('', xytext=(tx, ty), xy=(x, y), **wedge)
				
			elif atype == 'transform':
				plt.plot(*pos.T, **self.transform_props)
			
			elif atype == 'build':
				raise NotImplementedError
		
		return out

	def _best_arrow_coords(self, starts, targets):
		D = distance_matrix(starts, targets)
		ind = np.unravel_index(np.argmin(D, axis=None), D.shape)
		return [*starts[ind[0]], *targets[ind[1]]]
	
	def draw_special(self):
		pass

@fig.AutoModifier('canal')
class Canal(MapArtist):
	def __init__(self, A, **kwargs):

		canal_props = A.pull('canal-props', {})
		
		super().__init__(A, **kwargs)

		self.canal_props = canal_props

	def draw_special(self):
		super().draw_special()
		
		for loc, info in self.graph.items():
			if 'canal' in info and 'coast-pos' in info:
				pts = np.array(list(info['coast-pos'].values()))
				plt.plot(*pts[:,::-1].T, **self.canal_props)
		

@fig.Script('viz-map')
def viz_map(A):
	mlp_backend = A.pull('mlp_backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	path = A.pull('graph-path')
	data = load_yaml(path)
	
	save_path = A.pull('save-path', 'test.png')
	
	fpath = A.pull('fields-img')
	if fpath is None:
		raise NotImplementedError
	else:
		im = load_image(fpath)
	
	bgpath = A.pull('bg-path', None)
	bgs = None if bgpath is None else load_yaml(bgpath)
	
	# poi_path = A.pull('poi-path', None)
	# pois = None if poi_path is None else load_yaml(poi_path)
	
	fill_units = A.pull('fill-units', False)
	
	state = None
	if not fill_units:
		state_path = A.pull('state-path', None)
		state = None if state_path is None else load_yaml(state_path)
	
	
	player_path = A.pull('player-path', None)
	players = None if player_path is None else load_yaml(player_path)
	
	action_path = A.pull('action-path', None)
	actions = None if action_path is None else load_yaml(action_path)
	
	# opath = A.pull('overlay-path', None)
	# if opath is not None:
	# 	raise NotImplementedError
	
	if A.pull('force-border', True):
		bounds = get_borders_from_expanded(im)
		im[bounds == 1] = 0
	
	clean = im.copy().clip(max=1)
	clean = np.stack(3 * [clean], 2) * 255
	
	
	skip_bgs = A.pull('skip-bgs', False)
	skip_filling = A.pull('skip-filling', False)
	skip_labeling = A.pull('skip-labeling', False)
	skip_units = A.pull('skip-units', False)
	
	artist = A.pull('artist')
	
	artist.include_graph(graph=data, state=state, players=players, bgs=bgs, actions=actions)
	artist.include_image(im)
	
	dpi = 1200
	H, W = im.shape
	fg, ax = plt.subplots(figsize=(W / dpi, H / dpi))
	
	if bgs is not None and not skip_bgs:
		todo = tqdm(bgs.keys(), total=len(bgs), desc='Coloring in background')
		for loc in todo:
			artist.fill_region(clean, loc)
			
	if not skip_filling:
		todo = tqdm(data.keys(), total=len(data), desc='Coloring in nodes')
		for loc in todo:
			artist.fill_region(clean, loc)
	
	plt.imshow(clean, zorder=0)
	
	if not skip_labeling:
		todo = tqdm(data.items(), total=len(data), desc='Labeling nodes')
		for loc, info in todo:
			artist.draw_label(loc)
			# poi = pois[name] if pois is not None and name in pois else None
			
			if 'sc' in info and info['sc'] > 0:
				artist.draw_sc(loc, info['sc'])
			
	if fill_units:
		# print('Filling all unit locs')
		todo = tqdm(data.keys(), total=len(data), desc='Filling unit locs')
		for loc in todo:
			artist.draw_all_unit_locs(loc)
	elif not skip_units:
		artist.draw_units(pbar=tqdm)
		
	
	artist.draw_special()
	
	plt.axis('off')
	plt.subplots_adjust(0, 0, 1, 1)
	
	# plt.show()
	plt.savefig(save_path, dpi=dpi)




