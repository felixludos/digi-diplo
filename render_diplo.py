import sys, os
from itertools import chain
from omnibelt import load_yaml, save_yaml, create_dir
from tqdm import tqdm

import omnifig as fig

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import figaspect
import matplotlib.patheffects as pe

import numpy as np
from PIL import ImageColor, Image

from src.colors import lighter, dimmer, fill_region

import pydip

_SEASONS = ['', 'Spring', 'Autumn', 'Winter']


@fig.Script('render', description='Render a Diplomacy state')
def render_diplo_state(A):
	save_path = A.pull('save-path', None)
	view = A.pull('view', save_path is None)
	
	mlp_backend = A.pull('mlp-backend', 'qt5agg' if view else 'agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
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
	
	for loc, owner in tiles.items():
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

@fig.Script('render-traj')
def render_traj(A):
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
	

# save_path = A.pull('save-path', None)
# if save_path is None:
#
# 	save_root = A.pull('save-root', None)
#
# 	if save_root is not None:
# 		turn, season = new['time']['turn'], new['time']['season']
# 		r = '-r' if 'retreat' in new['time'] else ''
# 		new_name = f'{turn}-{season}{r}.yaml'
#
# 		save_path = os.path.join(save_root, new_name)
#
# if save_path is not None:
# 	save_yaml(new, save_path, default_flow_style=None)


if __name__ == '__main__':
	fig.entry('render')
