
import sys, os

import omnifig as fig

# import numpy as np
from omnibelt import load_yaml, save_yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .. import util


def _post_process(pos, nodes, retreat_pos):  # make sure coastal tiles also have a retreat pos
	
	for ID, p in pos.items():
		if retreat_pos and 'retreat' not in p:
			
			if ID not in nodes:
				print(f'WARNING: could not find node {ID}')
			elif 'coast-of' in nodes[ID]:
				base = nodes[ID]['coast-of']
				if base in pos and 'retreat' in pos[base]:
					p['retreat'] = pos[base]['retreat']

@fig.Script('collect-pos', 'Collects Pixel coordinates on the map for diplomacy')
def _collect_map_pos(A):
	'''
	Collects the coordinates on an image for units, texts, and centers used when rendering game states.

	Left-click - select coordinates for the named node
	Right-click - clear all specified coordinates for the current node and return to the previous node
	All selected coordinates are saved automatically to the an pos yaml file when the window is closed.
	'''
	
	mlp_backend = A.pull('mlp-backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	root = A.pull('root', None)
	
	image_path = A.pull('image-path')
	
	out_path = A.pull('out-path', '<>out')
	pos_path = A.pull('pos-path', out_path) # past pos file to update/overwrite
	
	if root is not None:
		pos_path = os.path.join(root, pos_path)
		out_path = os.path.join(root, out_path)
	
	A.push('map._type', 'map', overwrite=False)
	world = A.pull('map')
	
	nodes = world.nodes
	nodes.update(util.separate_coasts(nodes))
	edges = util.list_edges(world.edges)
	
	
	base_pos = A.pull('base', True)
	text_pos = A.pull('text', True)
	center_pos = A.pull('center', True)
	retreat_pos = A.pull('retreat', True)
	edge_pos = A.pull('edge', False)
	coast_retreat_pos = A.pull('coast-retreat', True)
	if coast_retreat_pos:
		retreat_pos = True
	
	try:
		pos = load_yaml(pos_path)
		if pos is None:
			pos = {}
		print(f'Found previous pos file with {len(pos)} entries')
	except FileNotFoundError:
		pos = {}
		print('No pos file found, starting from scratch')
	
	locs_key = A.pull('node-pos-key', None)
	if locs_key is not None:
		for ID, node in nodes.items():
			if ID not in pos:
				pos[ID] = {}
			if 'pos' in node:
				pos[ID][locs_key] = node['pos']
		
		for ID, coast in util.separate_coasts(nodes, dir_only=True).items():
			if ID not in pos:
				pos[ID] = {}
			origin = nodes[coast['coast-of']]
			if 'coast-pos' in origin:
				pos[ID][locs_key] = origin['coast-pos'][coast['dir']]
	
	font_size = A.pull('font-size', 6)
	use_text = A.pull('use-text', True)
	
	fig, ax = plt.subplots(figsize=(12, 8))
	
	img = mpimg.imread(image_path)
	
	plt.imshow(img)
	plt.axis('off')
	
	plt.title('test')

	plt.tight_layout()
	
	todo_nodes = list(nodes.keys()) if base_pos or text_pos or retreat_pos else []
	todo_edges = list(edges) if edge_pos else []
	done_nodes = []
	done_edges = []
	
	container = None
	key = None
	
	etype = None
	current = None
	
	name = None
	tname = None
	
	ms = A.pull('marker-size', 10)
	mew = A.pull('marker-edge-width', 3)
	pad = A.pull('text-pad', 2.)
	clear_bg = A.pull('clear-bg', False)
	font_weight = A.pull('font-weight', 'normal')
	font_size = A.pull('font-size', 6)
	
	def _next_prompt():
		nonlocal current, key, container, name, tname
		
		while len(todo_nodes) or isinstance(current, str):
			
			if current is None:
				current = todo_nodes.pop()
			
			if current not in pos:
				pos[current] = {}
			
			node = nodes[current]
			name = current.upper() if nodes is None or current not in nodes else node['name']
			
			container = pos[current]
			key = current
			
			if text_pos and ('coast-of' not in node or 'dir' in node):
				if 'text' not in container:
					plt.title(f'{name} - text')
					key = 'text'
					target = None
					
					plt.draw()
					plt.pause(0.0001)
					return
				else:
					x, y = container['text']
					# plt.scatter([x], [y], marker='s', color='g')
					if use_text:
						txt = node['dir'].upper() if 'dir' in node else current.upper()
						plt.text(x, y, s=txt, size=font_size,
						         color='k',
						         # ma='center',
						         ha='center', va='center',
						         bbox=dict(facecolor='' if clear_bg else 'w', edgecolor='k', pad=pad),
						         fontweight=font_weight,
						         zorder=3,
						         # backgroundcolor='w',
						         )
					else:
						plt.plot([x], [y], marker='o', ms=ms, mfc='g', mec='k', mew=mew, ls='')
			
			if base_pos:
				if 'base' not in container:
					plt.title(f'{name} - base')
					key = 'base'
					target = None

					plt.draw()
					plt.pause(0.0001)
					return
				else:
					x,y = container['base']
					plt.plot([x], [y], marker='o', ms=ms, mfc='w', mec='k', mew=mew, ls='')
			
			if center_pos and 'sc' in node:
				if 'center' not in container:
					plt.title(f'{name} - center')
					key = 'center'
					target = None
					
					plt.draw()
					plt.pause(0.0001)
					return
				else:
					x, y = container['center']
					plt.plot([x], [y], marker='o', ms=ms, mfc='y', mec='k', mew=mew, ls='')
			
			if retreat_pos and 'coast-of' not in nodes[current]:
				if 'retreat' not in container:
					plt.title(f'{name} - retreat')
					key = 'retreat'
					target = None

					plt.draw()
					plt.pause(0.0001)
					return
				else:
					x,y = container['retreat']
					plt.plot([x], [y], marker='x', ms=ms, mfc='r', mec='k', mew=mew, ls='')
					# plt.scatter([x], [y], marker='x', color='r')
				

			if coast_retreat_pos and 'coast-of' in nodes[current]:
				if 'retreat' not in container:
					plt.title(f'{name} - retreat')
					key = 'retreat'
					target = None

					plt.draw()
					plt.pause(0.0001)
					return
				else:
					x,y = container['retreat']
					plt.plot([x], [y], marker='x', ms=ms, mfc='r', mec='k', mew=mew, ls='')
					# plt.scatter([x], [y], marker='x', color='r')
				
			# plt.draw()
			# plt.pause(0.0001)
			
			# if coast_retreat_pos:
			# 	if 'retreat' not in container:
			# 		plt.title(f'{name} - retreat')
			# 		key = 'retreat'
			# 		target = None
			#
			# 		plt.draw()
			# 		plt.pause(0.0001)
			# 		return
			# 	else:
			# 		x,y = container['retreat']
			# 		plt.scatter([x], [y], marker='x', color='y')
			

			done_nodes.append(current)
			print(f'Done with {current}')
			current = None
			
		# print('Moving on to edges')
		
		while len(todo_edges) or isinstance(current, dict):
			
			if current is None:
				current = todo_edges.pop()
			
			if edge_pos:
				start = current['start']
				etype = current['type']
				end = current['end']
				
				name = start.upper() if nodes is None or start not in nodes else nodes[start]['name']
				
				if start not in pos:
					pos[start] = {}
				
				mypos = pos[start]
				
				if 'edges' not in mypos:
					mypos['edges'] = {}
				
				if etype not in mypos['edges']:
					mypos['edges'][etype] = {}

				

				container = mypos['edges'][etype]
				
				if end not in container:
					tname = end if end not in nodes else nodes[end]['name']
					
					key = end
					
					plt.title(f'{name} - edge with {tname}')
					
					# if key not in pos[start]:
					# 	pos[start][key] = {}
					# if end not in pos:
					# 	pos[end] = {}
					# if key not in pos[end]:
					# 	pos[end][key] = {}
					
					plt.draw()
					plt.pause(0.0001)
					return
				
				else:
					x1,y1 = container[end]
					plt.scatter([x1], [y1], marker='v', color='b')
					
					if end in pos and 'edges' in pos[end] and etype in pos[end]['edges'] \
							and start in pos[end]['edges'][etype]:
						x2,y2 = pos[end]['edges'][etype][start]
						
						plt.scatter([x2], [y2], marker='v', color='b')
						plt.plot([x1,x2], [y1,y2], marker='', ls='-', color='b')
						
				
			# current is done
			done_edges.append(current)
			# print(f'Done with {current}')
			current = None
		
		# no more todo
		
		fig.canvas.mpl_disconnect(cid)
		
		plt.title('Done with all positions! (you can close the window)')
		
		plt.draw()
		plt.pause(0.0001)
		
		# save_yaml(pos, out_path)
		# fig.close()
		# quit()
		# raise Exception('done')
		
		pass
	
	def _prev_node():
		
		nonlocal current
		
		if current is None:
			pass
		elif isinstance(current, str):
			
			if len(pos[current]) == 0 and len(done_nodes):
				
				if len(done_nodes):
					todo_nodes.append(current)
					prev = done_nodes.pop()
					pos[prev].clear()
					todo_nodes.append(prev)
				else:
					print('Cant go back anymore')
			
			else:
			
				pos[current].clear()
				
				todo_nodes.append(current)
			current = None
		
		else:
			
			
			if len(done_edges):
				todo_edges.append(current)
				prev = done_edges.pop()
				del pos[prev['start']]['edges'][prev['type']][prev['end']]
				todo_edges.append(prev)
				
				
			else:
				print('Cant go back anymore')
			
			
			current = None
		
		_next_prompt()
	
	def onclick(event):
		btn = event.button  # 1 is left, 3 is right
		try:
			xy = [float(event.xdata), float(event.ydata)]
		except:
			return
		
		if btn == 1:
			
			of = key
			if tname is not None:
				of = f'edge with {tname}'
			
			print(f'{name} [{of}]: {xy}')
			
			container[key] = xy
			
			_next_prompt()
		
		elif btn == 3:
			print('Going back')
			_prev_node()
	
		else: # invalid button
			print(f'unknown button: {btn}')
			return
	
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	
	_next_prompt()
	
	plt.show(block=True)
	
	_post_process(pos, nodes, retreat_pos)
	
	print(f'All positions collected, saved pos to: {out_path}')
	save_yaml(pos, out_path, default_flow_style=None )
	
	return pos
	
if __name__ == '__main__':
	fig.entry('collect-pos')
	
	
	