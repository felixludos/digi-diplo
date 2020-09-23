

import sys, os

import omnifig as fig

import numpy as np

# import numpy as np
from omnibelt import load_yaml, save_yaml
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

from PIL import Image

from src import util

def _pick_edges(ename, locs, pos, names, edges, edges_path, img,
                acolor='b'):
	
	adj = edges[ename]
	
	todo = locs.copy()
	done = []
	
	fg, ax = plt.subplots(figsize=(12, 8))
	
	plt.imshow(img)
	plt.axis('off')
	
	points = {n:plt.plot([pos[n][0]], [pos[n][1]], picker=10, color='r', ls='', markersize=12,
	               markeredgewidth=3, marker='o', markeredgecolor='k', zorder=5)[0]
	          for n in locs}
	
	for n, p in points.items():
		p.loc = n
	
	# for s, es in adj.items():
	# 	x1, y1 = pos[s]
	# 	for e in es:
	# 		x2, y2 = pos[e]
	# 		plt.arrow(x1, y1, x2 - x1, y2 - y1, lw=3, color=acolor, zorder=3,
	# 		          head_width=15, head_length=25,
	# 		          length_includes_head=True)
	
	plt.title('test')
	plt.ylabel(f'{ename}')
	plt.xlabel('Press space to continue to next node')
	
	plt.tight_layout()
	
	current = None
	
	elines = []
	
	def onpick(event):
		
		loc = event.artist.loc
		# print(loc)
		
		if current not in adj:
			adj[current] = []#set()
		
		# adj[current].add(loc)
		if loc not in adj[current]:
			adj[current].append(loc)
		
		x1, y1 = pos[current]
		x2, y2 = pos[loc]
		
		# elines.append(plt.plot([x1, x2], [y1, y2], lw=5, color='b', zorder=2)[0])
		elines.append(plt.arrow(x1, y1, x2 - x1, y2 - y1, lw=3, color=acolor, zorder=3,
		          head_width=15, head_length=25,
		          length_includes_head=True))
		
		plt.draw()
	
	def onkey(event=None):
		
		nonlocal current
		
		key = None if event is None else event.key
		
		if key is None or key == ' ':
			
			if not len(todo):
				plt.title('Done')
				plt.draw()
				return
			
			if current is not None:
				
				if len(elines):
					for l in elines:
						l.remove()
					elines.clear()
				
				points[current].set_color('r')
				done.append(current)
				
			current = todo.pop()
			points[current].set_color('y')
			
			plt.title(f'{names[current]} ({ename})')
			
			if current in adj:
				x1, y1 = pos[current]
				for e in adj[current]:
					x2, y2 = pos[e]
					elines.append(plt.arrow(x1, y1, x2 - x1, y2 - y1, lw=3, color=acolor, zorder=3,
					          head_width=15, head_length=25,
					          length_includes_head=True))
		
			plt.draw()
		
		elif key == 'backspace':
			
			ns = adj[current]
			
			if len(ns):
				

				if len(elines):
					for l in elines:
						l.remove()
					elines.clear()
				
				ns.clear()
			else:
				todo.append(current)
				if len(done):
					current = done.pop()
					adj[current].clear()
					todo.append(current)
	
	cid = fg.canvas.mpl_connect('pick_event', onpick)
	bid = fg.canvas.mpl_connect('key_press_event', onkey)
	
	onkey()
	
	plt.show(block=True)
	
	edges = {t:{n:list(e) for n, e in es.items()} for t, es in edges.items()}
	
	print(f'All edges collected, saved to: {edges_path}')
	save_yaml(edges, edges_path, default_flow_style=None)
	
	return edges


@fig.Script('mapping', 'Collects map adjacencies for diplomacy')
def _collect_adj(A):
	'''
	Select the coordinates of the nodes and edges for a map
	
	Instructions:
	If the node locations have not been specified yet, then the "node-locs" script is run.
	Once all the node locations are specified, you can specify the edges between nodes first for armies then fleets.
	
	First, a window will open for the edges of armies.
	Click on a node location to add an edge between
	Press space to move to the next node
	Press backspace to clear all selected edges and move to the previous node.
	All selected coordinates are saved automatically to the an edges yaml file when the window is closed.
	After the army edges are specified, another window opens for the fleets.
	'''
	mlp_backend = A.pull('mlp-backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	nodes_path, edges_path = util.get_map_paths(A, 'nodes', 'edges')
	
	image_path = A.pull('image-path')
	
	nodes = load_yaml(nodes_path)
	for name, node in nodes.items():
		if 'pos' not in node or ('dirs' in node and ('coast-pos' not in node
		                                             or len(node['coast-pos'])!=len(node['dirs']))):
			nodes = fig.run('node-locs', A)
			break
	
	arrow_color = A.pull('arrow_color', 'b')
	
	edges = load_yaml(edges_path) if os.path.isfile(edges_path) else {'army':{}, 'fleet':{}}
	
	coasts = util.separate_coasts(nodes, dir_only=True)
	
	pos = {n:node['pos'] for n, node in nodes.items()}
	pos.update({c:nodes[coast['coast-of']]['coast-pos'][coast['dir']] for c, coast in coasts.items()})
	
	img = np.array(Image.open(image_path).convert("RGB"))
	
	locs = [name for name, node in nodes.items() if node['type'] in {'land', 'coast'}]
	names = {name:node['name'] if 'name' in node else name for name, node in nodes.items()
	         if node['type'] in {'land', 'coast'}}
	edges = _pick_edges('army', locs, pos, names, edges, edges_path, img, acolor=arrow_color)
	
	locs = [name for name, node in nodes.items()
	        if node['type'] in {'coast', 'sea'} and 'dirs' not in node] + list(coasts)
	names = {**{name:node['name'] if 'name' in node else name for name, node in nodes.items()
	         if node['type'] in {'sea', 'coast'}},
			**{ID:coast['name'] if 'name' in coast else ID for ID, coast in coasts.items()}}
	edges = _pick_edges('fleet', locs, pos, names, edges, edges_path, img, acolor=arrow_color)

	return edges
	



@fig.Script('node-locs', 'Set positions for nodes in a Diplomacy map')
def _node_locs(A):
	'''
	Select the coordinates of the nodes to then specify all the edges (using the "mapping" script)
	
	Instructions:
	Left-click - select coordinates for the named node
	Right-click - return to the previous node
	All selected coordinates are saved automatically to the provided nodes yaml file when the window is closed.
	'''

	mlp_backend = A.pull('mlp_backend', 'qt5agg')
	if mlp_backend is not None:
		plt.switch_backend(mlp_backend)
	
	nodes_path = util.get_map_paths(A, 'nodes')
	image_path = A.pull('image-path')
	
	nodes = load_yaml(nodes_path)
	coasts = util.separate_coasts(nodes, dir_only=True)
	
	fg, ax = plt.subplots(figsize=(12, 8))
	
	img = np.array(Image.open(image_path).convert("RGB"))
	
	plt.imshow(img)
	plt.axis('off')
	
	plt.title('test')
	
	plt.tight_layout()
	
	todo = list(nodes) + list(coasts)
	done = []
	
	current = None
	
	def _next_prompt():
		
		nonlocal current
		
		while len(todo):
			
			if current is None:
				current = todo.pop()
			
			node = nodes[current] if current in nodes else coasts[current]
			
			if 'pos' in node or ('coast-of' in node and 'coast-pos' in nodes[node['coast-of']]
			                     and node['dir'] in nodes[node['coast-of']]['coast-pos']):
				x,y = node['pos'] if 'pos' in node else nodes[node['coast-of']]['coast-pos'][node['dir']]
				# plt.scatter([x], [y], color='k', marker='o', s=15)
				plt.plot([x], [y], picker=10, color='r', ls='', markersize=12,
				         markeredgewidth=3, marker='o', markeredgecolor='k', zorder=5)
				plt.draw()
			else:
				name = node['name'] if 'name' in node else current
				plt.title(f'Node: {name}')
				plt.draw()
				return
			
			done.append(current)
			current = None
		
		plt.title('Done!')
		plt.draw()
	
	def onclick(event):
		
		nonlocal current
		
		btn = event.button  # 1 is left, 3 is right
		try:
			xy = [float(event.xdata), float(event.ydata)]
		except:
			return
		
		if btn == 1:
			
			print(f'{current}: {xy}')
			
			if current in nodes:
				nodes[current]['pos'] = xy
			else:
				coasts[current]['pos'] = xy
				cst = coasts[current]['coast-of']
				if 'coast-pos' not in nodes[cst]:
					nodes[cst]['coast-pos'] = {}
				nodes[cst]['coast-pos'][coasts[current]['dir']] = xy
			
			_next_prompt()
		
		elif btn == 3:
			print('Going back')
			
			if current is not None:
				if 'pos' in nodes[current]:
					del nodes[current]['pos']
				todo.append(current)
				
			if len(done):
				
				current = done.pop()
				
				if 'pos' in nodes[current]:
					del nodes[current]['pos']
				
				todo.append(current)
			
			
		else:  # invalid button
			print(f'unknown button: {btn}')
			return
		
	
	cid = fg.canvas.mpl_connect('button_press_event', onclick)
	
	_next_prompt()
	
	plt.show(block=True)
	
	print(f'All node locs collected, saved to: {nodes_path}')
	save_yaml(nodes, nodes_path, default_flow_style=None)
	
	return nodes

if __name__ == '__main__':
	fig.entry('mapping')
