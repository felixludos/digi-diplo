import sys, os
from omnibelt import primitives
import omnifig as fig
from pydip.player.unit import UnitTypes


class Versioned(fig.Configurable):
	__version__ = (0,0)
	
	@classmethod
	def get_version(cls):
		return '.'.join(map(str,cls.__version__)) if isinstance(cls.__version__, tuple) else cls.__version__


UNIT_TYPES = {
	'army': UnitTypes.TROOP,
	'fleet': UnitTypes.FLEET,
}
UNIT_ENUMS = {v:k for k,v in UNIT_TYPES.items()}

COAST_NAMES = {
	'ec': 'Eastern Coast',
	'sc': 'Southern Coast',
	'wc': 'Western Coast',
	'nc': 'Northern Coast',
}

def get_map_paths(A, *keys):
	
	if len(keys):
		
		root = A.pull('root', None)
		name = A.pull('name', None)
		
		paths = []
		
		for key in keys:
			path = A.pull(f'{key}-path', None)
			if path is None:
				path = f'{key}.yaml' if name is None else f'{name}_{key}.yaml'
			if root is not None:
				path = os.path.join(root, path)
			paths.append(path)
		
		if len(paths) == 1:
			return paths[0]
		return paths
	

def is_coast(name):
	return name.endswith('-c') or name.endswith('-ec') or name.endswith('-sc') \
		or name.endswith('-wc') or name.endswith('-nc')

def convert_to_coast(ID):
	return ID if is_coast(ID) else f'{ID}-c'
	
def fix_loc(loc, utype, ltype='coast'):
	utype = UNIT_ENUMS[utype] if utype in UNIT_ENUMS else utype
	return convert_to_coast(loc) if ltype == 'coast' and utype == 'fleet' else loc

def make_node_dictionary(nodes):
	names = {node['name']: ID for ID, node in nodes.items()}
	
	coasts = separate_coasts(nodes, dir_only=True)
	names.update({coast['name'].replace('thern', 'th'): ID for ID, coast in coasts.items()})
	
	names['Mid-Atlantic Ocean'] = 'mao'
	names['Western Mediterranean'] = 'wes'
	names['Eastern Mediterranean'] = 'eas'
	names['North Sea'] = 'nth'
	names['Bulgaria North Coast'] = 'bul-ec'
	names['Tunis'] = 'tun'
	names['Yorkshire'] = 'yor'
	
	def locs(loc):
		loc = loc.replace('(', '').replace(')', '').replace('St ', 'St. ')
		return names.get(loc, loc)
	
	return locs

def separate_coasts(nodes, long_coasts=True, dir_only=False):
	coasts = {}
	
	for ID, node in nodes.items():
		name = node['name']
		if 'dirs' in node:
			for c in node['dirs']:  # add coasts as separate nodes
				cname = COAST_NAMES[c] if long_coasts else f'({c.upper()})'
				coasts[f'{ID}-{c}'] = {'name': f'{name} {cname}', 'type': 'coast', 'coast-of': ID, 'dir':c}
			# node['coasts'] = [f'{ID}-{c}' for c in node['coasts']]
	
		elif not dir_only and node['type'] == 'coast':
			coasts[f'{ID}-c'] = {'name': f'{name} Coast', 'type': 'coast', 'coast-of': ID}
			# node['coasts'] = [f'{ID}-c']
	
	return coasts


def list_edges(raw_edges, nodes=None):
	
	full = []
	
	for etype, edge_group in raw_edges.items():
		for s, es in edge_group.items():
			for e in es:
				full.append({'start': s, 'end': e, 'type': etype})
				
	return full


_unit_codes = {'army':'A', 'fleet':'F'}
def print_player(player):
	
	name = player['name']
	control = sorted(player['control'], key=lambda x: (x not in player['centers'], x))
	
	tiles = ', '.join(f'*{tile}' if tile in player['centers'] else tile for tile in control)
	
	units = ', '.join('{t}:[{l}]'.format(t=_unit_codes[unit['type']], l=unit['loc'],
									  s='*' if unit['loc'] in player['centers'] else '')
			 for unit in player['units'])
	
	scs = player['centers']
	
	print(f'{name} ({len(scs)}):')
	print(f'\tControl: {tiles}')
	print(f'\tUnits: {units}')

def _flatten_kwargs(tree):
	
	leaves = {}
	nodes = {}
	
	for name,node in tree.items():
		if node is None or isinstance(node, primitives):
			leaves[name] = node
		else:
			nodes[name] = _flatten_kwargs(node)
	
	for node in nodes.values():
		node.update(leaves)
	
	if len(nodes):
		return nodes
	return leaves

@fig.component('flatten-kwargs')
def flatten_kwargs(A):
	
	A.push('_type', None, silent=True)
	
	raw = A.pull_self()
	
	ignore_hidden = A.pull('_ignore_hidden', '<>skip-hidden', True)
	if ignore_hidden:
		raw = {k:v for k,v in raw.items() if not k.startswith('_')}
	
	flat = _flatten_kwargs(raw)
	
	return flat


import hashlib

_hashes = {'md5': hashlib.md5, 'sha1': hashlib.sha1}

def hash_file(path, hash='md5', buffer_size=65536):
	if hash in _hashes:
		hash = _hashes[hash]()
	
	with open(str(path), 'rb') as f:
		while True:
			data = f.read(buffer_size)
			if not data:
				break
			hash.update(data)
	return hash.hexdigest()

def str_conjunction(terms, delimiter=', ', conj='and', oxford=True):
	"""
	Converts a list of terms into a human-readable string.

	:param terms: The terms to be joined.
	:param delimiter: The delimiter to use between terms.
	:param conj: The conjunction to use before the last term.
	:param oxford: Whether to include the Oxford comma.
	:return: A human-readable string.
	"""

	if len(terms) == 1:
		return str(terms[0])
	elif len(terms) == 2:
		return f'{terms[0]} {conj} {terms[1]}'
	elif len(terms) > 2:
		if oxford:
			return f"{delimiter.join(terms[:-1])}{delimiter}{conj} {terms[-1]}"
		else:
			return f"{delimiter.join(terms[:-1])} {conj} {terms[-1]}"
