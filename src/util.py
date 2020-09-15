
from pydip.player.unit import UnitTypes

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


