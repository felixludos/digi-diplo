import random
import traceback
import discord
from omnibelt import load_yaml, load_txt, unspecified_argument, save_yaml
import omnifig as fig
from tabulate import tabulate
from tqdm import tqdm

from .base import DiscordBot, as_command, as_event, as_loop
from ..managers import ParsingFailedError
from ..util import hash_file, Versioned

#
# class ParsingFailedError(Exception):
# 	def __init__(self, line, reason=None, num=None):
# 		info =f'"{line}"' if reason is None else f'{reason}: "{line}"'
# 		if num is not None:
# 			info = f'({num}) {info}'
# 		super().__init__(info)



# class UnknownIdentError(Exception):
# 	def __init__(self, ident, related=None, player=None, unit=None):
# 		super().__init__(f'{ident} - {related} - {player} - {unit}')
#
#
#
# class SpecialActionError(Exception):
# 	pass



@fig.Component('diplomacy-bot')
class DiplomacyBot(Versioned, DiscordBot):
	__version__ = (1, 0)
	def __init__(self, A, manager=unspecified_argument, intents=None, **kwargs):
		if manager is unspecified_argument:
			manager = A.pull('manager', None)
		
		bot_data_path = A.pull('bot-data-path', None)
		
		if intents is None:
			intents = discord.Intents.default()
			intents.members = True
			
		super().__init__(A, intents=intents, **kwargs)
		self.manager = manager
		if self.manager is None:
			print('No manager provided, but the bot has started')
		else:
			self.manager.load_status()
		
		if bot_data_path is None:
			bot_data_path = self.manager.root / 'bot-data.yaml'
		
		self.bot_data_path = bot_data_path
	
	async def on_ready(self):
		await super().on_ready()
		self._load_bot_data(self.bot_data_path)
		
	
	@staticmethod
	def _find_discord_objects(base, options, silent=False):
		missing = []
		matches = {}
		for name, player in base.items():
			if '#' in name:
				ident, num = name.split('#')
				match = discord.utils.get(options, name=ident, discriminator=num)
			else:
				match = discord.utils.get(options, name=name)
			if match is None:
				missing.append(name)
			else:
				matches[match] = player
		if not silent and len(missing):
			print('WARNING: missing {}'.format(
				', '.join(f'{user} ({base[user]})' for user in missing)))
		return matches
	
	def _load_bot_data(self, path):
		self.persistent = load_yaml(path) if path.exists() else {}
		if 'players' not in self.persistent:
			self.persistent['players'] = {}
		if 'roles' not in self.persistent:
			self.persistent['roles'] = {}
		if 'channels' not in self.persistent:
			self.persistent['channels'] = {}

		self.player_users = self._find_discord_objects(self.persistent['players'], self.guild.members)
		if len(self.player_users):
			print(f'Found {len(self.player_users)} members that are players.')
			
		self.player_roles = self._find_discord_objects(self.persistent['roles'], self.guild.roles)
		if len(self.player_roles):
			print(f'Found {len(self.player_roles)} roles for players.')
		
		self.player_channels = self._find_discord_objects(self.persistent['channels'], self.guild.channels)
		if len(self.player_channels):
			print(f'Found {len(self.player_channels)} channels for player orders.')
		
	
	def _store_bot_data(self):
		save_yaml(self.persistent, self.bot_data_path)
		
		
	@as_command('version', brief='(admin) Print out bot/map version info')
	async def on_version(self, ctx): # hash the game files to make sure they are correct
		
		info = {}
		root = self.manager.root
		itr = tqdm(list(root.glob('*')))
		for path in itr:
			if not path.is_dir() and path != self.bot_data_path:
				itr.set_description(f'Hashing {path.stem}')
				info[path.stem] = hash_file(path)
		
		lines = []
		if isinstance(self, Versioned):
			lines.append(f'Bot: {self.manager.get_version()}')
		if isinstance(self.manager, Versioned):
			lines.append(f'Manager: {self.manager.get_version()}')
		if isinstance(self.manager.gamemap, Versioned):
			lines.append(f'Map: {self.manager.gamemap.get_version()}')
		if isinstance(self.manager.renderer, Versioned):
			lines.append(f'Renderer: {self.manager.renderer.get_version()}')
		
		lines.extend(f'{name}: {info[name][-5:].upper()}' for name in sorted(info.keys()))
		
		print('\n'.join(lines))
		await self._batched_send(ctx, lines)
		
	
	# @as_command('find-orders', brief='(admin) Checks orders channel of players for new orders')
	async def on_find_orders(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		raise NotImplementedError
	
	# @as_command('designate-channel', brief='(admin) Designate a channel for a player to submit orders')
	async def on_designate_channel(self, ctx, channel, player, force=False):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		if len(ctx.message.channel_mentions):
			channel = ctx.message.channel_mentions[0]
		else:
			await ctx.send(f'Channel "{channel}" not found (mention an existing channel)')
			return
		
		if player not in self.manager.state['players']:
			await ctx.send(f'Unknown player: {player}')
			return
		
		if channel in self.player_channels and not force:
			await ctx.send(f'{channel.mention} is already used by {self.player_channels[channel]} (use force to replace)')
			return
		
		raise NotImplementedError
		
		self.player_channels[channel] = player
		self.persistent['channels'][str(channel)] = player
		self._store_bot_data()
		await ctx.send(f'{channel.mention} is used by {player}.')
		
	
	@as_command('designate-player', brief='(admin) Designate a user for a player')
	async def on_designate_player(self, ctx, user, player, force=False):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		if len(ctx.message.mentions):
			user = ctx.message.mentions[0]
		else:
			await ctx.send(f'User "{user}" not found (mention an existing member)')
			return

		if player not in self.manager.state['players']:
			await ctx.send(f'Unknown player: {player}')
			return
		
		if user in self.player_users and not force:
			await ctx.send(f'{user.mention} is already playing {self.player_users[user]} (use force to replace)')
			return
		
		self.player_users[user] = player
		self.persistent['players'][str(user)] = player
		self._store_bot_data()
		await ctx.send(f'{user.mention} is now playing {player}.')
	
	
	@as_command('designate-role', brief='(admin) Designate a role for a player')
	async def on_designate_role(self, ctx, role, player, force=False):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		if len(ctx.message.role_mentions):
			role = ctx.message.role_mentions[0]
		else:
			await ctx.send(f'Role "{role}" not found (mention an existing role)')
			return
		
		if player not in self.manager.state['players']:
			await ctx.send(f'Unknown player: {player}')
			return
		
		if role in self.player_roles and not force:
			await ctx.send(f'{role.mention} is already used for {self.player_roles[role]} (use force to replace)')
			return
		
		self.player_roles[role] = player
		self.persistent['roles'][str(role)] = player
		self._store_bot_data()
		await ctx.send(f'{role.mention} is now designated for {player}.')
	
	
	@as_command('status', brief='Prints out the current season and number of missing commands')
	async def on_status(self, ctx):
		status = self.manager.get_status()
		if self._insufficient_permissions(ctx.author):
			user = str(ctx.author)
			if user in self.users and self.users[user] in status:
				await ctx.send(f'Missing {status[self.users[user]]}/{self.manager.units[self.users[user]]} commands.')
			else:
				await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return

		await ctx.send(f'Current turn: **{self.manager.format_date()}**')
		await ctx.send(f'Missing {sum(num for num in status.values())} commands.')
		return status
	
	@as_command('season', brief='Prints out the current season')
	async def on_season(self, ctx):
		await ctx.send(f'Current turn: **{self.manager.format_date()}**')
	
	
	@as_command('missing', brief='(admin) Prints out the number of missing commands per nation')
	async def on_fullstatus(self, ctx):
		status = self.manager.get_status()
		if not self._insufficient_permissions(ctx.author):
			await ctx.send('```' + tabulate(sorted(status.items(), key=lambda x: (-x[1], x[0])),
			                                headers=['Nation', 'Missing'])
			               + '```')
			# await ctx.send('\n'.join(f'{player}: {num}' for player, num in status.items()))
		return status

	@as_command('order-format', brief='(admin) Print out format for orders')
	async def on_order_format(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		rules = self.manager.action_format()
		
		lines = [f'{name.capitalize()} order: "{format}"' for name, format in rules.items()]
		await self._batched_send(ctx, lines)
		

	@as_command('region', brief='Prints out info about a specific territory')
	async def on_region(self, ctx, name=None):
		
		info = self.manager.check_region(name)
		
		if info is None:
			await ctx.send('You must provide a valid region name to check.')
			return
	
		
		node = info['node']
		
		lines = [''.join(['__', '{base}'.format(**info) + '*'*node.get('sc', 0), '__'])]
		
		if 'home' in info:
			home = 'Home of {home}'.format(**info)
			if 'capital' in info:
				home += ' (capital)'
			lines.append(home)
		
		if 'control' in info:
			owner = 'Controlled by **{control}**'.format(**info)
			lines.append(owner)
		
		if 'disband' in info:
			coast = self.manager.gamemap.decode_region_name(info['disband']['loc'])[1]
			unit = '{demo} *{utype}*'.format(
				demo=self.manager.get_demonym(info['disband']['player']),
				utype=info['disband']['type'])
			if isinstance(coast, str):
				unit += f' ({coast})'
			unit += ' disbanding'
			lines.append(unit)
		
		if 'retreat' in info:
			coast = self.manager.gamemap.decode_region_name(info['retreat']['loc'])[1]
			unit = '{demo} *{utype}*'.format(
				demo=self.manager.get_demonym(info['retreat']['player']),
				utype=info['retreat']['type'])
			if isinstance(coast, str):
				unit += f' ({coast})'
			unit += ' in retreat'
			lines.append(unit)
		
		if 'unit' in info:
			coast = self.manager.gamemap.decode_region_name(info['unit']['loc'])[1]
			unit = 'Occupied by: **{demo}** *{utype}*'.format(
				demo=self.manager.get_demonym(info['unit']['player']),
				utype=info['unit']['type'])
			if isinstance(coast, str):
				unit += f' ({coast})'
			lines.append(unit)
		else:
			lines.append('- Not occupied -')
		
		await self._batched_send(ctx, lines)
		
	

	@as_command('step', brief='(admin) Adjudicates current season and updates game state')
	async def on_step(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		old = self.manager.format_date()
		print(f'Adjudicating: {self.manager.format_date()}')
		self.manager.take_step(True)
		await ctx.send(f'Finished adjudicating {old}.')
		await ctx.send(f'Current turn: **{self.manager.format_date()}**')
		
	
	@as_command('render-state', brief='(admin) Renders game map and state')
	async def on_render_state(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return

		print(f'Rendering state: {self.manager.format_date()}')
		path = self.manager.render_latest(include_actions=False)
		await ctx.send(file=discord.File(str(path)))
	
	
	@as_command('render-orders', brief='(admin) Renders game map with current orders')
	async def on_render_actions(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		path = self.manager.render_latest(include_actions=True)
		await ctx.send(file=discord.File(str(path)))
	
	
	@as_command('print-state', brief='(admin) Prints out the game state (centers and units)')
	async def on_print_state(self, ctx, player=None):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return

		state = self.manager.format_state(player=player)
		
		if player is not None:
			state = {player: state}
		else:
			await ctx.send(f'State of **{self.manager.format_date()}**')
			
		lines = self._line_table(state)
		await self._batched_send(ctx, lines)
	
	def _line_table(self, info):
		lines = []
		for title, ls in info.items():
			if len(ls):
				lines.append(f'__{title}__')
				lines.extend(ls)
		return lines
	
	
	@as_command('print-orders', brief='(admin) Prints out the orders of the current season')
	async def on_print_actions(self, ctx, player=None):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		actions = self.manager.format_all_actions()
		
		if player is not None:
			actions = {player: actions[player]}
		else:
			await ctx.send(f'Orders for **{self.manager.format_date()}**')
		
		lines = self._line_table(actions)
		await self._batched_send(ctx, lines)


	@as_command('generate', brief='(admin) Generate N orders for the given player')
	async def on_random(self, ctx, player, num=1):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		actions = self.manager.sample_action(player, num)
		
		lines = [f'Generated {len(actions)} actions for {player}.']
		lines.extend(self.manager.format_action(player, action) for action in actions)
		await ctx.send('\n'.join(lines))
	
	
	@as_command('generate-all', brief='(admin) Generates all missing actions')
	async def on_random_all(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		num = 0
		for player in self.manager.actions:
			actions = self.manager.sample_action(player, -1)
			if actions is not None:
				num += len(actions)
		await ctx.send(f'Generated {num} actions for all players.')
		
	
	@as_command('set-order', brief='(admin) Submit orders for a given player (1 per line)')
	async def on_set_order(self, ctx, player, *, lines):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		if player not in self.manager.state['players']:
			await ctx.send(f'Unknown player: {player}')
			return
		
		results = []
		num = 0
		for line in lines.splitlines():
			line = line.strip()
			if len(line):
				try:
					action = self.manager.record_action(player, line)
				except ParsingFailedError as e:
					results.append(f'"{line}" failed with {type(e).__name__}: {str(e)}')
					print(traceback.format_exc())
					# raise e
				else:
					num += 1
					results.append(f'{self.manager.format_action(player, action)}')
		
		results = [f'Recorded {num} action/s for {player}:', *results]
		await self._batched_send(ctx, results)
		
	
	def _to_player(self, member):
		if member in self.player_users:
			return self.player_users[member]
		for role in member.roles:
			if role in self.player_roles:
				return self.player_roles[role]


	@as_command('order', brief='(player) Submit order/s (1 per line)')
	async def on_order(self, ctx, *, lines):
		player = self._to_player(ctx.author)
		if player is None:
			await ctx.send(f'{ctx.author.display_name} is not a player (admins should use `.set-order`).')
			return

		await self.on_set_order(ctx, player, lines=lines)
		
