import random
import traceback
import discord
from discord.ext import commands as command_util
from omnibelt import load_yaml, load_txt, unspecified_argument, save_yaml, get_now
import omnifig as fig
from tabulate import tabulate
from tqdm import tqdm

from .base import DiscordBot, as_command, as_event, as_loop
from ..managers import ParsingFailedError
from ..util import hash_file, Versioned


@fig.Component('diplomacy-bot')
class DiplomacyBot(Versioned, DiscordBot):
	__version__ = (1, 1)
	def __init__(self, A, manager=unspecified_argument, intents=None,
	             private_commands=None, log_commands=None, **kwargs):
		if manager is unspecified_argument:
			manager = A.pull('manager', None)
		
		bot_data_path = A.pull('bot-data-path', None)
		
		if private_commands is None:
			private_commands = A.pull('private-commands', False)
		
		if log_commands is None:
			log_commands = A.pull('log-commands', False)
		
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
		
		bot_log_path = self.manager.root / 'bot-log.txt' if log_commands else None
		self.bot_log_path = bot_log_path
		if log_commands:
			print(f'Will log all commands to {str(self.bot_log_path)}')
		
		self.private_commands = private_commands
		if private_commands:
			print('Players are only allowed to run commands in their private channels.')
		
		self.frozen = False
	
	
	# @command_util.Cog.listener(name='on_command')
	@as_event
	async def on_command(self, ctx):
		if self.bot_log_path is not None and ctx.guild == self.guild:
			line = f'[ {get_now()} ] {ctx.guild.name} #{ctx.channel} @{ctx.author}: {repr(ctx.message.clean_content)}\n'
			with self.bot_log_path.open('a+') as f:
				f.write(line)
			print(line)
		
	
	async def on_ready(self):
		await super().on_ready()
		self._load_bot_data(self.bot_data_path)
		
		if self.bot_log_path is not None:
			lines = ['',
			         f'[ {get_now()} ] Starting digi-diplo bot in server {self.guild}. Game: {str(self.manager.root)}',
			         *self._get_version_lines(pbar=False),
			         '']
			with self.bot_log_path.open('a+') as f:
				f.write('\n'.join(lines))
			
		
	@as_command('shutdown', brief='(admin) Shuts down the bot')
	async def shutdown(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		await ctx.send('Goodbye!')
		await self.close()
		
	
	@as_command('reload', brief='(admin) Reload state and orders from files')
	async def on_reload(self, ctx, name=None):
		print('Reloading state and orders')
		self.manager.load_status(name=name)
		await ctx.send(f'Loaded **{self.manager.format_date()}**')
	
	
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
		
	def _get_version_lines(self, pbar=True):
		
		info = {}
		root = self.manager.root
		itr = list(root.glob('*'))
		if pbar:
			itr = tqdm(itr)
		for path in itr:
			if not path.is_dir() and path != self.bot_data_path:
				if pbar:
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

		return lines
	
	@as_command('version', brief='(admin) Print out bot/map version info')
	async def on_version(self, ctx): # hash the game files to make sure they are correct
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		lines = self._get_version_lines()
		print('\n'.join(lines))
		await self._batched_send(ctx, lines)
		
	_magic_stop_scan_char = '\u2800' # used to prevent the scanning from reusing past orders
	
	@as_command('scan-orders', brief='(admin) Checks orders channel of players for new orders')
	async def on_find_orders(self, ctx, limit=100):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		lines = []
		
		itr = tqdm(self.player_channels.items())
		for channel, player in itr:
			itr.set_description(f'Scanning {channel.name} for {player}')
			messages = []
			for message in await channel.history(limit=limit).flatten():
				if self._magic_stop_scan_char in message.clean_content:
					break
				messages.append(message)
				
			included = []
			total = 0
			
			for message in reversed(messages):
				if message.clean_content.startswith('.order') and self._to_player(message.author) == player:
					raw = message.clean_content.replace('.order', '\n')
					results = self._register_orders(player, raw.splitlines())
					
					num = int(results[0].split('Recorded ')[-1].split(' action')[0])
					total += num
					included.extend(results[1:])
				
				if self._magic_stop_scan_char in message.clean_content:
					break

			lines.append(f'Found {total} new actions in {channel.mention} ({player}).')
			lines.extend(included)
			included = [f'Recorded {total} new actions for {player}.', *included]
			included[-1] = included[-1] + self._magic_stop_scan_char
			await self._batched_send(channel, included)
			
		
		if len(lines):
			if ctx not in self.player_channels:
				await self._batched_send(ctx, lines)
		else:
			await ctx.send('No new orders found in ' + ', '.join(f'{channel.mention}'
			                                                     for channel in self.player_channels))
			
	
	def _designate_mentions(self, mentions, player, table, persistent, force=False):
		lines = []
		for ref in mentions:
			if ref in table and not force:
				lines.append(
					f'{ref.mention} is already used by {table[ref]} (use force to replace)')
			# return
			else:
				table[ref] = player
				persistent[str(ref)] = player
				lines.append(f'{ref.mention} is now associated with {player}.')
		
		self._store_bot_data()
		return lines
		
	
	@as_command('designate-channel', brief='(admin) Designate a channel for a player to submit orders')
	async def on_designate_channel(self, ctx, player, *channels, force=False):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return

		player = self.manager.fix_player(player)
		if player not in self.manager.state['players']:
			await ctx.send(f'Unknown player: {player}')
			return
		
		lines = self._designate_mentions(ctx.message.channel_mentions, player,
		                                 self.player_channels, self.persistent['channels'], force=force)
		await self._batched_send(ctx, lines)
		
	
	@as_command('designate-player', brief='(admin) Designate a user for a player')
	async def on_designate_player(self, ctx, player, *users, force=False):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return

		player = self.manager.fix_player(player)
		if player not in self.manager.state['players']:
			await ctx.send(f'Unknown player: {player}')
			return
		
		lines = self._designate_mentions(ctx.message.mentions, player,
		                                 self.player_users, self.persistent['players'], force=force)
		await self._batched_send(ctx, lines)
		
	
	@as_command('designate-role', brief='(admin) Designate a role for a player')
	async def on_designate_role(self, ctx, player, *roles, force=False):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return

		player = self.manager.fix_player(player)
		if player not in self.manager.state['players']:
			await ctx.send(f'Unknown player: {player}')
			return
		
		lines = self._designate_mentions(ctx.message.role_mentions, player,
		                                 self.player_roles, self.persistent['roles'], force=force)
		await self._batched_send(ctx, lines)
	
	
	def _format_missing(self, player, todo):
		if isinstance(todo, int):
			return ' '.join([abs(todo), 'disbands' if todo < 0 else 'builds'])
		elif isinstance(todo, list):
			return 'Units: ' + ', '.join(todo)
		
		# retreats/disbands
		clauses = []
		if 'retreats' in todo:
			clauses.append('Retreats: ' + ', '.join(todo['retreats']))
		if 'disbands' in todo:
			clauses.append('Disbands: ' + ', '.join(todo['disbands']))
		return '. '.join(clauses)
	
	
	@as_command('status', brief='Prints out the current season and missing orders')
	async def on_status(self, ctx, player=None):
		admin = self._is_admin(ctx.author)
		name = self._to_player(ctx.author)
		if name is None and not admin:
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		if self.frozen:
			await ctx.send('The bot is currently frozen, ask the admins to unfreeze to run commands.')
			return
		
		if admin and player is not None:
			name = player

		status = self.manager.get_missing()
		missing = {player: abs(missing) if isinstance(missing, int)
			             else (len(missing) if isinstance(missing, list)
			                   else sum(map(len, missing.values())))
			             for player, missing in status.items()}
		
		lines = [f'Current turn: **{self.manager.format_date()}**']
		
		if name is None:
			lines.append(f'Missing {sum(missing.values())} orders.')
		elif missing.get(name):
			lines.append(f'{name} is missing {missing[name]} orders.' if admin
			             else f'You ({name}) are missing {missing[name]} orders.')
			todo = status[name]
			lines.append(self._format_missing(name, todo))
		else:
			lines.append(f'{name} has submitted all orders.' if admin
			             else f'You ({name}) have submitted all orders.')
		
		await self._batched_send(ctx, lines)
		
	# @as_command('centers', brief='(admins) Prints out the currently occupied centers')
	async def on_centers(self, ctx):
		raise NotImplementedError
	
	
	@as_command('season', brief='Prints out the current season')
	async def on_season(self, ctx):
		if self.frozen:
			await ctx.send('The bot is currently frozen, ask the admins to unfreeze to run commands.')
			return
		
		await ctx.send(f'Current turn: **{self.manager.format_date()}**')
	
	
	@as_command('missing', brief='(admin) Prints out missing orders for all players')
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
		

	@as_command('step', brief='(admin) Adjudicates current season and updates game state')
	async def on_step(self, ctx, silent=False):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		old = self.manager.format_date()
		print(f'Adjudicating: {self.manager.format_date()}')
		self.manager.take_step(True)
		
		msg = f'Finished adjudicating {old}.{self._magic_stop_scan_char}\n'\
		      f'Current turn: **{self.manager.format_date()}**'
		
		if not silent:
			for channel in self.player_channels:
				if channel != ctx.channel:
					await channel.send(msg)
		await ctx.send(msg)
		
	
	@as_command('prompt', brief='(admin) Prompt all player channels to submit orders')
	async def on_prompt(self, ctx, deadline=None, ignore_above=False, mention_missing=True):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		lines = [f'Current turn: **{self.manager.format_date()}**']
		
		if deadline is not None:
			lines.append(f'You have {deadline} to submit any missing orders.')
		
		if ignore_above:
			lines.append(f'(note any orders above this message will be ignored, '
			             f'unless they have already been recorded by me, '
			             f'use `.submitted` to check){self._magic_stop_scan_char}')
		
		if mention_missing:
			status = self.manager.get_missing()
		
		for channel, player in self.player_channels.items():
			prompt = lines.copy()
			if mention_missing and player in status:
				objs = [role for role, p in self.player_roles.items() if p == player] \
				       + [user for user, p in self.player_users.items() if p == player]
				prompt.append('{} You are missing orders:'.format(', '.join(f'{obj.mention}' for obj in objs)))
				prompt.append(self._format_missing(player, status[player]))
			
			await self._batched_send(channel, prompt)
	
	
	@as_command('freeze', brief='(admin) Stop players from running commands (eg. while admins test/organize things)')
	async def on_freeze(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
	
		self.frozen = True
		await ctx.send('Bot is frozen (only admins can run commands, until unfrozen with `.unfreeze`)')
		
	
	@as_command('unfreeze', brief='(admin) Unfreeze the bot to allow players to run commands again')
	async def on_unfreeze(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return

		self.frozen = False
		await ctx.send('Bot is unfrozen (so players can run commands again)')
		
	
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
			actions = {player: actions[self.manager.fix_player(player)]}
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
		
		
	def _register_orders(self, player, lines):
		results = []
		num = 0
		for line in lines:
			line = line.strip()
			if len(line):
				try:
					action = self.manager.record_action(player, line)
				except Exception as e:
					results.append(f'"{line}" failed with {type(e).__name__}: {str(e)}')
					print(traceback.format_exc())
				# raise e
				else:
					num += 1
					results.append(f'{self.manager.format_action(player, action)}')
		
		return [f'Recorded {num} actions for {player}:', *results]
		
	
	@as_command('set-order', brief='(admin) Submit orders for a given player (1 per line)')
	async def on_set_order(self, ctx, player, *, lines):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		if self.manager.fix_player(player) not in self.manager.state['players']:
			await ctx.send(f'Unknown player: {player}')
			return
		
		await self._batched_send(ctx, self._register_orders(player, lines.splitlines()))
	
	
	@as_command('region', brief='Prints out info about a specific territory')
	async def on_region(self, ctx, name=None):
		
		if self.frozen:
			await ctx.send('The bot is currently frozen, ask the admins to unfreeze to run commands.')
			return
		
		info = self.manager.check_region(name)
		
		if info is None:
			await ctx.send('You must provide a valid region name to check.')
			return
		
		node = info['node']
		
		lines = [''.join(['__', '{base}'.format(**info) + '*' * node.get('sc', 0), '__'])]
		
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
			unit = '{demo} {utype}'.format(
				demo=self.manager.get_demonym(info['disband']['player']),
				utype=info['disband']['type'])
			if isinstance(coast, str):
				unit += f' ({coast})'
			unit += ' disbanding'
			lines.append(unit)
		
		if 'retreat' in info:
			coast = self.manager.gamemap.decode_region_name(info['retreat']['loc'])[1]
			unit = '{demo} {utype}'.format(
				demo=self.manager.get_demonym(info['retreat']['player']),
				utype=info['retreat']['type'])
			if isinstance(coast, str):
				unit += f' ({coast})'
			unit += ' in retreat'
			lines.append(unit)
		
		if 'unit' in info:
			coast = self.manager.gamemap.decode_region_name(info['unit']['loc'])[1]
			unit = 'Occupied by: **{demo}** {utype}'.format(
				demo=self.manager.get_demonym(info['unit']['player']),
				utype=info['unit']['type'])
			if isinstance(coast, str):
				unit += f' ({coast})'
			lines.append(unit)
		else:
			lines.append('- Not occupied -')
		
		await self._batched_send(ctx, lines)
	
	
	def _to_player(self, member):
		if member in self.player_users:
			return self.player_users[member]
		for role in member.roles:
			if role in self.player_roles:
				return self.player_roles[role]

	@as_command('ping', brief='Pings the bot')
	async def on_ping(self, ctx):
		msg = [f'Hello, {ctx.author.display_name}']
		player = self._to_player(ctx.author)
		if player is not None:
			msg.append(f'({player})')
		if self._is_admin(ctx.author):
			msg.append('(admin)')
		await ctx.send(' '.join(msg))


	@as_command('order', brief='(player) Submit order/s (1 per line)')
	async def on_order(self, ctx, *, lines):
		player = self._to_player(ctx.author)
		if player is None:
			await ctx.send(f'{ctx.author.display_name} is not a player (admins should use `.set-order`).')
			return

		if self.private_commands and self.player_channels.get(ctx.channel) != player:
			await ctx.send(f'You can only run this command in the channel designated for your orders.')
			return
		
		if self.frozen:
			await ctx.send('The bot is currently frozen, ask the admins to unfreeze to run commands.')
			return
		
		await self._batched_send(ctx, self._register_orders(player, lines.splitlines()))
	
	
	@as_command('submitted', brief='(player) Prints out submitted orders for this season')
	async def on_submitted(self, ctx):
		
		player = self._to_player(ctx.author)
		
		if player is None:
			await ctx.send(f'{ctx.author.display_name} is not a player (admins should use `.print-orders`).')

		if self.private_commands and self.player_channels.get(ctx.channel) != player:
			await ctx.send(f'You can only run this command in the channel designated for your orders.')
			return

		if self.frozen:
			await ctx.send('The bot is currently frozen, ask the admins to unfreeze to run commands.')
			return
		
		actions = self.manager.format_all_actions()
		lines = [f'{player} orders for **{self.manager.format_date()}**', *actions[player]]
		await self._batched_send(ctx, lines)
