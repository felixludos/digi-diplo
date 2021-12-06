import random

import discord
from omnibelt import load_yaml, load_txt, unspecified_argument
import omnifig as fig
from tabulate import tabulate

from .base import DiscordBot, as_command, as_event, as_loop
from .managers import ParsingFailedError

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
class DiplomacyBot(DiscordBot):
	def __init__(self, A, manager=unspecified_argument, **kwargs):
		if manager is unspecified_argument:
			manager = A.pull('manager', None)
		super().__init__(A, **kwargs)
		self.manager = manager
		if self.manager is None:
			print('No manager provided, but the bot has started')
		else:
			self.manager.load_status()
	
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
	
	
	@as_command('missing', brief='(admin) Prints out the number of missing commands per nation')
	async def on_fullstatus(self, ctx):
		status = self.manager.get_status()
		if not self._insufficient_permissions(ctx.author):
			await ctx.send('```' + tabulate(sorted(status.items(), key=lambda x: (-x[1], x[0])),
			                                headers=['Nation', 'Missing'])
			               + '```')
			# await ctx.send('\n'.join(f'{player}: {num}' for player, num in status.items()))
		return status
	

	@as_command('step', brief='(admin) Prints out the number of missing commands per nation')
	async def on_step(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		old = self.manager.format_date()
		self.manager.take_step(True)
		await ctx.send(f'Finished adjudicating {old}.')
		await ctx.send(f'Current turn: **{self.manager.format_date()}**')
		
	
	@as_command('render', brief='(admin) Renders game map and state (optionally with actions)')
	async def on_render(self, ctx, action=None):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		path = self.manager.render_latest(include_actions=bool(action))
		await ctx.send(file=discord.File(str(path)))
	
	async def _batched_send(self, ctx, lines):
		
		def _group_lines():
			total = 0
			batch = []
			for line in lines:
				total += len(line)
				if total > (1900 - len(batch)):
					yield batch
					batch.clear()
					total = 0
				batch.append(line)
			if len(batch):
				yield batch
		
		for batch in _group_lines():
			await ctx.send('\n'.join(batch))
	
	@as_command('print-state', brief='(admin) Prints out the player state (centers and units)')
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
	
	
	@as_command('print-actions', brief='(admin) Prints out the actions of the current season')
	async def on_print_actions(self, ctx, include_missing=False):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return

		# status = self.manager.get_status()
		# actions = self.manager.actions
		
		# full = []
		# for player, actions in self.manager.format_all_actions():
		# 	missing = status.get(player, 0)
		# 	lines = []
		# 	# lines.append(f'__{player}__')
		# 	# if len(actions):
		# 	# 	lines.extend(actions)
		# 	if self.manager.season == 3:
		# 		delta = self.manager.state.get('adjustments', {}).get(player, 0)
		# 		if delta > 0:
		# 			lines.append(f'- {delta} build/s deferred -')
		# 		elif delta < 0:
		# 			lines.append(f' - **Must still disband {-delta} units** -')
		# 		# else:
		# 		# 	lines.append(f' - no orders needed -')
		# 	elif self.re
		
		lines = self._line_table(self.manager.format_all_actions())
		
		await ctx.send(f'Orders for **{self.manager.format_date()}**')
		await self._batched_send(ctx, lines)


	@as_command('generate', brief='(admin) "[nation name]" [num]')
	async def on_random(self, ctx, player, num=1):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		actions = self.manager.sample_action(player, num)
		
		lines = [f'Generated {len(actions)} actions for {player}.']
		lines.extend(self.manager.format_action(player, action) for action in actions)
		await ctx.send('\n'.join(lines))
	
	@as_command('generate-all', brief='(admin) Generates all missing actions')
	async def on_random(self, ctx):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		num = 0
		for player in self.manager.actions:
			actions = self.manager.sample_action(player, -1)
			if actions is not None:
				num += len(actions)
		await ctx.send(f'Generated {num} actions for all players.')
		
	
	@as_command('set-order', brief='(admin) "[nation name]" [order]')
	async def on_order(self, ctx, player, *terms):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		assert player in self.manager.state['players'], f'Unknown nation: {player}'
		
		try:
			action = self.manager.record_action(player, ' '.join(terms))
		except ParsingFailedError as e:
			await ctx.send(f'Parsing the action failed: {type(e).__name__}: {str(e)}')
			# print(e)
			raise e
		else:
			await ctx.send(f'Recorded action for "{player}": {self.manager.format_action(player, action)}')
		