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
		self.manager.load_status()
	
	@as_command('status', brief='Prints out the missing commands')
	async def on_status(self, ctx):
		status = self.manager.get_status()
		if self._insufficient_permissions(ctx.author):
			user = str(ctx.author)
			if user in self.users and self.users[user] in status:
				await ctx.send(f'Missing {status[self.users[user]]}/{self.manager.units[self.users[user]]} commands.')
			else:
				await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return

		rmsg = ' (retreats)' if self.manager.retreat else ''
		await ctx.send(f'Current time: Year {self.manager.year} Season {self.manager.season}{rmsg}')
		await ctx.send(f'Missing {sum(num for num in status.values())} commands.')
		return status
	
	
	@as_command('missing', brief='(privileged) Prints out the number of missing commands per nation')
	async def on_fullstatus(self, ctx):
		status = self.manager.get_status()
		if not self._insufficient_permissions(ctx.author):
			await ctx.send('```' + tabulate(sorted(status.items(), key=lambda x: (-x[1], x[0])),
			                                headers=['Nation', 'Missing'])
			               + '```')
			# await ctx.send('\n'.join(f'{player}: {num}' for player, num in status.items()))
		return status
	

	@as_command('step')
	async def on_step(self, ctx):
		self.manager.take_step(True)
		rmsg = ' (retreats)' if self.manager.retreat else ''
		await ctx.send(f'Finished Adjudicated: Year {self.manager.year} Season {self.manager.season}{rmsg}')

	
	@as_command('test')
	async def on_test(self, ctx, *terms):
		print(ctx.author)
		print(ctx, test)
		
	
	@as_command('generate', brief='(privileged) "[nation name]" [num]')
	async def on_random(self, ctx, player, num=1):
		if self._insufficient_permissions(ctx.author):
			await ctx.send(f'{ctx.author.display_name} does not have sufficient permissions for this.')
			return
		
		actions = self.manager.sample_action(player, num)
		await ctx.send(f'Generated {len(actions)} actions for {player}.')
		for action in actions:
			await ctx.send(self.manager.format_action(player, action))
	
	
	@as_command('set-order', brief='(privileged) "[nation name]" [order]')
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
		