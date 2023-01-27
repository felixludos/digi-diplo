import random
import discord

from omnibelt import get_printer, unspecified_argument
import omnifig as fig

from .compat import OmniBot, as_command, as_event, as_loop


class AdminError(Exception):
	def __init__(self, name):
		super().__init__(f'{name} doesn')



# @fig.Component('discord-bot')
class DiscordBot(OmniBot):
	def __init__(self, server_name=None, admins=(), admin_role=None,
	             intents=unspecified_argument, seed=None, **kwargs):
		
		if intents is unspecified_argument:
			intents = discord.Intents.default()
			intents.members = True
		
		super().__init__(intents=intents, **kwargs)
		
		self.admin_role = admin_role
		self.admins = admins
		
		if admin_role is None:
			print('WARNING: it is recommended that you specify an admin role.')
			if admins is None:
				print('WARNING: no admins specified, so everyone is treated as an admin!')
			else:
				self.admins = set(admins)
				if len(admins):
					print(f'Enabling {len(self.admins)} admin/s: {self.admins}')
				# if self.admin_role is not None:
					# print(f'All users with the role: {self.admin_role} are treated as admins')
		else:
			print(f'Treating all users with the role {admin_role} as admins.')
		
		self._rng = random.Random()
		if seed is not None:
			self._rng.seed(seed)
		
		self._server_name = server_name
	
	_char_limit = 1500
	
	@classmethod
	async def _batched_send(cls, ctx, lines, char_lim=None):
		
		if char_lim is None:
			char_lim = cls._char_limit
		
		def _group_lines():
			total = 0
			batch = []
			for line in lines:
				total += len(line)
				if total > (char_lim - len(batch)):
					yield batch
					batch.clear()
					total = 0
				batch.append(line)
			if len(batch):
				yield batch
		
		for batch in _group_lines():
			await ctx.send('\n'.join(batch))
	
	def _is_admin(self, user):
		return (self.admin_role is None and (self.admins is None or str(user) in self.admins)
					or (self.admin_role is not None and self.admin_role in user.roles))
	
	def _insufficient_permissions(self, user, permission=None):
		return not self._is_admin(user)
	
	
	# @as_command('ping', brief='Pings the bot')
	# async def on_ping(self, ctx):
	# 	role = ' (admin)' if self._is_admin(ctx.author) else ''
	# 	await ctx.send(f'Hello, {ctx.author.display_name}{role}')
	
	
	async def on_ready(self):
		print(f'Logged on as {self.user}')
		
		self._status = 'No game running.'
		
		self.guild = self.guilds[0] if self._server_name is None else discord.utils.get(self.guilds,
		                                                                                name=self._server_name)
	
		if self.admin_role is not None:
			self.admin_role = discord.utils.get(self.guild.roles, name=self.admin_role)

