import os
import random

from omnibelt import get_printer
import omnifig as fig

from . import bot
from . import managers

prt = get_printer(__file__)

_import_worked = False
try:
	
	import discord
	from discord.ext import commands
	
	_import_worked = True

except ImportError:
	prt.warning('discord package missing')
else:
	from . import bot


def _start_bot(A):
	TOKEN = A.pull('disord-token', os.getenv('LUDOS_TOKEN'), silent=True)
	if TOKEN is None:
		raise Exception('No discord token found')
	
	A.push('client._type', 'discord-bot', silent=True, overwrite=False)
	client = A.pull('client')
	
	client.run(TOKEN)


# def _start_bot(A):
# 	TOKEN = A.pull('disord-token', os.getenv('LUDOS_TOKEN'))
# 	if TOKEN is None:
# 		raise Exception('No discord token found')
#
# 	bot = commands.Bot(command_prefix='.')
#
# 	@bot.event
# 	async def on_ready():
# 		print('Online')
#
# 	# await ctx.send('Message sent!')
#
# 	@bot.command(name='ping', description='Check if the bot is active and your permissions')
# 	async def ping(ctx):
# 		await ctx.send(f'Hello, {ctx.author.display_name}')
#
# 	bot.run(TOKEN)


if _import_worked:
	fig.Script('start-bot')(_start_bot)

