import os
import random

from omnibelt import get_printer
import omnifig as fig

from . import bot

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
	TOKEN = A.pull('disord-token', os.getenv('DISCORD_TOKEN'), silent=True)
	if TOKEN is None:
		raise Exception('No discord token found (should be an environment variable "DISCORD_TOKEN" '
		                'or passed in using "--discord-token")')
	
	A.push('client._type', 'diplomacy-bot', silent=True, overwrite=False)
	client = A.pull('client')
	
	client.run(TOKEN)


if _import_worked:
	fig.Script('start-bot', description='Starts the discord bot')(_start_bot)

