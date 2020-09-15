
from pydip.turn.resolve import _resolve, _init_resolution, CommandMap, compute_retreats



def resolve_turn(game_map, commands):
    command_map = CommandMap(game_map, commands)
    _init_resolution()
    resolutions = {command.unit.position: _resolve(game_map, command_map, command) for command in commands}
    return compute_retreats(game_map, command_map, commands, resolutions)





