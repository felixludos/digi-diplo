
class ParsingFailedError(Exception):
	pass


class AmbiguousOrderError(ParsingFailedError):
	pass

class InvalidActionError(ParsingFailedError):
	pass


class UnknownUnitTypeError(ParsingFailedError):
	pass


class MissingCoastError(ParsingFailedError):
	pass


class NoUnitFoundError(ParsingFailedError):
	pass


class LocationError(ParsingFailedError):
	pass


class BadGraphError(Exception):
	pass


class BadNamesError(BadGraphError):
	def __init__(self, names):
		super().__init__('These region names are ambiguous: {}'.format(', '.join(names)))

