import re
from typing import Iterable, Union, Optional
from .errors import ParsingFailedError

# ----------------------------------------------------------------------------------
# 1) HELPER PATTERNS & SYNONYMS
# ----------------------------------------------------------------------------------

# Simple synonyms and patterns for units, locations, etc.
UNIT_PATTERN = r"(?:a|t|f|n|ar|tr|fl|na|army|troop|fleet|navy)"
LOCATION_PATTERN = r"[A-Za-z]{3}(?:-(?:nc|sc|ec|wc))?"

HOLD_SYNONYMS = r"(?:hold|holds|h|hd|holding)"
MOVE_SYNONYMS = r"(?:move|moves|m|mv|moving)"
TO_SYNONYMS    = r"(?:to|->|-|>)"
SUPPORT_SYNONYMS = r"(?:support|supports|su|s|sp|supp)"
CONVOY_SYNONYMS  = r"(?:convoy|convoys|convey|c|conv|convoying)"
BUILD_SYNONYMS   = r"(?:build|builds|b|bd|building|bldg)"
DISBAND_SYNONYMS = r"(?:disband|disbands|d|db|disb|disbanding)"
RETREAT_SYNONYMS = r"(?:retreat|retreats|r|rt|retreating)"

# A small convenience: optionally capture "in|at" between unit & location
# e.g., "army in mun" or "fleet at lon"
IN_AT_OPT = r"(?:\s+(?:in|at))?"

A_AN_OPT = r"(?:\s+(?:a|an))?"

# If you want parentheses or brackets around location, you might do:
#   LOCATION_WRAPPED = rf"[\(\[]?\s*{LOCATION_PATTERN}\s*[\)\]]?"
LOCATION_WRAPPED = LOCATION_PATTERN  # (simpler version here)


# ----------------------------------------------------------------------------------
# 2) MULTIPLE PATTERNS PER ORDER TYPE
# ----------------------------------------------------------------------------------

#
# 2.1 HOLD ORDERS
#
HOLD_PATTERN_A = rf"""^
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
(?:\s+(?P<hold_verb>{HOLD_SYNONYMS}))?
$"""

HOLD_PATTERN_B = rf"""^
(?P<hold_verb>{HOLD_SYNONYMS})
\s+
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
$"""

HOLD_PATTERNS = [
	re.compile(HOLD_PATTERN_A, re.IGNORECASE | re.VERBOSE),
	re.compile(HOLD_PATTERN_B, re.IGNORECASE | re.VERBOSE),
]

#
# 2.2 MOVE ORDERS
#   We'll allow something like:
#     "[unit loc] move [unit loc]" or "[unit loc] -> [unit loc]"
#     "move [unit loc] to [unit loc]"
#
MOVE_PATTERN_A = rf"""^
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
\s+
((?P<move_verb>{MOVE_SYNONYMS})
\s+)?
(?P<dest>{LOCATION_WRAPPED})
$"""

MOVE_PATTERN_B = rf"""^
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
\s+
(?:{TO_SYNONYMS})
\s+
(?P<dest>{LOCATION_WRAPPED})
$"""

MOVE_PATTERNS = [
	re.compile(MOVE_PATTERN_A, re.IGNORECASE | re.VERBOSE),
	re.compile(MOVE_PATTERN_B, re.IGNORECASE | re.VERBOSE),
]

#
# 2.3 SUPPORT (MOVE) ORDERS
#   e.g. "F lon supports A wal to lvp", "lon sup wal -> lvp"
#   or "support f bre moves pic"
#
SUPPORT_MOVE_PATTERN_A = rf"""^
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
\s+
(?P<sup_verb>{SUPPORT_SYNONYMS})
\s+
((?P<tunit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<target>{LOCATION_WRAPPED})
\s+
((?:{TO_SYNONYMS})
\s+)?
(?P<dest>{LOCATION_WRAPPED})
$"""

SUPPORT_MOVE_PATTERN_B = rf"""^
((?P<sup_verb>{SUPPORT_SYNONYMS})
\s+)?
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED}):?
\s+
((?P<tunit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<target>{LOCATION_WRAPPED})
\s+
((?:{TO_SYNONYMS})
\s+)?
(?P<dest>{LOCATION_WRAPPED})
$"""

SUPPORT_MOVE_PATTERNS = [
	re.compile(SUPPORT_MOVE_PATTERN_A, re.IGNORECASE | re.VERBOSE),
	re.compile(SUPPORT_MOVE_PATTERN_B, re.IGNORECASE | re.VERBOSE),
]

#
# 2.4 SUPPORT (HOLD) ORDERS
#   e.g., "F lon support hold lvp", "lon s h lvp"
#
SUPPORT_HOLD_PATTERN_A = rf"""^
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
\s+
(?P<sup_verb>{SUPPORT_SYNONYMS})
(\s*
(?:hold|h|holds|holding))?
\s+
(?P<dest>{LOCATION_WRAPPED})
$"""

SUPPORT_HOLD_PATTERN_B = rf"""^
(?P<sup_verb>{SUPPORT_SYNONYMS})
(\s*
(?:hold|h|holds|holding))?
\s+
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
\s+
((?:hold|h|holds|to hold)
\s+)?
(?P<dest>{LOCATION_WRAPPED})
$"""

SUPPORT_HOLD_PATTERNS = [
	re.compile(SUPPORT_HOLD_PATTERN_A, re.IGNORECASE | re.VERBOSE),
	re.compile(SUPPORT_HOLD_PATTERN_B, re.IGNORECASE | re.VERBOSE),
]

#
# 2.5 CONVOY ORDERS
#   e.g. "[unit loc] convoy [tunit location] to [target loc]"
#
CONVOY_PATTERN_A = rf"""^
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
\s+
(?P<cv_verb>{CONVOY_SYNONYMS})
\s+
((?P<tunit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<target>{LOCATION_WRAPPED})
\s+
((?:{TO_SYNONYMS})
\s+)?
(?P<dest>{LOCATION_WRAPPED})
$"""

CONVOY_PATTERN_B = rf"""^
(?P<cv_verb>{CONVOY_SYNONYMS})
\s+
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED}):?
:?\s+
((?P<tunit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<target>{LOCATION_WRAPPED})
\s+
((?:{TO_SYNONYMS})
\s+)?
(?P<dest>{LOCATION_WRAPPED})
$"""

CONVOY_PATTERNS = [
	re.compile(CONVOY_PATTERN_A, re.IGNORECASE | re.VERBOSE),
	re.compile(CONVOY_PATTERN_B, re.IGNORECASE | re.VERBOSE),
]

#
# 2.6 BUILD ORDERS
#   e.g. "mun build", "build f mun", "army at par b"
#
BUILD_PATTERN_A = rf"""^
(?P<loc>{LOCATION_WRAPPED})
\s+
((?P<build_verb>{BUILD_SYNONYMS})
{A_AN_OPT}?
\s+)?
(?P<unit>{UNIT_PATTERN})
$
"""

BUILD_PATTERN_B = rf"""^
((?P<build_verb>{BUILD_SYNONYMS})
\s+)?
(?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+
(?P<loc>{LOCATION_WRAPPED})
$
"""

BUILD_PATTERN_C = rf"""^
(?P<loc>{LOCATION_WRAPPED})
\s+
(?P<unit>{UNIT_PATTERN})
\s+
(?P<build_verb>{BUILD_SYNONYMS})
$
"""

BUILD_PATTERN_D = rf"""^
(?P<build_verb>{BUILD_SYNONYMS})
\s+
(?P<loc>{LOCATION_WRAPPED})
{A_AN_OPT}?
\s+
(?P<unit>{UNIT_PATTERN})
$
"""


BUILD_PATTERNS = [
	re.compile(BUILD_PATTERN_A, re.IGNORECASE | re.VERBOSE),
	re.compile(BUILD_PATTERN_B, re.IGNORECASE | re.VERBOSE),
	re.compile(BUILD_PATTERN_C, re.IGNORECASE | re.VERBOSE),
	re.compile(BUILD_PATTERN_D, re.IGNORECASE | re.VERBOSE),
]

#
# 2.7 DISBAND ORDERS
#
DISBAND_PATTERN_A = rf"""^
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
\s+
(?P<disband_verb>{DISBAND_SYNONYMS})
$
"""

DISBAND_PATTERN_B = rf"""^
((?P<disband_verb>{DISBAND_SYNONYMS})
\s+)?
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
$
"""

DISBAND_PATTERNS = [
	re.compile(DISBAND_PATTERN_A, re.IGNORECASE | re.VERBOSE),
	re.compile(DISBAND_PATTERN_B, re.IGNORECASE | re.VERBOSE),
]

#
# 2.8 RETREAT ORDERS
#   e.g. "retreat f bel to hol", "r a mun hol"
#
RETREAT_PATTERN_A = rf"""^
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
\s+
(((?P<rverb>{RETREAT_SYNONYMS})
\s+)?
(?:{TO_SYNONYMS})
\s+)?
(?P<dest>{LOCATION_WRAPPED})
$"""

RETREAT_PATTERN_B = rf"""^
(?P<rverb>{RETREAT_SYNONYMS})
\s+
((?P<unit>{UNIT_PATTERN})
{IN_AT_OPT}?
\s+)?
(?P<loc>{LOCATION_WRAPPED})
\s+
((?:{TO_SYNONYMS})
\s+)?
(?P<dest>{LOCATION_WRAPPED})
$"""

RETREAT_PATTERNS = [
	re.compile(RETREAT_PATTERN_A, re.IGNORECASE | re.VERBOSE),
	re.compile(RETREAT_PATTERN_B, re.IGNORECASE | re.VERBOSE),
]

# ----------------------------------------------------------------------------------
# 3) A DICTIONARY MAPPING ORDER TYPE -> LIST OF COMPILED PATTERNS
# ----------------------------------------------------------------------------------
ORDER_PATTERNS = {
	"hold": HOLD_PATTERNS,
	"move": MOVE_PATTERNS,
	"support_move": SUPPORT_MOVE_PATTERNS,
	"support_hold": SUPPORT_HOLD_PATTERNS,
	"convoy": CONVOY_PATTERNS,
	"build": BUILD_PATTERNS,
	"disband": DISBAND_PATTERNS,
	"retreat": RETREAT_PATTERNS,
}

ACTION_PATTERNS = {
	"hold": HOLD_PATTERNS,
	"move": MOVE_PATTERNS,
	"support_move": SUPPORT_MOVE_PATTERNS,
	"support_hold": SUPPORT_HOLD_PATTERNS,
	"convoy": CONVOY_PATTERNS,
}

GAIN_PATTERNS = {
	"build": BUILD_PATTERNS,
}

LOSE_PATTERNS = {
	"disband": DISBAND_PATTERNS,
}

RETREAT_PATTERNS = {
	"retreat": RETREAT_PATTERNS,
	"disband": DISBAND_PATTERNS,
}

ORDER_GROUPS = {
	'action': ['hold', 'move', 'support_move', 'support_hold', 'convoy'],
	'gain': ['build'],
	'lose': ['disband'],
	'retreat': ['retreat', 'disband'],
}


def standardize_order(order_info: dict[str, Optional[str]]) -> dict[str, Optional[str]]:
	"""
	Standardizes the order information by removing leading/trailing whitespace and converting to lowercase.

	:param order_info: A dictionary containing order information.
	:return: A standardized dictionary with cleaned values.
	"""

	unit_types = {
			'f': 'fleet', 'a': 'army', 'n': 'fleet', 't': 'army',
			'ar': 'army', 'tr': 'army', 'fl': 'fleet', 'na': 'fleet',
			'army': 'army', 'troop': 'army', 'fleet': 'fleet', 'navy': 'fleet',
		}

	fixed = {}
	for key, value in order_info.items():
		if key.endswith('unit') and value is not None:
			assert value in unit_types
			fixed[key] = unit_types.get(value.lower(), value.lower())
		else:
			fixed[key] = value

	return fixed



def parse_order(order_text: str, group: Union[str,Iterable[str]],
				order_patterns: dict[str,Iterable[re.Pattern]] = None) -> dict[str,dict[str, Optional[str]]]:
	"""
	Parses a given order text string into its components using the specified patterns.

	`group` should be in ['action', 'gain', 'lose', 'retreat'] or a list of valid order types.

	`order_patterns` is a dictionary mapping all order types to their respective compiled regex patterns, this defaults
	to the ORDER_PATTERNS dictionary (containing all order types of the classic game).

	:return: A dictionary mapping order types to their respective parsed components.
	"""
	if isinstance(group, str):
		assert group in ORDER_GROUPS, f"Group '{group}' not found in ORDER_GROUPS: {ORDER_GROUPS.keys()}"
		group = ORDER_GROUPS[group]
	if order_patterns is None:
		order_patterns = ORDER_PATTERNS

	text = order_text.strip()

	patterns = {key: order_patterns[key] for key in group}

	matches = {}
	for order_type, pats in patterns.items():
		for pat in pats:
			match = pat.match(text)
			if match:
				matches[order_type] = match.groupdict()
	if matches:
		return matches

	raise ParsingFailedError(f"Could not parse order: '{order_text}'.")













