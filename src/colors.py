import webcolors
import numpy as np
from skimage.segmentation import flood_fill

def fill_region(img, loc, val=None, threshold=0.1, dist='rmse', make_copy=False):
	
	px = img[loc]
	
	if val is None:
		val = px
	
	if dist == 'rmse':
		diff = np.sqrt(((img - px.reshape(1, 1, -1)) ** 2).sum(-1))
	else:
		diff = (np.abs(img - img[loc].reshape(1, 1, -1))).sum(-1)
	
	mask = flood_fill(diff, loc, -1, tolerance=threshold) < 0
	
	if make_copy:
		img = img.copy()
	img[mask] = val
	return img


def process_color(raw):
	full = webcolors.HTML5SimpleColor(*raw) if isinstance(raw, (list,tuple)) \
		else webcolors.html5_parse_legacy_color(raw)
	return webcolors.rgb_to_hex(full)

def rgb_to_hex(raw):
	full = webcolors.html5_parse_legacy_color(raw)
	return webcolors.rgb_to_hex(full)

def hex_to_rgb(raw):
	if isinstance(raw, str):
		full = webcolors.html5_parse_legacy_color(raw)
		return [full.red, full.green, full.blue]
	return raw


def greyit(color, pull=0.5, base=0.1):
	c = webcolors.html5_parse_legacy_color(color)
	
	vals = (c.red, c.green, c.blue)
	
	if base is not None:
		vals = [max(v, base) for v in vals]
	
	avg = sum(vals) / 3
	
	assert 0 < pull <= 1, pull
	
	vals = [max(0, min(int(pull * avg + (1 - pull) * v), 255)) for v in vals]
	
	return webcolors.rgb_to_hex(webcolors.HTML5SimpleColor(*vals))


def dimmer(color, dim=0.3):
	c = webcolors.html5_parse_legacy_color(color)
	
	vals = (c.red, c.green, c.blue)
	
	assert 0 < dim <= 1, dim
	
	vals = [int(v * (1 - dim)) for v in vals]
	
	return webcolors.rgb_to_hex(webcolors.HTML5SimpleColor(*vals))

def lighter(color, light=0.3):
	c = webcolors.html5_parse_legacy_color(color)
	
	vals = (c.red, c.green, c.blue)
	
	assert 0 < light <= 1, light
	
	vals = [int(255*light + v * (1 - light)) for v in vals]
	
	return webcolors.rgb_to_hex(webcolors.HTML5SimpleColor(*vals))


def light_shade(raw):
	return dimmer(raw, 0.2)


def dark_shade(raw):
	return dimmer(raw, 0.4)

# '#{:02x}{:02x}{:02x}'.format(*rgb) for rgb in rgbs])

def gen_colors(N=None): # 269
	raw = [
        "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        
        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
        "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",

        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
        
        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]
	return raw if N is None else raw[:N]

def gen_colors2(N=None): # 64
	raw = ['#000000', '#010067', '#d5ff00', '#ff0056', '#9e008e', '#0e4ca1', '#ffe502', '#005f39',
	       '#00ff00', '#95003a', '#ff937e', '#a42400', '#001544', '#91d0cb', '#620e00', '#6b6882',
	       '#0000ff', '#007db5', '#6a826c', '#00ae7e', '#c28c9f', '#be9970', '#008f9c', '#5fad4e',
	       '#ff0000', '#ff00f6', '#ff029d', '#683d3b', '#ff74a3', '#968ae8', '#98ff52', '#a75740',
	       '#01fffe', '#ffeee8', '#fe8900', '#bdc6ff', '#01d0ff', '#bb8800', '#7544b1', '#a5ffd2',
	       '#ffa6fe', '#774d00', '#7a4782', '#263400', '#004754', '#43002c', '#b500ff', '#ffb167',
	       '#ffdb66', '#90fb92', '#7e2dd2', '#bdd393', '#e56ffe', '#deff74', '#00ff78', '#009bff',
	       '#006401', '#0076ff', '#85a900', '#00b917', '#788231', '#00ffc6', '#ff6e41', '#e85ebe']
	return raw if N is None else raw[:N]

