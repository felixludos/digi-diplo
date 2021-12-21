import sys, os
import numpy as np
import matplotlib.pyplot as plt
import omnifig as fig
from PIL import Image

@fig.Script('render-state')
def _render_state(A):
	
	manager = A.pull('manager', None)
	manager.load_status()
	
	check_image = A.pull('check-image', None)
	if check_image is not None:
		path = manager.images_root / f'{check_image}.png'
		if path.exists():
			os.remove(str(path))
	
	include_actions = A.pull('include-actions', False)
	path = manager.render_latest(include_actions=include_actions)
	
	print(f'Path: {str(path)}')
	
	# img = np.array(Image.open(str(path)))
	#
	# plt.imshow(img)
	# plt.axis('off')
	# plt.show()





