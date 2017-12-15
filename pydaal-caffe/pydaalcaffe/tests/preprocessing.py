def crop_center(img,cropx,cropy):
	y = img.shape[1]
	x = img.shape[2]

	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)
	
	return img[:,starty:starty+cropy,startx:startx+cropx]