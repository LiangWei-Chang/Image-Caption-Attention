from PIL import Image
import os
from tqdm import tqdm

def resize_image(image):
	width, height = image.size
	if width > height:
		left = (width - height) / 2
		right = width - left
		top = 0
		bottom = height
	else:
		top = (height - width) / 2
		bottom = height - top
		left = 0
		right = width

	image = image.crop((left, top, right, bottom))
	image = image.resize([224, 224], Image.ANTIALIAS)
	return image

def main():
	splits = ['train', 'val']
	for split in splits:
		folder_path = './dataset/%s2014' % split
		resize_folder_path = './dataset/%s2014_resized/' % split
		if not os.path.exists(resize_folder_path):
			os.makedirs(resize_folder_path)
		print('Start to resize %s images' % split)

		image_files = os.listdir(folder_path)
		for i, image_file in tqdm(enumerate(image_files)):
			with open(os.path.join(folder_path, image_file), 'r+b') as f:
				with Image.open(f) as image:
					image = resize_image(image)
					image.save(os.path.join(resize_folder_path, image_file), image.format)

if __name__ == '__main__':
	main()
