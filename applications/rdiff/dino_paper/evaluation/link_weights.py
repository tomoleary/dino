import os

weights_dir = 'trained_weights/'
target_weights_dir = 'trained_weights/'

files_ = os.listdir(weights_dir)

files = []
for file in files_:
	if file.endswith('.pkl'):
		files.append(file)


ndatas = ['16','64','256','1024','4096']

for ndata in ndatas:
	os.makedirs(target_weights_dir+ndata,exist_ok=True)
	for file in files:
		if ndata in file:
			command = 'ln '+weights_dir+file+' '+target_weights_dir+ndata+'/'+file
			os.system(command)
