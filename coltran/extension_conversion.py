from PIL import Image
import glob, os

directory = r'D:\videos_cloudy'

for filename in glob.glob(directory+"\**\*"):
    if filename.endswith(".tif"):
        im = Image.open(filename)
        name = filename[:-4]+'.jpeg'
        im.save(name)
        print(os.path.join(directory, filename))
        continue
    else:
        continue

for filename in glob.glob(directory+"\**\*"):
    if filename.endswith(".tif"):
        os.remove(filename)
        continue
    else:
        continue