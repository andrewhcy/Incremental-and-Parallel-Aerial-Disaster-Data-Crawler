import os
import cv2
import imghdr

data_dir = 'Data'
image_exts = ['jpeg','jpg', 'bmp', 'png']
counter = 0
for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
                continue
            os.rename(image_path, "Data/" + str(counter) + "." + tip)
            counter += 1
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)