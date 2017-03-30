# Thank you to Vivek Yadav for his super write up
# https://medium.com/self-driving-cars/image-augmentation-bc75fd02a0ff
import cv2
import numpy as np
import math

PATH = 'datasets/data2/0001.jpg'
new_size_col,new_size_row = 64, 64

def augment_brightness_camera_images(image):
	image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	image1 = np.array(image1, dtype = np.float64)
	random_bright = .5+np.random.uniform()
	image1[:,:,2] = image1[:,:,2]*random_bright
	image1[:,:,2][image1[:,:,2]>255]  = 255
	image1 = np.array(image1, dtype = np.uint8)
	image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
	return image1


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()

    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    #tr_y = 40*np.random.uniform()-40/2
    tr_y = 0
    rows, cols, channels, = image.shape
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang


def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    #image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)    
    #image = image/255.-.5
    return image


# def preprocess_image_file_train(line_data):
#     image = cv2.imread(path_file)
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     image,y_steer,tr_x = trans_image(image,y_steer,100)
#     image = augment_brightness_camera_images(image)
#     image = preprocessImage(image)
#     image = np.array(image)
#     ind_flip = np.random.randint(2)
#     if ind_flip==0:
#         image = cv2.flip(image,1)
#         y_steer = -y_steer
#     return image,y_steer


def main():
	while(1):
		src = cv2.imread(PATH)
		cv2.imshow("img", src)
		aug1 = augment_brightness_camera_images(src)
		cv2.imshow("brightness", aug1)
		aug2 = add_random_shadow(src)
		cv2.imshow("shadow", aug2)
		# print src.shape
		aug3, angle = trans_image(src,20,40)
		cv2.putText(aug3, str(angle),(0,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
		cv2.imshow("translate", aug3)
		cv2.imshow("preprocess", preprocessImage(src))
		cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
