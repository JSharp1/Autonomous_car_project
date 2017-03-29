# Thank you to Vivek Yadav for his super write up
# https://medium.com/self-driving-cars/image-augmentation-bc75fd02a0ff
import cv2
import numpy as np

PATH = 'datasets/data2/0001.jpg'

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


# def trans_image(image,steer,trans_range):
#     # Translation
#     tr_x = trans_range*np.random.uniform()-trans_range/2
#     steer_ang = steer + tr_x/trans_range*2*.2
#     tr_y = 40*np.random.uniform()-40/2
#     #tr_y = 0
#     Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
#     image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
#     return image_tr,steer_ang

def main():
	while(1):
		src = cv2.imread(PATH)
		cv2.imshow("img", src)
		aug1 = augment_brightness_camera_images(src)
		cv2.imshow("brightness", aug1)
		aug2 = add_random_shadow(src)
		cv2.imshow("shadow", aug2)

		cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
