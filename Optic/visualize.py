import numpy as np
import cv2
import tensorflow as tf

# ______________________________________________________________
# HSV 	-> H = Hue : [0..360] of in cv2 [0..180]?
# 		-> S = Saturation : [0..1] (met 0 grijze pixels) [0..255]
# 		-> V = Value : [0..1] (brightness of pixel) [0..255]
# ______________________________________________________________
def visualize(opticflow,batchsize,image_size_x,image_size_y,filename="/users/start2012/r0298867/Thesis/implementation1/build_new/Optic/testfilename.bmp"):
	opticflow = opticflow[2,...]
	hsv = np.zeros([image_size_x, image_size_y, 3])
	hsv[...,1] = 255
	mag, ang = cv2.cartToPolar(opticflow[...,0], opticflow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	# datatype of hsv is float64, convert to float32 to avoid error
	# hsv = hsv.astype(np.float32)
	hsv = np.asarray(hsv, dtype= np.float32)
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	# imshow will display image on specified window after waitKey(x) -> bugs after showing first image
	# cv2.namedWindow('optic flow')
	# cv2.imshow('optic flow',bgr)
	# cv2.waitKey(10)
	cv2.imwrite(filename,bgr)
	return bgr


# ______________________________________________________________
# Test visualize, make color map
# -> Set hsv[...,1] = 1
# -> Set hsv[...,0] = ang*180/np.pi
# ______________________________________________________________

# _width = 224

# x = np.linspace(- _width/2, _width/2 - 1.0,  _width)
# y = np.linspace(- _width/2, _width/2 - 1.0,  _width)
# xv, yv = np.meshgrid(x, y)

# optic_visual = np.zeros([1,_width,_width,2])
# optic_visual[0,:,:,0] = xv
# optic_visual[0,:,:,1] = yv
# optic_visual /= _width

# test = visualize(optic_visual,1,_width,_width,filename='/users/start2012/r0298867/Thesis/implementation1/build_new/Optic/colorcodemap.bmp')
	
# hsv_test = np.zeros([255, 360, 3])
# hsv_test[...,1] = 1
# hue = np.linspace(0, 360 - 1.0,  360)
# value = np.linspace(0, 255 - 1.0, 255)
# hue, value = np.meshgrid(hue, value)
# print hue
# print value
# hsv_test[...,0] = hue
# hsv_test[...,2] = value
# hsv_test = np.asarray(hsv_test, dtype= np.float32)
# bgr = cv2.cvtColor(hsv_test,cv2.COLOR_HSV2BGR)
# filename = '/users/start2012/r0298867/Thesis/implementation1/build_new/Optic/colorcodemap.bmp'

# cv2.imshow('image', bgr)  
# cv2.waitKey(0)          
# cv2.destroyAllWindows() 

# cv2.imwrite(filename,bgr)
