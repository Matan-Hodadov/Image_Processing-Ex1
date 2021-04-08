# Image_Processing-Ex1

Python_ver: 3.8  

Platform: PyCharm

## Files used: 
### ex1_main
Main class of the program.  
Used to see the the program works fine and helps with the understading of the process in it.  
The main function also call the helping functions which plot the images before and after the image processing

### ex1_utils:
The class which contain all the processing functions.  

This class have the following main functions:  
imReadAndConvert(filename: str, representation: int) - convert image path and a given representation(grayScale or RGB) into a np.ndarray image  
imDisplay(filename: str, representation: int) - plot the image to the screen in the wanted representation  
transformRGB2YIQ(imgRGB: np.ndarray) - transform an RGB image into a YIQ image  
transformYIQ2RGB(imgYIQ: np.ndarray) - transform an YIQ image into a RGB image  
hsitogramEqualize(imgOrig: np.ndarray) - return a tuple of (the image after equalization, the histogram of the original image, the histogram of the image after equalization)   
quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) - return a list of the image of each iteration and a list of the error given each iteration  

There are more helping functions in this class for the quantizeImage function:  
find_new_q(z: np.array, image_hist: np.ndarray) - finding the new q for the q cacl in each iteration  
find_new_z(q: np.array) - finding the new z for the z cacl in each iteration  
findCenters(orig_hist: np.ndarray, num_colors: int, n_iter: int) - holds z,q's values after the new cacl and add them to list. return both lists  
convertToImg(imOrig: np.ndarray, histOrig: np.ndarray, yiq_im: np.ndarray, arrayQuantize: np.ndarray) - return the new image after the iteration and the error with it  

### gamma:
This class represent the gamma changes in the images.  
In this class we plot an image and change the gamma (brightness) in the image  

The main function in the class:  
gammaDisplay(img_path: str, rep: int) - this function plot the image with the gamma bar and change the gamma in the image by the changes we do in the bar  

There is an helping function to the main gammaDisplay function:  
adjust_gamma(org_image: np.ndarray, gamma: float) - this function return the new image after given the new gamma value
