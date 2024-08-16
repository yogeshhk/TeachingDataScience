from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte
from skimage import io
import numpy as np


image = img_as_float(io.imread("monalisa_noisy.jpg")).astype(np.float32)

# estimate the noise standard deviation from the noisy image
sigma_est = np.mean(estimate_sigma(image, multichannel=True))

print("estimated noise standard deviation = {}".format(sigma_est))

#Define a dictionary for the input parameters to NLM algorithm.
patch_kw = dict(patch_size=10,      # 5x5 patches
                patch_distance=3,  # 13x13 search area
                multichannel=True)


denoise_img = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
#The denoise image is float 64 type, so we need to convert to 8 byte 
#for desktop viewing. 

denoise_img_as_8byte = img_as_ubyte(denoise_img)

#Save the output file to current directory
io.imsave("NLM.jpg", denoise_img_as_8byte)