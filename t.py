from collections import Counter
import random
from matplotlib import pyplot as plt


# n = 1000000000
# c = Counter(random.randint(0, 112 - 48) for _ in range(n))
# for i in range(1,11):
#     print ('%2s  %02.10f%%' % (i, c[i] * 100.0 / n))

# n = 10000
# a = [random.gauss((112-48)//2, (112-48)//6) for _ in range(n)]
# b = [i for i in a if i > (112-48)]
# print(b)
# plt.hist(a, bins=100)
# plt.show()

import cv2
import numpy as np

img1 = cv2.imread('/home/intern2/test_dir/ComputerVision/FaceAntispoofing_3d/public/CASIA-SURF/Training/fake_part/CLKJ_AS0005/04_en_b.rssdk/depth/131.jpg', 1)
img2 = cv2.imread('/home/intern2/test_dir/ComputerVision/FaceAntispoofing_3d/public/CASIA-SURF/Training/real_part/CLKJ_AS0005/real.rssdk/depth/101.jpg', 1)
# fimg = open('/home/intern2/test_dir/ComputerVision/FaceAntispoofing_3d/public/CASIA-SURF/Training/fake_part/CLKJ_AS0005/04_en_b.rssdk/depth/131.jpg')
# fimg.dtype
# exit()
# print(img.shape)
# print(img[..., 0])
# print(np.array_equal(img[..., 0], img[..., 1]))
# print(img.dtype)
# img[..., 1] == img[..., 0] == img[..., 2]
np.savetxt('depth_casia_surf_fake.txt', img1[..., 0], fmt='%.1f')
np.savetxt('depth_casia_surf_real.txt', img2[..., 0], fmt='%.1f')
