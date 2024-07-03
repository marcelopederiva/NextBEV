import numpy as np
import matplotlib.pyplot as plt



img = np.zeros((256,256,1))


img[:3,:3,:] = 1

plt.imshow(img)
plt.show()
exit()