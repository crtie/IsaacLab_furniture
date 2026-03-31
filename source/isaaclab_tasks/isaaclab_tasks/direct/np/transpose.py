import numpy as np
from scipy.spatial.transform import Rotation as R


frame2_r = R.from_euler('xyz', [1.5707, -1.5707, 0.0])
print("Relative Rotation:\n", frame2_r.as_matrix())