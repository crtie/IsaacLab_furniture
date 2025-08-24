# import rotation
import numpy as np
from scipy.spatial.transform import Rotation as R

frame1_t = np.array([-0.16,-0.15,0.74])
frame2_t = np.array([0.1444,-0.033,1.20])
frame1_r = R.from_euler('xzy', [-180, 0.0, -180], degrees=True)
frame2_r = R.from_euler('xzy', [0, 0, 0.0], degrees=True)

# compute the pose of frame2 in frame1's coordinate system
relative_translation_xzy = (frame2_t - frame1_t) @ frame1_r.as_matrix()
relative_translation = np.array([relative_translation_xzy[0], relative_translation_xzy[2], relative_translation_xzy[1]])
relative_rotation = frame2_r * frame1_r.inv()

print("Relative Rotation:\n", relative_rotation.as_matrix())
print("Relative Translation:", relative_translation_xzy)