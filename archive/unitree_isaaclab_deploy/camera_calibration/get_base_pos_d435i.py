import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

tree = ET.parse('go2_with_d435i.urdf')
root = tree.getroot()

# 读取 front_camera joint (base→front_camera)
for j in root.findall('joint'):
    if j.get('name') == 'front_camera_joint':  # base→front_camera
        origin = j.find('origin')
        xyz1 = list(map(float, origin.get('xyz').split()))
        rpy1 = list(map(float, origin.get('rpy', '0 0 0').split()))

# 读取 d435i_joint (front_camera→d435i_link)
for j in root.findall('joint'):
    if j.get('name') == 'd435i_joint':
        origin = j.find('origin')
        xyz2 = list(map(float, origin.get('xyz').split()))
        rpy2 = list(map(float, origin.get('rpy', '0 0 0').split()))

# 合并变换
T1 = np.eye(4)
T1[:3, 3] = xyz1
T1[:3, :3] = Rotation.from_euler('xyz', rpy1).as_matrix()
T2 = np.eye(4)
T2[:3, 3] = xyz2
T2[:3, :3] = Rotation.from_euler('xyz', rpy2).as_matrix()
T = T1 @ T2

xyz_final = T[:3, 3]
q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
print(f'pos=({xyz_final[0]:.6f}, {xyz_final[1]:.6f}, {xyz_final[2]:.6f}),')
print(f'rot=({q[3]:.6f}, {q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}),  # wxyz')
