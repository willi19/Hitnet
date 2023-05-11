############################################################################################################
# Visualization from raw data
#
# TODO:
# 3) Implement a function: camera No./camera lookat vec. -> image
#    Use cameras.txt(K)
# + Remove outlier
# + Dense reconstruction (Poisson)
# 
############################################################################################################

import open3d as o3d
import numpy as np
import sys

TEST_NUM = 0
FRONT = [ 0.14929731980286653, 0.010257306466837358, 0.98873914556050935 ]
LOOKAT = [ 0.33555091098447065, -0.46720913009533144, 0.030660489566716142 ]
UP = [ 0.047966274022422893, -0.99884408506696898, 0.0031193401763148004 ]

# Visualize with .ply
def test_ply():
    path = "./snu.ply"
    pcd = o3d.io.read_point_cloud(path, format='ply')
    print(pcd)
    coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0.0, 0.0, 0.0]))
    o3d.visualization.draw_geometries([pcd, coord], zoom=0.02, front=FRONT, lookat=LOOKAT, up=UP)



# get_pcd - general ver. (format: 'ply', 'xyz', 'xyzrgb' etc)
def get_pcd(path, format):
    if format == 'ply':
        pcd = get_pcd_ply(path)
    elif format == 'xyz':
        pcd = get_pcd_xyz(path)
    elif format == 'xyzrgb':
        pcd = get_pcd_xyzrgb(path)
    else:
        print("Error: No matching format")
        print("Possible format: 'ply', 'xyz', 'xyzrgb'")
        sys.exit(1)
    print(pcd)
    return pcd

def get_pcd_ply(path):
    pcd = o3d.io.read_point_cloud(path, format='ply')
    return pcd

def get_pcd_xyz(path):
    try:
        f = open(path, 'r')

        points = []
        while True:
            line = f.readline()
            if not line: break
            data = line.split()
            if data[0][0] == '#': continue
            xyz = list(map(float, data[0:3]))
            points.append(xyz)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        f.close()

    except IOError:
        print("IOError: Could not open file", path)
        sys.exit(1)

    return pcd

def get_pcd_xyzrgb(path):
    return

# Construct 3D point cloud geometries
def get_pcd_colmap():
    filename = f"./test{TEST_NUM}/points3D.txt"

    try:
        f = open(filename, 'r')
        for _ in range(3): f.readline()

        points, colors = [], []
        while True:
            line = f.readline()
            if not line: break
            data = line.split()
            xyz = list(map(float, data[1:4]))
            rgb = list(map(float, data[4:7]))
            points.append(xyz)
            colors.append(rgb)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors) / 255)
        f.close()

    except IOError:
        print("IOError: Could not open file", filename)
        sys.exit()

    print(pcd)
    return pcd


# Construct camera geometries from extrinsic parameters
# Hamilton convention (https://fzheng.me/2017/11/12/quaternion_conventions_en/)
def get_cameras():
    filename = f"./test{TEST_NUM}/images.txt"

    try:
        f = open(filename, 'r')
        for _ in range(4): f.readline()

        cameras = []
        while True:
            line = f.readline()
            if not line: break
            data = line.split()
            Q = list(map(float, data[1:5]))
            T = np.array(list(map(float, data[5:8])))
            R = np.array([[1-2*Q[2]*Q[2]-2*Q[3]*Q[3], 2*Q[1]*Q[2]-2*Q[0]*Q[3], 2*Q[1]*Q[3]+2*Q[0]*Q[2]],
                          [2*Q[1]*Q[2]+2*Q[0]*Q[3], 1-2*Q[1]*Q[1]-2*Q[3]*Q[3], 2*Q[2]*Q[3]-2*Q[0]*Q[1]],
                          [2*Q[1]*Q[3]-2*Q[0]*Q[2], 2*Q[2]*Q[3]+2*Q[0]*Q[1], 1-2*Q[1]*Q[1]-2*Q[2]*Q[2]]])
            
            
            points = []
            c = 0.15
            for X_cam in [[0, 0, 0], [0.5, 1, 1], [0.5, -1, 1], [-0.5, 1, 1], [-0.5, -1, 1]]:
                X_world = np.linalg.solve(R, c * np.array(X_cam) - T)   # R * X_world + T = X_cam
                points.append(X_world.tolist())

            lines = [[0, 1], [0, 2], [0, 3], [0, 4]]
            colors = [[1, 0, 0] for _ in range(len(lines))]
            camera = o3d.geometry.LineSet()
            camera.points = o3d.utility.Vector3dVector(points)
            camera.lines = o3d.utility.Vector2iVector(lines)
            camera.colors = o3d.utility.Vector3dVector(colors)
            cameras.append(camera)

            vertices = points[1:]
            triangles = [[0, 2, 1], [1, 2, 3], [0, 1, 2], [1, 3, 2]]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.paint_uniform_color([1, 0.3, 0.3])
            cameras.append(mesh)
            
            f.readline()
            
        f.close()

    except IOError:
        print("IOError: Could not open file", filename)
        sys.exit()

    return cameras


# Construct coordinate frame geometry
def get_coord(size=0.5):
    return o3d.geometry.TriangleMesh().create_coordinate_frame(size=size, origin=np.array([0.0, 0.0, 0.0]))



# Visualize 3D scene from raw data
def visualize():
    # Prepare geometries
    pcd = get_pcd()
    cameras = get_cameras()
    coord = get_coord()
    
    # Visualization
    o3d.visualization.draw_geometries([pcd, coord] + cameras, zoom=0.03, front=FRONT, lookat=LOOKAT, up=UP)


def get_pcd_test(filename):
    try:
        f = open(filename, 'r')

        points, colors = [], []
        while True:
            line = f.readline()
            if not line: break
            data = line.split()
            xyz = list(map(float, data[0:3]))
            rgb = list(map(float, data[3:6]))
            points.append(xyz)
            colors.append(rgb)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors) / 255)
        f.close()

    except IOError:
        print("IOError: Could not open file", filename)
        sys.exit()

    print(pcd)
    return pcd

def test():
    filename = "./point_cloud"
    pcd = get_pcd_test(filename)
    o3d.visualization.draw_geometries([pcd], zoom=1, front=FRONT, lookat=LOOKAT, up=UP)
    return

def main():
    TEST = True
    if TEST:
        test()
    else:
        #test_ply()
        visualize()

if __name__ == "__main__":
    main()