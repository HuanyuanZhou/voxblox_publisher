import numpy as np
import torch

from geometry_msgs.msg import Vector3, Quaternion, Transform, Point, Pose
from geometry_msgs.msg import TransformStamped
from sensor_msgs.point_cloud2 import PointCloud2, PointField


# ROS param functions
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# ROS data format transformation fucntions
def pixel_to_3D(depth, scale, K):
    u = range(0, depth.shape[1])
    v = range(0, depth.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    res = np.transpose(np.vstack((X, Y, Z)))
    
    return res


def pixel_to_3DRGB(depth, img, scale, K):
    u = range(0, depth.shape[1])
    v = range(0, depth.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    Z = depth.astype(np.float32) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    R = np.ravel(img[:, :, 0])[valid]
    G = np.ravel(img[:, :, 1])[valid]
    B = np.ravel(img[:, :, 2])[valid]

    C = np.zeros((X.shape[0], 4), dtype=np.uint8) + 255
    C[:, 0] = B
    C[:, 1] = G
    C[:, 2] = R

    C = C.view("uint32")

    res = np.zeros(
        (X.shape[0], 1), 
        dtype={"names":("x", "y", "z", "rgb"), "formats":("f4", "f4", "f4", "u4")}
    )
    res["x"] = X.reshape((-1, 1))
    res["y"] = Y.reshape((-1, 1))
    res["z"] = Z.reshape((-1, 1))
    res["rgb"] = C
    
    return res


def tq_to_TransformStamped(t, q, child_frame_id, stamp=None, frame_id=None, seq=None):
    transform = Transform(
        translation=t,
        rotation=q
    )

    msg = TransformStamped()

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq

    msg.child_frame_id = child_frame_id
    msg.transform = transform

    return msg


def xyz_array_to_pointcloud2(points, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq: 
        msg.header.seq = seq

    if len(points.shape) == 3:
        msg.height = points.shape[0]
        msg.width = points.shape[1]
    else:
        msg.height = 1
        msg.width = len(points)

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()

    return msg


def xyzrgb_array_to_pointcloud2(points, stamp=None, frame_id=None, seq=None):
    msg = PointCloud2()

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq: 
        msg.header.seq = seq

    if len(points.shape) == 3:
        msg.height = points.shape[0]
        msg.width = points.shape[1]
    else:
        msg.height = 1
        msg.width = points.shape[0]

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points.tostring()

    return msg


# cuda utils
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


import sys

def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()
