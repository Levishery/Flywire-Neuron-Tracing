import numpy as np

####################################################################
## Process image stacks.
####################################################################

def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def crop_volume(data, sz, st=(0, 0, 0)):  # C*D*W*H, C=1
    st = np.array(st).astype(np.int32)
    return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

def crop_volume_mul(data, sz, st=(0, 0, 0)):  # C*D*W*H, for multi-channel input
    return data[:, st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

def select_points(bbox, points):
    [[bx1, by1, bz1], [bx2, by2, bz2]] = bbox
    ll = np.array([bx1, by1, bz1])  # lower-left
    ur = np.array([bx2, by2, bz2])  # upper-right

    inidx = np.all(np.logical_and(ll <= points, points <= ur), axis=1)
    return inidx

def get_crop_index(center, half_patch_size, volume_size):
    x_bound = np.clip(np.asarray([center[0]-half_patch_size[0], center[0] + half_patch_size[0]+1]), 0, volume_size[0])
    y_bound = np.clip(np.asarray([center[1] - half_patch_size[1], center[1] + half_patch_size[1] + 1]), 0, volume_size[1])
    z_bound = np.clip(np.asarray([center[2] - half_patch_size[2], center[2] + half_patch_size[2] + 1]), 0, volume_size[2])
    return [x_bound, y_bound, z_bound]