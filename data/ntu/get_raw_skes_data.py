# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import logging


def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger):
    """
    Get raw bodies data from a skeleton sequence.

    Each body's data is a dict that contains the following keys:
      - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
      - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
      - interval: a list which stores the frame indices of this body.
      - motion: motion amount (only for the sequence with 2 or more bodyIDs).

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """
    ske_file = osp.join(skes_path, ske_name + '.skeleton')
    assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
    # Read all data from .skeleton file into a list (in string format)
    print('Reading data from %s' % ske_file[-29:]) # 看到的形式是：Reading data from S017C003P020R002A060.skeleton
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()

    num_frames = int(str_data[0].strip('\r\n')) # 表示帧数
    frames_drop = []
    bodies_data = dict() # bodies的数据要放到 raw_skes_data里面
    valid_frames = -1  # 0-based index
    current_line = 1

    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n')) # 表明有多少个身体
        current_line += 1

        if num_bodies == 0:  # no data in this frame, drop it
            frames_drop.append(f)  # 0-based index # 这一帧没有数据 删除掉这一帧的数据 表明这里有缺失值
            continue

        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32) # num_bodies个身体，每个身体有25个关节，每个关节有3个坐标 关节矩阵
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32) # num_bodies个身体，每个身体有25个关节，每个关节有2个color坐标 色素矩阵

        for b in range(num_bodies):
            bodyID = str_data[current_line].strip('\r\n').split()[0] # 第一个是bodyID
            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
            current_line += 1

            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                current_line += 1

            if bodyID not in bodies_data:  # Add a new body's data
                body_data = dict()
                body_data['joints'] = joints[b]  # ndarray: (25, 3)
                body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2) 为什么要增加一个维度？ np.newaxis是增加一个维度 为了能够让维度统一
                body_data['interval'] = [valid_frames]  # the index of the first frame
            else:  # Update an already existed body's data
                body_data = bodies_data[bodyID]
                # Stack each body's data of each frame along the frame order
                body_data['joints'] = np.vstack((body_data['joints'], joints[b])) # n*3列
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis])) # 
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index # 这个身体一共承载了多少帧

            bodies_data[bodyID] = body_data  # Update bodies_data

    num_frames_drop = len(frames_drop)
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)
    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=np.int)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                    frames_drop))

    # Calculate motion (only for the sequence with 2 or more bodyIDs)
    if len(bodies_data) > 1:
        for body_data in bodies_data.values(): # 多人运动的话 给合并成一个 使用求和的方式
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0)) # 按照列求平均 然后再求和 这一部分改进？？？

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data():
    # # save_path = './data'
    # # skes_path = '/data/pengfei/NTU/nturgb+d_skeletons/'
    # stat_path = osp.join(save_path, 'statistics')
    #
    # skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    # save_data_pkl = osp.join(save_path, 'raw_skes_data.pkl')
    # frames_drop_pkl = osp.join(save_path, 'frames_drop_skes.pkl')
    #
    # frames_drop_logger = logging.getLogger('frames_drop')
    # frames_drop_logger.setLevel(logging.INFO)
    # frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'frames_drop.log')))
    # frames_drop_skes = dict()

    skes_name = np.loadtxt(skes_name_file, dtype=str)

    num_files = skes_name.size
    print('Found %d available skeleton files.' % num_files)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=np.int) # 创造一个56578的文件 

    for (idx, ske_name) in enumerate(skes_name):
        bodies_data = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger) # 读取每一个文件的内容 skes_path路径 ske_name文件名 frame_drop的文件名 frames_drop_logger的日志
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames'] # 这个代表的是可用的帧
        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d)' % \
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    save_path = './'

    skes_path = './nturgb+d_skeletons/' # 找到对应的骨骼数据文件
    stat_path = osp.join(save_path, 'statistics') # 将读取的相关信息保存在statistics文件夹里面
    if not osp.exists('./raw_data'):
        os.makedirs('./raw_data')

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt') # 可用的文件名
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl') # 保存的data文件
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl') # frame_drop文件

    frames_drop_logger = logging.getLogger('frames_drop') # 打印相关的log内容
    frames_drop_logger.setLevel(logging.INFO) # 记录的是info的信息
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'raw_data', 'frames_drop.log'))) # 将记录的日志放到文件当中
    frames_drop_skes = dict() # 生成一个对象

    get_raw_skes_data()

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)
        
