"""
Isaac Berrios
4/9/2024

Stream data from .bag files in the RaDICAL Dataset: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9361086

This does not require ROS, it uses rosbags to decode the data
"""

import numpy as np
import rosbags
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
import matplotlib.pyplot as plt

from radar_config import read_radar_params
from ros_bridge import rosmsg_to_radarcube_v1


# ============================================================================
# set constants
TOPICS_MAP = {'radar':'/radar_data',
              'depth':'/camera/aligned_depth_to_color/image_raw',
              'rgb':'/camera/color/image_raw',
              'imu_accel':'/camera/accel/sample',
              'imu_gyro':'/camera/gyro/sample',
             }

NAMES_MAP = { v:k for (k, v) in TOPICS_MAP.items() }

TOPICS = list(TOPICS_MAP.keys())

# configure robags typestore to read .bag file
TYPESTORE = get_typestore(Stores.ROS1_NOETIC)

# get custom type store for Radar data
RADARINFO_DEFINITION = """ int16[] data """

RADAR_TYPE_NAME = 'mmWave/msg/data_frame'
TYPESTORE.register(get_types_from_msg(RADARINFO_DEFINITION, RADAR_TYPE_NAME),)

# ============================================================================
# helper functions to convert decoded ROS messages
# units ref: https://www.ros.org/reps/rep-0103.html
def rosmsg_to_depth(msg, endian='big'):
    """ Converts to uint16 depth image
        Inputs: 
            msg - Decoded ROS 1 Noetic message
            endian - endianess of the message ('big' or 'little')
        Outputs:
            img - uint16 depth image
    """
    endian = endian.lower()
    assert endian in ['big', 'little']

    h, w = msg.height, msg.width
    img = msg.data.reshape((-1, 2)) 
    
    # get uint8 bytes pairs
    img_lo = img[:, 0]
    img_hi = img[:, 1]

    # concat uint8 byte pairs to get uint16
    if endian == 'big':
        img = (img_hi.astype(np.uint16) << 8) | img_lo.astype(np.uint16)
    else:
        img = (img_lo.astype(np.uint16) << 8) | img_hi.astype(np.uint16)

    img = img.reshape((h, w))

    return img

def rosmsg_to_rgb(msg):
    """ Converts uint8 RGB image 
        Inputs: 
            msg - Decoded ROS 1 Noetic message
        Outputs:
            img - uint8 RGB image
    """
    h, w = msg.height, msg.width
    img = msg.data.reshape((h, w, 3))
    return img

def rosmsg_to_accel(msg):
    """ Convert all IMU accleration data from rosbag to numpy arrays 
        Inputs: 
            msg - Decoded ROS 1 Noetic message
        Outputs:
            accel - linear (x,y,z) acceleration components (m/sec^2)
            accel_cov - flattened (3x3) accel covariance matrix
    """
    # accel (m/sec^2 ?)
    accel = np.array([msg.linear_acceleration.x, 
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.x])
    accel_cov = msg.linear_acceleration_covariance

    return accel, accel_cov

def rosmsg_to_gyro(msg):
    """ Convert all IMU gyro data from rosbag to numpy arrays 
        Inputs: 
            msg - Decoded ROS 1 Noetic message
        Outputs:
            gyro - angular velocity (x,y,z) components (rad/sec)
            gyro_cov - flattened (3x3) accel covariance matrix
    """
    # gyro (rad/sec?)
    gyro = np.array([msg.angular_velocity.x, 
                     msg.angular_velocity.y,
                     msg.angular_velocity.z])
    gyro_cov = msg.angular_velocity_covariance

    return gyro, gyro_cov


# main geenerator to stream rosbag
def stream_rosbag(bag_path, radar_confg_path, 
                  *, 
                  typestore=TYPESTORE, 
                  topics=TOPICS,
                  max_idx=np.infty,
                  verbose=True):
    """
        Streams Radical ROS 1 Noetic BAG file

        NOTE: The '*' in args enforces keyword args for:
            typestore, topics, max_idx, verbose
        Inputs:
            bag_path - filepath to ROS Noetic .bag file
            radar_confg_path - filepath to Radar config (.cfg) file
            typestore - rosbags typestore to read RaDICaL .bag files
            topics - desired topics to return from yeild
            max_idx - maximum number of indexes (default: read whole bag)
            verbose - determine whether to print timestamp and topic
        Outputs:
            idx - index of topic as it appears in the data
            timestamp - time stamp 
            topic - (str) topic name
            data - decoded bag data
    """
    # get Radar config
    radar_cfg = read_radar_params(radar_confg_path)

    # open bag and stream
    with Reader(bag_path) as reader:

        # Iterate over messages.
        for idx, \
            (connection, timestamp, rawdata) in enumerate(reader.messages()):

            # get msg and topic
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            topic = NAMES_MAP[connection.topic]
            
            # weird stuff happens here
            # don't continue if topic isn't present
            if topic not in topics:
                continue
            # check for exit condition
            elif idx >= max_idx:
                return None

            if verbose:
                print(timestamp, " ", topic)

            # process applicable message type and get data
            msg_type = connection.msgtype.lower()
            if topic == "imu_accel":
                # accel, accel_cov = rosmsg_to_accel(msg)
                data = rosmsg_to_accel(msg)

            elif topic == "imu_gyro":
                # gyro, accel_gyro = rosmsg_to_gyro(msg)
                data = rosmsg_to_gyro(msg)

            elif topic == "radar":
                # radar_cube_raw = rosmsg_to_radarcube_v1(msg, radar_cfg)
                data = rosmsg_to_radarcube_v1(msg, radar_cfg)
            
            elif topic == "depth":
                # depth_image = rosmsg_to_depth(msg)
                data = rosmsg_to_depth(msg)

            elif topic == "rgb":
                # rgb_image = rosmsg_to_rgb(msg)
                data = rosmsg_to_rgb(msg)

            else:
                print("UNKOWN MSG TYPE: ", msg_type)
                data = None

            # package data
            yield (idx, timestamp, topic, data)
