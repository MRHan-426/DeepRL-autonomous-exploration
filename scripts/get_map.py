import rospy
import os
import numpy as np
import math
import time
import cv2
import tf
from std_msgs.msg import Float32
from actionlib_msgs.msg import GoalStatusArray
from gazebo_msgs.msg import ModelState,ModelStates,ContactsState
from move_base_msgs.msg import MoveBaseActionResult,MoveBaseGoal,MoveBaseAction
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Twist
from nav_msgs.srv import GetMap, GetMapRequest
from nav_msgs.msg import Odometry

from std_srvs.srv import Empty,EmptyRequest

HEIGHT = 162
WIDTH = 322

def get_map():
    get_map_data = rospy.ServiceProxy("dynamic_map",GetMap)
    get_map_data.wait_for_service()
    map_data_request = GetMapRequest()
    map_data_response = get_map_data.call(map_data_request)
    map_data = np.array(map_data_response.map.data)
    # map data process
    height = map_data_response.map.info.height
    width = map_data_response.map.info.width
    print("current H W",height,width)
    map_data = map_data.reshape((height,width),order='C')
    temp = np.zeros((HEIGHT,WIDTH))

    for i in range(HEIGHT - height,HEIGHT):
        for j in range(WIDTH - width,WIDTH):
            temp[i,j] = temp[i,j] + map_data[i-HEIGHT+height,j-WIDTH+width]
    for i in range(HEIGHT - height):
        for j in range(WIDTH):
            temp[i,j] = -1
    for i in range(HEIGHT):
        for j in range(WIDTH-width):
            temp[i,j] = -1
    map_data = temp

    for i in range(HEIGHT):
        for j in range(WIDTH):
            # unknown zone
            if(map_data[i,j] == -1):
                map_data[i,j] = 100
            
            elif(map_data[i,j] == 100):
                map_data[i,j] = 0

            else:
                map_data[i,j] = 255
    map_data = cv2.flip(map_data, 0)

    cv2.imwrite('/home/hzq/multirobot_formation/src/formation/maps/' + str(time.time()) + '.jpg', map_data)


if __name__ == '__main__':
    rospy.init_node('get_map',anonymous=True)

    get_map()

    