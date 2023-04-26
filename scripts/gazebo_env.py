# -*- coding: utf-8 -*-

'''
Set and reset gazebo environment
Collision & Done
author@ziqi han
'''
import rospy
import os
import numpy as np
import math
import time
import cv2
import tf
from std_msgs.msg import Float32, Int8
from actionlib_msgs.msg import GoalStatusArray
from gazebo_msgs.msg import ModelState,ModelStates,ContactsState
from move_base_msgs.msg import MoveBaseActionResult,MoveBaseGoal,MoveBaseAction
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Twist
from nav_msgs.srv import GetMap, GetMapRequest
from nav_msgs.msg import Odometry

from std_srvs.srv import Empty,EmptyRequest

class gazebo_env():
    def __init__(self):
        '''
        @param  last_trajectory_length:  上一状态下小车行驶的总路程
        @param  current_trajectory_length:  当前状态下小车行驶的总路程
        @param  path_length:  路程差，用于计算奖励
        @param  h:  当前状态下地图的香农熵，暂时修改为当前状态下地图上已知点的个数
        @param  last_h:  上一状态下地图的香农熵，暂时修改为上一状态下地图上已知点的个数
        @param  delta_h:  已知点的个数差，用于计算奖励
        @param  map_data:  地图数据，二维数组
        @param  roi:  地图上已知点的占比
        @param  alpha:  系数，用于调整奖励的大小
        @param  beta:  系数，用于调整delta_h的大小
        @param  reward:  奖励，大小介于-1与1之间
        @param  target_point:  目标点坐标[x,y]，即当前状态下小车的位置

        '''
        rospy.init_node('gazebo_control_node', anonymous=True)
        self.robotstate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.gazebo_model_states = ModelStates()
        self.last_trajectory_length = 0.0
        self.last_angular_accumulate = 0.0
        self.current_trajectory_length = 0.0
        self.current_angular_accumulate = 0.0
        self.path_length = 0.0
        self.angular_change = 0.0
        self.h = 0
        self.last_h = 0
        self.delta_h = 0
        self.delta_h_max = 0
        self.valid = 0
        self.map_data = []
        self.roi = 0.0
        # self.alpha = 0.9
        # self.beta = 1.0 / 1300
        self.reward = 0.0
        self.target_point = [0.0,0.0]
        self.agent_name = 'ares1'
        self.done_list = False
        self.goal_target = 0
        self.width = 324
        self.height = 164
        self.origin_x = -7.51
        self.origin_y = -3.51
        self.resetval()
        # self.agent_state_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_states_callback)
        rospy.Subscriber("move_base/result", MoveBaseActionResult, self.goal_callback_1)
        rospy.Subscriber("move_base/status", GoalStatusArray, self.goal_fail_callback)
        rospy.Subscriber("front/bumper_states", ContactsState, self.collision_front_callback)
        rospy.Subscriber("back/bumper_states", ContactsState, self.collision_back_callback)
        # rospy.Subscriber("/robot/arrive",Int8,self.goal_callback_2)
        
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
   

    def resetval(self):
        self.robotstate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.gazebo_model_states = ModelStates()
        self.last_trajectory_length = 0.0
        self.last_angular_accumulate = 0.0
        self.current_trajectory_length = 0.0
        self.current_angular_accumulate = 0.0
        self.path_length = 0.0
        self.angular_change = 0.0
        self.h = 0
        self.last_h = 0
        self.delta_h = 0
        self.delta_h_max = 0

        self.roi = 0.0
        # self.alpha = 0.9
        # self.beta = 1.0 / 1300
        self.reward = 0.0
        self.target_point = [0,0]
        self.agent_name = 'ares1'
        self.done_list = False
        self.goal_target = 0

    def get_map_size(self):
        get_map_data = rospy.ServiceProxy("dynamic_map",GetMap)
        get_map_data.wait_for_service()
        map_data_request = GetMapRequest()
        map_data_response = get_map_data.call(map_data_request)
        self.origin_x = map_data_response.map.info.origin.position.x
        self.origin_y = map_data_response.map.info.origin.position.y

        # map data process
        height = map_data_response.map.info.height
        width = map_data_response.map.info.width

        if width % 4 != 0:
            self.width = 4*(int(width/4) + 1)
        else:
            self.width = width
        if height % 4 != 0:
            self.height = 4*(int(height/4) + 1)
        else:
            self.height = height
        return self.width, self.height

    # def get_origin_xy(self):
    #     get_map_data = rospy.ServiceProxy("dynamic_map",GetMap)
    #     get_map_data.wait_for_service()
    #     map_data_request = GetMapRequest()
    #     map_data_response = get_map_data.call(map_data_request)
    #     self.origin_x = map_data_response.map.info.origin.position.x
    #     self.origin_y = map_data_response.map.info.origin.position.y

    # # 获取agent robot的回报值
    def get_reward(self):
        '''
        @param  path_length:  路程差，用于计算奖励
        @param  map_data:  地图数据，二维数组
        @param  roi:  地图上已知点的占比
        @param  alpha:  系数，用于调整奖励的大小
        @param  beta:  系数，用于调整delta_h的大小
        @param  reward:  奖励，大小介于-1与1之间
        @param  target_point:  目标点坐标，即当前状态下小车的位置
        '''
        self.reward = 0
        self.calculate_path_length()
        self.calculate_delta_h()
        l = 1.4
        resolution = 0.025
        self.delta_h_max = (np.pi * (l**2) + 2 * (0.7 * self.path_length + 0.3*self.angular_change) * l)/(resolution ** 2)
        # self.delta_h_max = (np.pi * (l**2) + 2 * (self.path_length + 0.3*self.angular_change) * l)/(resolution ** 2)
        
        # print(np.pi * (l**2), 2 * (self.path_length + 0.3*self.angular_change) * l,self.path_length,0.3*self.angular_change)
        # 假如目标点处于未知区域
        # print(self.target_point[0],self.target_point[1])
        if self.roi >= 0.85:
            self.reward = 1
            done = 2
            print("SLAM Finished!!!! Restart in Seconds")
            # cv2.imwrite('/home/hanziqi/even_ws/src/formation/maps/' + str(time.time()) + '.jpg', self.map_data)
        
        elif self.valid == 0:
            self.reward = -1
            done = 1

        # navigation failed
        elif self.goal_target == 2:
            self.reward =-1
            done = 1
        
        elif self.path_length == 0:
            self.reward = -1
            print("LAZY! LAZY! LAZY! LAZY! LAZY! LAZY!")
            done = 1  
        else:
            # self.reward = self.alpha * (self.beta*self.delta_h - self.path_length)
            print("goal reached!")
            if (self.delta_h - 2500) >= 0:
                self.reward = float(self.delta_h - 2000) / float(self.delta_h_max)
            else :
                self.reward = -(0.07 * self.path_length + 0.03*self.angular_change) 

            # print(self.delta_h,self.delta_h_max)
            if self.reward < 0:
                self.reward = max(-1,self.reward)
            else:
                self.reward = min(1,self.reward)
            done = 0

        print("Current ROI is :",self.roi)

        print("Reward of this step is :",self.reward)

        return self.reward, done

    def get_map(self):
        get_map_data = rospy.ServiceProxy("dynamic_map",GetMap)
        get_map_data.wait_for_service()
        map_data_request = GetMapRequest()
        map_data_response = get_map_data.call(map_data_request)
        map_data = np.array(map_data_response.map.data)
        height = map_data_response.map.info.height
        width = map_data_response.map.info.width
        map_data = map_data.reshape((height,width),order='C')

        odom = rospy.wait_for_message("odom",Odometry)
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        x = int((x-self.origin_x)/0.025)
        y = int((y-self.origin_y)/0.025)
            
        temp = np.zeros((self.height,self.width))
        edge_data = np.zeros((self.height,self.width))
        position_data = 255 * np.ones((self.height,self.width))

        for i in range(self.height - height,self.height):
            for j in range(self.width - width,self.width):
                temp[i,j] = temp[i,j] + map_data[i- self.height + height,j - self.width + width]
        for i in range(self.height - height):
            for j in range(self.width):
                temp[i,j] = -1
        for i in range(self.height):
            for j in range(self.width-width):
                temp[i,j] = -1
        map_data = temp

        for i in range(self.height):
            for j in range(self.width):
                if(map_data[i,j] == -1):
                    map_data[i,j] = 100
                elif(map_data[i,j] == 100):
                    map_data[i,j] = 0
                else:
                    map_data[i,j] = 255

        for i in range(self.height-1):
            for j in range(self.width-1):
                if map_data[i,j] == 100:
                    if map_data[i+1,j]==255:
                        edge_data[i+1,j] = 255
                    elif  map_data[i,j+1]==255:
                        edge_data[i,j+1] = 255
                elif map_data[i,j] == 255:
                    if map_data[i+1,j]==100 or map_data[i,j+1]==100:
                        edge_data[i,j] = 255

        for i in range(y-3,y+3):
            for j in range(x-5,x+5):
                position_data[i,j] = 0

        self.data = np.zeros((self.height,self.width,3))
        self.data[:,:,0] = map_data
        self.data[:,:,1] = edge_data
        self.data[:,:,2] = position_data
        self.map_data = map_data
        # cv2.imwrite('/home/hanziqi/even_ws/src/formation/map/' + str(time.time()) + '1.jpg', self.data[:,:,0])
        # cv2.imwrite('/home/hanziqi/even_ws/src/formation/map/' + str(time.time()) + '2.jpg', self.data[:,:,1])
        # cv2.imwrite('/home/hanziqi/even_ws/src/formation/map/' + str(time.time()) + '3.jpg', self.data[:,:,2])

        return self.data

    def calculate_delta_h(self):
        # 已知点的个数
        self.h = 0
        for i in range(self.height):
            for j in range(self.width):
                # unknown zone & P = 0
                if(self.map_data[i,j]) == 100:
                    pass
                else:
                    self.h = self.h + 1
        self.roi = float(self.h) / float(self.height * self.width)
        self.delta_h = self.h - self.last_h
        self.last_h = self.h

        '''
        熵值算法:源码里有问题，无法输出0-100之间的数字
        # for i in range(height):
        #     for j in range(width):
        #         # unknown zone & P = 0
        #         if(map_data[i,j] == -1 or map_data[i,j] == 0):
        #             pass
        #         else:
        #             self.h = self.h - (map_data[i,j]/100) *np.log2(map_data[i,j]/100)
        # self.delta_h = self.last_h - self.h
        # self.last_h = self.h
        # print("map h(m)=",self.h)
        '''
        
    def calculate_path_length(self):
        data = rospy.wait_for_message('/trajectory_length', Odometry)
        self.current_trajectory_length = data.pose.pose.position.x
        self.current_angular_accumulate = data.pose.pose.position.y

        self.path_length = self.current_trajectory_length - self.last_trajectory_length
        self.angular_change = self.current_angular_accumulate - self.last_angular_accumulate
        
        self.last_trajectory_length = self.current_trajectory_length
        self.last_angular_accumulate = self.current_angular_accumulate


    # 重置environment & Restart Karto Slam
    def reset_all(self):
        os.system("rosnode kill /move_base")
        vel_reset = rospy.Publisher("cmd_vel", Twist,queue_size = 1)
        t_2 = time.time() + 2
        while(time.time() <= t_2):
            msg = Twist()
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
            vel_reset.publish(msg) 

        self.resetval()
        gazebo_reset = rospy.ServiceProxy("/gazebo/reset_world",Empty)
        gazebo_reset.wait_for_service()
        gazebo_reset_request = EmptyRequest()
        gazebo_reset.call(gazebo_reset_request)
        os.system("rosnode kill /slam_karto")
        rospy.sleep(0.2)
        print("Gazebo Env has been reset!")  

    def reset_karto(self):
        os.system("rosnode kill /slam_karto")
        self.resetval()
        gazebo_reset = rospy.ServiceProxy("/gazebo/reset_world",Empty)
        gazebo_reset.wait_for_service()
        gazebo_reset_request = EmptyRequest()
        gazebo_reset.call(gazebo_reset_request)
        rospy.sleep(0.2)
        print("Gazebo Env has been reset!")  

    def goal_callback_1(self,msg):
        if msg.status.status == 3:
            self.goal_target = 1

    def goal_callback_2(self,msg):
        if msg.data == 1:
            self.goal_target = 1

    def goal_fail_callback(self,msg):
        if msg.status_list != []:
            if msg.status_list[0].status == 4:
                self.goal_target = 2
    
    def collision_front_callback(self,msg):
        if msg.states != []:
            if abs(msg.states[0].total_wrench.force.x) >= 30:
                print("Bump in the front, navigation failed",abs(msg.states[0].total_wrench.force.x))
                self.goal_target = 2

    def collision_back_callback(self,msg):
        if msg.states != []:
            if abs(msg.states[0].total_wrench.force.x) >= 30:
                print("Bump in the back, navigation failed",abs(msg.states[0].total_wrench.force.x))
                self.goal_target = 2

    def take_action(self, action, WIDTH):
        self.target_point[0] = int((action) / WIDTH)
        self.target_point[1] = int((action) % WIDTH)
        x = self.target_point[1] * 0.025 + self.origin_x
        y = self.target_point[0] * 0.025 + self.origin_y
        print("Matrix Point is :",self.target_point[0],self.target_point[1])
        print("Actual Point is :",x,y)

        # Can not navigate
        if (self.map_data[int(self.target_point[0]), int(self.target_point[1])] == 100) :
            self.valid = 0
            return 0
        elif (self.map_data[int(self.target_point[0]), int(self.target_point[1])] == 0) :
            self.valid = 0
            return 0
        else:
            msg = rospy.wait_for_message('odom',Odometry)
            agent_x = msg.pose.pose.position.x
            agent_y = msg.pose.pose.position.y
            # yaw = math.atan2((y - agent_y),(x - agent_x))
            # q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            # point = Pose(Point(x, y, 0.000), Quaternion(q[0], q[1], q[2], q[3]))
            if x >= agent_x:
                point = Pose(Point(x, y, 0.000), Quaternion(0,0,0,1))
            else:
                point = Pose(Point(x, y, 0.000), Quaternion(0,0,1,0))

            print("Target Point is :",point)
            target_position = PoseStamped()
            target_position.header.frame_id = "map"
            target_position.pose = point
            self.goal_pub.publish(target_position)
            self.valid = 1
            return 1

# def gazebo_states_callback(self, data):
#     self.gazebo_model_states = data
#     for i in range(len(data.name)):
#         if data.name[i] == self.agent_name:
#             # robotstate--->x,y,v,w,yaw,vx,vy
#             self.robotstate[0] = data.pose[i].position.x
#             self.robotstate[1] = data.pose[i].position.y
#             v = math.sqrt(data.twist[i].linear.x**2 + data.twist[i].linear.y**2)
#             self.robotstate[2] = v
#             self.robotstate[3] = data.twist[i].angular.z
#             rpy = self.euler_from_quaternion(data.pose[i].orientation.x,data.pose[i].orientation.y,
#             data.pose[i].orientation.z,data.pose[i].orientation.w)
#             self.robotstate[4] = rpy[2]
#             self.robotstate[5] = data.twist[i].linear.x
#             self.robotstate[6] = data.twist[i].linear.y

# def get_env(self):
#     # env_info---> [x,y,map_data,reward,done_list]
#     env_info=[]
#     # robotstate---> x,y,v,w,yaw,vx,vy
#     env_info.append(self.robotstate[0])  # x
#     env_info.append(self.robotstate[1])  # y
#     self.get_reward()
#     env_info.append(self.map_data)
#     env_info.append(self.reward)
#     # 判断是否终止
#     self.done_list = True
#     if self.roi < 0.85:
#         self.done_list = False  # 不终止
#     else:
#         self.done_list = True  # 终止
#         print("Map Completed!")
#     env_info.append(self.done_list)
#     return env_info



