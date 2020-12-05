from numpy.lib.function_base import quantile
import rospy
import numpy as np
import tensorflow as tf

from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose


class GazeboEnv():
    # initialize the environment
    def __init__(self):
        rospy.init_node("GazeboWorld", anonymous=False)

        self.observation_shape = (1, 22)
        self.action_shape = (4, )
        self.max_speed = 1300

        # action space specification
        high_end = np.ones(self.action_shape) * self.max_speed
        low_end = np.zeros(self.action_shape)
        self.half_point = np.ones(self.action_shape) * self.max_speed * 0.5
        self.action_space = np.array([low_end, high_end])
        self.last_action = [0, 0, 0, 0]
        # self.observation_space = np.zeros()

        # initializing parameters
        self.reference_frame = 'world'
        self.ns = 'bebop'
        self.self_state = np.zeros(21)
        self.min_reward = 0
        self.pause_duration = 0.0001
        self.boundx = 1
        self.boundy = 1
        self.boundz = 1

        self.last_speed = [0.0, 0.0, 0.0, 0.0]

        # -----------Publisher & Subscribers--------------
        self.actuators_publisher = rospy.Publisher(
            self.ns + '/command/motors', Actuators, queue_size=1)
        self.state_publisher = rospy.Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=1)

        self.object_state_subscriber = rospy.Subscriber(
            'gazebo/model_states', ModelStates, self.model_state_cb)

        # -----------Physic and World Services------------
        self.pause_sim = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.pause_sim()
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.reset_World = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.init_state()

    # initilize the starting state
    def init_state(self):
        self.pause_sim()

        rotor_vel = Actuators()
        rotor_vel.angular_velocities = np.random.uniform(
            100.0, self.max_speed, size=4)
        self.actuators_publisher.publish(rotor_vel)

        model_state = ModelState()

        model_state.model_name = 'bebop'
        model_state.pose.position.x = 0
        model_state.pose.position.y = 0
        model_state.pose.position.z = 1

        # self.goal = np.random.uniform(size=(3, 1), low=-5.0, high=5.0)
        self.goal = [0, 0, 1]

        model_state.pose.orientation.x = 0
        model_state.pose.orientation.y = 0
        model_state.pose.orientation.z = 0
        model_state.pose.orientation.w = 0

        # [x, y, z] = np.random.uniform(size=(3), low=-5, high=5)
        # model_state.twist.angular.x = x
        # model_state.twist.angular.y = y
        # model_state.twist.angular.z = z

        # [x, y, z] = np.random.uniform(size=(3), low=-5, high=5)
        # model_state.twist.linear.x = x
        # model_state.twist.linear.y = y
        # model_state.twist.linear.z = z

        # rot = self.rot_mat_from_quaternion(model_state.pose.orientation)
        quaternion = model_state.pose.orientation
        euler = self.quaternion_to_euler_angle_vectorized1(quaternion.w, quaternion.x, quaternion.y, quaternion.z)

        self.lastState = [
            self.goal[0] - model_state.pose.position.x,
            self.goal[1] - model_state.pose.position.y,
            self.goal[2] - model_state.pose.position.z,            
            model_state.twist.linear.x,
            model_state.twist.linear.y,
            model_state.twist.linear.z,
            model_state.twist.angular.x,
            model_state.twist.angular.y,
            model_state.twist.angular.z,
            euler[0]/180,
            euler[1]/180,
            euler[2]/180,
            self.last_action[0],
            self.last_action[1],
            self.last_action[2],
            self.last_action[3]
        ]

        self.state_publisher.publish(model_state)

    def model_state_cb(self, data):
        index = data.name.index('bebop')

        # rot = self.rot_mat_from_quaternion(data.pose[index].orientation)
        quaternion = data.pose[index].orientation
        euler = self.quaternion_to_euler_angle_vectorized1(quaternion.w, quaternion.x, quaternion.y, quaternion.z)

        self.lastState = [
            self.goal[0] - data.pose[index].position.x,
            self.goal[1] - data.pose[index].position.y,
            self.goal[2] - data.pose[index].position.z,
            data.twist[index].linear.x,
            data.twist[index].linear.y,
            data.twist[index].linear.z,
            data.twist[index].angular.x,
            data.twist[index].angular.y,
            data.twist[index].angular.z,
            euler[0]/180,
            euler[1]/180,
            euler[2]/180,
            self.last_action[0],
            self.last_action[1],
            self.last_action[2],
            self.last_action[3]
        ]

    def rot_mat_from_quaternion(self, quaternion):
                      
        euler = self.quaternion_to_euler_angle_vectorized1(quaternion.w, quaternion.x, quaternion.y, quaternion.z)

        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        rotation_matrix = np.array(
            [
                [
                    np.cos(yaw) * np.cos(pitch),
                    np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
                    np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
                ],
                [
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
                    np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
                ],
                [
                    -1 * np.sin(pitch),
                    np.cos(pitch) * np.sin(roll),
                    np.cos(pitch) * np.cos(roll)
                ]
            ]
        )

        return rotation_matrix

    def quaternion_to_euler_angle_vectorized1(self, w, x, y, z):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2>+1.0,+1.0,t2)
        #t2 = +1.0 if t2 > +1.0 else t2

        t2 = np.where(t2<-1.0, -1.0, t2)
        #t2 = -1.0 if t2 < -1.0 else t2
        Y = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.degrees(np.arctan2(t3, t4))

        return X, Y, Z 

    def get_state(self):
        return self.lastState

    # publish actions to motors
    def publish_action(self, action):
        motor_actions = Actuators()
        motor_actions.angular_velocities = action
        self.actuators_publisher.publish(motor_actions)

    def get_reward(self):
        [dx, dy, dz] = self.lastState[:3]
        [ax, ay, az, aw] = self.lastState[3:7]
        [roll, pitch, yaw] = self.lastState[9:12]

        reward = 0
        dp = abs(dx) + abs(dy) + abs(dz)
        da = abs(ax) + abs(ay) + abs(az) + abs(aw)
        dq = abs(roll) + abs(pitch)
        ac = abs(self.last_action[0]) + abs(self.last_action[1]) + abs(self.last_action[2]) + abs(self.last_action[3])
        reward = -dp * 1e-1 - da * 1e-1 - ac * 1e-4 - dq * 1e-1 + 1

        done = dp > 10 or da > 8 or dq > 0.7 or dz > 0.05
        if done:
            reward = -1000.0

        return reward, done

    # publish action and observe result
    def step(self, action):
        action = action * self.half_point
        action = action + self.last_action
        action = np.clip(action, -self.max_speed, self.max_speed)
        self.last_action = action

        self.unpause_sim()
        self.publish_action(action)

        rospy.sleep(self.pause_duration)
        self.pause_sim()

        # create reward & done signal here
        reward, done = self.get_reward()
        state = self.get_state()

        return state, reward, done,

    # reset the environment
    def reset(self):
        self.init_state()
        return self.get_state()
