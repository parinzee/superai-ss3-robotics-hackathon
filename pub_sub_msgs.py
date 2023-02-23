import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

from std_msgs.msg import String

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from tf_transformations import euler_from_quaternion, quaternion_from_euler

from math import degrees

import numpy as np

from geometry_msgs.msg import Twist


class LearningNode(Node):
    def __init__(self):
        super().__init__('test2')
        self.timer_period = .5 # seconds
        self.timer = self.create_timer(self.timer_period, self.call_back)
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)


    def call_back(self):
        x,y,yaw = self.odom_receive()
        ranges_arr, range_max, angle_max, angle_increment = self.scan_receive()
        v,w = self.rl_func(x,y,yaw,ranges_arr, range_max, angle_max, angle_increment)
        self.publisher_vel(v,w)
        
        
    def odom_receive(self):
        _,msg_odom=self.wait_for_message('/odom', Odometry)
         
        x = msg_odom.pose.pose.position.x
        y = msg_odom.pose.pose.position.y
        # z = msg_odom.pose.pose.position.z
        
        orientation_q = msg_odom.pose.pose.orientation
        orientation_list = [ orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w ]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        return x,y,yaw
        
        
    def scan_receive(self):
        _,msg_scan=self.wait_for_message('/scan', LaserScan)
        
        ranges_arr = np.frombuffer(msg_scan.ranges)
        # len_ranges = len(ranges_arr)
        range_max = msg_scan.range_max 
        # range_min = msg_scan.range_min
        angle_max = msg_scan.angle_max
        # angle_min = msg_scan.angle_min
        angle_increment = msg_scan.angle_increment
        # scan_time = msg_scan.scan_time
        # time_increment = msg_scan.time_increment
        # intensities = msg_scan.intensities
        # header = msg_scan.header
        
        return ranges_arr, range_max, angle_max, angle_increment
        
    
    def rl_func(self,x,y,yaw,ranges_arr, range_max, angle_max, angle_increment):
        """
        x = float : current position of robot compared to initial position in meter unit(x increase whun go forward, decrase when go backward)
        y = float : current position of robot compared to initial position in meter unit(y increase when go left, decrease when go right)
        yaw = float : rotation of robot in radian unit, increase when rotate counter-clock wise(up to 3.14 when flip 180 degree), switch to -3.14 when rotate greater than 180 degree and approach to 0 when rotate back to initial
        ranges_arr = numpy.ndarray : contain 450 float values of distance each angles in meter unit
        ranges_max = 15 meter, maximum measureable length of lidar sensor
        angle_max = around 6.28 in unit radian
        angle_increment = each steps of angle lidar beam in radian unit
        ------------------------
        v = linear velocity of robot ()
        
        """
        v=0
        w=0
        
        
        
        
        
        
        return v,w
    
    
    def publisher_vel(self,v,w):
        velMsg = Twist()
        velMsg.linear.x = v
        # velMsg.linear.y = 0.
        # velMsg.linear.z = 0.
        # velMsg.angular.x = 0.
        # velMsg.angular.y = 0.
        velMsg.angular.z = w
        self.velPub.publish(velMsg)


    def wait_for_message(
        node,
        topic: str,
        msg_type,
        time_to_wait=-1
    ):
        """
        Wait for the next incoming message.
        :param msg_type: message type
        :param node: node to initialize the subscription on
        :param topic: topic name to wait for message
        :time_to_wait: seconds to wait before returning
        :return (True, msg) if a message was successfully received, (False, ()) if message
            could not be obtained or shutdown was triggered asynchronously on the context.
        """
        context = node.context
        wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
        wait_set.clear_entities()

        sub = node.create_subscription(msg_type, topic, lambda _: None, 1)
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return (False, None)

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                return (True, msg_info[0])

        return (False, None)


def main():
    rclpy.init()
    ln_obj=LearningNode()
    rclpy.spin(ln_obj)
    
    ln_obj.destroy_node()
    rclpy.shutdown()



if __name__=='__main__':
    main()
