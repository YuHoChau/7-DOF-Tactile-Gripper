#!/usr/bin/env python
import rospy
from wsg_32_common.msg import Cmd
from wsg_32_common.msg import Status
from std_msgs.msg import Float32
import time
import sys,tty,termios

global contact_area_
global gripper_ini_flag_

contact_area_ = 0
gripper_width = 0
gripper_ini_flag_ = False

def contact_area_cb(msg):
    global contact_area_
    contact_area_ = msg.data

def gripper_state_cb(data):
    global gripper_width
    global gripper_ini_flag_
    gripper_width = data.width
    gripper_ini_flag_ = True

if __name__ == "__main__":
    rospy.init_node('PC_tactile_servo_pd_publisher', anonymous=True)

    tactile_servo_pub = rospy.Publisher('tactile_servo_control', Float32, queue_size=1)

    rospy.Subscriber('/tactile_state/contact_area', Float32, contact_area_cb)

    old_attr = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    frequency = 30
    init_posi = 180
    desired_contact_area = 8000

    kp = 10
    kd = 1.5
    scale_factor = 0.000001
    prev_error = 0

    del_t = 1 / frequency
    tactile_servo_cmd = Float32()
    tactile_servo_cmd.data = init_posi
    rate = rospy.Rate(frequency)

    try:
        # while not gripper_ini_flag_:
        #     print('Wait for initializing the gripper.')

        # while (sys.stdin.read(1) != 'l'):
        #     print('Wait for starting! Press l to start')
        #     time.sleep(0.1)

        while not rospy.is_shutdown():
            error = desired_contact_area - contact_area_
            derivative = (error - prev_error) / del_t
            control_signal = kp * error + kd * derivative
            tactile_servo_cmd.data -= scale_factor * control_signal
            tactile_servo_cmd.data = min(180, max(40, tactile_servo_cmd.data))
            prev_error = error

            tactile_servo_pub.publish(tactile_servo_cmd)
            rate.sleep()

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)
    except KeyboardInterrupt:
        print('Interrupted!')