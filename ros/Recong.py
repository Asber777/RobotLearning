#!/usr/bin/env python

from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
from torch import nn
import torchvision.models as models
import numpy as np
from geometry_msgs.msg import Twist
# CNN model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layers3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(20*15*64,300),
            nn.ReLU(inplace=True),
            nn.Linear(300,5),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

#config my CNN model
        
pthfile = '../abcd.pth'
cnn = CNN()
cnn.load_state_dict(torch.load(pthfile))
        
    
#reference:
#http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/depth/image_raw",Image,self.callback)
    self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=100)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
    except CvBridgeError as e:
      print(e)
    #downSample
    rows,cols = cv_image.shape
    cv_image = cv2.pyrDown(cv2.pyrDown(cv_image))
    
    #the value of cv_image might be nan.So,we need to pretreatment
    #because depthmap's unit is metres,but png is 0~255 
    #the range of Kinect is 0.8~4.0m 255/3.2=79.6875
    #so 
    img = torch.from_numpy(np.nan_to_num(cv_image))
    img = img.view(1,1,160,120)*79.6875
    #print(img)
    out = cnn(img.float())
    pre = torch.max(out.data,1)[1]    
    move_cmd = Twist()
	# let's go forward at 0.2 m/s
    move_cmd.linear.x = 0.2
	# let's turn at 0 radians/s
    if pre==0:
        move_cmd.angular.z = -1.5
        print("turning-full-right(0)")
    elif pre == 1:
        move_cmd.angular.z = -0.75
        print("turning-half-right(1)")
    elif pre == 2:
        move_cmd.angular.z = 0
        print("go-straightforward(2)")
    elif pre == 3:
        move_cmd.angular.z = 0.75
        print("turning-half-left(3)")
    else:
        move_cmd.angular.z = 1.5
        print("turning-full-left(4)")
        
    self.cmd_vel.publish(move_cmd)
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

def shutdown(self):
    # stop turtlebot
    rospy.loginfo("Stop TurtleBot")
	# a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
    self.cmd_vel.publish(Twist())
	# sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
    rospy.sleep(1)
    # shut down AllWindows
    cv2.destroyAllWindows()
        
def main(args):
  ic = image_converter()
  rospy.init_node('image_converter_And_Go_forward', anonymous=True)
  rospy.on_shutdown(shutdown)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
