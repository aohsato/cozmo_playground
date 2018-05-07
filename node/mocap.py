#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import rospy, tf
import paho.mqtt.client as mqtt

# mqtt
host = "127.0.0.1"
port = 1883
topic = "/cozmo/mocap"
client = mqtt.Client(protocol=mqtt.MQTTv311)
client.connect(host, port=port, keepalive=60)

# ros
rospy.init_node("cozmo_mocap")
rate = rospy.Rate(10)
tf_listener = tf.TransformListener()
from_frame_id = rospy.get_param("from_frame_id", "/ar_marker_1")
to_frame_id = rospy.get_param("to_frame_id", "/ar_marker_0")

# main loop
while not rospy.is_shutdown():
    try:
        (trans, rot) = tf_listener.lookupTransform(
            from_frame_id, to_frame_id, rospy.Time(0))
        msg = json.dumps({
            "time" : rospy.Time.now().to_sec(),
            "translation" : trans,
            "rotation" : tf.transformations.euler_from_quaternion(rot) })
        client.publish(topic, msg)
        rate.sleep()
    except Exception as e:
        print(e)
        rospy.Rate(1).sleep()
client.disconnect()
