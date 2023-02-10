# from mbientlab.metawear import MetaWear
# from mbientlab.metawear.cbindings import *
# from mbientlab.warble import *
#
# address = "EB:50:2A:9D:89:8A"
# device = MetaWear(address)
# device.connect()

#####

# from mbientlab.metawear import MetaWear, libmetawear
# from mbientlab.metawear.cbindings import *
#
# from mbientlab.warble import *
# from mbientlab.metawear import *
# from threading import Event
#
# e = Event()
# address = None
# def device_discover_task(result):
#     global address
#     if (result.has_service_uuid(MetaWear.GATT_SERVICE)):
#         # grab the first discovered metawear device
#         address = result.mac
#         e.set()
#
# BleScanner.set_handler(device_discover_task)
# BleScanner.start()
# e.wait()
#
# BleScanner.stop()

#####


# Requires: sudo pip3 install metawear
# usage: sudo python3 scan_connect.py
from mbientlab.metawear import MetaWear, libmetawear
from mbientlab.metawear.cbindings import *
from mbientlab.warble import *

from time import sleep

import six

selection = -1
devices = None

while selection == -1:
    print("scanning for devices...")
    devices = {}
    def handler(result):
        devices[result.mac] = result.name

    BleScanner.set_handler(handler)
    BleScanner.start()

    sleep(10.0)
    BleScanner.stop()

    i = 0
    for address, name in six.iteritems(devices):
        print("[%d] %s (%s)" % (i, address, name))
        i+= 1

    msg = "Select your device (-1 to rescan): "
    selection = int(input(msg) if platform.python_version_tuple()[0] == '2' else input(msg))

address = list(devices)[selection]
address = "EB:50:2A:9D:89:8A"
print("Connecting to %s..." % (address))
device = MetaWear(address)
device.connect()
#
# print("Connected")
# print("Device information: " + str(device.info))
# # sleep(5.0)
#
# pattern= LedPattern(repeat_count= Const.LED_REPEAT_INDEFINITELY)
# libmetawear.mbl_mw_led_load_preset_pattern(byref(pattern), LedPreset.SOLID)
#
# time_to_sleep = 0.05
#
# for _ in range(20):
#     libmetawear.mbl_mw_led_stop_and_clear(device.board)
#     libmetawear.mbl_mw_led_write_pattern(device.board, byref(pattern), LedColor.GREEN)
#     libmetawear.mbl_mw_led_play(device.board)
#     sleep(time_to_sleep)
#     libmetawear.mbl_mw_led_stop_and_clear(device.board)
#     libmetawear.mbl_mw_led_write_pattern(device.board, byref(pattern), LedColor.RED)
#     libmetawear.mbl_mw_led_play(device.board)
#     sleep(time_to_sleep)
# libmetawear.mbl_mw_led_write_pattern(device.board, byref(pattern), LedColor.BLUE)
# libmetawear.mbl_mw_led_write_pattern(device.board, byref(pattern), LedColor.GREEN)
# sleep(5.0)
# libmetawear.mbl_mw_led_stop_and_clear(device.board)
# sleep(1.0)
#
# device.disconnect()
# sleep(1.0)
# print("Disconnected")

##############

# usage: python stream_acc.py [mac1] [mac2] ... [mac(n)]

from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from time import sleep

address = "EB:50:2A:9D:89:8A"

class State:
    # init
    def __init__(self, device):
        self.device = device
        self.samples = 0
        self.callback = FnVoid_VoidP_DataP(self.data_handler)
    # callback
    def data_handler(self, ctx, data):
        print("%s -> %s" % (self.device.address, parse_value(data)))
        self.samples+= 1

states = []
# connect

d = MetaWear(address)
d.connect()
print("Connected to " + d.address)
states.append(State(d))

# configure
for s in states:
    print("Configuring device")
    # setup ble
    libmetawear.mbl_mw_settings_set_connection_parameters(s.device.board, 7.5, 7.5, 0, 6000)
    sleep(1.5)
    # setup acc
    libmetawear.mbl_mw_acc_set_odr(s.device.board, 100.0)
    libmetawear.mbl_mw_acc_set_range(s.device.board, 16.0)
    libmetawear.mbl_mw_acc_write_acceleration_config(s.device.board)
    # get acc and subscribe
    signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_subscribe(signal, None, s.callback)
    # start acc
    libmetawear.mbl_mw_acc_enable_acceleration_sampling(s.device.board)
    libmetawear.mbl_mw_acc_start(s.device.board)

# sleep
sleep(2.0)

# tear down
for s in states:
    # stop acc
    libmetawear.mbl_mw_acc_stop(s.device.board)
    libmetawear.mbl_mw_acc_disable_acceleration_sampling(s.device.board)
    # unsubscribe
    signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_unsubscribe(signal)
    # disconnect
    libmetawear.mbl_mw_debug_disconnect(s.device.board)

# recap
print("Total Samples Received")
for s in states:
    print("%s -> %d" % (s.device.address, s.samples))


