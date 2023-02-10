from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from mbientlab.warble import *

import platform

from time import sleep, time
from array import array


class State:
    # init
    def __init__(self, device=None):
        self.device = device
        self.samples = 0
        self.dh_callback = FnVoid_VoidP_DataP(self.data_handler)
        self.ts_callback = FnVoid_VoidP_VoidP(self.time_stamp)

        self.x = array('f', [])
        self.y = array('f', [])
        self.z = array('f', [])

    def add_device(self, d):
        self.device = d

    # callback
    def data_handler(self, ctx, data):
        xyz = parse_value(data)
        # print("%s -> %s" % (self.device.address, xyz))

        self.x.append(xyz.x)
        self.y.append(xyz.y)
        self.z.append(xyz.z)
        self.samples+= 1

    def time_stamp(self, ctx, data):
        print(data)
        # self.samples += 1

def scanning_for_devices():
    devices = {}

    def handler(result):
        devices[result.mac] = result.name

    BleScanner.set_handler(handler)
    BleScanner.start()

    sleep(10.0)
    BleScanner.stop()

    return devices

def connect_and_get_state(_address, s=None):
    d = MetaWear(_address)
    d.connect()
    if s is None:
        s = State(d)
        sleep(2.0)
        return s
    else:
        s.add_device(d)

def setup_acc(s):
    libmetawear.mbl_mw_settings_set_connection_parameters(s.device.board, 7.5, 7.5, 0, 6000)
    sleep(1.5)

    # setup acc
    libmetawear.mbl_mw_acc_set_odr(s.device.board, 100.0)
    libmetawear.mbl_mw_acc_set_range(s.device.board, 16.0)
    libmetawear.mbl_mw_acc_write_acceleration_config(s.device.board)

    # get acc and subscribe
    signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_subscribe(signal, None, s.dh_callback)
    # libmetawear.mbl_mw_dataprocessor_accounter_create(signal, None, s.ts_callback)

def start_acc(s, t=None):
    # start acc
    libmetawear.mbl_mw_acc_enable_acceleration_sampling(s.device.board)
    libmetawear.mbl_mw_acc_start(s.device.board)

    if t is not None:
        # sleep
        sleep(t)

def stop_acc(s):
    # stop acc
    libmetawear.mbl_mw_acc_stop(s.device.board)
    libmetawear.mbl_mw_acc_disable_acceleration_sampling(s.device.board)


def disconnect_acc(s):
    # unsubscribe
    signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_unsubscribe(signal)

    # disconnect
    print("disconnect")
    libmetawear.mbl_mw_debug_disconnect(s.device.board)

def set_light(s, color):
    color_dict = {'r': LedColor.RED,
                  'g': LedColor.GREEN,
                  'b': LedColor.BLUE,}

    pattern = LedPattern(repeat_count=Const.LED_REPEAT_INDEFINITELY)
    libmetawear.mbl_mw_led_load_preset_pattern(byref(pattern), LedPreset.SOLID)

    libmetawear.mbl_mw_led_stop_and_clear(s.device.board)
    libmetawear.mbl_mw_led_write_pattern(s.device.board, byref(pattern), color_dict[color])
    libmetawear.mbl_mw_led_play(s.device.board)

def reset_light(s):
    libmetawear.mbl_mw_led_stop_and_clear(s.device.board)

if __name__ == "__main__":
    address = "EB:50:2A:9D:89:8A"
    state = connect_and_get_state(address)
    setup_acc(state)
    start_acc(state, t=2)
    stop_acc(state)
    set_light(state, 'g')
    disconnect_acc(state)

    print("Total Samples Received")
    print("%s -> %d" % (state.device.address, state.samples))