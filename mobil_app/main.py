from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, Property

from mobil_app.metawear import scanning_for_devices, connect_and_get_state, setup_acc, start_acc, stop_acc,\
    disconnect_acc, State
import threading


address = "EB:50:2A:9D:89:8A"
state = State()
thread_measure = None



class MainWindow(Screen):
    def disconnect(self):
        disconnect_acc(state)


class NewWindow(Screen):
    measurement = str("")

    def start_measure(self):
        global thread_measure
        thread_measure = threading.Thread(target=start_acc, args=(state,))
        thread_measure.start()

    def stop_measure(self):
        global thread_measure
        stop_acc(state)
        thread_measure.join()

        print(state.x)

class ConnectionWindow(Screen):
    log = Property("")

    def print_connected(self):
        self.log = str("connecting...")

    def connect_to_device(self):
        global state
        connect_and_get_state(address, state)
        setup_acc(state)
        self.log = str(state)




class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("my.kv")


class MyMainApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    MyMainApp().run()
