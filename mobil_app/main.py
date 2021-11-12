from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy_garden.graph import Graph, LinePlot
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, Property

from mobil_app.metawear import scanning_for_devices, connect_and_get_state, setup_acc, start_acc, stop_acc,\
    disconnect_acc, State, set_light, reset_light

import threading
import numpy as np
from time import sleep


address = "EB:50:2A:9D:89:8A"
state = State()
thread_measure = None
thread_plot = None


class MainWindow(Screen):
    @staticmethod
    def disconnect():
        global thread_measure, thread_plot

        disconnect_acc(state)

        if thread_measure is not None:
            thread_measure.join()

        if thread_plot is not None:
            thread_plot.join()



class NewWindow(Screen):
    measurement = str("")

    @staticmethod
    def start_measure():
        global thread_measure

        if state.device is not None:
            thread_measure = threading.Thread(target=start_acc, args=(state,))
            thread_measure.start()
            set_light(state, 'g')

    @staticmethod
    def stop_measure():
        global thread_measure

        if state.device is not None:
            stop_acc(state)
            thread_measure.join()
            reset_light(state)


class ConnectionWindow(Screen):
    log = Property("")

    def print_connected(self):
        self.log = str("connecting...")

    def connect_to_device(self):
        global state
        connect_and_get_state(address, state)
        setup_acc(state)
        self.log = str(state)


class MainGrid(BoxLayout):
    zoom = NumericProperty(1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inited = False
        self.run_plot = False

    def init(self):
        if not self.inited:
            self.inited = True
            self.max_samples = 512
            self.zoom = 1
            self.graph = Graph(y_ticks_major=0.5,
                               x_ticks_major=64,
                               border_color=[0, 1, 1, 1],
                               tick_color=[0, 1, 1, 0.7],
                               x_grid=True, y_grid=True,
                               xmin=0, xmax=self.max_samples,
                               ymin=-3.0, ymax=3.0,
                               draw_border=False,
                               x_grid_label=True, y_grid_label=True)

            self.ids.modulation.add_widget(self.graph)
            self.plot_x = LinePlot(color=[1, 1, 0, 1], line_width=1.2)
            self.plot_y = LinePlot(color=[1, 0, 1, 1], line_width=1.2)
            self.plot_z = LinePlot(color=[0, 1, 1, 1], line_width=1.2)
            self.graph.add_plot(self.plot_x)
            self.graph.add_plot(self.plot_y)
            self.graph.add_plot(self.plot_z)
            self.plot_x.points = []
            self.plot_y.points = []
            self.plot_z.points = []


    def update_plot(self):
        while self.run_plot:
            x = state.x[-self.max_samples:]
            y = state.y[-self.max_samples:]
            z = state.z[-self.max_samples:]
            t = np.linspace(0, len(x), len(x))
            self.plot_x.points = list(zip(t, x))
            self.plot_y.points = list(zip(t, y))
            self.plot_z.points = list(zip(t, z))
            sleep(1)

    def start_plot(self):
        global thread_plot
        self.init()
        self.run_plot = True
        thread_plot = threading.Thread(target=self.update_plot)
        thread_plot.start()

    def stop_plot(self):
        global thread_plot
        self.run_plot = False
        thread_plot.join()


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("my.kv")


class MyMainApp(App):
    @staticmethod
    def build():
        return kv


if __name__ == "__main__":
    MyMainApp().run()
