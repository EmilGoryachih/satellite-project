import matplotlib.pyplot as plt
import numpy as np
import PIL
from scipy.integrate import odeint
import math
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from motplotlib import Ui_MainWindow


def Rot(x, y, a):
    a = math.radians(a)
    x2 = x * round(math.cos(a), 5) - y * math.sin(a)
    y2 = x * math.sin(a) + y * round(math.cos(a), 5)
    return x2, y2


def odefun(s: np.ndarray, _):
    x, y, z, vec_x, vec_y, vec_z = s
    r = np.array([0 - x, 0 - y, 0 - z])
    mr = np.linalg.norm(r) ** 3
    xx = G * (Earth_drawing().mass * (0 - x) / mr + satellite.mass * (0 - x) / mr)
    yy = G * (Earth_drawing().mass * (0 - y) / mr + satellite.mass * (0 - y) / mr)
    zz = G * (Earth_drawing().mass * (0 - z) / mr + satellite.mass * (0 - z) / mr)
    return np.array([vec_x, vec_y, vec_z, xx, yy, zz])


class Earth_drawing:
    def __init__(self, quality=1):
        self.poligons = 100 * quality
        self.R = 6371
        self.R_atm = 6489
        self.mass = 5 * (10 ** 16)
        self.g = 9.8

    def draw(self):
        u = np.linspace(-np.pi, np.pi, self.poligons)
        v = np.linspace(0, np.pi, self.poligons)

        x = self.R * np.outer(np.cos(u), np.sin(v))
        y = self.R * np.outer(np.sin(u), np.sin(v))
        z = self.R * np.outer(np.ones(np.size(u)), np.cos(v))

        im = PIL.Image.open('min_earth.png')
        im = np.array(im.resize([self.poligons, self.poligons])) / 255

        ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=im, antialiased=True, shade=False)
        ax.scatter(15000, 15000, 15000, alpha=0)
        ax.scatter(-15000, 15000, 15000, alpha=0)
        ax.scatter(15000, -15000, 15000, alpha=0)
        ax.scatter(15000, 15000, -15000, alpha=0)


class Satellite_trajectory:
    def __init__(self, lat, lon):
        self.height = 1300
        x = self.height + Earth_drawing().R
        y = 0
        z = 0
        self.lat = lat
        self.lon = lon
        self.x, self.z = Rot(x, z, lat)
        self.x, self.y = Rot(self.x, y, lon)
        self.x, self.y, self.z = round(self.x), round(self.y), round(self.z)
        self.velocity = 8000
        self.mass = 10000
        self.vec_x = 0
        self.vec_y = 2
        self.vec_z = 24

    def create_trajectory(self):
        x, z = Rot(self.x, self.z, 180 - self.lat)
        x, y = Rot(x, self.y, 180 - self.lon)
        ts = np.linspace(0, 5000, 1000)
        state0 = np.array([x, y, z, self.vec_x, self.vec_y, self.vec_z])

        sol = odeint(odefun, state0, ts)
        coord = min(sol, key=lambda xc: math.sqrt((xc[0] - self.x) ** 2 + (xc[1] - self.y) ** 2 +
                                                  (xc[2] - self.z) ** 2))

        ax.scatter(coord[0], coord[1], coord[2], color="black")
        ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], 'g', label='Trajectory', linewidth=2.0)
        z = np.linspace(-1 * coord[2] - 2000, coord[2] + 2000)
        x = 0 * z
        y = 0 * z

        ax.plot(x, y, z, 'r', linewidth=2, label="Earth axis")
        ax.legend()

        kinetic_energy = []
        potential_energy = []

        for i in sol:
            v = math.sqrt(i[3] ** 2 + i[4] ** 2 + i[5] ** 2)
            h = math.sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2)
            kinetic_energy.append(0.5 * satellite.mass * v ** 2)
            potential_energy.append(satellite.mass * 9.8 * h)

        ax2.plot(ts, potential_energy, 'b', label="potential")
        ax2.plot(ts, kinetic_energy, 'r', label="kinetic")
        ax2.legend()


class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)
        self.ui.pushButton.clicked.connect(self.Start)

    def show(self):
        self.main_win.show()

    def Start(self):
        new_value = str(self.ui.horizontalSlider.value())
        earth = Earth_drawing(int(new_value))
        earth.draw()
        plt.show()
        self.main_win.close()


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
ax2.legend()
ax.set_box_aspect((1, 1, 1))
ax.margins(8000, 8000, 8000)
ax.legend()
ax.autoscale(enable=False, tight=True)

ax2.autoscale(enable=True, tight=False)

G = 6.6743015 * (10 ** (-11))
LAT, LON = 90, 90

satellite = Satellite_trajectory(LAT, LON)
satellite.create_trajectory()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
