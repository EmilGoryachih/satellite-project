import matplotlib.pyplot as plt
import numpy as np
import PIL
from scipy.integrate import odeint
import math
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from Design import Ui_MainWindow


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
        self.r = 6371
        self.atm = 6489
        self.mass = 5 * (10 ** 16)
        self.g = 9.8
        self.h_atm = self.r + 400

    def draw(self):
        u = np.linspace(-np.pi, np.pi, self.poligons)
        v = np.linspace(0, np.pi, self.poligons)

        x = self.r * np.outer(np.cos(u), np.sin(v))
        y = self.r * np.outer(np.sin(u), np.sin(v))
        z = self.r * np.outer(np.ones(np.size(u)), np.cos(v))

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x1 = np.cos(u) * np.sin(v) * self.h_atm
        y1 = np.sin(u) * np.sin(v) * self.h_atm
        z1 = np.cos(v) * self.h_atm
        ax.plot_wireframe(x1, y1, z1, color="b", alpha=0.3)

        im = PIL.Image.open('earthmap.jpg')
        im = np.array(im.resize([self.poligons, self.poligons])) / 255

        ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=im, antialiased=True, shade=False)


class Satellite_trajectory:
    def __init__(self, lat, lon):
        self.height = 1300
        x = self.height + Earth_drawing().r
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
        tspan = np.linspace(0, 5000, 1000)
        state0 = np.array([x, y, z, self.vec_x, self.vec_y, self.vec_z])

        solution = odeint(odefun, state0, tspan)
        coord = min(solution, key=lambda xc: math.sqrt((xc[0] - self.x) ** 2 + (xc[1] - self.y) ** 2 +
                                                       (xc[2] - self.z) ** 2))

        ISS_height = min(solution, key=lambda xw: math.sqrt((xw[0]) ** 2 + (xw[1]) ** 2 + (xw[2]) ** 2))
        ISS_height = math.sqrt((ISS_height[0]) ** 2 + (ISS_height[1]) ** 2 + (ISS_height[2]) ** 2)

        if ISS_height > 6371 + 400:
            print('Спутник не входит в атмосферу земли')
        else:
            print('Спутник входит в атмосферу земли')

        if ISS_height > 6371:
            print('Спутник не падает')
        else:
            print('Спутник падает')


        ax.scatter(coord[0], coord[1], coord[2], color="black", label="Satellite")
        ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], 'gray', label='Trajectory', linewidth=2.0)
        z = np.linspace(-1 * coord[2] - 2000, coord[2] + 2000)
        x = 0 * z
        y = 0 * z

        ax.plot(x, y, z, 'r', linewidth=2, label="Earth axis")
        ax.legend()

        kinetic = []
        potential = []
        tot = []

        for i in solution:
            v = math.sqrt(i[3] ** 2 + i[4] ** 2 + i[5] ** 2)
            h = math.sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2)
            kinetic.append(0.5 * satellite.mass * v ** 2)
            potential.append(-1 * G * 5 * (10 ** 16) * satellite.mass / h)
            tot.append(kinetic[-1] + potential[-1])

        ax2.plot(tspan, potential, 'b', label="potential")
        ax2.plot(tspan, tot, 'k', label='total')
        ax2.plot(tspan, kinetic, 'r', label="kinetic")
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
