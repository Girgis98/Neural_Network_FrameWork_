#use this line in jupyter
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

class Plot():
    def __init__(self,name,colour):
        self.name = name
        self.colour = colour
        self.values = []
    def add_value(self,value):
        self.values.append(value)


class Visualizer():
    def __init__(self):
        self.iteration = 0
        self.fig = None
        self.ax = None
        self.plots = []

    def add_plot(self, name, colour):
        plot = Plot(name=name, colour=colour)
        self.plots.append(plot)

    # plot values are a list
    def update(self, plot_values):
        for i in range(len(self.plots)):
            self.plots[i].add_value(plot_values[i])

        # if first iteration
        if (self.iteration == 0):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
            plt.ion()
            self.fig.show()
            plt.show()
            self.fig.canvas.draw()
            for plot in self.plots:
                self.ax.plot(plot.values, color=plot.colour, label=plot.name)
            legend = self.ax.legend()
            self.fig.canvas.draw()
        else:
            for plot in self.plots:
                self.ax.plot(plot.values, color=plot.colour, label=plot.name)
            self.fig.canvas.draw()

        self.iteration += 1