# -*- utf-8 -*-

import numpy as np
import pylab as pl
import LdsManager

FRAMERATE = 60
MOTION_START = 0
MOTION_END = MOTION_START + FRAMERATE * 60

T_MIN = int(FRAMERATE * 1.0)
THRESH_MICRO = 0.02
THRESH_MACRO = 0.001
LATTENT_DIM = 12


class MotionTexture:

    def __init__(self, data, name):
        self.train_data = data
        self.name = name

    def initialization(self):

        print "minimum len  =", T_MIN
        print "lattent dim  =", LATTENT_DIM
        print "micro thresh =", THRESH_MICRO
        print "macro thresh =", THRESH_MACRO

        lm = LdsManager.LdsManager(T_MIN, LATTENT_DIM, THRESH_MICRO, THRESH_MACRO, FRAMERATE)
        lm.init_fit(self.train_data)

        self.lm = lm

        path = "C:/Users/mkmk0_000/Documents/Git/MOSE/Data/segment/" + self.name + "_texture_init.csv"
        lm.write_segment(path)

    def optimization(self):

        self.lm.fit()

        path = "C:/Users/mkmk0_000/Documents/Git/MOSE/Data/segment/" + self.name + "_texture_final.csv"
        self.lm.write_segment(path)


def main():

    fn = "data/burenai_q.csv"
    #fn = "data/perfume_q.csv"
    fn_core = fn.split(".")[0]
    fn_core = fn_core.split("/")[-1]

    data = np.loadtxt(fn, delimiter=",")
    data = data[MOTION_START:MOTION_END, 1:]
    print "data size = ", data.shape

    import sklearn
    data2 = sklearn.preprocessing.scale(data)
    data3 = sklearn.preprocessing.normalize(data2)

    # pl.figure()
    # pl.plot(data3)
    # pl.show()

    mt = MotionTexture(data3, fn_core)
    mt.initialization()
    mt.optimization()

    return


if __name__ == '__main__':
    main()
