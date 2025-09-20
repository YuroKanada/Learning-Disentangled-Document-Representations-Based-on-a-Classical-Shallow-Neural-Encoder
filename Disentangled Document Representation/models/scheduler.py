from common import np

class SigmoidAnnealingScheduler:

#KLDに関するロスを計算する際のシグモイド関数によるスケジューラ
#表現性能と各次元の独立化のバランスを調整するために使用

    def __init__(self, max_value=1.0, k=0.1, t0=47380, min_value=0.0):
        self.max_value = max_value
        self.k = k
        self.t0 = t0
        self.min_value = min_value

    def get_value(self, t):
        return self.min_value + (self.max_value - self.min_value) / (1 + np.exp(-self.k * (t - self.t0)))
