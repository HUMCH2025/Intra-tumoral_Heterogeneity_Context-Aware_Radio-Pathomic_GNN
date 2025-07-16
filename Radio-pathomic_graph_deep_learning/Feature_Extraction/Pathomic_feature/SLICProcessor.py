import numpy as np


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, f1=0, f2=0, f3=0, f4=0):
        self.update(h, w, f1, f2, f3, f4)
        self.pixels = []
        self.no = self.cluster_index
        self.cluster_index += 1

    def update(self, h, w, f1, f2, f3, f4):
        self.h = h
        self.w = w
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4

    def print_no(self):
        return self.no

    def __str__(self):
        return "{},{},{}:{} {} {} {}".format(self.h, self.w, self.f1, self.f2, self.f3, self.f4)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def get_data(data):
        return data

    @staticmethod
    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2],
                       self.data[h][w][3]
                       )

    def __init__(self, data, K, M):
        self.K = K
        self.M = M
        self.data = self.get_data(data)
        self.height = self.data.shape[0]
        self.width = self.data.shape[1]
        self.N = self.height * self.width
        self.S = int((self.N / self.K) ** 0.5)

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.height, self.width), np.inf)

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.height:
            while w < self.width:
                self.clusters.append(self.make_cluster(self, h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.width:
            w = self.width - 2
        if h + 1 >= self.height:
            h = self.height - 2

        gradient = (self.data[h + 1][w][0] + self.data[h][w + 1][0] - 2 * self.data[h][w][0]) + \
                   (self.data[h + 1][w][1] + self.data[h][w + 1][1] + self.data[h][w][1] - 2 * self.data[h][w][1]) + \
                   (self.data[h + 1][w][2] + self.data[h][w + 1][2] + self.data[h][w][2] - 2 * self.data[h][w][2]) + \
                   (self.data[h + 1][w][3] + self.data[h][w + 1][3] + self.data[h][w][3] - 2 * self.data[h][w][3])
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w,
                                       self.data[_h][_w][0],
                                       self.data[_h][_w][1],
                                       self.data[_h][_w][2],
                                       self.data[_h][_w][3])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            h_list = list(range(cluster.h - 2 * self.S, cluster.h + 2 * self.S))
            w_list = list(range(cluster.w - 2 * self.S, cluster.w + 2 * self.S))
            for h in h_list:
                for w in w_list:
                    if h < 0 or h >= self.height:
                        continue
                    if w < 0 or w >= self.width:
                        continue
                    f1, f2, f3, f4 = self.data[h][w]
                    "feature"
                    Dc = (
                                 (f1 - cluster.f1) ** 2 +
                                 (f2 - cluster.f2) ** 2 +
                                 (f3 - cluster.f3) ** 2 +
                                 (f4 - cluster.f4) ** 2) ** 0.5
                    "position"
                    Ds = (
                                 (h - cluster.h) ** 2 +
                                 (w - cluster.w) ** 2) ** 0.5
                    D = ((Dc / self.M) ** 2 + (Ds / self.S) ** 2) ** 0.5
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                cluster.update(_h, _w,
                               self.data[_h][_w][0],
                               self.data[_h][_w][1],
                               self.data[_h][_w][2],
                               self.data[_h][_w][3])

    def save_current_cluster(self):
        arr = np.zeros((self.height, self.width))
        n = 0
        for cluster in self.clusters:
            if len(cluster.pixels) != 0:
                n += 1
            for p in cluster.pixels:
                arr[p[0]][p[1]] = n

        return arr

    def iterates(self):
        arr = []
        self.init_clusters()
        self.move_clusters()
        for i in range(20):
            self.assignment()
            self.update_cluster()
        arr = self.save_current_cluster()
        return arr