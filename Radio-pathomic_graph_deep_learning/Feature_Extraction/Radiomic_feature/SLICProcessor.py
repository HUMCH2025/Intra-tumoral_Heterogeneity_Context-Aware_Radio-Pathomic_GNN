import numpy as np
import random

class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, d, f1=0, f2=0, f3=0):
        self.update(h, w, d, f1, f2, f3)
        self.pixels = []
        self.no = self.cluster_index
        self.cluster_index += 1

    def update(self, h, w, d, f1, f2, f3):
        self.h = h
        self.w = w
        self.d = d
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

    def print_no(self):
        return self.no

    def __str__(self):
        return "{},{},{}:{} {} {}".format(self.h, self.w, self.d, self.f1, self.f2, self.f3)

    def __repr__(self):
        return self.__str__()

    class SLICProcessor(object):
        @staticmethod
        def get_data(data):
            return data

        @staticmethod
        def make_cluster(self, h, w, d):
            h = int(h)
            w = int(w)
            d = int(d)
            return Cluster(h, w, d,
                           self.data[h][w][d][0],
                           self.data[h][w][d][1],
                           self.data[h][w][d][2]
                           )

        def __init__(self, data, K, M):
            self.K = K
            self.M = M
            self.data = self.get_data(data)
            self.height = self.data.shape[0]
            self.width = self.data.shape[1]
            self.depth = self.data.shape[2]
            self.N = self.height * self.width * self.depth
            self.S = int((self.N / self.K) ** 0.34)

            self.clusters = []
            self.label = {}
            self.dis = np.full((self.height, self.width, self.depth), np.inf)

        def init_clusters(self):
            h = self.S / 2
            w = self.S / 2
            d = self.S / 2
            while h < self.height:
                while w < self.width:
                    while d < self.depth:
                        self.clusters.append(self.make_cluster(self, h, w, d))
                        d += self.S
                    d = self.S / 2
                    w += self.S
                w = self.S / 2
                h += self.S

        def get_gradient(self, h, w, d):
            if d + 1 >= self.depth:
                d = self.depth - 2
            if w + 1 >= self.width:
                w = self.width - 2
            if h + 1 >= self.height:
                h = self.height - 2

            gradient = (self.data[h + 1][w][d][0] + self.data[h][w + 1][d][0] + self.data[h][w][d + 1][0] - 3 *
                        self.data[h][w][d][0]) + \
                       (self.data[h + 1][w][d][1] + self.data[h][w + 1][d][1] + self.data[h][w][d + 1][1] - 3 *
                        self.data[h][w][d][1]) + \
                       (self.data[h + 1][w][d][2] + self.data[h][w + 1][d][2] + self.data[h][w][d + 1][2] - 3 *
                        self.data[h][w][d][2])
            return gradient

        def move_clusters(self):
            for cluster in self.clusters:
                cluster_gradient = self.get_gradient(cluster.h, cluster.w, cluster.d)
                for dh in range(-1, 2):
                    for dw in range(-1, 2):
                        for dd in range(-1, 2):
                            _h = cluster.h + dh
                            _w = cluster.w + dw
                            _d = cluster.d + dd
                            new_gradient = self.get_gradient(_h, _w, _d)
                            if new_gradient < cluster_gradient:
                                cluster.update(_h, _w, _d,
                                               self.data[_h][_w][_d][0],
                                               self.data[_h][_w][_d][1],
                                               self.data[_h][_w][_d][2])
                                cluster_gradient = new_gradient

        def assignment(self):
            for cluster in self.clusters:
                n = 1
                h_list = list(range(cluster.h - 2 * self.S, cluster.h + 2 * self.S))
                w_list = list(range(cluster.w - 2 * self.S, cluster.w + 2 * self.S))
                d_list = list(range(cluster.d - 2 * self.S, cluster.d + 2 * self.S))
                while n < 1000:
                    h = random.choice(h_list)
                    w = random.choice(w_list)
                    d = random.choice(d_list)
                    if h < 0 or h >= self.height:
                        continue
                    if w < 0 or w >= self.width:
                        continue
                    if d < 0 or d >= self.depth:
                        continue
                    f1, f2, f3 = self.data[h][w][d]
                    "feature"
                    Dc = (
                                 (f1 - cluster.f1) ** 2 +
                                 (f2 - cluster.f2) ** 2 +
                                 (f3 - cluster.f3) ** 2 ) ** 0.5
                    "position"
                    Ds = (
                                 (h - cluster.h) ** 2 +
                                 (w - cluster.w) ** 2 +
                                 (d - cluster.d) ** 2) ** 0.5
                    D = ((Dc / self.M) ** 2 + (Ds / self.S) ** 2) ** 0.5
                    if D < self.dis[h][w][d]:
                        if (h, w, d) not in self.label:
                            self.label[(h, w, d)] = cluster
                            cluster.pixels.append((h, w, d))
                        else:
                            self.label[(h, w, d)].pixels.remove((h, w, d))
                            self.label[(h, w, d)] = cluster
                            cluster.pixels.append((h, w, d))
                        self.dis[h][w][d] = D
                    n += 1

        def update_cluster(self):
            for cluster in self.clusters:
                sum_h = sum_w = sum_d = number = 0
                for p in cluster.pixels:
                    sum_h += p[0]
                    sum_w += p[1]
                    sum_d += p[2]
                    number += 1
                    _h = int(sum_h / number)
                    _w = int(sum_w / number)
                    _d = int(sum_d / number)
                    cluster.update(_h, _w, _d,
                                   self.data[_h][_w][_d][0],
                                   self.data[_h][_w][_d][1],
                                   self.data[_h][_w][_d][2])

        def save_current_cluster(self):
            arr = np.zeros((self.height, self.width, self.depth))
            n = 0
            for cluster in self.clusters:
                if len(cluster.pixels) != 0:
                    n += 1
                for p in cluster.pixels:
                    arr[p[0]][p[1]][p[2]] = n

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