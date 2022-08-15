import random
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tag = -1


class Center:
    def __init__(self, p):
        self.p = p  # 记录均值向量


def distance(p1, p2):
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2


def k_means():
    k = 2
    n = int(input("input number of points: "))  # 输入点的数量
    points = []
    for i in range(n):  # 随机产生一系列点
        if i % 2 == 0:
            x = random.random() / 2
            y = random.random() / 2
        else:
            x = random.random() / 2 + 0.5
            y = random.random() / 2 + 0.5
        points.append(Point(x, y))

    centers = []
    for i in range(k):
        centers.append(Center(points[i]))  # 随机选k个样本作为均值向量初始值

    plt.ion()  # 开启交互模式
    plt.subplots()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    while 1:
        for i in range(k):
            print("[{0:.2f},{1:.2f}]".format(centers[i].p.x, centers[i].p.y), end=" ")
        print()
        for e in points:
            min_dis = float("inf")
            min_k = -1
            for i in range(k):
                dis = distance(centers[i].p, e)
                if dis < min_dis:
                    min_dis = dis
                    min_k = i

            e.tag = min_k  # 记录改点所在的组
            print(e.tag)

        new_centers = []
        for i in range(k):
            new_centers.append(Center((Point(0, 0))))

        nums = [0] * k
        x = [e.x for e in points]
        y = [e.y for e in points]
        c = [e.tag for e in points]
        for e in points:
            nums[e.tag] += 1
            new_centers[e.tag].p.x += e.x
            new_centers[e.tag].p.y += e.y

        for i in range(k):
            print(i, new_centers[i].p.x, new_centers[i].p.y)
            if nums[i] != 0:
                new_centers[i].p.x /= nums[i]
                new_centers[i].p.y /= nums[i]
                print(i, new_centers[i].p.x, new_centers[i].p.y)
            else:
                new_centers[i].p.x = 0.5
                new_centers[i].p.y = 0.5
        for i in range(k):
            print(i, new_centers[i].p.x, new_centers[i].p.y)

        plt.scatter(x, y, c=c)
        plt.pause(0.5)
        plt.clf()  # 清空画布
        plt.xlim(0, 1)  # 因为清空了画布，所以要重新设置坐标轴的范围
        plt.ylim(0, 1)

        flag = False
        for i in range(k):
            if abs(distance(centers[i].p, new_centers[i].p)) > 0.000000001:
                flag = True

        if not flag:
            break
        else:
            centers = new_centers

        for i in range(k):
            print("[{0:.2f},{1:.2f}]".format(centers[i].p.x, centers[i].p.y), end=" ")
        print()


    for i in range(k):
        print(i, end=":")
        for e in points:
            if e.tag == i:
                print("[{0:.2f},{1:.2f}]".format(e.x, e.y), end=" ")

        print()
    x = [e.x for e in points]
    y = [e.y for e in points]
    c = [e.tag for e in points]
    plt.scatter(x, y, c=c)
    plt.ioff()
    plt.show()


def main():
    k_means()


if __name__ == "__main__":
    main()
