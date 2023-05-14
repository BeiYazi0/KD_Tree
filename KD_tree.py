from typing import List
from collections import namedtuple
import time
import random

from matplotlib import pyplot as plt
import numpy as np


class Point(namedtuple("Point", "x y")):
    def __repr__(self) -> str:
        return f'Point{tuple(self)!r}'


class Rectangle(namedtuple("Rectangle", "lower upper")):
    def __repr__(self) -> str:
        return f'Rectangle{tuple(self)!r}'

    def is_contains(self, p: Point) -> bool:
        return self.lower.x <= p.x <= self.upper.x and self.lower.y <= p.y <= self.upper.y


class Node(namedtuple("Node", "location left right")):
    """
    location: Point
    left: Node
    right: Node
    """

    def __repr__(self):
        return f'{tuple(self)!r}'


class KDTree:
    """k-d tree"""

    def __init__(self):
        self._root = None # 根节点 
        self._n = 0       # 节点数

    def insert(self, p: List[Point]):
        """insert a list of points"""
        def x_insert(p: List[Point]):
            n = len(p)
            # 排序确定中位数节点
            p.sort(key = lambda point:point.x) 
            mid = n//2
            if mid > 0:
                pl = p[0:mid]
                # 根据节点划分双支数据
                if mid+1 < n:
                    pr = p[mid+1:]
                    curnode = Node(p[mid], y_insert(pl), y_insert(pr))
                else:
                    curnode = Node(p[mid], y_insert(pl), None)
            else:
                # 根据节点划分双支数据
                if mid+1 < n:
                    pr = p[mid+1:]
                    curnode = Node(p[mid], None, y_insert(pr))
                else:
                    curnode = Node(p[mid], None, None)
            return curnode

        def y_insert(p: List[Point]):
            n = len(p)
            # 排序确定中位数节点
            p.sort(key = lambda point:point.y)
            mid = n//2
            if mid > 0:
                pl = p[0:mid]
                # 根据节点划分双支数据
                if mid+1 < n:
                    pr = p[mid+1:]
                    curnode = Node(p[mid], x_insert(pl), x_insert(pr))
                else:
                    curnode = Node(p[mid], x_insert(pl), None)
            else:
                # 根据节点划分双支数据
                if mid+1 < n:
                    pr = p[mid+1:]
                    curnode = Node(p[mid], None, x_insert(pr))
                else:
                    curnode = Node(p[mid], None, None)
            return curnode
        self._root = x_insert(p)
        self._n = len(p)

    def range(self, rectangle: Rectangle) -> List[Point]:
        """range query"""
        ans = []
        # 确定区域的范围
        x1 = rectangle.lower.x
        x2 = rectangle.upper.x
        y1 = rectangle.lower.y
        y2 = rectangle.upper.y
        def x_find(curnode: Node):
            if curnode == None:
                return
            x = curnode.location.x
            y = curnode.location.y
            # 当前节点的 x 坐标小于 xmin，说明需要查询右子树
            if x < x1:
                y_find(curnode.right)
            # 当前节点可能在区域内（还需要判断y轴），需要查询所有子树
            elif x >= x1 and x <= x2:
                if y >= y1 and y <= y2:
                    ans.append(curnode.location)
                y_find(curnode.left)
                y_find(curnode.right)
            # 当前节点的 x 坐标大于 xmax，说明需要查询左子树
            else:
                y_find(curnode.left)

        def y_find(curnode: Node):
            if curnode == None:
                return
            x = curnode.location.x
            y = curnode.location.y
            # 当前节点的 y 坐标小于 ymin，说明需要查询右子树
            if y < y1:
                x_find(curnode.right)
            # 当前节点可能在区域内（还需要判断 x 轴），需要查询所有子树
            elif y >= y1 and y <= y2:
                if x >= x1 and x <= x2:
                    ans.append(curnode.location)
                x_find(curnode.left)
                x_find(curnode.right)
            # 当前节点的 y 坐标大于 ymax，说明需要查询左子树
            else:
                x_find(curnode.left)
        x_find(self._root)
        return ans

    def search_nearest(self, pos: Point):
        x0, y0 = pos.x, pos.y
        global n_dst, ans
        n_dst = 1e9
        ans = None
        def x_search(curnode: Node):
            x, y = curnode.location.x, curnode.location.y
            # 计算目标节点与当前节点的距离
            dst = ((x - x0)**2 + (y - y0)**2)**0.5
            global n_dst, ans
            # 修改最小距离和相应的最近节点
            if dst < n_dst:
                n_dst = dst
                ans = curnode.location
            # 目标节点在分割轴右侧
            if x0 > x:
                # 检索右子树
                if curnode.right != None:
                    y_search(curnode.right)
                dst = abs(x-x0) # 目标节点与分割轴的距离
                # 如果目标节点与分割轴的距离小于目前的最小距离，说明在另一子树中可能存在更小距离，否则不探索另一子树
                if dst < n_dst:
                    if curnode.left != None:
                        y_search(curnode.left)
            # 目标节点在分割轴左侧
            else:
                # 检索左子树
                if curnode.left != None:
                    y_search(curnode.left)
                dst = abs(x-x0) # 目标节点与分割轴的距离
                # 如果目标节点与分割轴的距离小于目前的最小距离，说明在另一子树中可能存在更小距离，否则不探索另一子树
                if dst < n_dst:
                    if curnode.right != None:
                        y_search(curnode.right)

        def y_search(curnode: Node):
            x, y = curnode.location.x, curnode.location.y
            # 计算目标节点与当前节点的距离
            dst = ((x - x0)**2 + (y - y0)**2)**0.5
            global n_dst, ans
            # 修改最小距离和相应的最近节点
            if dst < n_dst:
                n_dst = dst
                ans = curnode.location
            # 目标节点在分割轴右侧
            if y0 > y:
                # 检索右子树
                if curnode.right != None:
                    x_search(curnode.right)
                dst = abs(y-y0) # 目标节点与分割轴的距离
                # 如果目标节点与分割轴的距离小于目前的最小距离，说明在另一子树中可能存在更小距离，否则不探索另一子树
                if dst < n_dst:
                    if curnode.left != None:
                        x_search(curnode.left)
            else:
                # 检索左子树
                if curnode.left != None:
                    x_search(curnode.left)
                dst = abs(y-y0) # 目标节点与分割轴的距离
                # 如果目标节点与分割轴的距离小于目前的最小距离，说明在另一子树中可能存在更小距离，否则不探索另一子树
                if dst < n_dst:
                    if curnode.right != None:
                        x_search(curnode.right)
        x_search(self._root)
        return ans

# 范围检索正确性测试
def range_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6)))
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])


# 范围检索正确性测试，比较直接检索和 KD 树检索的花费时间
def performance_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]

    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)
    #  naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    print(f'Naive method search range: {end - start}ms')

    kd = KDTree()
    kd.insert(points)
    # k-d tree
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    print(f'K-D tree search range: {end - start}ms')

    assert sorted(result1) == sorted(result2)


# 直接检索最邻近的点
def naive_search(pos: Point, points: List[Point]):
    x0, y0 = pos.x, pos.y
    n_dst = 1e9
    ans = None
    for point in points:
        x, y = point.x, point.y
        dst = ((x - x0)**2 + (y - y0)**2)**0.5
        if dst < n_dst:
            n_dst = dst
            ans = point
    return ans


# 最近邻检索正确性测试，比较直接检索和 KD 树检索的花费时间
def search_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]
    
    kd = KDTree()
    kd.insert(points)
    
    kdresult = []
    result = []
    
    start = int(round(time.time() * 1000))
    for i in range(1,1000, 50):
        point = Point(i+0.1,i+0.1)
        result.append(naive_search(point, points))
    end = int(round(time.time() * 1000))
    print(f'Naive method search nearest: {end - start}ms')
    
    start = int(round(time.time() * 1000))
    for i in range(1,1000, 50):
        point = Point(i+0.1,i+0.1)
        kdresult.append(kd.search_nearest(point))
    end = int(round(time.time() * 1000))
    print(f'K-D tree search nearest: {end - start}ms')

    assert result == kdresult


# 图像绘制
def plot_table(row, col, vals1, vals2):
    xticks = np.arange(len(row))
    fig, ax = plt.subplots(figsize=(10, 7))

    plt.rcParams["font.sans-serif"]=['SimHei']
    plt.rcParams["axes.unicode_minus"]=False
    ax.bar(xticks, vals1, fc="r", width=0.3, label=col[0])
    ax.bar(xticks + 0.3,vals2, fc="b", width=0.3, label=col[1])
 
    ax.set_title("Naive method和K-D tree耗时对比---柱状图")
    ax.set_xlabel("数量级")
    ax.set_ylabel("耗时(ms)")
    ax.legend()

    ax.set_xticks(xticks+0.3)
    ax.set_xticklabels(row)
    
    plt.show()


# 指定数量级的点数进行范围检索，比较直接检索和 KD 树检索的花费时间
def Rectangle_contain_test(points: List[Point], rectangle: Rectangle):
    #  naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    tn = end - start

    kd = KDTree()
    kd.insert(points)
    # k-d tree
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    tkd = end - start

    assert sorted(result1) == sorted(result2)
    return tn, tkd


# 不同数量级下进行范围检索，比较直接检索和 KD 树检索的花费时间
def Rectangle_test():
    lower = Point(500, 500)
    upper = Point(600, 600)
    rectangle = Rectangle(lower, upper)

    tn_lst = []
    tkd_lst = []
    row = []
    for k in range(3, 8):
        row.append(k)
        nums = 10**k
        points = [Point(random.randint(0,1000), random.randint(0,1000)) for _ in range(nums)]
        tn, tkd = Rectangle_contain_test(points, rectangle)
        tn_lst.append(tn)
        tkd_lst.append(tkd)

    col = [ 'naive method', 'K-D tree']
    plot_table(row, col, tn_lst, tkd_lst)


if __name__ == '__main__':
    random.seed(42)
    range_test()
    performance_test()
    search_test()
    Rectangle_test()
