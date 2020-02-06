import numpy
import pandas


def partition_column(column, n=3):
    partitioned = column.apply(lambda x: numpy.random.dirichlet(numpy.ones(n),size=1).ravel() * x).values
    return [[i[j] for i in partitioned] for j in range(n)]


def hierarchical_sine_data(start, end, n=10000):
    dts = (end - start).total_seconds()
    dti = pandas.DatetimeIndex([start + pandas.Timedelta(numpy.random.uniform(0, dts), 's')
                                for _ in range(n)]).sort_values()
    time = numpy.arange(0, len(dti), 0.01)
    amplitude = numpy.sin(time) * 10
    amplitude += numpy.random.normal(2 * amplitude + 2, 5)
    df = pandas.DataFrame(index=dti, data={'total': amplitude[0:len(dti)]})
    df['a'], df['b'], df['c'], df['d'] = partition_column(df.total, n=4)
    df['aa'], df['ab'] = partition_column(df.a, n=2)
    df['aaa'], df['aab'] = partition_column(df.aa, n=2)
    df['ba'], df['bb'], df['bc'] = partition_column(df.b, n=3)
    df['ca'], df['cb'], df['cc'], df['cd'] = partition_column(df.c, n=4)
    return df


class Node:
    def __init__(self, val):
        self.val = val
        self.children = []


def createNode(tree, root, b=None, stack=None):
    if stack is None:
        stack = []  # stack to store children values
        root = Node(root)  # root node is created
        b = root  # it is stored in b variable
    x = root.val  # root.val = 2 for the first time

    if len(tree[x]) > 0:  # check if there are children of the node exists or not
        for i in range(len(tree[x])):  # iterate through each child
            y = Node(tree[x][i])  # create Node for every child
            root.children.append(y)  # append the child_node to its parent_node
            stack.append(y)  # store that child_node in stack
            if y.val == 0:  # if the child_node_val = 0 that is the parent = leaf_node
                stack.pop()  # pop the 0 value from the stack
        if len(stack):  # iterate through each child in stack
            if len(stack) >= 2:  # if the stack length >2, pop from bottom-to-top
                p = stack.pop(0)  # store the popped val in p variable
            else:
                p = stack.pop()  # pop the node top_to_bottom
        createNode(tree, p, b, stack)  # pass p to the function as parent_node
    return b
