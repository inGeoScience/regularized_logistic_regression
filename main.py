import numpy
import pandas
from matplotlib import pyplot


# 特征映射
def map_feature(X_to_map, degree=6):
    """

    :param X_map: (ndarray Shape(m, n)) dataset, m samples by n features
    :param degree: the degree of the polynomial you would like to map to
    :return: a mapped dataset with more features of higher degree
    """
    out = []
    for i in range(1, degree+1):
        for j in range(i+1):
            out.append((X_to_map[:, 0]**(i-j) * (X_to_map[:, 1]**j)))
    return numpy.stack(out, axis=1)


# sigmoid function
def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


# cost function for logistic regression
def compute_regularized_cost(X, y, w_in, b_in, lambda_=0.):
    """
    compute the cost over all examples
    :param X: (ndarray Shape(m, n)) dataset, m samples by n features
    :param y: (array_like Shape(m,)) target values for all samples
    :param w_in: (array_like Shape(n,)) values of parameters of the model
    :param b_in: scalar Values of bias parameter of the model
    :param lambda_: the parameter of regularization
    :return: the regularized loss
    """
    m, n = X.shape
    cost_without_reg = 0
    for i in range(m):
        cost_without_reg += (y[i] * numpy.log(sigmoid(numpy.dot(w_in, X[i]) + b_in)) + (1 - y[i]) * numpy.log(1 - sigmoid(numpy.dot(w_in, X[i]) + b_in)))
    cost_without_reg = (-1 / m) * cost_without_reg
    reg_cost = 0
    for j in range(n):
        reg_cost += w_in[j]**2
    reg_cost = (lambda_ / (2 * m)) * reg_cost
    return cost_without_reg + reg_cost


def compute_regularized_gradient(X, y, w_in, b_in, lambda_=0):
    """
    compute gradient for each iteration
    :param X: (ndarray Shape(m, n)) dataset, m samples by n features
    :param y: (array_like Shape(m,)) target values for all samples
    :param w_in: (array_like Shape(n,)) values of parameters of the model
    :param b_in: scalar Values of bias parameter of the model
    :param lambda_: the parameter of regularization
    :return: the gradient dj_dw and dj_db
    """
    m, n = X.shape
    dj_dw = numpy.zeros(n)
    dj_db = 0
    # 计算w的梯度
    for j in range(n):
        for i in range(m):
            dj_dw[j] += (sigmoid(numpy.dot(w_in, X[i]) + b_in) - y[i]) * X[i][j]
    dj_dw /= m
    # 为每一个特征的梯度末尾添加正则化项的偏导数
    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w_in[j]
    # 计算截距项的梯度
    for i in range(m):
        dj_db += (sigmoid(numpy.dot(w_in, X[i]) + b_in) - y[i])
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, iters, lambda_=0.):
    """
    run gradient descent
    :param X: (ndarray Shape(m, n)) dataset, m samples by n features
    :param y: (array_like Shape(m,)) target values for all samples
    :param w_in: (array_like Shape(n,)) values of parameters of the model
    :param b_in: scalar value of bias parameter of the model
    :param alpha: learning rate α
    :param iters: iterations for gradient descent
    :param lambda_: the parameter of regularization
    :return:
    """
    m_in, n_in = X.shape
    for inum in range(iters):
        dj_dw, dj_db = compute_regularized_gradient(X, y, w_in, b_in, lambda_)
        for j in range(n_in):
            w_in[j] = w_in[j] - alpha * dj_dw[j]
        b_in = b_in - alpha * dj_db
    loss_in = compute_regularized_cost(X, y, w_in, b_in, lambda_)
    return w_in, b_in, loss_in


def predict(X_pred, w_pred, b_pred):
    """
    make predictions with learned w and b
    :param X_pred: data set with m samples and n features
    :param w_pred: values of parameters of the model
    :param b_pred: scalar value of bias parameter of the model
    :return:
    """
    predictions = sigmoid(numpy.dot(X_pred, w_pred) + b_pred)
    p = [1 if item >= 0.5 else 0 for item in predictions]
    return numpy.array(p)


def plot_data(X_train, y, ax, positive_label='y=1', negative_label='y=0'):
    # 正例列表，当y=1时为True，否则为False。
    # 需要注意的是shape为(m,1)，用它过滤shape不为(m,1)或(m,)的ndarray时会匹配不上。
    # 而这里X_train为(100,2)，因此需要将Boolean list通过.reshape(-1)转为(m,)
    positive = y == 1
    # 负例列表，当y=0时为True，否则为False。需要注意的是shape为(m,1)
    negative = y == 0
    # 通过上述Boolean list绘制正例。用pyplot.scatter()也行
    ax.plot(X_train[positive.reshape(-1), 0], X_train[positive.reshape(-1), 1], 'k+', label=positive_label)
    # 通过上述Boolean list绘制负例
    ax.plot(X_train[negative.reshape(-1), 0], X_train[negative.reshape(-1), 1], 'yo', label=negative_label)


def plot_decision_boundary(w_db, b_db, ax):
    # 画等高线
    x1_ax_contour = numpy.linspace(-1, 1.5, 100)
    x2_ax_contour = numpy.linspace(-1, 1.5, 100)
    values_contour = numpy.zeros((len(x1_ax_contour), len(x2_ax_contour)))
    for i in range(len(x1_ax_contour)):
        for j in range(len(x2_ax_contour)):
            values_contour[i][j] = sigmoid(numpy.dot(map_feature(numpy.array([x1_ax_contour[i], x2_ax_contour[j]]).reshape((1, 2))), w_db) + b_db)
    ax.contour(x1_ax_contour, x2_ax_contour, values_contour.T, levels=[0.5], colors='red')


# 导入数据到一个DataFrame
data = pandas.read_csv('data/data_regular.txt', header=None)
# 获取列数（包括输出变量）
colNum = data.shape[1]
# 定义数据集（不含输出变量）
X_train = data.iloc[:, :colNum - 1]
X_train = numpy.array(X_train.values)
# 特征映射
X_mapped = map_feature(X_train)
# 定义输出空间
y_train = data.iloc[:, colNum - 1: colNum]
y_train = numpy.array(y_train.values)
# 获取样本量和特征数
m, n = X_mapped.shape
# 初始化w, b
numpy.random.seed(1)
w_init = numpy.random.rand(X_mapped.shape[1]) - 0.5
b_init = 1.
# 正则化梯度下降求解模型参数
w, b, loss = gradient_descent(X_mapped, y_train, w_init, b_init, alpha=0.01, iters=10000, lambda_=0.01)
print(f'w = {w}\nb = {b}\nloss = {loss}')
# 根据学习得到的参数进行目标变量的预测
pred = predict(X_mapped, w, b)
# 将预测的目标变量与真实目标变量相比，得到准确率
accuracy = numpy.mean(pred == y_train.reshape(-1)) * 100
print(f'Train Accuracy: {accuracy}')
# 初始化figure和axis
fig, ax = pyplot.subplots()
# 展示数据
plot_data(X_train=X_train, y=y_train, ax=ax)
# 绘决策边界
plot_decision_boundary(w, b, ax)
pyplot.show()
