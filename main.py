# coding=utf-8
import tensorflow as tf
import input_data

# 下载MNINST数据集到MNIST_data文件夹并解压
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 设置权重weights和偏置biases作为优化变量，初始值设为0
weights = tf.Variable(tf.zeros([784, 10]))                                                          # 权重初始化为784 * 10的零矩阵
biases = tf.Variable(tf.zeros([10]))                                                                # 偏置项置为1 * 10的零矩阵

# 构架模型
x = tf.placeholder("float", [None, 784])
y = tf.nn.softmax(tf.matmul(x,weights) + biases)                                                    # 模型的预测值
y_real = tf.placeholder("float", [None, 10])                                                        # 真实值
cross_entropy = -tf.reduce_sum(y_real * tf.log(y))                                                  # 预测值和真实值的交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)                        # 使用梯度下降优化最小交叉熵

# 开始训练
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)                                                  # 每次随机选取100个数据进行训练，运用随机梯度下降算法
    sess.run(train_step, feed_dict={x: batch_xs, y_real: batch_ys})                                   # 将x,y_real给定，并执行上面构建的模型train_step

    if i % 50== 0:                                                                                    # 每训练50次后评估模型
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_real, 1))                        # 比较预测值和真实值是否一致
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                               # 在每100个测试样本中计算出正确的个数的占比（准确率）
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_real: mnist.test.labels}))        # 将x,y_real给定，并执行上面构建的模型accuracy