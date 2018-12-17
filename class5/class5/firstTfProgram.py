from tensorflow.examples.tutorials.mnist import input_data
# 如果没有会去官网下载 数据路径class5/MNIST_data one_hot=True 十位onehot编码
mnist = input_data.read_data_sets('class5/MNIST/', one_hot=True)
# 加上one_hot=True会使得图像的标签读成10位的onehot编码，不加是0-9的标量
#训练集
# print("Train data ")
# print(mnist.train.images.shape,mnist.train.labels.shape)
# #验证集
# print("validation data ")
# print(mnist.validation.images.shape,mnist.validation.labels.shape)
# #测试集
# print("test data ")
# print(mnist.test.images.shape,mnist.test.labels.shape)
# #
# print("Train data example",mnist.train.images[0])
#
#
# #matplotlib展示前8张图
# import matplotlib.pyplot as plt
# from tensorflow.contrib import learn
#
# for i in range(0,8):
#
#     plt.subplot(2, 8,i+1)
#     plt.axis('off')
#     plt.imshow(mnist.train.images[i].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
#     print(mnist.train.labels[i])

import tensorflow as tf
def add_layer(input,input_size,output_size,activation_function=None): # activation_function 激活函数 None时是线性变换
    with tf.name_scope('weight'):
        w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))# 正太分度初始化 矩阵 大小是上一层乘下一层
        variable_summary(w)
    with tf.name_scope('bias'):
        b = tf.Variable(tf.constant(0.1, shape=[output_size]))#确定wx+b维度，神经元个数
        variable_summary(b)
    with tf.name_scope('Z'):
        Z = tf.add(tf.matmul(input, w), b)#Z = wx+b
        variable_summary(Z)
    if activation_function == None:#激活
        a = Z
    else:
        a = activation_function(Z)#a = f(z)
    with tf.name_scope('activation'):
        variable_summary(a)
    return a
# 变量日志
def variable_summary(var):
    with tf.name_scope('summary'):#关注均值所有节点的均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)#写入summary以标量的形式
        with tf.name_scope('stddev'):#求标差
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram',var)#变量以直方图记录下
#输入层
with tf.name_scope("input_layer"):
    input_images = tf.placeholder(tf.float32,[None,784]) #placeholder占位不知道到底有多少数据 none表示来多少是多少 任意行784列
#隐藏层
with tf.name_scope("layer1"):
    hidden1 = add_layer(input_images,784,16,tf.nn.relu)
    # hidden2 = add_layer(hidden1,522,174,tf.nn.relu)
# hidden3 = add_layer(hidden2,174,58,tf.nn.relu)





# 隐藏层数量 (in+out)*2/3 还有 开根号加10
#输出层
with tf.name_scope("output_layer"):
    output_layer = add_layer(hidden1,16,10)
# output_layer = add_layer(hidden1,784,10)# 无隐藏层时  不激活因为直接输出和激活后输出的0-1之间的差不多输出层直接得到结果  a=z
    output = tf.nn.softmax(output_layer)#softmax 输出符合概率分布的结果e^x/Σe^x

# 预测
def predict(output,label):
    #argmax获取到one-hot编码中最大值的索引，与label里正确答案的索引对比，判定单样本是否预测正确
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))# axis=0纵向是有多少数据，axis=1横向的是对应的每个特征argmax指最大的索引 相等记1不等记0
    #求多样本平均正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #cast转换数据形式
    return accuracy

# 损失函数
def crossentropy(label,logit):
    # reduction_indices求沿1方向的
    return -tf.reduce_sum(tf.multiply(label,tf.log(logit)),reduction_indices=[1])

with tf.name_scope("accuracy"):
    label = tf.placeholder(tf.float32,[None,10])
    accuracy = predict(output,label)# 输入灌进去，求正确率
    tf.summary.scalar("accuracy", accuracy)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(crossentropy(label, output))
    tf.summary.scalar('loss', loss)

with tf.name_scope("optimizer"):#梯度下降优化器
    op = tf.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(loss)

# 合并所有summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("class5/log/", sess.graph)
    for i in  range(10000):
        #调用封装好的API构建minibatch
        batch_image,batch_label = mnist.train.next_batch(100)
        #使用minibatch训练
        _= sess.run(op, feed_dict={input_images: batch_image, label: batch_label}) #占位目前数据不处理良好习惯
        # op1 = sess.run(op, feed_dict={input_images: batch_image, label: batch_label})
        # 每100次迭代，检查一次
        if i % 100 == 0:
            train_result = sess.run(accuracy, feed_dict={input_images: mnist.train.images, label: mnist.train.labels})
            test_result, cost, merged_summary = sess.run([accuracy, loss, merged],
                                                         feed_dict={input_images: mnist.test.images,

                                                                 label: mnist.test.labels})
            writer.add_summary(merged_summary, i)
            print("step %d,train_accuracy= %g ,test_accuracy= %g,loss is:%f" % (i, train_result, test_result, cost))




            # 在没有激活函数的情况下跑的准确率（10000）大概是step 9900,train_accuracy= 0.924418 ,test_accuracy= 0.9228,loss is:0.274043
            # 加了一层step 9900,train_accuracy= 0.9594 ,test_accuracy= 0.9537,loss is:0.164562
            # 1/3 情况下 step 9900,train_accuracy= 0.999727 ,test_accuracy= 0.9802,loss is:0.068632