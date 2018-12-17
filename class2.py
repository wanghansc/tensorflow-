import tensorflow as tf
z = tf.get_variable("my_variable", [1,2,3])
y2 = tf.get_variable("y",shape=[3,3])
# y3 = tf.get_variable("y",shape=[1,3])

# 创建卷积神经网络中层
def conv_relu(input, kernel_shape, bias_shape):
     # Create variable named "weights".
     weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
     # Create variable named "biases".
     biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
     conv = tf.nn.conv2d(input, weights,
            strides=[1,1,1,1], padding='SAME')
     return tf.nn.relu(conv + biases)
# 错误示范
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
# x = conv_relu(input1, kernel_shape = [5,5,32,32], bias_shape=[32])
# x = conv_relu(x, kernel_shape = [5,5,32,32], bias_shape=[32])
#1 使用variable scope实现变量共享
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variable created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5,5,32,32],[32])
    with tf.variable_scope("conv2"):
        # Variable created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
x1 = my_image_filter(input1)
#2 reuse
with tf.variable_scope("scope1"):
    v = tf.get_variable('v',[1],initializer=tf.constant_initializer(1.0))
    print(v.name)
with tf.variable_scope("scope1",reuse = True):
    v1 = tf.get_variable('v',[1])
    print(v1.name)
# 复用时scope里变量名必须是出现过的，不然报错
with tf.variable_scope("scpoe2", reuse=True):
    v8 = tf.get_variable('v',[1])
 

