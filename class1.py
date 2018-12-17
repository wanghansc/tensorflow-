# 默认图里创操作
import tensorflow as tf
c = tf.constant(1.0)#创建常量
a = 1.0
print(c.graph)
print(tf.get_default_graph)
print(c.name)
####################
#上下文管理器选择图
# 将新图加载到我想要的图里
g = tf.Graph()#创建一个类
print("g:",g)
with g.as_default():
    d = tf.constant(1.0)
    print(d.graph)

# 位图分配不同的运行设备
# 1 面向过程
g2 = tf.Graph()
with g.device('/gpu:0'):
    a = tf.constant(1.5)
    with g.device('/cpu:1'):
        pass

with g2.device('/cpu:0'):
    b = 1

#2 面向对象
def op_on_gpu(n):
    if n.type == "gpu":
        return '/gpu:0'
    else:
        return '/cpu:0'
with g.device(op_on_gpu):
    pass



# namescope
with tf.name_scope("A"):
    v1 = tf.Variable([1],name="v1")
    print (v1)
    with tf.name_scope("B"):
        v2 = tf.Variable([1],name="v2")
with tf.name_scope("C"):
    v3 = tf.Variable([1],name="v3")
print ('v1.name: ', v1.name)
print ('v2.name: ', v2.name)
print ('v3.name: ', v3.name)

# Session
node1 = tf.constant(value = 1.5)
node2 = tf.constant(3.0)
sum1 = tf.add(node1, node2)#node1+node2
print (node1,node2,sum1)
#不显示原因：session未运行，即静态
# 1
# 一对一
sess  = tf.Session()
print(sess.run(node1))
print(sess.run(node2))
print(sess.run(sum1))
sess.close()

# 2
with tf.Session() as sess:
    print(sess.run(sum1))
# 报错示范
# print(sess.run(sum1))#当其跳出上下文管理器时session是关闭的sess运行不了：RuntimeError: Attempted to use a closed Session.

# 3
sess2 = tf.InteractiveSession()
# 一对多，避免session被锁死
#互动式session,好处后面的默认使用这个session，便于单步调试
# tf.InteractiveSession()
# = with tf.Session() as sess:
# = with sess.set default()
print(node1.eval())

# Variable
# 先声明一个variable变量，用tf.ones作为初始赋值
# tf.ones(shape1)表示生成一个shape1的张量，所有值都为1.0
var1 = tf.Variable(tf.ones([3,3]))
print(var1)
# 输出所有未初始化的变量名
print(sess2.run(tf.report_uninitialized_variables()))
# 初始化变量值
sess2.run(tf.global_variables_initializer())
# 输出变量
print(sess2.run(var1))
# tf.ones(shape1)表示生成一个shape1的张量，所有值都为0.0
var2 = tf.Variable(tf.zeros([3,3]))
sess2.run(var2.initializer)
print(sess2.run(var2))

# 变量共享
z = tf.get_variable("my_variable",[1, 2, 3])#防止生成重复的变量名
y2 = tf.get_variable("y",shape= [3,3])



# #placeholder
# # 声明p1,p2两个占位符，数据格式为float32
# p1 = tf.placeholder(tf.float32)
# p2 = tf.placeholder(tf.float32)
# # result = p1*p2
# result = tf.multiply(p1, p2)
# print(p1,p2,result)
# # 通过feed dict的方式,把数据以字典的形式传入占位符
# print(sess2.run(result,feed_dict={p1:[13.5,2.0],p2:[3.2,5.0]}))
