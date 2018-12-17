#tensorflow 常用函数
import tensorflow as tf
sess = tf.InteractiveSession()
# cast
x = tf.constant([3.9,1.2])
print(x)
# Tensor("Const:0", shape=(2,), dtype=float32)
y = tf.cast(x,tf.int32)
print(sess.run(y))
# [3 1]

# rank表示数据的维度
y0 = tf.constant(3)
ranky0 = tf.rank(y0)

print('rank y0 = '+str(sess.run(ranky0)))
# rank y0 = 0
y1 = [3,3],[2,2]
ranky1 = tf.rank(y1)
print('rank y1 = '+str(sess.run(ranky1)))
# rank y1 = 2
y2 = [[3,3],[4,4]],[[3,3],[4,4]]
ranky2 = tf.rank(y2)
print('rank y2 = '+str(sess.run(ranky2)))
# rank y2 = 3

#shape表示每个数据维度有多少个元素
t = [[[1,1,1],[2,2,2]],[[1,2,3],[4,5,6]]]
print(sess.run(tf.shape(t)))
# [2 2 3]
# placeholder数据维度=喂入数据的维度
t1 = tf.placeholder(tf.int32)
print(t1.get_shape())
print(sess.run(tf.shape(t1),feed_dict={t1:[[0,2],[1,5]]}))
# <unknown>
# [2 2]
# size 获得输入张量的标量元素个数 = shape 每层数字的乘积
print(sess.run(tf.size(t)))
# 12

# reshape改变数据的shape
image = tf.Variable([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
init = image.initializer
sess.run(init)
print(image.shape)
print(image.eval())
# (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

image1 = tf.reshape(image,[4,3])
init = tf.global_variables_initializer()
sess.run(init)
print(image1.shape)
print(image1.eval())
# (4, 3)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

# reshape参数shape中如果有-1，表示沿-1所在轴展开
image2 = tf.reshape(image,[-1,2])
init = tf.global_variables_initializer()
sess.run(init)
print(image2.shape)
print(image2.eval())
# (6, 2)
# [[ 1  2]
#  [ 3  4]
#  [ 5  6]
#  [ 7  8]
#  [ 9 10]
#  [11 12]]

image3 = tf.reshape(image, [2,-1])
init = tf.global_variables_initializer()
sess.run(init)

print(image3.eval())
print(sess.run(tf.shape(image3)))
# [[ 1  2  3  4  5  6]
#  [ 7  8  9 10 11 12]]
# [2 6]

# 升维expand dims沿axis方向升一维
a = [1,2,3]

print(sess.run(tf.shape(a)))
# [3]
a0 = tf.expand_dims(a,0)

print(sess.run(tf.shape(a0)))
print(sess.run(a0))
# [1 3]
# [[1 2 3]]
print("#######")
a1 = tf.expand_dims(a,1)
print(sess.run(tf.shape(a1)))
print(sess.run(a1))
# [3 1]
# [[1]
#  [2]
#  [3]]

# slice将张量切片，从begin(索引位置)参数位置开始，size表示切片大小

slice_source = tf.Variable([[      [1,2,3],[4,5,6]     ],
                            [      [7,8,9],[10,11,12]  ],
                            [      [13,14,15],[16,17,18] ]
                            ])
sess.run(tf.shape(slice_source))
slice1 = tf.slice(slice_source,[1,0,0],[2,1,2])
# [[[1 2]]
#  [[7 8]]]

# slice1 = tf.slice(slice_source,[1,0,0],[2,1,2])
# [[[ 7  8]]
#  [[13 14]]]
init = tf.global_variables_initializer()
sess.run(init)
print(slice1.eval())

# split planA num or size splits为标量a时表示均分为大小相等的a份，为向量时必须等于axis轴上元素数目
split_source = tf.Variable(tf.zeros([5,50,15]))
split0,split1,split2 = tf.split(split_source,num_or_size_splits=[15,15,20],axis=1)
# axis=1 表示切的地方（0，1，2）三轴
print(sess.run(tf.shape(split0)))
print(sess.run(tf.shape(split1)))
print(sess.run(tf.shape(split2)))
# [ 5 15 15]
# [ 5 15 15]
# [ 5 20 15]
# planB
split_source1 = tf.Variable(tf.zeros([12,60]))
spli0,spli1,spli2 = tf.split(split_source1,num_or_size_splits=3,axis=1)
print(sess.run(tf.shape(spli0)))
print(sess.run(tf.shape(spli1)))
print(sess.run(tf.shape(spli2)))

# concat沿axis轴连接
c1 = [[1,2],[3,4],[5,6]]
c2 = [[7,8],[9,10],[11,12]]

c = tf.concat([c1,c2],axis=1)
print(sess.run(tf.shape(c)))
print(sess.run(c))
# [3 4]
# [[ 1  2  7  8]
#  [ 3  4  9 10]
#  [ 5  6 11 12]]
c = tf.concat([c1,c2],axis=0)
print(sess.run(tf.shape(c)))
print(sess.run(c))
# [6 2]
# [[ 1  2]
#  [ 3  4]
#  [ 5  6]
#  [ 7  8]
#  [ 9 10]
#  [11 12]]

# stack沿axis轴堆叠成维度+1的张量

c = tf.stack([c1,c2],axis = 0)
print(sess.run(tf.shape(c)))
print(sess.run(c))
# [2 3 2]
# [[[ 1  2]
#   [ 3  4]
#   [ 5  6]]
#  [[ 7  8]
#   [ 9 10]
#   [11 12]]]
c = tf.stack([c1,c2],axis = 1)
print(sess.run(tf.shape(c)))
print(sess.run(c))
# [3 2 2]
# [[[ 1  2]
#   [ 7  8]]
#  [[ 3  4]
#   [ 9 10]]
#  [[ 5  6]
#   [11 12]]]
c = tf.stack([c1,c2],axis = 2)
print(sess.run(tf.shape(c)))
print(sess.run(c))
# [3 2 2]
# [[[ 1  7]
#   [ 2  8]]
#  [[ 3  9]
#   [ 4 10]]
#  [[ 5 11]
#   [ 6 12]]]

# gather根据indict提取元素合并
g1 = [1,2,3,4,5,6,7]
gathered = tf.gather(g1,[0,5,0])
print(gathered.eval())

# one hot
# 减小比重数值的影响
labels = [0,1,2,3,4,5,6,7,8,9]
batch_size = tf.size(labels)
print(sess.run(batch_size))
vector1 = [1,3,5]
# 矩阵表示
vector_matrix1 = tf.expand_dims(vector1,1)
print(sess.run(vector_matrix1))
# 两种写法1
concated = tf.concat([vector_matrix1, vector_matrix1],1)
print(sess.run(concated))
# 两种写法2
concated = tf.stack([vector1,vector1],axis = 1 )
print(sess.run(concated))
# use matrix
# sparse to dense 稠密矩阵与疏密矩阵转化
# spare indices输入值，output shape输出格式，sparse values个别元素，default value=0，
onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 10]), 1.0, 0.0)
print(sess.run(onehot_labels))
# use vector
onehot_labels1 = tf.sparse_to_dense(vector1, tf.stack([10]), 1.0, 0.0) #tf.stack([10])希望输出的就是10这个格式向量
print(sess.run(tf.stack([batch_size, 10])))
print(sess.run(onehot_labels1))
# one hot()
onehot_labels2 = tf.one_hot(labels,depth=9,on_value=1.0,off_value=0.0)
print(sess.run(onehot_labels2))
# [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# 最后一个以0代表，不常用

# matmul 矩阵乘法
w = [[1,1,1],[1,1,1]]
x = [[1,10],[2,20],[3,30]]
b = 0
z = tf.matmul(w,x)+b
print(sess.run(z))
x2 = [[1,2,3],[10,20,30]]
z2 = tf.matmul(w,x2,transpose_b=True)
# matmul里的参数transpose_a 转制w transpose_b转x2
print(sess.run(z2))

# reduce_sum 沿axis归约求和
x1 = [[1,2,3,4,5,6,7,8,9],[11,10,35,40,50,60,0,80,90],[100,22,33,44,55,66,77,88,99]]

reduce_sum1 = tf.reduce_sum(x1)
print(sess.run(reduce_sum1))
reduce_sum2 = tf.reduce_sum(x1,1)
print(sess.run(reduce_sum2))

# reduce prod 乘积
reduce_prod1 = tf.reduce_prod(x1,0)
print(sess.run(reduce_prod1))

# reduce max 归约最大值
reduce_max1 = tf.reduce_max(x1,1)
print(sess.run(reduce_max1))
# tf.argmax(input, dimension, name=None)归约最大值的索引
index_max = tf.argmax(x1, 1)
print(sess.run(index_max))
# reduce min 归约最小值
reduce_min1 = tf.reduce_min(x1,0)
print(sess.run(reduce_min1))
# tf.argmin(input, dimension, name=None)归约最小值的索引
index_min = tf.argmin(x1)
print(sess.run(index_min))

# reduce all/any 归约与运算/或运算
x2 = [[True,False],[False,True]]
reduce_all1 = tf.reduce_all(x2,1,keep_dims=True)
print(sess.run(reduce_all1))
reduce_any1 = tf.reduce_any(x2,1,keep_dims=True)
print(sess.run(reduce_any1))

# accumulate n 张量元素级(标量级)求和
# 叠加
x3 = [[1,2,3],[4,5,6]]
x4 = [[7,8,9],[10,11,12]]
a_n = tf.accumulate_n([x3,x3,x4,x3,x3])
print(sess.run(tf.shape(a_n)))
print(sess.run(a_n))
# [2 3]
# [[11 16 21]
#  [26 31 36]]

##tf.unique(x) 去重。返回去重后的tuple与原数组索引
x5 = [1,2,1,1,3,5,5,3,6,6,7,7,7,7,8]
y,idx = tf.unique(x5)
init = tf.unique(x5)
init = tf.global_variables_initializer()
sess.run(init)
print(y.eval())
# 去重后的数组
# [1 2 3 5 6 7 8]
print(idx.eval())
# 原数组里对应的新数组的位置
# [0 1 0 0 2 3 3 2 4 4 5 5 5 5 6]

##nn.activate
#sigmoid激活
p1 = tf.constant([-3,-1,-0.5,-0.3,0,0.3,1,3,7,9])
print(sess.run(tf.sigmoid(p1)))
# [0.04742587 0.26894143 0.37754068 0.4255575  0.5   0.5744425   0.7310586  0.95257413 0.999089   0.9998766 ]
#tanh
print(sess.run(tf.nn.tanh(p1)))
# [-0.9950547  -0.7615942  -0.46211717 -0.2913126   0.   0.2913126   0.7615942   0.9950547   0.99999833  1.    ]
#relu
print(sess.run(tf.nn.relu(p1)))
# [0.  0.  0.  0.  0.  0.3 1.  3.  7.  9. ]























