# 计算图的三种构建方式：静态计算图，动态计算图， Autograph
# 静态为tf1中的格式，动态为tf2中的格式，效率相比静态会低一些，但是方便调试
# Autograph则是在tf2中使用 @tf.function 装饰器将普通Python函数转换成对应的TensorFlow计算图构建代码
# 效率相当于静态图

import tensorflow as tf
import time
import datetime
from pathlib import Path

st = time.time()
# eager mode
x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x, y], separator=" ")

print(z)
central = time.time()
print('time cost: {}'.format((central-st)/1000))

# Autograph mode
@tf.function
def str_join(x, y):
    z = tf.strings.join([x, y], separator=" ")
    print(z)

    return z


print(str_join(x, y))
print('time cost: {}'.format((time.time()-central)/1000))

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path('../data/autograph/'+stamp))

writer = tf.summary.create_file_writer(logdir)

# autograph trace
tf.summary.trace_on(graph=True, profiler=True)

# execute autograph
res = str_join("hello", "world")

with writer.as_default():
    tf.summary.trace_export(
        name='autograph',
        step=0,
        profiler_outdir=logdir
    )

