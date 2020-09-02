import tensorflow as tf
import random
import zipfile
import numpy as np


def load_data_jay_lyrics():
    with zipfile.ZipFile('jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')

    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    # print(corpus_chars[:1000])

    # 建立字符索引
    idx_to_char = list(set(corpus_chars))
    # 以列表形式构建字典
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

    # print(char_to_idx)

    # 利用字符找出索引
    corpus_index = [char_to_idx[char] for char in corpus_chars]
    # sample = corpus_index[:20]
    # print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    # print('indices:', sample)

    return corpus_index, char_to_idx, idx_to_char, len(char_to_idx)


# 对于时序数据的采样
# 随机采样
# 相邻两个随机小批量在原始序列上的位置不一定相毗邻，每次都要初始化隐藏状态

def data_iter_random(corpus_indices, batch_size, num_steps):
    num_examples = (len(corpus_indices) - 1) // num_steps

    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))

    random.shuffle(example_indices)

    def _data(pos):
        # 返回从pos开始的长为num_steps的序列
        return corpus_indices[pos:pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size

        batch_indices = example_indices[i:i + batch_size]
        x = [_data(j * num_steps) for j in batch_indices]
        y = [_data(j * num_steps + 1) for j in batch_indices]

        yield np.array(x), np.array(y)


# my_seq = list(range(30))
# for x,y in data_iter_random(my_seq, 2, 6):
#     print(x, y)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    # 每个批次都是基于上一个批次的位置
    corpus_indices = np.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


# my_seq = list(range(30))
# for x,y in data_iter_consecutive(my_seq, 2, 6):
#     print(x, y)

# 将字符变成长度为N的one-hot向量，N为字符总长（感觉不好拓展）
def to_onehot(X, size):
    # X: (batch, 1) res (batch, n_class)
    return [tf.one_hot(x, size, dtype=tf.float32) for x in X.T]


if __name__ == '__main__':
    corpus_index, char_to_idx, idx_to_char, size = load_data_jay_lyrics()
    print(corpus_index)
