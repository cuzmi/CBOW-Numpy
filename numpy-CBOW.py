import numpy as np
# 目的是为了预测词汇 - 而词向量的分布式表示只是中间的产物 // 其实one-hot也是CBOW出现的一大关键因素
test = '. you say goodbye and i say hi.' # 添加起始和结尾表示 这里用.来表示

# 预处理words -> word
def preprocess(texts):
    texts = texts.lower()
    texts = texts.replace(',',' ,')
    texts = texts.replace('.',' .')
    words = texts.split(' ')

    # word - idx
    words_to_idx = {}
    idx_to_words = {}
    for word in words:
        if word not in words_to_idx:
            idx = len(words_to_idx)
            words_to_idx[word] = idx
            idx_to_words[idx] = word

    corpus = [words_to_idx[word] for word in words]

    return corpus, words_to_idx, idx_to_words

# 将全部变成one-hot向量表示
corpus, words_to_idx, idx_to_words = preprocess(test)

def onehot(corpus, words_to_idx):
    # corpus - 整个句子的排列 // words_to_idx - 特征维度的数量
    vocab_size = len(words_to_idx)
    corpus_size = len(corpus)
    onehot_matrix = np.zeros((corpus_size, vocab_size), dtype= np.int32)

    for idx, word_id in enumerate(corpus):
        # id - 横坐标 // word_id - 纵坐标
        onehot_matrix[idx, word_id] = 1

    return onehot_matrix

onehot_matrix = onehot(corpus,words_to_idx)

from common.layers import Affine, SoftmaxWithLoss


# CBOW应该是根据窗口输入确定
class CBOW:
    # 整个模型结构就是在构建 <计算图> 的过程 -> 之前都是在loss.backward是因为他的loss层加在了外面 // 而他的loss层同样有backward函数，且dout也是默认的1
    # 所以实际运行中，只需要直接从loss部分backward就行，注意dout从1开始
    def __init__(self, input_size, hidden_size, output_size):  # input_sizeh和output_size就是语料库的大小，hidden自定义
        # init 里面把要用的层都写上
        self.w1 = np.random.randn(input_size, hidden_size)  # N *
        self.b1 = np.zeros(hidden_size)  # 会广播
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

        self.layer1 = Affine(self.w1, self.b1)
        self.layer2 = Affine(self.w2, self.b2)
        self.layers = [self.layer1, self.layer2]
        self.last_layer = SoftmaxWithLoss()

        # 模型的参数也要记下来
        self.params = [self.w1, self.b1, self.w2, self.b2]
        self.grads = []  # 需要在backward的过程中把层的grad添加进来

        self.word_vec = self.w1

    def forward(self, n_x, t):
        # 向前传播，整个网络结构包含了loss部分
        # n_x np.array (2,7) * Affine 1 (7,3) -> h(2,3) -> h1(1,3) -> Affine (3,7) -> o(1,7)
        x = np.sum(self.layer1.forward(n_x), axis=0, keepdims=True) / n_x.shape[0]  # 数值上除样本数
        x = self.layer2.forward(x)
        loss = self.last_layer.forward(x, t)
        return loss

    def backward(self, dout=1):  # 计算图一开始算得的都是dout=1
        # 反向传播 //从loss开始 网络中包含了loss，也就是最后直接得到了loss，先forward等于算到loss，然后loss.backward就可以来算
        dout = self.last_layer.backward(dout)
        dout = self.layer2.backward(dout)  # 不能自动广播 (7,2) 和 (1,3) 没法自动广播
        dout = np.tile(dout, (2, 1))
        dout = self.layer1.backward(dout)
        self.grads.extend(self.layer1.grads)
        self.grads.extend(self.layer2.grads)
        return None

input_size, output_size = len(words_to_idx), len(words_to_idx)
hidden_size = 3
cbow = CBOW(input_size, hidden_size, output_size)
cbow_w1 = [cbow.w1.copy()]

from common.optimizer import Adam

optimizer = Adam()

for epoch in range(10):
    # 训练cbow要走完1轮，即根据上下文预测每一个的值
    for batch in range(1, len(corpus) - 1):  # 把每个词当成一个batch，同时忽略起始和结尾标识符
        # forward构建计算图
        corpus_size = len(corpus)
        target = np.zeros((1, corpus_size))
        target[0, batch] = 1

        input_data = onehot_matrix[[batch - 1, batch + 1], :]
        # 构建计算图
        cbow.forward(input_data, target)
        cbow.backward()
        # 优化
        optimizer.update(cbow.params, cbow.grads)
    if epoch % 3 == 0:
        print(f'epoch {epoch} is running')

cbow_w1.append(cbow.w1)
print(cbow.w1)