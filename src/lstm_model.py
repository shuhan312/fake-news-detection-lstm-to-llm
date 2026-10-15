"""
LSTM / BiLSTM 模型 / LSTM and BiLSTM models.

结构和超参数完全沿用原论文（Liu, 2023，Table 2）和 02 notebook 的"优化版"，
区别只在于：数据来自统一管线（src/data.py），并且所有设置都写在一个配置里，
方便做"每次只改一项"的消融实验。
Architecture and hyper-parameters follow the original paper (Liu, 2023,
Table 2) and the "optimised" version in notebook 02. The differences: data
comes from the shared pipeline, and every setting lives in one config so
ablations change exactly one thing at a time.

消融顺序 / Ablation ladder (each step adds ONE change to the previous one)
    A0  原论文 LSTM / paper LSTM          Tokenizer(5000), maxlen 200, LSTM(128), Dropout 0.5
    A1  + <OOV> 标记 / + OOV token        未知词替换为 <OOV>，而不是直接丢掉
    A2  + 双向 / + Bidirectional          Bidirectional(LSTM(128))
    A3  + maxlen 300                      保留更多正文
    A4  + Dropout 0.6                     = 02 notebook 的"优化版 BiLSTM" / 02's optimised BiLSTM
    A5  + 保留开头 / + keep the start     truncating='post'

关于 EarlyStopping：所有配置都使用。原论文固定训练 10 轮、没有验证集选择，那样的话
"训练了几轮"会成为另一个混杂因素。统一用验证集 loss 选最佳轮数，比较才公平。
EarlyStopping is used by EVERY config. The paper trained a fixed 10 epochs
with no validation-based selection; using it everywhere keeps "how many
epochs" from becoming another confounder.

关于截断方向（A5）：Keras 的 pad_sequences 默认 truncating='pre'，即超长时 *删掉开头、
保留结尾*。原论文和 02 都是这个默认值。A5 改成保留开头，单独检验它的影响。
pad_sequences defaults to truncating='pre' (drop the START, keep the END) —
what the paper and notebook 02 used. A5 keeps the start instead, tested alone.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np

# 训练设置（所有配置共用）/ Training settings shared by all configs
MAX_EPOCHS = 10           # 原论文的 10 轮作为上限 / the paper's 10 epochs as the cap
BATCH_SIZE = 64           # 原论文 / paper
PATIENCE = 3              # 验证集 loss 连续 3 轮不下降就停 / stop after 3 epochs without val-loss gain


@dataclass(frozen=True)
class LSTMConfig:
    """一个 LSTM 实验配置 / One LSTM experiment configuration."""
    name: str
    description: str
    vocab_size: int = 5000          # Tokenizer(num_words=5000)
    oov_token: bool = False         # 是否使用 <OOV> / map unknown words to <OOV>
    maxlen: int = 200               # 每篇保留多少个词 / tokens kept per article
    truncating: str = 'pre'         # 'pre' = 保留结尾（Keras 默认）/ keep the end (Keras default)
    embedding_dim: int = 100
    lstm_units: int = 128
    bidirectional: bool = False
    dense_dropout: float = 0.5
    # 以下设置只有 A6 会改；默认值等于原论文 / Keras 默认，所以 A0–A5 不受影响
    # The settings below are changed only by A6; defaults equal the paper / Keras
    # defaults, so A0–A5 are unaffected
    padding: str = 'pre'            # 短文章在前面补 0（Keras 默认）/ pad short articles at the front
    mask_zero: bool = False         # 是否让模型忽略补齐的 0 / let the model skip padding zeros
    learning_rate: float = 1e-3     # Adam 默认学习率 / Adam's default
    clipnorm: float = None          # 梯度裁剪，None = 不裁剪 / gradient clipping, None = off


# 消融阶梯：每一步只比上一步多改一项 / Ablation ladder: one change per step
A0 = LSTMConfig('A0', 'Paper LSTM (Liu, 2023)')
A1 = replace(A0, name='A1', description='+ OOV token', oov_token=True)
A2 = replace(A1, name='A2', description='+ Bidirectional', bidirectional=True)
A3 = replace(A2, name='A3', description='+ maxlen 300', maxlen=300)
A4 = replace(A3, name='A4', description='+ Dropout 0.6 (= notebook 02 BiLSTM)', dense_dropout=0.6)
A5 = replace(A4, name='A5', description='+ keep the start (truncating=post)', truncating='post')

# A6：调优版 BiLSTM。它 *不是* 消融阶梯的一级（一次改了多项），而是回答"把 LSTM 尽量调好，
# 上限在哪里"。每项改动都针对 A0–A5 中诊断出的具体问题，并且只根据 ISOT 验证集来判断：
# A6: tuned BiLSTM. NOT a ladder step (it changes several things); it asks how
# good an LSTM can get. Each change targets a problem diagnosed in A0–A5:
#   - 词表 5,000 → 20,000：太多词被丢掉或变成 <OOV>
#     vocab 5k → 20k: too many words dropped / mapped to <OOV>
#   - maxlen 400、在后面补 0、忽略补齐的 0：文章中位数约 368 词；短文章前面大量的 0
#     也会被 LSTM 当成内容去读
#     maxlen 400, pad at the end, mask padding: median article ~368 words, and
#     front-padding zeros were being read as content
#   - 学习率减半 + 梯度裁剪：部分种子在第 1 个 epoch 后就开始过拟合，训练不稳定
#     half learning rate + gradient clipping: some seeds overfit after epoch 1
A6 = replace(A5, name='A6', description='Tuned BiLSTM (vocab 20k, maxlen 400, masking, lr 5e-4, clipnorm 1)',
             vocab_size=20000, maxlen=400, padding='post', mask_zero=True,
             learning_rate=5e-4, clipnorm=1.0)

LADDER = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5')     # 消融阶梯 / the ablation ladder
CONFIGS = {c.name: c for c in (A0, A1, A2, A3, A4, A5, A6)}


def build_model(config: LSTMConfig):
    """按配置搭建 Keras 模型 / Build the Keras model for a config.

    层的顺序和原论文一致 / Same layer order as the paper:
        Embedding → (Bi)LSTM(dropout 0.2, recurrent_dropout 0.2) → Dense(64, relu)
        → Dropout → Dense(1, sigmoid)
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    lstm = layers.LSTM(config.lstm_units, dropout=0.2, recurrent_dropout=0.2)
    model = keras.Sequential([
        keras.Input(shape=(config.maxlen,)),
        # mask_zero=True：编号为 0 的位置（补齐用的空白）会被后面的 LSTM 跳过
        # mask_zero=True: positions with index 0 (padding) are skipped by the LSTM
        layers.Embedding(config.vocab_size, config.embedding_dim, mask_zero=config.mask_zero),
        layers.Bidirectional(lstm) if config.bidirectional else lstm,
        layers.Dense(64, activation='relu'),
        layers.Dropout(config.dense_dropout),
        layers.Dense(1, activation='sigmoid'),
    ])
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=config.clipnorm)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def make_encoder(config: LSTMConfig, train_texts):
    """在 *训练集* 上建立词表，返回把文本转成定长数字序列的函数。
    Fit the Tokenizer on TRAINING texts only; return a text → padded-sequence function.

    词表只看训练集：验证集、测试集、外部数据集里的新词会变成 <OOV>（A1 之后）或被丢掉（A0）。
    The vocabulary sees training data only; unseen words in val / test /
    external data become <OOV> (A1+) or are dropped (A0).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=config.vocab_size,
                          oov_token='<OOV>' if config.oov_token else None)
    tokenizer.fit_on_texts(train_texts)

    def encode(texts):
        return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=config.maxlen,
                             padding=config.padding, truncating=config.truncating)
    return encode


def train_and_predict(config: LSTMConfig, seed: int, train, val, eval_sets: dict):
    """训练一个 (配置, 种子)，返回训练信息和每个评估集上的"真新闻概率"。
    Train one (config, seed); return training info and P(real) on every eval set.

    Args:
        train, val: ISOT 的训练集和验证集（DataFrame，读取 model_text 和 label）。
                    ISOT train / val DataFrames (model_text, label).
        eval_sets:  {名称: DataFrame}，例如 ISOT test、WELFake test。
                    {name: DataFrame}, e.g. ISOT test and WELFake test.

    Returns:
        (info, probs)：info 记录停在第几轮、最佳轮数、训练时间；probs 是 {名称: 概率数组}。
        (info, probs): info has stopping epoch, best epoch, train time; probs maps
        each eval-set name to an array of P(real).
    """
    import time

    from tensorflow import keras

    from src.utils import set_seed

    set_seed(seed)
    encode = make_encoder(config, train['model_text'])
    x_train, x_val = encode(train['model_text']), encode(val['model_text'])
    y_train, y_val = train['label'].to_numpy(), val['label'].to_numpy()

    model = build_model(config)
    # restore_best_weights=True：训练结束后恢复验证集 loss 最低那一轮的权重
    # restore_best_weights=True: end with the weights of the lowest-val-loss epoch
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                               restore_best_weights=True)
    start = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[early_stop], verbose=2)
    info = {
        'epochs_run': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
        'train_seconds': round(time.time() - start, 1),
        'history': {k: [round(float(v), 5) for v in vals] for k, vals in history.history.items()},
    }
    probs = {'ISOT val': model.predict(x_val, batch_size=256, verbose=0).ravel()}
    for name, df in eval_sets.items():
        probs[name] = model.predict(encode(df['model_text']), batch_size=256, verbose=0).ravel()
    return info, probs


def config_dict(config: LSTMConfig) -> dict:
    """配置转成字典，保存到结果文件里 / Config as a dict, stored with each result."""
    return asdict(config)
