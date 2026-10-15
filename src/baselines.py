"""
TF-IDF + 逻辑回归基线与评估工具 / TF-IDF + logistic-regression baseline and evaluation.

为什么用这么简单的模型？/ Why such a simple model?
    跨数据集实验要回答的是"在 ISOT 上学到的东西能不能迁移"，这是数据的问题，不是模型的问题。
    TF-IDF + 逻辑回归训练只要几秒，结果确定（同样的输入永远得到同样的输出），
    而且它在 ISOT 上已经达到 98.3%，和深度模型差不多，足以回答这个问题。
    The cross-dataset question — does what ISOT teaches transfer? — is about
    the data, not the model. TF-IDF + LR trains in seconds, is deterministic,
    and already reaches 98.3% on ISOT, so it is enough to answer it.

设置和 src/probes.py 的探针完全相同（1-2 gram、最多 5 万个特征、min_df=2、sublinear_tf），
所以这里的数字可以和 04_leakage_audit 直接对照。
Settings match src/probes.py (1-2 grams, 50k features, min_df=2, sublinear tf),
so numbers are directly comparable with 04_leakage_audit.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)

# 正则化强度 C 的候选值。C 越大，正则化越弱，模型越贴合训练数据。
# 最初的范围到 16 为止，但两个数据集都选中了 16（落在边界上，说明范围可能不够），
# 所以扩展到 256，确认最优值落在范围内部。
# Candidate regularisation strengths (larger C = weaker regularisation). The
# first grid stopped at 16 and both datasets picked the edge value, so it was
# extended to 256 to make sure the optimum lies inside the grid.
C_GRID = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0)


def fit_tfidf_lr(train: pd.DataFrame, C: float = 4.0):
    """在 train 的 model_text 上训练 TF-IDF + 逻辑回归。
    Fit TF-IDF + logistic regression on train['model_text'].

    词表只从训练数据建立：测试数据里的新词会被直接忽略，这正是跨数据集时会发生的情况。
    The vocabulary comes from training data only; unseen words in test data are
    ignored — exactly what happens across datasets.

    Returns:
        (vectorizer, classifier)
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, min_df=2,
                                 sublinear_tf=True)
    x = vectorizer.fit_transform(train['model_text'])
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(x, train['label'])
    return vectorizer, clf


def predict_proba_real(vectorizer, clf, df: pd.DataFrame) -> np.ndarray:
    """返回每篇文章是真新闻（label = 1）的概率 / Probability that each article is real (label 1)."""
    return clf.predict_proba(vectorizer.transform(df['model_text']))[:, 1]


def select_C(train: pd.DataFrame, val: pd.DataFrame, grid=C_GRID):
    """在 *同一个数据集* 的验证集上选择 C，按 AUC 挑最好的。
    Choose C on the validation split of the SAME dataset, by AUC.

    为什么只能用训练数据所在数据集的验证集：如果用目标数据集（例如 WELFake）来调参，
    等于偷看了答案，跨数据集的分数就不再是"没见过的数据"上的表现了。
    Only the training dataset's own validation split may be used: tuning on the
    target dataset would be peeking, and the cross-dataset score would no
    longer measure unseen data.

    为什么按 AUC 而不是准确率：AUC 不依赖 0.5 这个阈值，比较稳定。
    Why AUC: it does not depend on the 0.5 threshold, so it is more stable.

    Returns:
        (best_C, table)：table 记录每个 C 在验证集上的 accuracy 和 AUC。
        (best_C, table): validation accuracy and AUC for every C.
    """
    rows = []
    for C in grid:
        vectorizer, clf = fit_tfidf_lr(train, C)
        prob = predict_proba_real(vectorizer, clf, val)
        rows.append({'C': C,
                     'val accuracy %': 100 * accuracy_score(val['label'], prob > 0.5),
                     'val AUC': roc_auc_score(val['label'], prob)})
    table = pd.DataFrame(rows).set_index('C').round(4)
    # idxmax 在并列时取第一个，也就是较小的 C：效果相同时选正则化更强、更不容易过拟合的设置
    # On ties idxmax returns the first, i.e. the smaller C: with equal scores,
    # prefer stronger regularisation (less overfitting)
    return float(table['val AUC'].idxmax()), table


def evaluate(y_true, prob_real, threshold: float = 0.5) -> dict:
    """计算一组互相补充的指标 / A set of complementary metrics.

    - majority baseline %：永远猜多数类能得到的准确率。准确率必须和它比，
      例如真新闻占 57% 时，57% 的准确率等于什么都没学到。
      accuracy of always guessing the majority class — accuracy must be read
      against it (57% accuracy is nothing when 57% of articles are real).
    - accuracy % / balanced accuracy %：后者是两个类别各自准确率的平均值，不受类别比例影响。
      balanced accuracy averages the per-class accuracies, so class imbalance
      cannot inflate it (0.5 = chance).
    - AUC：随机抽一篇真新闻和一篇假新闻，模型给真新闻打分更高的概率。0.5 = 瞎猜，
      低于 0.5 = 方向反了。它不依赖阈值，所以换数据集后模型打分整体偏移时，仍然可靠。
      probability that a random real article outscores a random fake one; 0.5
      = chance, below 0.5 = systematically reversed. Threshold-free, so it stays
      meaningful when a model's scores shift on a new dataset.
    - F1 fake / F1 real：每个类别各自的 F1。
    - predicted real %：模型把多少比例判为真。和实际真新闻比例差很多，说明阈值整体偏了。
      share predicted real; far from the true share means the threshold has shifted.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob_real) > threshold).astype(int)
    real_share = y_true.mean()
    return {
        'n': len(y_true),
        'real %': round(100 * real_share, 1),
        'majority baseline %': round(100 * max(real_share, 1 - real_share), 1),
        'accuracy %': round(100 * accuracy_score(y_true, pred), 1),
        'balanced accuracy %': round(100 * balanced_accuracy_score(y_true, pred), 1),
        'AUC': round(roc_auc_score(y_true, prob_real), 3),
        'F1 fake': round(f1_score(y_true, pred, pos_label=0), 3),
        'F1 real': round(f1_score(y_true, pred, pos_label=1), 3),
        'predicted real %': round(100 * pred.mean(), 1),
    }


def bootstrap_ci(y_true, prob_real, n_boot: int = 1000, seed: int = 42,
                 threshold: float = 0.5) -> dict:
    """用自助法（bootstrap）估计 balanced accuracy 和 AUC 的 95% 置信区间。
    95% bootstrap confidence intervals for balanced accuracy and AUC.

    做法：有放回地重复抽样测试集 n_boot 次，每次重新计算指标，取第 2.5 和第 97.5 百分位。
    区间表示"换一批同样来源的测试文章，结果大概会在这个范围内"，
    用来判断两个实验的差距是不是真实存在，而不是测试集抽样带来的偶然。
    Resample the test set with replacement n_boot times, recompute, take the
    2.5th / 97.5th percentiles. The interval shows how much the number would
    move with a different sample of test articles, i.e. whether a gap between
    two experiments is real or sampling noise.
    """
    rng = np.random.default_rng(seed)
    y_true, prob_real = np.asarray(y_true), np.asarray(prob_real)
    pred = (prob_real > threshold).astype(int)
    bal, auc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():   # 抽到只有一个类别时无法计算 AUC，跳过
            continue                                 # single-class resample: AUC undefined, skip
        bal.append(balanced_accuracy_score(y_true[idx], pred[idx]))
        auc.append(roc_auc_score(y_true[idx], prob_real[idx]))
    pct = lambda v: (round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3))
    return {'balanced accuracy 95% CI': tuple(round(100 * x, 1) for x in pct(bal)),
            'AUC 95% CI': pct(auc)}


def confusion_table(y_true, prob_real, threshold: float = 0.5) -> pd.DataFrame:
    """带行列名的混淆矩阵：行是真实标签，列是预测标签。
    Labelled confusion matrix: rows = true label, columns = predicted label."""
    pred = (np.asarray(prob_real) > threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return pd.DataFrame(cm, index=['true fake', 'true real'],
                        columns=['predicted fake', 'predicted real'])


def coefficient_table(vectorizer, clf) -> pd.Series:
    """每个特征（词或词组）的逻辑回归权重：正数推向"真"，负数推向"假"。
    Logistic-regression weight per feature: positive → real, negative → fake."""
    return pd.Series(clf.coef_[0], index=vectorizer.get_feature_names_out())
