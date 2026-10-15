"""
DistilBERT 微调 / DistilBERT fine-tuning.

设置沿用 03 notebook（distilbert-base-uncased、max_length 128、batch 16、学习率 2e-5、
weight decay 0.01、3 个 epoch、前 10% 步数线性预热），只修正旧版的两个方法问题：
Settings follow notebook 03; only its two methodological problems are fixed:

    1. 旧版用 *测试集* 做 early stopping。这里改用 ISOT *验证集*：每个 epoch 结束后在验证集上
       计算 loss，保留 loss 最低那个 epoch 的权重。测试集只在最后评估一次。
       The old version early-stopped on the TEST set. Here the best epoch is
       chosen by ISOT VALIDATION loss; test sets are scored once at the end.
    2. 旧版没有去重（21% 的测试文章在训练集里出现过）。这里的数据来自统一管线。
       The old version had no de-duplication (21% of test articles were in
       train). Data now comes from the shared pipeline.

输入是 model_text（小写、无标点），和 LSTM 完全相同，这样两类模型看到的信息一样，对比才公平。
DistilBERT 是 uncased 模型，本来就会转小写，所以小写本身不损失信息。
Input is model_text (lower-case, no punctuation) — exactly what the LSTM sees,
so the comparison is fair. The model is uncased, so lower-casing loses nothing.

截断方向：分词器默认保留文章 *开头* 的 128 个 token（与 LSTM 的 A5 一致，与 A0–A4 相反）。
Truncation: the tokenizer keeps the FIRST 128 tokens (like LSTM A5, unlike A0–A4).
"""

import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DistilBertConfig:
    """DistilBERT 实验配置（与 03 notebook 相同）/ Config (same as notebook 03)."""
    name: str = 'DistilBERT'
    description: str = 'distilbert-base-uncased, max_length 128 (notebook 03 settings)'
    checkpoint: str = 'distilbert-base-uncased'
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_fraction: float = 0.1


CONFIG = DistilBertConfig()


def _device():
    """优先使用 Apple GPU（MPS），其次 CUDA，最后 CPU / Prefer Apple GPU, then CUDA, then CPU."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _encode(tokenizer, texts, max_length):
    """把文本转成 input_ids 和 attention_mask（统一补齐到 max_length）。
    Tokenise to input_ids / attention_mask, padded to max_length."""
    enc = tokenizer(list(texts), truncation=True, padding='max_length',
                    max_length=max_length, return_tensors='pt')
    return enc['input_ids'], enc['attention_mask']


def _predict(model, ids, mask, device, batch_size=64):
    """返回每篇文章是真新闻的概率（只做推理，不计算梯度）。
    Return P(real) for every article (inference only, no gradients)."""
    import torch
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            logits = model(input_ids=ids[i:i + batch_size].to(device),
                           attention_mask=mask[i:i + batch_size].to(device)).logits
            probs.append(torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs)


def train_and_predict(config: DistilBertConfig, seed: int, train, val, eval_sets: dict,
                      sample_weight=None):
    """微调一个 (配置, 种子)，返回训练信息和每个评估集上的"真新闻概率"。
    Fine-tune one (config, seed); return training info and P(real) per eval set.

    和 src/lstm_model.train_and_predict 的输入输出格式完全相同，方便统一管理实验。
    Same inputs / outputs as src/lstm_model.train_and_predict.

    sample_weight：每篇训练文章的权重（第 4 步 4b 去偏重加权用）。None 时所有文章权重相同，
    和原来的训练完全一样。
    sample_weight: per-article training weights (step 4b). None = equal weights,
    identical to the original training.
    """
    import torch
    from torch.optim import AdamW
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    from src.utils import set_seed

    set_seed(seed)
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(config.checkpoint, num_labels=2)
    model.to(device)

    train_ids, train_mask = _encode(tokenizer, train['model_text'], config.max_length)
    val_ids, val_mask = _encode(tokenizer, val['model_text'], config.max_length)
    train_y = torch.tensor(train['label'].to_numpy())
    train_w = torch.tensor(np.ones(len(train)) if sample_weight is None else np.asarray(sample_weight),
                           dtype=torch.float32)
    val_y = val['label'].to_numpy()

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = int(np.ceil(len(train_ids) / config.batch_size))
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.warmup_fraction * total_steps), total_steps)

    # 打乱顺序用固定种子的随机数生成器，保证同一个种子每次的训练顺序相同
    # A seeded generator makes the shuffling order identical for the same seed
    generator = torch.Generator().manual_seed(seed)
    history = {'loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss, best_state, best_epoch = float('inf'), None, 0
    start = time.time()

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(train_ids), generator=generator)
        epoch_loss = 0.0
        for i in range(0, len(order), config.batch_size):
            batch = order[i:i + config.batch_size]
            logits = model(input_ids=train_ids[batch].to(device),
                           attention_mask=train_mask[batch].to(device)).logits
            # 逐篇计算交叉熵，再按权重加权平均。权重全为 1 时，等于原来的平均 loss。
            # Per-article cross-entropy, then a weighted mean; with all weights = 1
            # this equals the original mean loss.
            per_article = torch.nn.functional.cross_entropy(logits, train_y[batch].to(device),
                                                            reduction='none')
            w = train_w[batch].to(device)
            loss = (per_article * w).sum() / w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * len(batch)

        # 每个 epoch 结束后在 *验证集* 上评估；保留验证 loss 最低的那一版权重
        # After each epoch, score the VALIDATION set; keep the lowest-val-loss weights
        val_prob = np.clip(_predict(model, val_ids, val_mask, device), 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(val_y * np.log(val_prob) + (1 - val_y) * np.log(1 - val_prob)))
        history['loss'].append(round(epoch_loss / len(train_ids), 5))
        history['val_loss'].append(round(val_loss, 5))
        history['val_accuracy'].append(round(float(((val_prob > 0.5) == val_y).mean()), 5))
        print(f'epoch {epoch + 1}: loss {history["loss"][-1]:.4f}  val_loss {val_loss:.4f}  '
              f'val_acc {history["val_accuracy"][-1]:.4f}  ({time.time() - start:.0f}s)', flush=True)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            # 权重复制到 CPU 内存保存，避免占用 GPU 显存
            # Keep a CPU copy of the best weights so GPU memory is not doubled
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    info = {'epochs_run': config.epochs, 'best_epoch': best_epoch,
            'train_seconds': round(time.time() - start, 1), 'history': history}

    probs = {'ISOT val': _predict(model, val_ids, val_mask, device)}
    for name, df in eval_sets.items():
        ids, mask = _encode(tokenizer, df['model_text'], config.max_length)
        probs[name] = _predict(model, ids, mask, device)

    del model, best_state
    if device.type == 'mps':
        torch.mps.empty_cache()
    return info, probs


def config_dict(config: DistilBertConfig) -> dict:
    return asdict(config)
