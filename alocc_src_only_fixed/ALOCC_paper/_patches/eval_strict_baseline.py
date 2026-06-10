"""Evaluate ALOCC original strict baseline checkpoints on MNIST test set.

Loads each `checkpoint_d{digit}_s{seed}/mnist_128_28_28/ALOCC_Model.model-39` and
scores the full MNIST test set under three score functions:
  (1) D(R(x))           -- paper Eq.7, the canonical OCC score
  (2) D(x)              -- what original test.py actually computes (transparency)
  (3) -L2(x - R(x))     -- reconstruction-error fallback score

Run from inside D:\\Trae_coding\\ALLOC\\ALOCC-original\\ via .venv-tf1 python so
relative checkpoint paths resolve.
"""
from __future__ import print_function
import os, sys, json, time, argparse
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf

_ALOCC_ROOT = r'D:\Trae_coding\ALLOC\ALOCC-original'
if _ALOCC_ROOT not in sys.path:
    sys.path.insert(0, _ALOCC_ROOT)
from models import ALOCC_Model
from tensorflow.examples.tutorials.mnist import input_data

DATASET_DIR = './dataset/mnist'
BS = 128
EPOCH_TAG = 39


def _auc(y, s):
    """ROC AUC via rank statistic (Mann-Whitney U), tie-aware."""
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(s).astype(np.float64)
    n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind='mergesort')
    s_sorted = s[order]; y_sorted = y[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    sum_ranks_pos = ranks[y_sorted == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ap_and_best_f1(y, s):
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(s).astype(np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None, 0.0
    order = np.argsort(-s, kind='mergesort')
    y_s = y[order]
    tp = np.cumsum(y_s); fp = np.cumsum(1 - y_s)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    ap = float((prec * y_s).sum() / n_pos)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return ap, float(f1.max())


def evaluate_digit(digit, seed, ckpt_root, x_test, y_test):
    tf.reset_default_graph()
    suffix = '_d{}_s{}'.format(digit, seed)
    ckpt_dir = ckpt_root + suffix
    sample_dir = 'export/mnist_28.28' + suffix
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        model = ALOCC_Model(
            sess,
            input_height=28, input_width=28,
            output_height=28, output_width=28,
            batch_size=BS, sample_num=BS,
            attention_label=digit, r_alpha=0.2, is_training=False,
            dataset_name='mnist', dataset_address=DATASET_DIR,
            input_fname_pattern='*',
            checkpoint_dir=ckpt_dir, log_dir='log', sample_dir=sample_dir,
            nd_patch_size=(28, 28), n_stride=28, n_per_itr_print_results=100,
            kb_work_on_patch=True, nd_input_frame_size=(28, 28), n_fetch_data=0,
        )

        sess.run(tf.global_variables_initializer())
        model.saver = tf.train.Saver()
        ckpt_inner = os.path.join(ckpt_dir, model.model_dir)
        ckpt = tf.train.get_checkpoint_state(ckpt_inner)
        if not (ckpt and ckpt.model_checkpoint_path):
            raise RuntimeError('No checkpoint at ' + ckpt_inner)
        target = os.path.join(ckpt_inner, 'ALOCC_Model.model-{}'.format(EPOCH_TAG))
        model.saver.restore(sess, target)
        print('  restored: ' + target)

        N = len(x_test)
        pad = (-N) % BS
        x_pad = np.concatenate([x_test, np.zeros((pad, 28, 28, 1), np.float32)], 0) if pad else x_test
        Np = len(x_pad)
        d_rx = np.zeros(Np, np.float32)
        d_x = np.zeros(Np, np.float32)
        recon = np.zeros(Np, np.float32)
        for i in range(0, Np, BS):
            b = x_pad[i:i + BS]
            g_out, drx_l, dx_l = sess.run(
                [model.G, model.D_logits_, model.D_logits],
                feed_dict={model.z: b, model.inputs: b},
            )
            d_rx[i:i + BS] = 1.0 / (1.0 + np.exp(-drx_l.reshape(-1)))
            d_x[i:i + BS] = 1.0 / (1.0 + np.exp(-dx_l.reshape(-1)))
            recon[i:i + BS] = ((b - g_out) ** 2).reshape(BS, -1).mean(axis=1)
        d_rx, d_x, recon = d_rx[:N], d_x[:N], recon[:N]

    y_pos = (y_test == digit).astype(np.int32)
    auc_drx = _auc(y_pos, d_rx)
    auc_dx = _auc(y_pos, d_x)
    auc_rec = _auc(y_pos, -recon)
    ap_drx, best_f1 = _ap_and_best_f1(y_pos, d_rx)
    mask67 = (y_test == digit) | np.isin(y_test, (6, 7))
    y67 = (y_test[mask67] == digit).astype(np.int32)
    auc_f6 = _auc(y67, d_rx[mask67])
    return {
        'digit': digit, 'seed': seed, 'epoch': EPOCH_TAG,
        'n_inlier_test': int(y_pos.sum()), 'n_outlier_test': int(N - y_pos.sum()),
        'auc_D_of_Rx': auc_drx, 'auc_D_of_x': auc_dx, 'auc_recon_neg': auc_rec,
        'ap_D_of_Rx': ap_drx, 'best_f1_D_of_Rx': best_f1,
        'auc_figure6_outlier_67': auc_f6,
        'mean_Drx_inlier': float(d_rx[y_pos == 1].mean()),
        'mean_Drx_outlier': float(d_rx[y_pos == 0].mean()),
        'mean_recon_inlier': float(recon[y_pos == 1].mean()),
        'mean_recon_outlier': float(recon[y_pos == 0].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--digits', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--checkpoint-root', default='checkpoint')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    digits = [int(x) for x in args.digits.split(',')]

    print('[eval] loading MNIST test set ...')
    mnist = input_data.read_data_sets(DATASET_DIR)
    x_test = mnist.test.images.reshape(-1, 28, 28, 1).astype(np.float32)
    y_test = mnist.test.labels.astype(np.int32)
    print('[eval] N_test={}, class counts={}'.format(
        len(y_test), np.bincount(y_test).tolist()))

    results = []
    t0 = time.time()
    for d in digits:
        print('\n=== digit {} seed={} ==='.format(d, args.seed))
        t = time.time()
        r = evaluate_digit(d, args.seed, args.checkpoint_root, x_test, y_test)
        r['elapsed_s'] = round(time.time() - t, 2)
        results.append(r)
        print('  AUC[D(R(x))]={:.4f}  AUC[D(x)]={:.4f}  AUC[-recon]={:.4f}  '
              'F1*={:.4f}  elapsed={:.1f}s'.format(
                  r['auc_D_of_Rx'] or float('nan'), r['auc_D_of_x'] or float('nan'),
                  r['auc_recon_neg'] or float('nan'), r['best_f1_D_of_Rx'], r['elapsed_s']))

    def _agg(key):
        vs = [r[key] for r in results if r[key] is not None]
        return (float(np.mean(vs)), float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0)
    summary = {
        'protocol': 'ALOCC original strict baseline (CVPR 2018)',
        'epoch_tag': EPOCH_TAG, 'seed': args.seed,
        'total_elapsed_s': round(time.time() - t0, 2),
        'mean_std_auc_D_of_Rx': _agg('auc_D_of_Rx'),
        'mean_std_auc_D_of_x': _agg('auc_D_of_x'),
        'mean_std_auc_recon_neg': _agg('auc_recon_neg'),
        'per_digit': results,
    }
    out_path = args.out or 'eval_strict_baseline_seed{}.json'.format(args.seed)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n[eval] wrote ' + out_path)
    for k in ('mean_std_auc_D_of_Rx', 'mean_std_auc_D_of_x', 'mean_std_auc_recon_neg'):
        m, s = summary[k]
        print('  {:<30s} = {:.4f} ± {:.4f}'.format(k, m, s))


if __name__ == '__main__':
    main()
