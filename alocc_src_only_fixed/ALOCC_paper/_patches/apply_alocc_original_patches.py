"""Apply minimal patches to ALOCC-CVPR2018 upstream code to enable MNIST baseline.

Scope:
  1) train.py -- switch dataset-specific block to be conditional on FLAGS.dataset;
                 add --seed flag, seed initialization, and per-(digit,seed)
                 suffixing of checkpoint_dir / sample_dir so parallel runs do
                 not collide.
  2) models.py -- guard SIFTETS.npy load (UCSD-only artifact) and provide MNIST
                  sample_test fallback so train loop runs end-to-end on MNIST.

Both edits are reversible (see *.orig backup). Nothing else is touched.
"""
from __future__ import print_function
import os
import shutil
import sys

ROOT = r"D:\Trae_coding\ALLOC\ALOCC-original"
TRAIN = os.path.join(ROOT, "train.py")
MODELS = os.path.join(ROOT, "models.py")


def backup(path):
    bak = path + ".orig"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print("[backup] %s -> %s" % (path, bak))
    else:
        print("[backup] %s already exists, skipping" % bak)


def patch_train():
    with open(TRAIN, "r", encoding="utf-8") as f:
        src = f.read()

    # absl-py 2.x rejects np.inf (float) as an integer flag default; replace with a large int.
    old_inf = 'flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")'
    new_inf = 'flags.DEFINE_integer("train_size", 10**9, "The size of train images [np.inf]")'
    inf_changed = False
    if old_inf in src:
        src = src.replace(old_inf, new_inf, 1)
        inf_changed = True
        print("[train.py] train_size flag patched (np.inf -> 10**9)")

    # absl.logging pre-registers a "log_dir" flag; rename ours to avoid DuplicateFlagError.
    log_renames = [
        ('flags.DEFINE_string("log_dir", "log",',
         'flags.DEFINE_string("alocc_log_dir", "log",'),
        ('FLAGS.log_dir', 'FLAGS.alocc_log_dir'),
    ]
    log_changed = False
    for a, b in log_renames:
        if a in src:
            src = src.replace(a, b)
            log_changed = True
    if log_changed:
        print("[train.py] log_dir flag renamed to alocc_log_dir")

    # Add --seed flag (after the "train" flag) for reproducible / parallel runs.
    seed_anchor = 'flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")\n'
    seed_flag_line = 'flags.DEFINE_integer("seed", 42, "Random seed for reproducibility [42]")\n'
    seed_flag_added = False
    if seed_flag_line not in src and seed_anchor in src:
        src = src.replace(seed_anchor, seed_anchor + seed_flag_line, 1)
        seed_flag_added = True
        print("[train.py] --seed flag added")

    # Seed Python / numpy / TF right after pp.pprint at the start of main().
    seed_init_anchor = "    pp.pprint(flags.FLAGS.__flags)\n"
    seed_init_block = (
        "    pp.pprint(flags.FLAGS.__flags)\n"
        "    import random as _random\n"
        "    _random.seed(FLAGS.seed)\n"
        "    np.random.seed(FLAGS.seed)\n"
        "    tf.set_random_seed(FLAGS.seed)\n"
    )
    seed_init_added = False
    if "_random.seed(FLAGS.seed)" not in src and seed_init_anchor in src:
        src = src.replace(seed_init_anchor, seed_init_block, 1)
        seed_init_added = True
        print("[train.py] seed initialization inserted")

    # Append _d{digit}_s{seed} suffix to checkpoint_dir and sample_dir for MNIST,
    # so parallel runs over (digit, seed) do not overwrite each other.
    suffix_anchor = (
        "    FLAGS.sample_dir = 'export/'+FLAGS.dataset +'_%d.%d'%(nd_slice_size[0],nd_slice_size[1])\n"
        "    FLAGS.input_fname_pattern = '*'\n"
    )
    suffix_block = (
        "    FLAGS.sample_dir = 'export/'+FLAGS.dataset +'_%d.%d'%(nd_slice_size[0],nd_slice_size[1])\n"
        "    FLAGS.input_fname_pattern = '*'\n"
        "    if FLAGS.dataset == 'mnist':\n"
        "        _run_suffix = '_d%d_s%d' % (FLAGS.attention_label, FLAGS.seed)\n"
        "        FLAGS.checkpoint_dir = FLAGS.checkpoint_dir + _run_suffix\n"
        "        FLAGS.sample_dir = FLAGS.sample_dir + _run_suffix\n"
    )
    suffix_added = False
    if "_run_suffix" not in src and suffix_anchor in src:
        src = src.replace(suffix_anchor, suffix_block, 1)
        suffix_added = True
        print("[train.py] checkpoint_dir/sample_dir suffix _d{digit}_s{seed} added")

    # Replace the hard-coded UCSD block with a conditional block keyed on FLAGS.dataset.
    old = (
        "    # DATASET PARAMETER : UCSD\n"
        "    #FLAGS.dataset = 'UCSD'\n"
        "    #FLAGS.dataset_address = './dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'\n"
        "\n"
        "    nd_input_frame_size = (240, 360)\n"
        "    nd_slice_size = (45, 45)\n"
        "    n_stride = 25\n"
        "    n_fetch_data = 600\n"
        "    # ---------------------------------------------------------------------------------------------\n"
        "    # # DATASET PARAMETER : MNIST\n"
        "    # FLAGS.dataset = 'mnist'\n"
        "    # FLAGS.dataset_address = './dataset/mnist'\n"
        "    # nd_input_frame_size = (28, 28)\n"
        "    # nd_slice_size = (28, 28)\n"
    )
    new = (
        "    # DATASET PARAMETER (conditional on --dataset flag)\n"
        "    if FLAGS.dataset == 'mnist':\n"
        "        FLAGS.dataset_address = './dataset/mnist'\n"
        "        nd_input_frame_size = (28, 28)\n"
        "        nd_slice_size = (28, 28)\n"
        "        n_stride = 28\n"
        "        n_fetch_data = 0\n"
        "    else:\n"
        "        FLAGS.dataset = 'UCSD'\n"
        "        FLAGS.dataset_address = './dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'\n"
        "        nd_input_frame_size = (240, 360)\n"
        "        nd_slice_size = (45, 45)\n"
        "        n_stride = 25\n"
        "        n_fetch_data = 600\n"
    )
    secondary_changed = (inf_changed or log_changed or seed_flag_added
                         or seed_init_added or suffix_added)
    if new.split("\n")[1].strip() in src:
        if secondary_changed:
            with open(TRAIN, "w", encoding="utf-8") as f:
                f.write(src)
            print("[train.py] UCSD/MNIST block already patched; flushed secondary fixes")
        else:
            print("[train.py] already patched, skipping")
        return
    if old not in src:
        raise RuntimeError("train.py: expected UCSD/MNIST block not found (upstream file changed?)")
    src2 = src.replace(old, new, 1)
    with open(TRAIN, "w", encoding="utf-8") as f:
        f.write(src2)
    print("[train.py] patched")


def patch_models():
    with open(MODELS, "r", encoding="utf-8") as f:
        src = f.read()

    old = "      sample_test = np.load('SIFTETS.npy').reshape([504,45,45,1])[0:128]\n"
    new = (
        "      if config.dataset == 'UCSD':\n"
        "        sample_test = np.load('SIFTETS.npy').reshape([504,45,45,1])[0:128]\n"
        "      else:\n"
        "        sample_test = sample_w_noise[:config.batch_size]\n"
    )
    if "if config.dataset == 'UCSD':\n        sample_test = np.load('SIFTETS.npy')" in src:
        print("[models.py] already patched, skipping")
        return
    if old not in src:
        raise RuntimeError("models.py: expected SIFTETS.npy line not found")
    src2 = src.replace(old, new, 1)
    with open(MODELS, "w", encoding="utf-8") as f:
        f.write(src2)
    print("[models.py] patched")


def main():
    backup(TRAIN)
    backup(MODELS)
    patch_train()
    patch_models()
    print("[done] all patches applied")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[error]", e, file=sys.stderr)
        sys.exit(1)
