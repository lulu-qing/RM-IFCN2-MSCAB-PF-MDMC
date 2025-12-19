import random
import torch
import os
import time
import numpy as np
import pprint as pprint
from sklearn.metrics import confusion_matrix

# Ensure non-interactive backend (safe for headless servers)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def count_acc_topk(x, y, k=4):
    _, maxk = torch.topk(x, k, dim=-1)
    total = y.size(0)
    test_labels = y.view(-1, 1)
    topk = (test_labels == maxk).sum().item()
    return float(topk / total)


def count_acc_taskIL(logits, label, args):
    basenum = args.base_class
    incrementnum = (args.num_classes - args.base_class) / args.way
    for i in range(len(label)):
        currentlabel = label[i]
        if currentlabel < basenum:
            logits[i, basenum:] = -1e9
        else:
            space = int((currentlabel - basenum) / args.way)
            low = basenum + space * args.way
            high = low + args.way
            logits[i, :low] = -1e9
            logits[i, high:] = -1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def _get_system_fonts_map(fontpaths=None, fontext='ttf'):
    """
    Return a mapping of font friendly name -> font file path for fonts available
    in the environment (best-effort).
    """
    try:
        system_fonts = fm.findSystemFonts(fontpaths=fontpaths, fontext=fontext)
    except Exception:
        system_fonts = fm.findSystemFonts(fontpaths=fontpaths)
    name_to_path = {}
    for fpath in system_fonts:
        try:
            prop = fm.FontProperties(fname=fpath)
            fname = prop.get_name()
            if fname:
                # prefer the first seen mapping for a given friendly name
                if fname not in name_to_path:
                    name_to_path[fname] = fpath
        except Exception:
            # skip unreadable font files
            continue
    return name_to_path


def list_available_fonts(fontpaths=None, fontext='ttf', verbose=True, limit=200):
    """
    Return and optionally print available font family names discovered in the environment.
    """
    name_to_path = _get_system_fonts_map(fontpaths=fontpaths, fontext=fontext)
    fams = sorted(list(name_to_path.keys()))
    if verbose:
        print(f"Found {len(fams)} fonts available in the environment. Showing up to {limit}:")
        for i, f in enumerate(fams[:limit]):
            print(f"  {i+1:03d}: {f} -> {name_to_path[f]}")
        if len(fams) > limit:
            print("  ...")
    return name_to_path


def configure_mpl_font(preferred_fonts=('Times New Roman', 'Arial', 'DejaVu Sans'),
                       fallback_sans=('DejaVu Sans', 'Arial', 'Liberation Sans', 'Noto Sans CJK JP'),
                       fontpaths=None,
                       rebuild_cache=False,
                       verbose=False):
    """
    Configure matplotlib to use a font available in the environment.

    - preferred_fonts: tuple/list of font family names in order of preference.
    - fallback_sans: list of sans-serif families to use if none of preferred_fonts found.
    - fontpaths: optional list of directories to search for additional fonts.
    - rebuild_cache: if True, attempts to rebuild matplotlib font cache (best-effort).
    - verbose: print debug information.
    Returns: (chosen_family, chosen_font_path_or_None)
    """
    try:
        if rebuild_cache:
            cache_dir = matplotlib.get_cachedir()
            if cache_dir:
                # attempt to remove common cache filenames so font list refreshes
                for fn in ('fontlist-v310.json', 'fontlist-v310.py3k.cache', 'fontList.cache'):
                    fp = os.path.join(cache_dir, fn)
                    try:
                        if os.path.exists(fp):
                            os.remove(fp)
                    except Exception:
                        pass
            try:
                fm.fontManager._reload()
            except Exception:
                pass
    except Exception:
        pass

    name_to_path = _get_system_fonts_map(fontpaths=fontpaths)
    if verbose:
        print(f"[configure_mpl_font] Discovered {len(name_to_path)} fonts (searched paths={fontpaths}).")

    # Try preferred fonts
    for fam in preferred_fonts:
        if fam in name_to_path:
            matplotlib.rcParams['font.family'] = fam
            # keep a sensible default font size if not set elsewhere
            matplotlib.rcParams.setdefault('font.size', 12)
            if verbose:
                print(f"[configure_mpl_font] Using preferred font: {fam} -> {name_to_path[fam]}")
            return fam, name_to_path[fam]

    # Fallback: use any available fallback sans-family that exists
    available_fallbacks = [f for f in fallback_sans if f in name_to_path]
    if available_fallbacks:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        # put the available_fallbacks at front of sans-serif list
        matplotlib.rcParams['font.sans-serif'] = available_fallbacks + list(matplotlib.rcParams.get('font.sans-serif', []))
        matplotlib.rcParams.setdefault('font.size', 12)
        if verbose:
            print(f"[configure_mpl_font] Using fallback sans-serif: {available_fallbacks[0]} -> {name_to_path[available_fallbacks[0]]}")
        return matplotlib.rcParams['font.sans-serif'][0], name_to_path[available_fallbacks[0]]

    # Last resort: if there is at least one discovered font, use its name
    if len(name_to_path) > 0:
        first = next(iter(name_to_path.items()))
        fam_name, fam_path = first[0], first[1]
        matplotlib.rcParams['font.family'] = fam_name
        matplotlib.rcParams.setdefault('font.size', 12)
        if verbose:
            print(f"[configure_mpl_font] No preferred fonts found; using {fam_name} -> {fam_path}")
        return fam_name, fam_path

    # If nothing found (very unlikely), leave rcParams as-is and return None
    if verbose:
        print("[configure_mpl_font] No fonts discovered; leaving matplotlib defaults.")
    return None, None


def confmatrix(logits, label, filename, font_verbose=False):
    """
    Compute confusion matrix (sklearn normalized) and save two PDF figures:
      - filename.pdf (no colorbar)
      - filename_cbar.pdf (with colorbar)
    This function will attempt to set a usable font from system fonts before plotting.
    """
    # Ensure a usable font is set before plotting
    font_used, font_path = configure_mpl_font(verbose=font_verbose)
    if font_verbose:
        print(f"[confmatrix] font_used={font_used}, font_path={font_path}")

    # set a reasonable font size for these plots
    matplotlib.rcParams.update({'font.size': 18})

    pred = torch.argmax(logits, dim=1)
    cm = confusion_matrix(label, pred, normalize='true')
    clss = len(cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    if clss <= 100:
        plt.yticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
        plt.xticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    elif clss <= 200:
        plt.yticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
        plt.xticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
    else:
        plt.yticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)
        plt.xticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)

    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=16)
    if clss <= 100:
        plt.yticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
        plt.xticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    elif clss <= 200:
        plt.yticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
        plt.xticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
    else:
        plt.yticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)
        plt.xticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + '_cbar.pdf', bbox_inches='tight')
    plt.close()

    return cm


def dummy_matrix(mat, filename, font_verbose=False):
    # Ensure a usable font is set before plotting
    configure_mpl_font(verbose=font_verbose)
    matplotlib.rcParams.update({'font.size': 18})

    cm = mat

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=16)
    plt.yticks([0, 19, 39, 59], [0, 20, 40, 60], fontsize=16)
    plt.xticks([0, 19, 39], [0, 20, 40], fontsize=16)

    plt.xlabel('Virtual Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()

    print('transpose')
    cm = np.transpose(mat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    cbar = plt.colorbar(cax, shrink=0.7)
    cbar.ax.tick_params(labelsize=16)
    plt.xticks([0, 19, 39, 59], [0, 20, 40, 60], fontsize=16)
    plt.yticks([0, 19, 39], [0, 20, 40], fontsize=16)

    plt.ylabel('Virtual Label', fontsize=20)
    plt.xlabel('True Label', fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + '_2.pdf', bbox_inches='tight')
    return cm


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


if __name__ == '__main__':
    # show available fonts in environment (useful for debugging)
    list_available_fonts(verbose=True, limit=100)
    configure_mpl_font(verbose=True)
    matplotlib.rcParams.update({'font.size': 18})

    cm = np.random.rand(100, 100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    plt.yticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    plt.xticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.tight_layout()
    plt.savefig('2.pdf', bbox_inches='tight')
    plt.close()