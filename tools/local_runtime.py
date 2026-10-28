"""Local Notebook setup. No downloads or package installation during setup."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'datasets'
MODEL_DIR = ROOT / 'models'
CACHE_DIR = ROOT / 'cache'


def setup(assignment):
    target = ROOT / f'assignments/assignment{assignment}/assignment{assignment}'
    if not target.is_dir():
        raise FileNotFoundError(target)
    for name, module in list(sys.modules.items()):
        if name == 'cs231n' or name.startswith('cs231n.'):
            origin = getattr(module, '__file__', None)
            if origin and not Path(origin).resolve().is_relative_to(target.resolve()):
                raise RuntimeError('内核已导入另一份 cs231n。请重启 Kernel，再运行本作业首单元。')
    # Use assignment code before any similarly named installed package.
    sys.path[:] = [str(target)] + [p for p in sys.path if p != str(target)]
    os.chdir(target)
    # Contain large model/dataset caches for explicit later cleanup.
    os.environ['CS231N_DATA_DIR'] = str(DATA_DIR)
    os.environ['TORCH_HOME'] = str(CACHE_DIR / 'torch')
    os.environ['TFDS_DATA_DIR'] = str(CACHE_DIR / 'tfds')
    from local_resources import link_data
    link_data()
    print(f'本地作业目录：{target}\nPython：{sys.executable}')
    print('首单元不会下载数据或安装软件。资源状态：python tools/local_resources.py status（仓库根目录执行）')
    return target


def require_resource(name):
    from local_resources import RESOURCES, local, present
    if not present(name):
        raise FileNotFoundError(f'尚未准备 {name}。在仓库根目录执行：python tools/local_resources.py fetch {name}')
    _, archive, extracted, _ = RESOURCES[name]
    return local(extracted or archive)


def load_clip(device, allow_download=False):
    import clip
    # Use CLIP's official model table to find and verify its own cache file.
    from clip.clip import _MODELS
    filename = _MODELS['ViT-B/32'].rsplit('/', 1)[-1]
    cache = CACHE_DIR / 'clip'
    if not (cache / filename).is_file() and not allow_download:
        raise FileNotFoundError('CLIP 权重未缓存。确认需使用后，将本单元 ALLOW_MODEL_DOWNLOADS 改为 True。缓存位于 cache/clip。')
    if not allow_download:
        import hashlib
        expected = _MODELS['ViT-B/32'].split('/')[-2]
        with (cache / filename).open('rb') as f:
            actual = hashlib.sha256()
            for block in iter(lambda: f.read(1024**2), b''):
                actual.update(block)
        if actual.hexdigest() != expected:
            raise ValueError('CLIP 缓存不完整。清理后显式允许重新下载。')
    return clip.load('ViT-B/32', device=device, download_root=str(cache))


def load_dino(allow_download=False):
    import torch
    hub = CACHE_DIR / 'torch/hub'
    torch.hub.set_dir(str(hub))
    source = hub / 'facebookresearch_dino_main'
    checkpoint = hub / 'checkpoints/dino_deitsmall8_pretrain.pth'
    if not allow_download and (not (source / 'hubconf.py').is_file() or not checkpoint.is_file()):
        raise FileNotFoundError('DINO 代码/权重未缓存。确认后将 ALLOW_MODEL_DOWNLOADS 改为 True。缓存位于 cache/torch。')
    if allow_download:
        return torch.hub.load('facebookresearch/dino:main', 'dino_vits8', trust_repo=True)
    # Instantiate locally without a pretrained=True call that could fetch again.
    model = torch.hub.load(str(source), 'dino_vits8', source='local', pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    return model
