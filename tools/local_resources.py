#!/usr/bin/env python3
"""Explicit downloads, shared course datasets, and dry-run-first cleanup. Stdlib only."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
A3 = Path('assignments/assignment3/assignment3')
RESOURCES = {
    'cifar10': ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'datasets/cifar-10-python.tar.gz', 'datasets/cifar-10-batches-py', '约 163 MiB；步骤 05–14、19–21'),
    'coco': ('https://cs231n.stanford.edu/coco_captioning.zip',
             'datasets/coco_captioning.zip', 'datasets/coco_captioning', '约 987 MiB 压缩；步骤 16、18、22'),
    'emoji': ('https://cs231n.stanford.edu/2025/storage/a3/emoji_data.npz',
              'datasets/emoji_data.npz', None, '约 29 MiB；步骤 25'),
    'text-embeddings': ('https://cs231n.stanford.edu/2025/storage/a3/text_embeddings.pt',
                        str(A3 / 'cs231n/datasets/text_embeddings.pt'), None, '约 3.5 MiB，已提供；步骤 25'),
    'simclr': ('https://cs231n.stanford.edu/2025/storage/a3/pretrained_simclr_model.pth',
               'models/simclr/pretrained_simclr_model.pth', None, '约 94 MiB；步骤 21'),
    'ddpm': ('https://cs231n.stanford.edu/2025/storage/a3/model-70000.pt',
             'models/ddpm/model-70000.pt', None, '约 143 MiB；步骤 25'),
    'imagenet-sample': ('https://cs231n.stanford.edu/imagenet_val_25.npz',
                        'datasets/imagenet_val_25.npz', None, '约 3.8 MiB，已提供；可选示例'),
}
CACHES = {'clip': 'cache/clip', 'dino': 'cache/torch', 'davis': 'cache/tfds',
          'fashion-mnist': 'datasets/discussion/FashionMNIST',
          'moving-mnist': 'datasets/discussion/MovingMNIST'}


def local(path):
    """Reject paths redirected outside this workspace, including symlinked parents."""
    p = ROOT / path
    if not p.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f'路径不在仓库内，拒绝操作：{p}')
    return p


def usage(path):
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def present(name):
    _, archive, extracted, _ = RESOURCES[name]
    p = local(extracted or archive)
    if not p.exists():
        return False
    if name == 'cifar10':
        return all((p / f).is_file() for f in ['batches.meta', 'test_batch'] + [f'data_batch_{i}' for i in range(1, 6)])
    if name == 'coco':
        return all((p / f).is_file() for f in ['coco2014_captions.h5', 'coco2014_vocab.json',
                   'train2014_vgg16_fc7_pca.h5', 'val2014_vgg16_fc7_pca.h5', 'train2014_urls.txt', 'val2014_urls.txt'])
    return p.is_file() and p.stat().st_size > 0


def link_data():
    """Only add aliases; never replace an existing real file/directory with a link."""
    for i in [1, 2, 3]:
        dest = local(f'assignments/assignment{i}/assignment{i}/cs231n/datasets')
        if not dest.is_dir():
            continue
        for name in ['cifar-10-batches-py', 'coco_captioning', 'imagenet_val_25.npz']:
            if name == 'coco_captioning' and i == 1:
                continue
            target = local('datasets/' + name)
            alias = dest / name
            if not target.exists() or alias.is_symlink():
                continue
            if alias.exists():
                print(f'保留现有数据（未复制）：{alias.relative_to(ROOT)}')
                continue
            try:
                alias.symlink_to(os.path.relpath(target, dest), target_is_directory=target.is_dir())
            except OSError as e:
                raise OSError('无法创建数据符号链接。Linux/WSL 可直接使用；Windows 需开启开发者模式。') from e


def validate_members(names, expected):
    for name in names:
        # Both separators are checked so archives remain safe across platforms.
        parts = name.replace('\\', '/').split('/')
        if not parts or parts[0] != expected or '..' in parts or name.startswith('/'):
            raise ValueError(f'归档条目超出预期目录：{name}')


def prepare_one(name):
    _, archive, extracted, _ = RESOURCES[name]
    if not extracted or present(name):
        return
    src, dest = local(archive), local(extracted)
    if not src.exists():
        return
    if dest.exists():
        raise FileExistsError(f'{dest} 已存在但数据不完整。先手动检查，不自动覆盖。')
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Staging is on the same filesystem; incomplete extraction never becomes ready data.
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=dest.parent) as temp:
        stage = Path(temp)
        expected = dest.name
        if src.name.endswith('.tar.gz'):
            digest = hashlib.md5()
            with src.open('rb') as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != 'c58f30108f718f92721af3b95e74349a':
                raise ValueError('CIFAR-10 归档 MD5 与官方值不同；不要解压残缺文件。')
            with tarfile.open(src) as f:
                members = f.getmembers()
                validate_members([m.name for m in members], expected)
                if any(not (m.isfile() or m.isdir()) for m in members):
                    raise ValueError('归档含非普通文件/目录条目')
                # Validated above; works on Python 3.10 without newer filter= support.
                f.extractall(stage, members=members)
        else:
            with zipfile.ZipFile(src) as f:
                infos = [m for m in f.infolist() if not m.filename.startswith('__MACOSX/')]
                validate_members([m.filename for m in infos], expected)
                if any(stat.S_ISLNK(m.external_attr >> 16) for m in infos):
                    raise ValueError('归档不应包含符号链接')
                f.extractall(stage, members=infos)
        staged = stage / expected
        # Validate expected contents before publishing.
        if name == 'cifar10':
            required = ['batches.meta', 'test_batch'] + [f'data_batch_{i}' for i in range(1, 6)]
        else:
            required = ['coco2014_captions.h5', 'coco2014_vocab.json', 'train2014_vgg16_fc7_pca.h5',
                        'val2014_vgg16_fc7_pca.h5', 'train2014_urls.txt', 'val2014_urls.txt']
        if not all((staged / f).is_file() for f in required):
            raise ValueError(f'{name} 归档缺少课程读取所需文件；未发布解压目录。')
        staged.replace(dest)
    print('已解压：', dest.relative_to(ROOT))


def fetch(name, keep_archive=False):
    url, archive, extracted, hint = RESOURCES[name]
    dest = local(archive)
    if present(name):
        print(f'{name} 已可用，跳过下载。')
        link_data()
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        part = dest.with_name(dest.name + '.part')
        offset = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        print(f'下载 {name}（{hint}），来源 {url}', flush=True)
        with urllib.request.urlopen(request, timeout=60) as response:
            # A server ignoring Range must restart rather than append duplicate bytes.
            resume = offset > 0 and response.status == 206
            if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                raise ValueError('服务器返回了不匹配的续传范围')
            expected = response.headers.get('Content-Length')
            count = 0
            with part.open('ab' if resume else 'wb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    count += len(chunk)
                    if count // (16 * 1024**2) != (count - len(chunk)) // (16 * 1024**2):
                        print(f'  本次已接收 {count / 1024**2:.0f} MiB', flush=True)
            if expected and count != int(expected):
                raise IOError('下载未完成，已保留 .part，可再次运行同一命令续传。')
        # NPZ/torch checkpoint are zip-based; reject HTML and empty error pages.
        with part.open('rb') as f:
            magic = f.read(20).lstrip().lower()
        if not magic or magic.startswith((b'<', b'<!doctype')):
            raise ValueError('响应不是数据文件，未改名为目标文件。')
        part.replace(dest)
    prepare_one(name)
    if extracted and not keep_archive:
        dest.unlink()  # Only remove the known archive after validated extraction.
        print('已删除完整归档，保留共享解压数据。')
    link_data()


def cleanup_paths(name, archives_only=False):
    if name in CACHES:
        if archives_only:
            raise ValueError('缓存没有单独归档；请去掉 --archives-only')
        paths = [local(CACHES[name])]
    else:
        _, archive, extracted, _ = RESOURCES[name]
        if archives_only and not extracted:
            raise ValueError('此项是直接使用的文件，不能按归档清理。')
        paths = [local(archive), local(archive + '.part')]
        if extracted and not archives_only:
            paths.append(local(extracted))
            # Remove only aliases pointing to this managed target; never real user directories.
            for alias in (ROOT / 'assignments').glob(f'assignment*/assignment*/cs231n/datasets/{Path(extracted).name}'):
                if alias.is_symlink() and alias.resolve() == local(extracted).resolve():
                    paths.insert(0, alias)
    return list(dict.fromkeys(paths))


def clean(name, archives_only=False, yes=False):
    paths = cleanup_paths(name, archives_only)
    total = 0
    for p in paths:
        size = usage(p)
        total += size
        print(f'{"删除" if yes else "预览"} {p.relative_to(ROOT)} ({size / 1024**2:.1f} MiB)')
        if yes:
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.exists():
                shutil.rmtree(p)
    print(f'{"已清理" if yes else "预计释放"} {total / 1024**2:.1f} MiB')
    if not yes:
        print('未删除任何文件；确认用途结束后，在相同命令末尾加 --yes 才执行。')


def status():
    for name, (_, archive, extracted, hint) in RESOURCES.items():
        size = sum(usage(local(p)) for p in [archive, archive + '.part'] + ([extracted] if extracted else []))
        print(f'{name:18} {"ready" if present(name) else "missing":7} {size / 1024**2:8.1f} MiB  {hint}')
    for name, relative in CACHES.items():
        print(f'{name:18} cache   {usage(local(relative)) / 1024**2:8.1f} MiB  {relative}')
    print('仅统计受管理路径，不含 conda 环境、系统缓存、课程资料和你的实验结果。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    get = sub.add_parser('fetch')
    get.add_argument('name', choices=RESOURCES)
    get.add_argument('--keep-archive', action='store_true')
    sub.add_parser('prepare')
    remove = sub.add_parser('clean')
    remove.add_argument('name', choices=list(RESOURCES) + list(CACHES))
    remove.add_argument('--archives-only', action='store_true')
    remove.add_argument('--yes', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'status':
            status()
        elif args.command == 'fetch':
            fetch(args.name, args.keep_archive)
        elif args.command == 'prepare':
            for name in ['cifar10', 'coco']:
                prepare_one(name)
            link_data()
        else:
            clean(args.name, args.archives_only, args.yes)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as e:
        parser.exit(1, f'操作未完成：{e}\n')


if __name__ == '__main__':
    main()
