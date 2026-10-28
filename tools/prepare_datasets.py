#!/usr/bin/env python3
"""Backward-compatible offline preparation command; downloads are explicit."""
from local_resources import link_data, prepare_one

if __name__ == '__main__':
    for name in ['cifar10', 'coco']:
        prepare_one(name)
    link_data()
    print('本地数据链接已检查；缺少的资源不会自动下载。')
