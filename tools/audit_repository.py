#!/usr/bin/env python3
"""Generate a reproducible file inventory and download audit; no network or model execution."""
from pathlib import Path
import collections,hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parents[1]
IGNORE={'.git','.agents','.codex','.aws','.venv','__pycache__','.download-parts'}
def included(p):return not any(x in IGNORE for x in p.relative_to(ROOT).parts) and not p.name.endswith('.part')
def main():
 manifest=json.loads((ROOT/'resources/download_manifest.json').read_text())
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and included(p))
 dirs=sorted(p for p in ROOT.rglob('*') if p.is_dir() and included(p))
 errors=[]
 for r in manifest:
  if r['status']=='ok':
   p=ROOT/r['path']
   if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=r.get('local_sha256',r['sha256']):errors.append(r['path'])
 slides=list((ROOT/'slides').rglob('*.pdf'))
 notebooks=list((ROOT/'assignments').rglob('*.ipynb'))
 for p in notebooks:json.loads(p.read_text())
 zipmissing=[]
 for z in (ROOT/'assignments').glob('*.zip'):
  with zipfile.ZipFile(z) as f:
   for n in f.namelist():
    if n.endswith('/') or '__MACOSX' in n or '.DS_Store' in n:continue
    p=z.parent/z.stem.replace('_colab','')/n
    if not p.exists():zipmissing.append(str(p.relative_to(ROOT)))
 assets=[]
 for p in (ROOT/'readings/notes').glob('*.md'):
  for ref in re.findall(r'(?:src=[\"\']|\]\()(assets/[^\"\'\s\)<>]+)',p.read_text()):
   if not (p.parent/ref).exists():assets.append((p.name,ref))
 lines=['# 资料审计与完整文件目录','', '资料来源核查：2026-10-19；本地运行改造核查：2026-10-28。运行 `python tools/audit_repository.py` 可更新本页；本次为静态结构与文件完整性检查，未执行未完成的作业、训练模型或隐藏评分。','',
 f'扫描得到 {len(files)} 个普通文件（不含 Git、环境配置、虚拟环境、Python 缓存和下载分片），{len(dirs)} 个目录；幻灯片 PDF {len(slides)} 份，作业 Notebook {len(notebooks)} 本（核心练习 14 本，提交工具 3 本）。', '',
 f'下载清单共 {len(manifest)} 项，成功 {sum(r["status"]=="ok" for r in manifest)} 项；延后下载 {sum(r["status"]=="deferred" for r in manifest)} 项，失败 {sum(r["status"]=="failed" for r in manifest)} 项。成功项 SHA-256 复核异常：{len(errors)}；原始 ZIP 有效条目缺失：{len(zipmissing)}；笔记本地图片缺失：{len(assets)}。','',
 '## 仍需区分的内容','',
 '- 2025 官网第 18 讲没有公开 slides 链接；2024 第 18 讲仅作历史补充。',
 '- Canvas 2025 视频需要课程权限；Ed、Gradescope 批改和隐藏测试不在公开存档内。',
 '- 网页快照未镜像全部图片、脚本、字体、视频；课件内外部视频也未镜像。',
 '- CLIP、DINO 与 DAVIS 首次获取由 Notebook 显式开关控制，默认不下载，集中在 cache/。',
 '- 已测试本地初始化、资源工具和 A1 导入；尚未安装完整 A3/GPU 环境，未执行未完成题目。','',
 '## 本次补充与修复','',
 '- 首次从 ZIP 恢复初始化文件；后续在解压目录改造本地初始化与路径，原 ZIP 保留上游版本。',
 '- 下载官网项目规范、讨论课、Efficient BackProp、辅助论文、外部阅读及笔记插图。',
 '- 笔记图片改为相对路径；两个交互演示保留官网地址。',
 '- 新增本地资源管理：共享数据、按需下载、归档校验和默认只预览的清理；来源哈希与本地修改哈希分开记录。',
 '- 已按用户要求停止大文件下载并清理约 176 MiB 临时分片。未下载资源的用途、获取和操作见 [按需资源指南](未下载资源与按需获取指南.md)。','',
 '## 下载状态与来源','', '| 路径 | 状态 | 大小 | 来源 / 失败原因 |','|---|---|---:|---|']
 for r in manifest:
  why=f'[来源]({r["url"]})'
  if r['status']!='ok':why+='；'+r.get('reason',r.get('error','')).replace('\n',' ').replace('|','/').replace('<','&lt;')
  lines.append(f'| `{r["path"]}` | {r["status"]} | {r.get("bytes",0):,} B | {why} |')
 lines+=['','完整 SHA-256 见 [JSON 清单](../resources/download_manifest.json)。','', '## 校验问题','',f'- 哈希不符：{errors or "无"}',f'- ZIP 缺失：{zipmissing or "无（忽略 macOS 元数据）"}',f'- 本地图片缺失：{assets or "无"}','',
 '## 文件夹清单','', '课程目录的具体职责见 [学习指南](CS231n_2025_仓库导览与学习路线.md)。以下包括运行资源和原包编译产物；隐藏 Git 内部对象不逐条展开。','']
 for p in dirs:lines.append('- `'+str(p.relative_to(ROOT))+'/`')
 lines+=['','## 普通文件清单','', '| 文件 | 大小（字节） |','|---|---:|']
 for p in files:lines.append(f'| `{p.relative_to(ROOT)}` | {p.stat().st_size:,} |')
 (ROOT/'docs/资料审计.md').write_text('\n'.join(lines)+'\n')
 print('FILES',len(files),'DOWNLOADS',len(manifest),'HASH_ERRORS',errors,'ZIP_MISSING',zipmissing,'ASSET_MISSING',assets)
 if errors or zipmissing or assets:raise SystemExit(1)
if __name__=='__main__':main()
