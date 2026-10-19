#!/usr/bin/env python3
"""Download missing public CS231n 2025 resources; keep a source/checksum manifest."""
import concurrent.futures, hashlib, json, pathlib, re, subprocess, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / 'resources/download_manifest.json'

def tasks(include_data=False):
    out = []
    def add(path, url, kind='course'): out.append((path, url, kind))
    for name in ['index','schedule','assignments','project','reports']:
        add(f'resources/official/{name}.html', f'https://cs231n.stanford.edu/2025/{name}.html')
    add('resources/official/cvpr2017AuthorKit.zip','https://cvpr2017.thecvf.com/files/cvpr2017AuthorKit.zip','template')
    add('resources/official/setup-instructions.html','https://cs231n.github.io/setup-instructions/')
    add('resources/official/notes-index.html','https://cs231n.github.io/')
    add('readings/handouts/1998_Efficient_BackProp.pdf','https://cs231n.stanford.edu/papers/lecun-98b.pdf')
    add('slides/2021/discussion_4_tensorflow.ipynb','https://cs231n.stanford.edu/slides/2021/discussion_4_tensorflow.ipynb','historical')
    for name, ident in [('section_2_backprop','1yjxfAugU5JrbgCb1TcCbXDMCKE_G_P-e'),('section_4_pytorch','1Dl_Xs5GKjEOAedQUVOjG_fPMB1eXtLSm'),('section_5_rnn_transformers','1mC5CWwekbZ2NrYv6Zfpuv55z8DuOZXVP')]:
        add(f'slides/2025/{name}.ipynb',f'https://drive.google.com/uc?export=download&id={ident}','notebook')
    for name, ident in [('2020_SimCLR','2002.05709'),('2020_DDPM','2006.11239'),('2021_CLIP','2103.00020'),('2022_Classifier_Free_Guidance','2207.12598'),('2013_VAE','1312.6114'),('2014_GAN','1406.2661'),('2015_Batch_Normalization','1502.03167')]:
        add(f'readings/papers/{name}.pdf',f'https://arxiv.org/pdf/{ident}','supplement')
    for name,url in [
        ('momentum','https://distill.pub/2017/momentum/'),('backprop','https://colah.github.io/posts/2015-08-Backprop/'),
        ('nndl-ch2','http://neuralnetworksanddeeplearning.com/chap2.html'),('lstm','https://colah.github.io/posts/2015-08-Understanding-LSTMs/'),
        ('illustrated-transformer','https://jalammar.github.io/illustrated-transformer/'),('attention','https://lilianweng.github.io/posts/2018-06-24-attention/'),
        ('self-supervised','https://lilianweng.github.io/posts/2019-11-10-self-supervised/'),('elbo','https://yunfanj.com/blog/2021/01/11/ELBO.html'),
        ('dlbook-rnn','https://www.deeplearningbook.org/contents/rnn.html')]:
        add(f'resources/blogs/{name}.html',url,'web_snapshot')
    refs=set()
    for p in (ROOT/'readings/notes').glob('*.md'):
        refs.update('/assets/'+x for x in re.findall(r'(?:/)?assets/([^\s\"\)\'<>]+)',p.read_text()))
    for ref in sorted(refs):
        if not ref.endswith('.html'):
            add('readings/notes'+ref,'https://cs231n.github.io'+ref,'note_asset')
    if include_data:
        base='assignments/assignment3/assignment3/'
        for fname in ['emoji_data.npz','text_embeddings.pt']:
            add(base+'cs231n/datasets/'+fname,'https://cs231n.stanford.edu/2025/storage/a3/'+fname,'data')
        add(base+'pretrained_model/pretrained_simclr_model.pth','https://cs231n.stanford.edu/2025/storage/a3/pretrained_simclr_model.pth','data')
        add(base+'cs231n/exp/pretrained/model-70000.pt','https://cs231n.stanford.edu/2025/storage/a3/model-70000.pt','data')
        add('datasets/cifar-10-python.tar.gz','https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz','data')
        add('datasets/coco_captioning.zip','https://cs231n.stanford.edu/coco_captioning.zip','data')
        add('datasets/imagenet_val_25.npz','https://cs231n.stanford.edu/imagenet_val_25.npz','data')
    return out

def download(t):
    path,url,kind=t; dest=ROOT/path; dest.parent.mkdir(parents=True,exist_ok=True)
    record=dict(path=path,url=url,kind=kind,checked_date='2026-10-19')
    try:
        if not dest.exists():
            part=dest.with_name(dest.name+'.part')
            r=subprocess.run(['curl','-L','--fail','--silent','--show-error','--connect-timeout','15','--max-filesize','268435456','--max-time','300' if kind=='data' else '240' if path.endswith('.pdf') else '45',url,'-o',str(part)],capture_output=True,text=True)
            if r.returncode: raise RuntimeError(r.stderr.strip()[-450:])
            data=part.read_bytes()
            if path.endswith('.pdf') and not data.startswith(b'%PDF-'): raise ValueError('Response is not PDF')
            if path.endswith('.ipynb'):
                obj=json.loads(data)
                if 'cells' not in obj: raise ValueError('Response is not notebook')
            part.replace(dest)
        record.update(status='ok',bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    except Exception as e:
        record.update(status='failed',error=str(e))
        part=dest.with_name(dest.name+'.part')
        if part.exists():part.unlink()
    return record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',action='store_true');ap.add_argument('--retry-deferred',action='store_true');ap.add_argument('--only',help='Download only a path containing this string');args=ap.parse_args()
    old={r['path']:r for r in json.loads(P.read_text())} if P.exists() else {}
    selected=[t for t in tasks(args.data) if (not args.only or args.only in t[0]) and (args.retry_deferred or old.get(t[0],{}).get('status')!='deferred')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(download,selected):
            old[r['path']]=r
            P.write_text(json.dumps(list(old.values()),ensure_ascii=False,indent=2)+'\n')
            print(r['status'],r['path'],r.get('bytes',r.get('error')),flush=True)
if __name__=='__main__':main()
