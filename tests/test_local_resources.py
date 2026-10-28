"""Infrastructure regression tests using tiny temporary data only; no network."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import local_resources as r


class LocalResourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(r, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        for i in [1, 2, 3]:
            (self.root / f'assignments/assignment{i}/assignment{i}/cs231n/datasets').mkdir(parents=True)

    def coco_archive(self, names):
        archive = self.root / r.RESOURCES['coco'][1]
        archive.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive, 'w') as z:
            for name in names:
                z.writestr(name, 'tiny test fixture')
        return archive

    def test_shared_data_links_and_cleanup_preserve_real_work(self):
        data = self.root / 'datasets/cifar-10-batches-py'
        data.mkdir(parents=True)
        (data / 'test').write_text('shared')
        real = self.root / 'assignments/assignment1/assignment1/cs231n/datasets/cifar-10-batches-py'
        real.mkdir()
        (real / 'my-data').write_text('do not delete')
        r.link_data()
        alias = self.root / 'assignments/assignment2/assignment2/cs231n/datasets/cifar-10-batches-py'
        self.assertTrue(alias.is_symlink())
        with contextlib.redirect_stdout(io.StringIO()):
            r.clean('cifar10')
        self.assertTrue(data.exists())  # dry run
        with contextlib.redirect_stdout(io.StringIO()):
            r.clean('cifar10', yes=True)
        self.assertFalse(data.exists())
        self.assertFalse(alias.is_symlink())
        self.assertEqual((real / 'my-data').read_text(), 'do not delete')

    def test_archive_traversal_does_not_publish_or_escape(self):
        self.coco_archive(['coco_captioning/../../escape.txt'])
        with self.assertRaises(ValueError):
            r.prepare_one('coco')
        self.assertFalse((self.root / 'escape.txt').exists())
        self.assertFalse((self.root / 'datasets/coco_captioning').exists())

    def test_valid_archive_is_reusable_and_archive_only_cleanup_retains_data(self):
        names=['coco2014_captions.h5', 'coco2014_vocab.json', 'train2014_vgg16_fc7_pca.h5',
               'val2014_vgg16_fc7_pca.h5', 'train2014_urls.txt', 'val2014_urls.txt']
        archive = self.coco_archive(['coco_captioning/' + n for n in names])
        r.prepare_one('coco')
        r.prepare_one('coco')
        self.assertTrue(r.present('coco'))
        with contextlib.redirect_stdout(io.StringIO()):
            r.clean('coco', archives_only=True, yes=True)
        self.assertFalse(archive.exists())
        self.assertTrue(r.present('coco'))

    def test_redirected_directory_outside_workspace_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / 'cache').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                r.clean('clip', yes=True)

    def test_setup_never_fetches_and_detects_other_assignment(self):
        import local_runtime as runtime
        from types import SimpleNamespace
        original = Path.cwd()
        old_path = list(sys.path)
        old_modules = {k:v for k,v in sys.modules.items() if k == 'cs231n' or k.startswith('cs231n.')}
        import os
        try:
            with patch.object(runtime, 'ROOT', self.root), patch.dict(os.environ), patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
                with contextlib.redirect_stdout(io.StringIO()):
                    runtime.setup(1)
                sys.modules['cs231n'] = SimpleNamespace(__file__=str(self.root/'assignments/assignment1/assignment1/cs231n/__init__.py'))
                with self.assertRaisesRegex(RuntimeError, '重启'):
                    runtime.setup(2)
        finally:
            os.chdir(original)
            sys.path[:] = old_path
            for k in list(sys.modules):
                if k == 'cs231n' or k.startswith('cs231n.'):
                    del sys.modules[k]
            sys.modules.update(old_modules)

    def test_short_http_transfer_never_publishes_incomplete_file(self):
        class Response(io.BytesIO):
            status = 200
            headers = {'Content-Length': '100'}
        with patch('urllib.request.urlopen', return_value=Response(b'PK tiny')):
            with self.assertRaises(IOError):
                r.fetch('emoji')
        self.assertFalse((self.root / 'datasets/emoji_data.npz').exists())
        self.assertTrue((self.root / 'datasets/emoji_data.npz.part').exists())

    def test_resume_and_server_ignoring_range_do_not_duplicate_bytes(self):
        class Response(io.BytesIO):
            pass
        for resume in [True, False]:
            with self.subTest(resume=resume):
                part = self.root / 'datasets/emoji_data.npz.part'
                part.parent.mkdir(exist_ok=True)
                part.write_bytes(b'PK first')
                response = Response(b' rest' if resume else b'PK first rest')
                response.status = 206 if resume else 200
                response.headers = {'Content-Length': str(len(response.getvalue()))}
                if resume:
                    response.headers['Content-Range'] = 'bytes 8-12/13'
                with patch('urllib.request.urlopen', return_value=response), contextlib.redirect_stdout(io.StringIO()):
                    r.fetch('emoji')
                dest = self.root / 'datasets/emoji_data.npz'
                self.assertEqual(dest.read_bytes(), b'PK first rest')
                self.assertFalse(part.exists())
                dest.unlink()

    def test_failed_extraction_keeps_archive_and_existing_work(self):
        archive = self.coco_archive(['coco_captioning/coco2014_vocab.json'])
        with self.assertRaises(ValueError):
            r.prepare_one('coco')
        self.assertTrue(archive.exists())
        self.assertFalse((self.root / 'datasets/coco_captioning').exists())
        self.assertFalse(list((self.root / 'datasets').glob('.extract-*')))


if __name__ == '__main__':
    unittest.main()
