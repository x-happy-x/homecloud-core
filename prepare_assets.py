"""Explicit online setup: download official model pack and public LFW test data."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def download(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print('Downloading:', url, flush=True)
        partial = target.with_suffix(target.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output)
        partial.replace(target)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    checksum = digest.hexdigest()
    if expected and checksum != expected:
        raise RuntimeError(f'Checksum mismatch: {target}')
    return {'url': url, 'sha256': checksum}


def main():
    archive = ROOT / 'downloads' / 'buffalo_l.zip'
    records = {'model': download(
        'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip', archive)}
    destination = ROOT / 'models' / 'buffalo_l'
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as pack:
        for name in ('det_10g.onnx', 'w600k_r50.onnx'):
            matches = [n for n in pack.namelist() if Path(n).name == name]
            if len(matches) != 1:
                raise RuntimeError(f'Missing or ambiguous model: {name}')
            with pack.open(matches[0]) as source, (destination / name).open('wb') as output:
                shutil.copyfileobj(source, output)
    archive = ROOT / 'downloads' / 'lfw.tgz'
    records['dataset'] = download('https://ndownloader.figshare.com/files/5976018', archive,
        '055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0')
    sample = ROOT / 'test-photos'
    sample.mkdir(exist_ok=True)
    count = 0
    with tarfile.open(archive, 'r:gz') as pack:
        people = {}
        for member in pack.getmembers():
            if member.isfile() and member.name.lower().endswith('.jpg'):
                people.setdefault(Path(member.name).parent.name, []).append(member)
        selected = sorted(person for person, images in people.items() if len(images) >= 20)[:20]
        for person in selected:
            folder = sample / person
            folder.mkdir(exist_ok=True)
            for member in sorted(people[person], key=lambda item: item.name)[:30]:
                with pack.extractfile(member) as source, (folder / Path(member.name).name).open('wb') as output:
                    shutil.copyfileobj(source, output)
                count += 1
    records['sample'] = {'people': len(selected), 'photos': count}
    (ROOT / 'assets-manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Assets ready:', records['sample'], flush=True)


if __name__ == '__main__':
    main()
