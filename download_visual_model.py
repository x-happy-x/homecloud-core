"""Download one visual-search model into HomeCloud's shared Hugging Face cache."""
import argparse
import os
from pathlib import Path

import envs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--cache', type=Path, default=envs.hf_home())
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(args.cache)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import snapshot_download

    print(f'Downloading {args.model}', flush=True)
    # Runtime uses PyTorch. Jina also publishes a multi-gigabyte ONNX copy of
    # the same model; do not download duplicate weights we never load.
    folder = snapshot_download(
        args.model, cache_dir=args.cache / 'hub', max_workers=2,
        ignore_patterns=['onnx/*', 'pytorch_model.bin'])
    # Interrupted retries leave random-suffixed partial blobs behind. Once the
    # snapshot is complete they are not referenced and only waste disk space.
    model_cache = args.cache / 'hub' / ('models--' + args.model.replace('/', '--'))
    for partial in (model_cache / 'blobs').glob('*.incomplete'):
        partial.unlink(missing_ok=True)
    print(f'Ready: {folder}', flush=True)


if __name__ == '__main__':
    main()
