"""Окружения моделей ядра: где лежат рабочие venv."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Общие для машины папки прежней раскладки; на PC-X это ссылки.
SHARED_MODELS = Path(r'C:\cv-models')
SHARED_OCR = Path(r'C:\cv-ocr')


def link_target(path):
    """Цель ссылки (junction или symlink) без прохода через неё; не ссылка — None."""
    try:
        target = os.readlink(path)
    except (OSError, ValueError, NotImplementedError):
        return None
    if target.startswith('\\\\?\\'):
        target = target[4:]
    target = Path(target)
    return target if target.is_absolute() else Path(path).parent / target


def real(path):
    """Путь, а если это ссылка — её цель."""
    return link_target(path) or Path(path)


def worker_root(root=ROOT):
    """Папка work ядра с окружениями моделей.

    На рабочей машине work бывает ссылкой на другой диск, а сеанс SSH, из
    которого хаб запускает ядро, через такую ссылку не ходит («untrusted mount
    point»). Поэтому сразу берём настоящий путь: чтение цели ссылку не открывает.
    """
    work = Path(root) / 'work'
    return link_target(work) or work


def python(venv, root=ROOT):
    """python.exe рабочего окружения: vision-venv, audio-venv, imgutils-venv…"""
    return worker_root(root) / venv / 'Scripts' / 'python.exe'


def models_root(root=ROOT):
    """Кэш моделей: models/cv-models ядра, иначе общий C:/cv-models (без прохода по ссылке)."""
    own = real(Path(root) / 'models') / 'cv-models'
    if own.is_dir():
        return own
    shared = real(SHARED_MODELS)
    # На новом ядре общей папки нет — модели ложатся в его собственную.
    return shared if shared.is_dir() else own


def hf_home(root=ROOT):
    """HF_HOME для окружений моделей."""
    return models_root(root) / 'huggingface'


def ocr_python(root=ROOT):
    """OCR: прежний общий C:/cv-ocr, а если его нет — work/cv-ocr ядра (туда его ставит components)."""
    shared = real(SHARED_OCR) / 'Scripts' / 'python.exe'
    return shared if shared.is_file() else worker_root(root) / 'cv-ocr' / 'Scripts' / 'python.exe'
