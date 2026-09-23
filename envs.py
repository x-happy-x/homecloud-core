"""Окружения моделей ядра: где лежат рабочие venv."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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
