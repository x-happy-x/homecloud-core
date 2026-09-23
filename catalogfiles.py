"""Файлы каталога рядом с базой: миниатюры лиц, обученные модели роутера.

На хабе они лежат в папке каталога, как и раньше. Ядро пишет их у себя в
рабочую папку (так их сразу видят следующие шаги этапа) и отправляет копию на
хаб; читает — из своей копии, а если её нет, забирает с хаба и кладёт рядом.
Без hub.json всё сводится к обычным файлам в папке каталога.
"""
from pathlib import Path

import hublink

ALLOWED = ('thumbnails/', 'router-models/')


def _relative(relative):
    value = str(relative).replace('\\', '/').lstrip('/')
    if '..' in value.split('/') or not value.startswith(ALLOWED):
        raise ValueError(f'Недопустимый файл каталога: {relative!r}')
    return value


def local(folder, relative):
    return Path(folder) / _relative(relative)


def publish(folder, relative):
    """Файл уже записан в рабочую папку — отправить копию на хаб."""
    link = hublink.link_for(folder)
    if link is None:
        return
    path = local(folder, relative)
    link.put_file(_relative(relative), path.read_bytes())


def write_bytes(folder, relative, data):
    path = local(folder, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    publish(folder, relative)
    return path


def read_bytes(folder, relative):
    path = local(folder, relative)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        link = hublink.link_for(folder)
        if link is None:
            raise
    data = link.get_file(_relative(relative))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError:
        pass
    return data


def fetch(folder, relative):
    """Путь к файлу на этой машине: с хаба забирается, если своей копии нет."""
    path = local(folder, relative)
    if not path.is_file():
        read_bytes(folder, relative)
    return path


def delete(folder, relative):
    local(folder, relative).unlink(missing_ok=True)
    link = hublink.link_for(folder)
    if link is not None:
        try:
            link.delete_file(_relative(relative))
        except hublink.HubError:
            pass
