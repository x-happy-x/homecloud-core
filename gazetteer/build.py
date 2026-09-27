"""Собирает gazetteer/places.tsv.gz — офлайн-справочник городов для подборок по месту.

Источник — GeoNames (CC BY 4.0): cities5000.zip и countryInfo.txt с
https://download.geonames.org/export/dump/. Скачиваются вручную один раз;
сервис в интернет не ходит. Название — русское из alternatenames (первое
кириллическое без украинских и белорусских букв), иначе латиницей.

    python gazetteer/build.py cities5000.zip countryInfo.txt
"""
import csv
import difflib
import gzip
import io
import re
import sys
import zipfile
from pathlib import Path

RUSSIAN_ONLY = re.compile(r'^[А-Яа-яЁё][А-Яа-яЁё \-]*$')
TRANSLIT = dict(zip('абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                    ['a', 'b', 'v', 'g', 'd', 'e', 'e', 'zh', 'z', 'i', 'y', 'k', 'l', 'm', 'n', 'o',
                     'p', 'r', 's', 't', 'u', 'f', 'kh', 'ts', 'ch', 'sh', 'shch', '', 'y', '', 'e',
                     'yu', 'ya']))


def latin(name):
    return ''.join(TRANSLIT.get(char, char) for char in name.lower())


def russian(names, ascii_name):
    """Русское имя из вариантов: только русские буквы и ближе всех к латинскому по транслитерации.

    Варианты в GeoNames отсортированы по алфавиту, и первое кириллическое
    часто аварское или казахское («МахIачхъала»), а не русское.
    """
    target = ascii_name.lower()
    best, best_score = '', 0.0
    for name in names.split(','):
        name = name.strip()
        if not name or len(name) > 40 or not RUSSIAN_ONLY.match(name):
            continue
        score = difflib.SequenceMatcher(None, latin(name), target).ratio()
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= 0.45 else ''


# Крупные города, чьё латинское имя на русское не похоже (Moscow — Москва).
KNOWN = {
    ('Moscow', 'RU'): 'Москва', ('Saint Petersburg', 'RU'): 'Санкт-Петербург', ('Cairo', 'EG'): 'Каир',
    ('Istanbul', 'TR'): 'Стамбул', ('Dubai', 'AE'): 'Дубай', ('Abu Dhabi', 'AE'): 'Абу-Даби',
    ('Paris', 'FR'): 'Париж', ('London', 'GB'): 'Лондон', ('Rome', 'IT'): 'Рим', ('Milan', 'IT'): 'Милан',
    ('Venice', 'IT'): 'Венеция', ('Berlin', 'DE'): 'Берлин', ('Munich', 'DE'): 'Мюнхен',
    ('Prague', 'CZ'): 'Прага', ('Vienna', 'AT'): 'Вена', ('Warsaw', 'PL'): 'Варшава',
    ('Barcelona', 'ES'): 'Барселона', ('Madrid', 'ES'): 'Мадрид', ('Athens', 'GR'): 'Афины',
    ('Baku', 'AZ'): 'Баку', ('Tbilisi', 'GE'): 'Тбилиси', ('Batumi', 'GE'): 'Батуми', ('Yerevan', 'AM'): 'Ереван',
    ('Minsk', 'BY'): 'Минск', ('Kyiv', 'UA'): 'Киев', ('Tashkent', 'UZ'): 'Ташкент', ('Almaty', 'KZ'): 'Алматы',
    ('Astana', 'KZ'): 'Астана', ('Bishkek', 'KG'): 'Бишкек', ('Tehran', 'IR'): 'Тегеран',
    ('Mecca', 'SA'): 'Мекка', ('Medina', 'SA'): 'Медина', ('Riyadh', 'SA'): 'Эр-Рияд', ('Doha', 'QA'): 'Доха',
    ('Hurghada', 'EG'): 'Хургада', ('Sharm el-Sheikh', 'EG'): 'Шарм-эш-Шейх', ('Alexandria', 'EG'): 'Александрия',
    ('Bangkok', 'TH'): 'Бангкок', ('Phuket', 'TH'): 'Пхукет', ('Beijing', 'CN'): 'Пекин',
    ('Tokyo', 'JP'): 'Токио', ('New York City', 'US'): 'Нью-Йорк', ('Ankara', 'TR'): 'Анкара',
    ('Antalya', 'TR'): 'Анталья', ('Kazan', 'RU'): 'Казань', ('Grozny', 'RU'): 'Грозный',
    ('Nalchik', 'RU'): 'Нальчик', ('Vladikavkaz', 'RU'): 'Владикавказ', ('Stavropol', 'RU'): 'Ставрополь',
    ('Krasnodar', 'RU'): 'Краснодар', ('Kislovodsk', 'RU'): 'Кисловодск', ('Astrakhan', 'RU'): 'Астрахань',
}


def main(cities_zip, country_info):
    countries = {}
    for line in Path(country_info).read_text(encoding='utf-8').splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split('\t')
        countries[parts[0]] = parts[4]
    rows = []
    with zipfile.ZipFile(cities_zip) as archive:
        name = next(item for item in archive.namelist() if item.endswith('.txt'))
        for line in io.TextIOWrapper(archive.open(name), encoding='utf-8'):
            parts = line.rstrip('\n').split('\t')
            ident, ascii_name, alternates = parts[0], parts[2], parts[3]
            # Районы и кварталы большого города (PPLX) тянули бы точку на себя: «Булак», а не Каир.
            if parts[7] in ('PPLX', 'PPLH', 'PPLQ', 'PPLW'):
                continue
            lat, lon, code, population = parts[4], parts[5], parts[8], parts[14]
            rows.append((ident, KNOWN.get((ascii_name, code)) or russian(alternates, ascii_name) or ascii_name, code,
                         f'{float(lat):.4f}', f'{float(lon):.4f}', population or '0'))
    target = Path(__file__).resolve().parent / 'places.tsv.gz'
    with gzip.open(target, 'wt', encoding='utf-8', newline='') as sink:
        writer = csv.writer(sink, delimiter='\t', lineterminator='\n')
        writer.writerow(('#countries', *[f'{code}={name}' for code, name in sorted(countries.items())]))
        writer.writerows(rows)
    print(f'{len(rows)} places -> {target} ({target.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main(*sys.argv[1:3])
