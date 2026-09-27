"""Где снято: ближайший город по координатам из EXIF — без интернета.

Справочник — gazetteer/places.tsv.gz (города от 5 тысяч жителей из GeoNames,
CC BY 4.0, собирает gazetteer/build.py). Грузится лениво, при первом
запросе: хабу, который подборки не собирает, он не нужен.
"""
import csv
import gzip
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'gazetteer' / 'places.tsv.gz'
# Дальше этого от ближайшего города — место без названия (море, горы, трасса).
MAX_KM = 40.0

# Русские названия стран; остальные — по-английски из справочника.
COUNTRIES_RU = {
    'RU': 'Россия', 'AZ': 'Азербайджан', 'GE': 'Грузия', 'AM': 'Армения', 'KZ': 'Казахстан',
    'UZ': 'Узбекистан', 'KG': 'Киргизия', 'TJ': 'Таджикистан', 'TM': 'Туркмения', 'BY': 'Беларусь',
    'UA': 'Украина', 'TR': 'Турция', 'EG': 'Египет', 'AE': 'ОАЭ', 'SA': 'Саудовская Аравия',
    'QA': 'Катар', 'IR': 'Иран', 'TH': 'Таиланд', 'CN': 'Китай', 'JP': 'Япония', 'IN': 'Индия',
    'VN': 'Вьетнам', 'ID': 'Индонезия', 'MV': 'Мальдивы', 'LK': 'Шри-Ланка', 'TN': 'Тунис',
    'MA': 'Марокко', 'IL': 'Израиль', 'JO': 'Иордания', 'CY': 'Кипр', 'GR': 'Греция', 'IT': 'Италия',
    'ES': 'Испания', 'FR': 'Франция', 'DE': 'Германия', 'AT': 'Австрия', 'CZ': 'Чехия',
    'PL': 'Польша', 'GB': 'Великобритания', 'NL': 'Нидерланды', 'CH': 'Швейцария', 'US': 'США',
    'ME': 'Черногория', 'RS': 'Сербия', 'BG': 'Болгария', 'HU': 'Венгрия', 'FI': 'Финляндия',
    'EE': 'Эстония', 'LV': 'Латвия', 'LT': 'Литва', 'MN': 'Монголия', 'KR': 'Южная Корея',
}


class Place:
    __slots__ = ('id', 'name', 'country', 'lat', 'lon', 'population')

    @property
    def country_name(self):
        return COUNTRIES_RU.get(self.country) or _gazetteer().countries.get(self.country, self.country)


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    value = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, value)))


class Gazetteer:
    def __init__(self, path=DATA):
        self.cells = {}
        self.countries = {}
        self.by_id = {}
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as source:
            for row in csv.reader(source, delimiter='\t'):
                if row and row[0] == '#countries':
                    self.countries = dict(item.split('=', 1) for item in row[1:])
                    continue
                place = Place()
                place.id, place.name, place.country = int(row[0]), row[1], row[2]
                place.lat, place.lon, place.population = float(row[3]), float(row[4]), int(row[5] or 0)
                self.by_id[place.id] = place
                self.cells.setdefault((math.floor(place.lat), math.floor(place.lon)), []).append(place)

    def nearest(self, lat, lon, max_km=MAX_KM):
        """Ближайший город не дальше max_km; из почти равноудалённых — крупнее."""
        if lat is None or lon is None:
            return None
        best, best_cost = None, None
        base = (math.floor(lat), math.floor(lon))
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                for place in self.cells.get((base[0] + dlat, base[1] + dlon), ()):
                    km = distance_km(lat, lon, place.lat, place.lon)
                    if km > max_km:
                        continue
                    # Пригород тянется к большому городу рядом, а не к посёлку в двух шагах.
                    cost = km / (1 + 0.15 * math.log10(max(place.population, 10)))
                    if best_cost is None or cost < best_cost:
                        best, best_cost = place, cost
        return best


_cache = {}


def _gazetteer():
    if 'value' not in _cache:
        _cache['value'] = Gazetteer()
    return _cache['value']


def nearest(lat, lon, max_km=MAX_KM):
    return _gazetteer().nearest(lat, lon, max_km)


def by_id(ident):
    return _gazetteer().by_id.get(int(ident))
