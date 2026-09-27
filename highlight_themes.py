"""Тематические подборки: котики, собаки, море, зоопарк, праздники…

Снимок относится к теме по визуальному индексу: его вектор сравнивается с
несколькими текстовыми описаниями темы (векторы описаний один раз на модель
считает `photo_curation.py prompts` в vision-venv и кладёт в curation_prompts
под именами `theme:<тема>:<n>`).

Абсолютный порог не годится: у SigLIP «сырая» похожесть на разные тексты
живёт на разных уровнях (у «скриншота» фон выше, чем у «кота» в лучшем кадре).
Поэтому порог относительный — z-оценка внутри кандидатов той же модели, плюс
отрыв от медианы. Снимок идёт в ту тему, где он выделяется сильнее всего.
Кандидаты — только годные снимки с камеры (оценка photo_curation), иначе в
«Котиков» набились бы мемы и скриншоты.
"""
import numpy as np

# (id, название, описания). Описания — на английском: SigLIP лучше понимает их.
THEMES = (
    ('cats', 'Котики', ('a photo of a cat', 'a cute kitten', 'a cat sleeping at home')),
    ('dogs', 'Собаки', ('a photo of a dog', 'a cute puppy', 'a dog playing outside')),
    ('zoo', 'Животные и зоопарк', ('animals at the zoo', 'a horse or a camel close up',
                                   'farm animals like sheep, goats and cows')),
    ('birds', 'Птицы', ('a bird close up', 'birds in the sky or on a branch')),
    ('sea', 'Море', ('the sea and a beach', 'people swimming in the sea', 'waves on the sea shore')),
    ('mountains', 'Горы', ('a mountain landscape', 'hiking in the mountains', 'a canyon or a waterfall')),
    ('nature', 'Природа', ('a forest landscape', 'a river or a lake in nature', 'a green meadow')),
    ('winter', 'Зима', ('a snowy winter landscape', 'children playing in the snow', 'a snowman')),
    ('flowers', 'Цветы', ('a bouquet of flowers', 'flowers blooming in a garden')),
    ('sunset', 'Закаты', ('a beautiful sunset sky', 'sunrise over the horizon')),
    ('newyear', 'Новый год', ('a decorated christmas tree with lights', 'new year celebration at home',
                              'new year fireworks')),
    ('birthday', 'Дни рождения', ('a birthday cake with candles', 'a birthday party with balloons')),
    ('wedding', 'Свадьбы', ('a wedding ceremony', 'a bride in a white wedding dress')),
    ('school', 'Школа', ('first day of school with flowers', 'children in school uniform')),
    ('food', 'Застолья', ('a festive table full of food', 'family dinner at a big table')),
    ('city', 'Прогулки по городу', ('a city street at night with lights', 'a famous landmark or monument',
                                    'a city square with people walking')),
)
# «Анти-темы»: не семейный снимок, а картинка из интернета. Похожее на них в
# темы не идёт — иначе «Море» и «Горы» набирают обои, а «Котики» — мемы.
NEGATIVE = (
    ('stock', ('a professional stock photo or a desktop wallpaper',
               'a postcard landscape picture downloaded from the internet')),
    ('art', ('digital art, an illustration or a painting', 'a meme picture with a caption',
             'a screenshot of a phone screen')),
)
TITLES = {theme: title for theme, title, _ in THEMES}
PREFIX = 'theme:'

PARAMS = {
    'theme_z': 2.6,          # насколько снимок выделяется в теме среди кандидатов той же модели
    'theme_margin': 0.018,   # и насколько похожесть выше медианной
    'theme_min_pool': 60,    # меньше кандидатов у модели — статистика ненадёжна, тем не даём
    'negative_z': 1.5,       # так сильно похоже на обои или рисунок — в темы не идёт
}


def _all_prompts():
    return [(theme, prompts) for theme, _, prompts in THEMES] + [(f'not-{name}', prompts) for name, prompts in NEGATIVE]


def prompt_rows():
    """[(имя в curation_prompts, текст)] для всех тем и анти-тем."""
    return [(f'{PREFIX}{theme}:{index}', text)
            for theme, prompts in _all_prompts() for index, text in enumerate(prompts)]


def theme_vectors(db, model):
    """{тема: матрица описаний} для модели; пусто, если описания не посчитаны или устарели."""
    rows = db.execute("SELECT name,prompt,embedding FROM curation_prompts WHERE model=? "
                      "AND name LIKE 'theme:%'", (model,)).fetchall()
    known = {name: (prompt, blob) for name, prompt, blob in rows}
    result = {}
    for theme, prompts in _all_prompts():
        vectors = []
        for index, text in enumerate(prompts):
            entry = known.get(f'{PREFIX}{theme}:{index}')
            if entry is None or entry[0] != text:
                vectors = []
                break
            vector = np.frombuffer(entry[1], dtype='<f4').astype(np.float32)
            vectors.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
        if vectors:
            result[theme] = np.stack(vectors)
    return result


def prompts_up_to_date(db, model):
    return len(theme_vectors(db, model)) == len(THEMES) + len(NEGATIVE)


def assign(items, vectors_by_model, params=None):
    """Проставляет item.theme (или None) и item.theme_z. Возвращает {тема: число}.

    items — кандидаты с нормированным item.vector и item.model.
    """
    params = {**PARAMS, **(params or {})}
    for item in items:
        item.theme, item.theme_z = None, 0.0
    counts = {}
    by_model = {}
    for item in items:
        if item.vector is not None and item.model in vectors_by_model:
            by_model.setdefault(item.model, []).append(item)
    for model, members in by_model.items():
        themes = vectors_by_model[model]
        if len(members) < params['theme_min_pool'] or not themes:
            continue
        matrix = np.stack([item.vector for item in members])
        negatives = [name for name in themes if name.startswith('not-')]
        names = [name for name in themes if not name.startswith('not-')]
        if not names:
            continue
        spoiled = np.zeros(len(members), dtype=bool)
        for name in negatives:
            values = (matrix @ themes[name].T).max(axis=1)
            spoiled |= (values - values.mean()) / max(float(values.std()), 1e-6) >= params['negative_z']
        # Похожесть на тему — лучшая из её описаний.
        similarity = np.stack([(matrix @ themes[name].T).max(axis=1) for name in names], axis=1)
        mean = similarity.mean(axis=0)
        std = np.maximum(similarity.std(axis=0), 1e-6)
        median = np.median(similarity, axis=0)
        z = (similarity - mean) / std
        for row, item in enumerate(members):
            if spoiled[row]:
                continue
            best = int(np.argmax(z[row]))
            if z[row, best] >= params['theme_z'] and \
                    similarity[row, best] - median[best] >= params['theme_margin']:
                item.theme, item.theme_z = names[best], float(z[row, best])
                counts[names[best]] = counts.get(names[best], 0) + 1
    return counts
