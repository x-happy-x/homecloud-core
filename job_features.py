"""Возможности задания двумя наборами: для снимков и для роликов.

Прежде набор был один и шёл на всё найденное сразу: включил описания — их
считали и для видео, хотя ролик стоит в разы дороже снимка. Теперь у каждой
фазы свой вид файлов, а задание получает объединение наборов плюс карту
«фаза → вид», по которой device_job.py передаёт скриптам --kinds.
"""

FEATURES = ('inventory', 'faces', 'visual', 'ocr', 'caption', 'adult', 'speech', 'diarize',
            'authenticity')

# Речь и разделение голосов существуют только для видео.
VIDEO_ONLY = ('speech', 'diarize')

# Опись обходит диск целиком и вида файлов не различает.
KIND_FREE = ('inventory',)


def _chosen(features, supported):
    """Набор одного вида с учётом зависимостей этапов."""
    chosen = {name: bool((features or {}).get(name)) for name in FEATURES}
    # OCR и описания строятся поверх визуального индекса, а описанию нужны ещё
    # рейтинг и уверенные теги из анализа 18+.
    if chosen['caption']:
        chosen['visual'] = True
        chosen['adult'] = bool(supported.get('adult'))
    if chosen['ocr']:
        chosen['visual'] = True
    return chosen


def resolve(features, video_features, supported):
    """Два набора → (объединение для задания, карта «фаза → вид файлов»).

    video_features=None — вызов с одним набором, как у прежних версий
    интерфейса: он идёт и на снимки, и на ролики.
    """
    photos = _chosen(features, supported)
    videos = _chosen(features if video_features is None else video_features, supported)
    for name in VIDEO_ONLY:
        photos[name] = False
    for name in KIND_FREE:
        photos[name] = videos[name] = photos[name] or videos[name]
    selected = {name: photos[name] or videos[name] for name in FEATURES}
    kinds = {}
    for name in FEATURES:
        if not selected[name] or name in KIND_FREE:
            continue
        kinds[name] = 'all' if photos[name] and videos[name] else (
            'photos' if photos[name] else 'videos')
    return selected, kinds
