"""Треки лиц в видео: не «лицо на кадре 417», а «лицо с 00:13 до 00:25».

Раньше на весь ролик бралась горстка равномерно расставленных кадров — у
короткого домашнего видео между ними могли пройти секунды, и одно и то же
лицо попадало в каталог россыпью несвязанных карточек. Здесь детектор
запускается часто (по умолчанию дважды в секунду), а последовательные
находки связываются в треки: сначала по пересечению рамок с прошлым
кадром — самый надёжный сигнал на соседних кадрах, — а если лицо на миг
пропало из виду (моргнуло, отвернулось, камеру тряхнуло), трек всё равно
подхватывается по схожести эмбеддинга с уже накопленным. Трек закрывается,
если продолжить его не получилось дольше `gap` секунд.

Для каждого закрытого трека берутся `best` лучших кадров — их эмбеддинги
усредняются (устойчивее одного случайного кадра), а превью берётся с лучшего
момента. «Лучший» — уверенность детектора, помноженная на резкость лица
(face_quality.py): смазанный в движении кадр детектор часто видит уверенно,
но узнавать человека по нему плохо.
"""
import heapq

import numpy as np

import face_quality


def iou(first, second):
    """Пересечение рамок по площади — узнаём то же лицо на соседнем кадре."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    common = (right - left) * (bottom - top)
    area = ((first[2] - first[0]) * (first[3] - first[1])
            + (second[2] - second[0]) * (second[3] - second[1]) - common)
    return common / area if area > 0 else 0.0


def cosine(first, second):
    denom = (np.linalg.norm(first) * np.linalg.norm(second)) or 1.0
    return float(np.dot(first, second) / denom)


class _Track:
    """Один трек в процессе накопления: хранит только `cap` лучших кадров."""

    def __init__(self, moment, box, embedding, score, extra, cap=8):
        self.start = self.stop = moment
        self.last_box = box
        self.cap = cap
        self._top = []  # мин-куча: (score, счётчик, moment, box, embedding, extra)
        self._counter = 0
        self._push(moment, box, embedding, score, extra)

    def _push(self, moment, box, embedding, score, extra):
        # Счётчик — только чтобы куча не пыталась сравнивать эмбеддинги между
        # собой при равном score; сам он в выборе кадра не участвует.
        entry = (score, self._counter, moment, box, embedding, extra)
        self._counter += 1
        if len(self._top) < self.cap:
            heapq.heappush(self._top, entry)
        elif score > self._top[0][0]:
            heapq.heapreplace(self._top, entry)

    def anchor(self):
        """Средний эмбеддинг лучших кадров — по нему узнаём трек после потери."""
        vectors = np.stack([entry[4] for entry in self._top])
        mean = vectors.mean(axis=0)
        return mean / (np.linalg.norm(mean) or 1)

    def extend(self, moment, box, embedding, score, extra):
        self.stop = moment
        self.last_box = box
        self._push(moment, box, embedding, score, extra)

    def finalize(self):
        entries = sorted(self._top, key=lambda entry: -entry[0])
        vectors = np.stack([entry[4] for entry in entries])
        weights = np.array([entry[0] for entry in entries])
        embedding = np.average(vectors, axis=0, weights=weights)
        embedding = embedding / (np.linalg.norm(embedding) or 1)
        best = entries[0]
        return {
            'start': self.start, 'stop': self.stop, 'frame_time': best[2],
            'box': best[3], 'embedding': embedding.astype('<f4'), 'extra': best[5],
            'frames': len(entries),
        }


class Tracker:
    """Копит открытые треки по кадрам и отдаёт закрытые, как только можно."""

    def __init__(self, gap=1.2, iou_threshold=0.3, embedding_threshold=0.55, best=8):
        self.gap = gap
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.best = best
        self._open = []

    def update(self, moment, detections):
        """detections — список (box, embedding, score, extra). Отдаёт закрытые треки."""
        closed = [track.finalize() for track in self._open
                  if moment - track.stop > self.gap]
        available = [track for track in self._open if moment - track.stop <= self.gap]

        remaining = list(detections)
        matched_tracks, matched_detections = set(), set()

        # Сначала — по пересечению рамок с прошлым кадром трека, это надёжнее.
        pairs = sorted(
            ((iou(track.last_box, box), ti, di)
             for ti, track in enumerate(available)
             for di, (box, _, _, _) in enumerate(remaining)),
            reverse=True)
        for value, ti, di in pairs:
            if value < self.iou_threshold or ti in matched_tracks or di in matched_detections:
                continue
            matched_tracks.add(ti); matched_detections.add(di)
            box, embedding, score, extra = remaining[di]
            available[ti].extend(moment, box, embedding, score, extra)

        # Что не нашлось по рамке — пробуем по голосу эмбеддинга: лицо
        # пропадало из кадра на миг, но осталось тем же человеком.
        pairs = sorted(
            ((cosine(available[ti].anchor(), remaining[di][1]), ti, di)
             for ti in range(len(available)) if ti not in matched_tracks
             for di in range(len(remaining)) if di not in matched_detections),
            reverse=True)
        for value, ti, di in pairs:
            if (value < self.embedding_threshold or ti in matched_tracks
                    or di in matched_detections):
                continue
            matched_tracks.add(ti); matched_detections.add(di)
            box, embedding, score, extra = remaining[di]
            available[ti].extend(moment, box, embedding, score, extra)

        # Всё, что осталось без трека, — начало нового.
        for di, (box, embedding, score, extra) in enumerate(remaining):
            if di not in matched_detections:
                available.append(_Track(moment, box, embedding, score, extra, self.best))

        self._open = available
        return closed

    def close_all(self):
        closed = [track.finalize() for track in self._open]
        self._open = []
        return closed


def sample_frames(path, step_seconds, stop_seconds=0.0, stop_check=None):
    """Кадры видео через равные промежутки времени — быстро, без перемотки.

    Перемотка (`CAP_PROP_POS_FRAMES`) на каждый нужный кадр — на плотной
    выборке в разы медленнее декодирования подряд: на реальном ролике это
    3 кадра/с против 10-45. Поэтому читаем последовательно, а лишние кадры
    между нужными не декодируем полностью — только `grab()`, без цветового
    преобразования и копирования.
    """
    import video as video_media
    info = video_media.probe(path)
    fps = info['fps'] or 25.0
    step_frames = max(1, round(fps * step_seconds))
    last_wanted = int(fps * stop_seconds) if stop_seconds else info['frames']
    capture = video_media._open(path)
    index = 0
    try:
        if not capture.isOpened():
            raise OSError('не удалось открыть видео')
        while last_wanted <= 0 or index < last_wanted:
            if stop_check and stop_check():
                return
            if index % step_frames == 0:
                ok, frame = capture.read()
                if not ok:
                    return
                yield index / fps, frame
            elif not capture.grab():
                return
            index += 1
    finally:
        capture.release()


def find_tracks(path, models, step_seconds=0.5, gap_seconds=1.2, best_frames=8,
                stop_seconds=0.0, stop_check=None, min_score=0.6):
    """Треки лиц по всему ролику — детектор, трекер и сборка результатов вместе."""
    import video as video_media
    from insightface.app.common import Face
    tracker = Tracker(gap=gap_seconds, embedding_threshold=0.55, best=best_frames)
    tracks = []
    for moment, frame in sample_frames(path, step_seconds, stop_seconds, stop_check):
        detections = []
        boxes, landmarks = models['detection'].detect(frame)
        for number, box in enumerate(boxes):
            if landmarks is None or box[4] < min_score:
                continue
            face = Face(bbox=box[:4], kps=landmarks[number], det_score=box[4])
            models['recognition'].get(frame, face)
            embedding = np.asarray(face.normed_embedding, dtype='<f4')
            if not np.all(np.isfinite(embedding)):
                continue
            # Вырезаем и уменьшаем кадр сразу: трек держит до `best_frames`
            # штук в памяти, и полные кадры (особенно 4K) там неуместны —
            # только маленькое превью, которое всё равно пойдёт в файл.
            crop = video_media.to_image(frame).crop(tuple(int(v) for v in box[:4]))
            crop.thumbnail((160, 160))
            rank = float(box[4]) * face_quality.sharpness_weight(face_quality.face_blur(crop))
            detections.append((box[:4].tolist(), embedding, rank, crop))
        tracks.extend(tracker.update(moment, detections))
    tracks.extend(tracker.close_all())
    return tracks
