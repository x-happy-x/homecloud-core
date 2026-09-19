"""Local PySide6 UI for reviewing and naming automatically clustered faces."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

import albums
from authenticity_photos import ANIME_THRESHOLD
import face_quality
import video_identities
from prototype import cluster_embeddings, database, limited_linkage
import settings as catalog_settings


class CatalogStore:
    # Порог притяжения нового лица к готовой группе. Проверено на каталоге:
    # 30% лиц отложили, группы собрали из остальных, отложенные притянули —
    # при 0.60 притянулось 595 лиц из 1592 с двумя ошибками, при прежних 0.66
    # только 509 с одной.
    def __init__(self, folder, min_cluster_size=8, thread_safe=False, max_faces=0,
                 chunk=4000, assign_threshold=0.60):
        self.folder = Path(folder).resolve()
        self.db = database(self.folder, check_same_thread=not thread_safe)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS people (
              id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
              created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS face_people (
              face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
              person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS face_exclusions (
              face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
              created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS group_avatars (
              group_key TEXT PRIMARY KEY,
              face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS face_clusters (
              face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
              label INTEGER NOT NULL, probability REAL NOT NULL DEFAULT 0,
              method TEXT NOT NULL DEFAULT 'hdbscan', computed_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS face_clusters_label ON face_clusters(label);
            CREATE TABLE IF NOT EXISTS label_history (
              id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
              description TEXT NOT NULL, before_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS video_people_hints (
              path TEXT PRIMARY KEY REFERENCES photos(path) ON DELETE CASCADE,
              count INTEGER NOT NULL, created_at TEXT NOT NULL);
        ''')
        if 'bigfam_id' not in {row[1] for row in self.db.execute('PRAGMA table_info(people)')}:
            self.db.execute('ALTER TABLE people ADD COLUMN bigfam_id TEXT')
            self.db.commit()
        # Обычно колонку ставит опись (catalog_index.py), но исключение по папке
        # нужно и там, где каталог собран в обход неё — например, в тестах.
        if 'dir' not in {row[1] for row in self.db.execute('PRAGMA table_info(photos)')}:
            self.db.execute('ALTER TABLE photos ADD COLUMN dir TEXT')
            self.db.commit()
        # Лицо из ролика помнит свою секунду; в каталоге без видео колонки ещё нет.
        face_columns = {row[1] for row in self.db.execute('PRAGMA table_info(faces)')}
        if 'frame_time' not in face_columns:
            self.db.execute('ALTER TABLE faces ADD COLUMN frame_time REAL')
            self.db.commit()
        if 'track_start' not in face_columns:
            # Трек — промежуток времени, а не одна секунда; у фото и старых
            # видеозаписей, снятых ещё покадрово, эти колонки пустые.
            self.db.execute('ALTER TABLE faces ADD COLUMN track_start REAL')
            self.db.execute('ALTER TABLE faces ADD COLUMN track_stop REAL')
            self.db.commit()
        # Мультяшных и игровых персонажей, которых детектор принял за лица,
        # отдельный этап помечает сюда — группировка их обходит стороной.
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS face_authenticity (
              face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
              real_score REAL NOT NULL, anime_score REAL NOT NULL,
              model TEXT NOT NULL, analyzed_at TEXT NOT NULL
            )''')
        self.db.commit()
        face_quality.ensure_schema(self.db)
        self.min_cluster_size = min_cluster_size
        self.chunk = max(200, chunk)
        self.assign_threshold = assign_threshold
        # Векторы в памяти не держим: на большом каталоге это сотни мегабайт.
        self.rows = self.db.execute(
            'SELECT id,path,NULL,thumbnail,frame_time FROM faces ORDER BY id').fetchall()
        if max_faces and len(self.rows) > max_faces:
            raise ValueError(
                f'В каталоге {len(self.rows)} лиц, а разрешено {max_faces}. '
                'Поднимите потолок: backend.ps1 -MaxFaces <число>')
        self.by_id = {row[0]: row for row in self.rows}
        if self.rows:
            self.auto_labels, self.auto_confidence = self._cluster()
        else:
            self.auto_labels, self.auto_confidence = {}, {}

    def reload_faces(self):
        """Refresh in-memory face rows after a scan or photo deletion."""
        self.rows = self.db.execute(
            'SELECT id,path,NULL,thumbnail,frame_time FROM faces ORDER BY id').fetchall()
        self.by_id = {row[0]: row for row in self.rows}
        if self.rows:
            self.auto_labels, self.auto_confidence = self._cluster()
        else:
            self.auto_labels, self.auto_confidence = {}, {}

    def _vectors(self, face_ids):
        """Векторы читаем порциями и только когда они действительно нужны."""
        found = {}
        for offset in range(0, len(face_ids), 900):
            batch = face_ids[offset:offset + 900]
            places = ','.join('?' * len(batch))
            for face_id, blob in self.db.execute(
                    f'SELECT id,embedding FROM faces WHERE id IN ({places})', batch):
                vector = np.frombuffer(blob, dtype='<f4').astype('<f4')
                found[face_id] = vector / max(float(np.linalg.norm(vector)), 1e-12)
        return np.stack([found[face_id] for face_id in face_ids if face_id in found]), \
            [face_id for face_id in face_ids if face_id in found]

    def vectors(self, face_ids):
        """Нормированные векторы лиц: (матрица, порядок id)."""
        return self._vectors(face_ids)

    def _centroids(self):
        """Средний вектор каждой группы — к нему притягиваются новые лица."""
        sums, counts = {}, {}
        blurry = self.blurry()
        for face_id, label, blob in self.db.execute(
                'SELECT faces.id,face_clusters.label,faces.embedding FROM face_clusters '
                'JOIN faces ON faces.id=face_clusters.face_id WHERE face_clusters.label>=0'):
            if face_id in blurry:
                continue
            vector = np.frombuffer(blob, dtype='<f4').astype('<f4')
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
            if label in sums:
                sums[label] += vector
                counts[label] += 1
            else:
                sums[label] = vector.copy()
                counts[label] = 1
        return sums, counts

    def _store_labels(self, assigned):
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.executemany(
                'INSERT INTO face_clusters(face_id,label,probability,method,computed_at) '
                'VALUES(?,?,?,?,?) ON CONFLICT(face_id) DO UPDATE SET label=excluded.label,'
                'probability=excluded.probability,method=excluded.method,'
                'computed_at=excluded.computed_at',
                [(face_id, int(label), float(probability), method, now)
                 for face_id, (label, probability, method) in assigned.items()])

    def quality_options(self):
        """Порог мыла и минимальный размер лица — из настроек каталога."""
        # Не через settings.read: тот на каждом вызове создаёт схему и делает
        # commit, а здесь — каждый опрос состояния, в том числе посреди работы.
        values = {'face_blur_threshold': face_quality.BLUR_THRESHOLD,
                  'face_min_size': face_quality.MIN_SIZE}
        try:
            stored = self.db.execute(
                "SELECT key,value FROM settings WHERE key IN "
                "('face_blur_threshold','face_min_size')").fetchall()
        except sqlite3.OperationalError:
            stored = []
        for key, raw in stored:
            try:
                values[key] = catalog_settings._cast(key, json.loads(raw))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        return float(values['face_blur_threshold']), float(values['face_min_size'])

    def measure_quality(self):
        """Оценить резкость лиц, у которых её ещё нет. Только по миниатюрам."""
        return face_quality.measure(self.db, self.folder)

    def blurry(self):
        """Мыльные лица: в группировку не идут и в карточках не показываются."""
        threshold, min_size = self.quality_options()
        return face_quality.blurry_ids(self.db, threshold, min_size)

    def ensure_labels(self):
        """Считаем метки только для новых лиц: старые уже лежат в каталоге."""
        # Оценка резкости досчитывается и без новых меток: у каталога, собранного
        # до её появления, метки есть у всех лиц, а оценок нет ни у одного.
        if self.db.execute('SELECT 1 FROM face_track_data WHERE active=1 LIMIT 1').fetchone():
            # The scanner / ReclusterController owns the CPU work and its connection.
            if not video_identities.needs_rebuild(self.db) and self.db.execute('SELECT 1 FROM faces f LEFT JOIN face_clusters c ON c.face_id=f.id WHERE c.face_id IS NULL LIMIT 1').fetchone():
                with self.db:
                    video_identities.mark_dirty(self.db)
            return 0
        self.measure_quality()
        pending = [row[0] for row in self.db.execute(
            'SELECT faces.id FROM faces LEFT JOIN face_clusters '
            'ON face_clusters.face_id=faces.id WHERE face_clusters.face_id IS NULL '
            'ORDER BY faces.id')]
        if not pending:
            return 0
        video_pending = {r[0] for r in self.db.execute(
            'SELECT f.id FROM faces f WHERE f.track_start IS NOT NULL OR f.frame_time IS NOT NULL')}
        pending = [i for i in pending if i not in video_pending]
        if not pending:
            return 0
        blurry = self.blurry()
        sums, counts = self._centroids()
        highest = max(sums) if sums else -1
        done = 0
        for offset in range(0, len(pending), self.chunk):
            batch = pending[offset:offset + self.chunk]
            # Мыльное лицо тянет группы друг к другу — ему метка «не в группе».
            assigned = {face_id: (-1, 0.0, 'blurry') for face_id in batch if face_id in blurry}
            vectors, batch = self._vectors([face_id for face_id in batch
                                            if face_id not in blurry])
            if not batch:
                self._store_labels(assigned)
                done += len(assigned)
                continue
            leftovers = list(range(len(batch)))
            if sums:
                labels = sorted(sums)
                centroids = np.stack([sums[label] / max(float(np.linalg.norm(sums[label])), 1e-12)
                                      for label in labels])
                scores = vectors @ centroids.T
                leftovers = []
                for index, face_id in enumerate(batch):
                    best = int(scores[index].argmax())
                    if float(scores[index][best]) >= self.assign_threshold:
                        label = labels[best]
                        assigned[face_id] = (label, float(scores[index][best]), 'nearest')
                        sums[label] += vectors[index]
                        counts[label] += 1
                    else:
                        leftovers.append(index)
            if len(leftovers) >= self.min_cluster_size:
                fresh, probabilities = cluster_embeddings(
                    vectors[leftovers], algorithm='average',
                    min_cluster_size=self.min_cluster_size)
                for position, index in enumerate(leftovers):
                    label = int(fresh[position])
                    if label < 0:
                        assigned[batch[index]] = (-1, 0.0, 'average')
                        continue
                    label += highest + 1
                    assigned[batch[index]] = (label, float(probabilities[position]), 'average')
                    if label in sums:
                        sums[label] += vectors[index]
                        counts[label] += 1
                    else:
                        sums[label] = vectors[index].copy()
                        counts[label] = 1
                highest = max(sums) if sums else highest
            else:
                for index in leftovers:
                    assigned[batch[index]] = (-1, 0.0, 'average')
            self._store_labels(assigned)
            done += len(assigned)
        return done

    def refresh_labels(self):
        """Досчитать метки для лиц, появившихся после последнего расчёта."""
        if not self.ensure_labels():
            return False
        self.auto_labels, self.auto_confidence = self._cluster()
        return True

    def _cluster(self):
        self.ensure_labels()
        labels = {row[0]: -1 for row in self.rows}
        confidence = {row[0]: 0.0 for row in self.rows}
        for face_id, label, probability in self.db.execute(
                'SELECT face_id,label,probability FROM face_clusters'):
            if face_id in labels:
                labels[face_id] = int(label)
                confidence[face_id] = float(probability)
        return labels, confidence

    def groups(self):
        excluded = {row[0] for row in self.db.execute('SELECT face_id FROM face_exclusions')}
        # Мыло не показывается нигде, кроме своей группы на проверке: даже у
        # названного человека такое лицо — каша в карточке, а не портрет.
        # Имя у лица при этом остаётся, и поиск по человеку снимок находит.
        blurry = self.blurry() - excluded
        assigned = defaultdict(list)
        for row in self.db.execute(
                'SELECT face_people.face_id,people.id,people.name,people.bigfam_id FROM face_people '
                'JOIN people ON people.id=face_people.person_id'):
            if row[0] not in blurry:
                assigned[(row[1], row[2], row[3])].append(row[0])
        # Мультяшный или игровой персонаж, которого детектор принял за лицо —
        # не человек, группировать не о чем. Имя, если уже назначено кем-то
        # вручную, важнее любой автоматической догадки — до сюда не доходит.
        anime = {row[0] for row in self.db.execute(
            'SELECT face_id FROM face_authenticity WHERE anime_score>=?', (ANIME_THRESHOLD,))}
        assigned_ids = {face_id for members in assigned.values() for face_id in members}
        automatic = defaultdict(list)
        for face_id, label in self.auto_labels.items():
            if (face_id not in assigned_ids and face_id not in excluded and face_id not in anime
                    and face_id not in blurry):
                automatic[label].append(face_id)

        result = [
            {'key': f'person:{person_id}', 'title': name, 'name': name,
             'bigfam_id': bigfam_id, 'face_ids': members, 'kind': 'person'}
            for (person_id, name, bigfam_id), members in assigned.items()
        ]
        result.sort(key=lambda group: group['name'].casefold())
        for group in result:
            group['face_ids'].sort(key=lambda face_id: self.auto_confidence.get(face_id, 0),
                                   reverse=True)
        auto_groups = [
            {'key': f'auto:{label}', 'title': f'Группа {label + 1}', 'name': '',
             'face_ids': members, 'kind': 'auto'}
            for label, members in automatic.items() if label != -1
        ]
        auto_groups.sort(key=lambda group: -len(group['face_ids']))
        for group in auto_groups:
            group['face_ids'].sort(key=lambda face_id: self.auto_confidence.get(face_id, 0),
                                   reverse=True)
        result.extend(auto_groups)
        if automatic.get(-1):
            result.append({'key': 'noise', 'title': 'Не сгруппированы', 'name': '',
                           'face_ids': automatic[-1], 'kind': 'noise'})
        shown_blurry = sorted(face_id for face_id in blurry if face_id in self.by_id)
        if shown_blurry:
            result.append({'key': 'blurry', 'title': 'Размытые лица', 'name': '',
                           'face_ids': shown_blurry, 'kind': 'blurry'})
        if excluded:
            result.append({'key': 'excluded', 'title': 'Исключённые вручную', 'name': '',
                           'face_ids': sorted(excluded), 'kind': 'excluded'})
        pinned = dict(self.db.execute('SELECT group_key,face_id FROM group_avatars'))
        quality = {face_id: (blur, size) for face_id, blur, size in self.db.execute(
            'SELECT face_id,blur,size FROM face_quality')}
        for group in result:
            chosen = pinned.get(group['key'])
            # выбранный кадр мог уехать в другую группу — тогда снова лучший портрет
            group['avatar_pinned'] = chosen in group['face_ids']
            group['avatar_face'] = (chosen if group['avatar_pinned']
                                    else self._portrait(group['face_ids'], quality))
        return result

    def _portrait(self, face_ids, quality, candidates=30):
        """Лицо для аватарки: резкое и крупное из самых уверенных в группе.

        Раньше бралось просто первое по уверенности — а это нередко крошечное
        лицо с заднего плана, которое в кружке на 400 точек превращалось в кашу.
        """
        best, best_score = None, -1.0
        for face_id in face_ids[:candidates]:
            blur, size = quality.get(face_id, (None, None))
            sharp = 1.0 - (blur if blur is not None else 0.5)
            scale = min(size or 60.0, 160.0) / 160.0
            score = sharp * (0.4 + 0.6 * scale) * (0.5 + 0.5 * self.auto_confidence.get(face_id, 0.0))
            if score > best_score:
                best, best_score = face_id, score
        return best

    def suggest_people(self, threshold=0.6):
        """Кого напоминает каждая автоматическая группа.

        Средний вектор группы сравнивается не со средним вектором человека, а
        с каждым уже названным лицом по отдельности: у человека, снятого в
        разном возрасте, свете и ракурсе, среднее смазывается, а отдельные
        кадры — нет. На реальном каталоге (27 человек, 306 посторонних групп)
        такой способ верно узнал 25 групп из 27 против 22 при голосовании по
        каждому лицу группы, и ни разу не ошибся человеком.

        Ничего не записывает: это подсказка, а решение остаётся за человеком.
        """
        named = self._named_bank()
        if not named:
            return []
        titles = {row[0]: (row[1], row[2]) for row in self.db.execute(
            'SELECT id,name,bigfam_id FROM people')}

        bank, bank_ids = self._vectors(sorted(named))
        if not bank_ids:
            return []
        owners = np.array([named[face_id] for face_id in bank_ids])

        groups = [group for group in self.groups() if group['kind'] == 'auto']
        wanted = [face_id for group in groups for face_id in group['face_ids']]
        if not wanted:
            return []
        # Один заход в базу на все группы: по отдельности это сотни запросов.
        matrix, order = self._vectors(wanted)
        place = {face_id: index for index, face_id in enumerate(order)}

        found = []
        for group in groups:
            rows = [place[face_id] for face_id in group['face_ids'] if face_id in place]
            if not rows:
                continue
            centroid = matrix[rows].mean(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
            scores = bank @ centroid
            best = int(scores.argmax())
            score = float(scores[best])
            if score < threshold:
                continue
            person_id = int(owners[best])
            name, bigfam_id = titles.get(person_id, ('', None))
            found.append({'key': group['key'], 'person_id': person_id, 'name': name,
                          'bigfam_id': bigfam_id, 'score': round(score, 3),
                          'faces': len(rows)})
        found.sort(key=lambda item: -item['score'])
        return found

    def _named_bank(self):
        """Названные лица, по которым узнаются остальные: без мыла."""
        blurry = self.blurry()
        return {face_id: person_id for face_id, person_id in self.db.execute(
            'SELECT face_id,person_id FROM face_people') if face_id not in blurry}

    def person_candidates(self, person_id, threshold=0.5, margin=0.03, limit=120, top=3):
        """Безымянные лица, похожие на этого человека, — по одному, не группой.

        Группа целиком узнаётся подсказкой `suggest_people`, но почти половина
        лиц ни в какую группу не попадает, а поиск по человеку находит только
        названные. Здесь каждое безымянное лицо сравнивается с `top` самыми
        похожими подписанными лицами каждого человека: среднее по нескольким
        устойчивее одного случайного совпадения. Лицо предлагается, только
        если этот человек у него лучший и опережает второго на `margin`.

        Проверка на каталоге (30% подписанных лиц спрятаны как «безымянные»):
        при пороге 0.5 узнано 221 из 279, ошибок две.
        """
        import numpy as np
        named = self._named_bank()
        mine = sorted(face_id for face_id, owner in named.items() if owner == person_id)
        if not mine:
            return []
        groups = self.groups()
        wanted = [face_id for group in groups if group['kind'] in {'auto', 'noise'}
                  for face_id in group['face_ids']]
        if not wanted:
            return []
        bank, bank_ids = self._vectors(sorted(named))
        owners = np.array([named[face_id] for face_id in bank_ids])
        matrix, order = self._vectors(wanted)
        scores = matrix @ bank.T
        people = sorted(set(owners.tolist()))
        per_person = np.full((len(order), len(people)), -1.0)
        for column, owner in enumerate(people):
            block = scores[:, owners == owner]
            depth = min(top, block.shape[1])
            per_person[:, column] = np.sort(block, axis=1)[:, -depth:].mean(axis=1)
        target = people.index(person_id)
        mine_scores = per_person[:, target]
        others = np.delete(per_person, target, axis=1)
        rival = others.max(axis=1) if others.shape[1] else np.full(len(order), -1.0)
        group_of = {face_id: group['key'] for group in groups
                    if group['kind'] in {'auto', 'noise'} for face_id in group['face_ids']}
        found = [(float(mine_scores[index]), face_id) for index, face_id in enumerate(order)
                 if mine_scores[index] >= threshold and mine_scores[index] - rival[index] >= margin]
        found.sort(reverse=True)
        return [{'face_id': face_id, 'score': round(score, 3), 'group': group_of.get(face_id, '')}
                for score, face_id in found[:limit]]

    def set_avatar(self, group_key, face_id):
        """Закрепить кадр как аватарку группы."""
        face_id = int(face_id)
        group = next((item for item in self.groups() if item['key'] == group_key), None)
        if group is None:
            raise KeyError('Группа больше не существует')
        if face_id not in group['face_ids']:
            raise ValueError('Это лицо не из выбранной группы')
        with self.db:
            self.db.execute(
                'INSERT INTO group_avatars(group_key,face_id) VALUES(?,?) '
                'ON CONFLICT(group_key) DO UPDATE SET face_id=excluded.face_id',
                (group_key, face_id))
        return face_id

    def clear_avatar(self, group_key):
        with self.db:
            self.db.execute('DELETE FROM group_avatars WHERE group_key=?', (group_key,))

    def _snapshot(self, face_ids, description):
        face_ids = sorted(set(face_ids))
        people = dict(self.db.execute(
            f'SELECT face_id,person_id FROM face_people WHERE face_id IN ({",".join("?" * len(face_ids))})',
            face_ids)) if face_ids else {}
        excluded = {row[0] for row in self.db.execute(
            f'SELECT face_id FROM face_exclusions WHERE face_id IN ({",".join("?" * len(face_ids))})',
            face_ids)} if face_ids else set()
        origins = {r[0]: r[1:] for r in self.db.execute('SELECT face_id,source,score,version FROM face_people')}
        state = [{'face_id': face_id, 'origin': origins.get(face_id), 'person_id': people.get(face_id),
                  'excluded': face_id in excluded} for face_id in face_ids]
        video_identities.mark_dirty(self.db)
        self.db.execute('INSERT INTO label_history(created_at,description,before_json) VALUES(?,?,?)',
                        (datetime.now(timezone.utc).isoformat(), description,
                         json.dumps(state, separators=(',', ':'))))

    def find_or_create_person(self, name, bigfam_id=None):
        """Человек по имени: та же запись, если уже есть, иначе новая."""
        name = name.strip()
        bigfam_id = (bigfam_id or '').strip() or None
        if not name:
            raise ValueError('Введите имя человека')
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute('INSERT INTO people(name,created_at) VALUES(?,?) ON CONFLICT(name) DO NOTHING',
                        (name, now))
        person_id = self.db.execute('SELECT id FROM people WHERE name=?', (name,)).fetchone()[0]
        if bigfam_id:
            # Один человек картотеки — одна запись каталога.
            self.db.execute('UPDATE people SET bigfam_id=NULL WHERE bigfam_id=? AND id!=?',
                            (bigfam_id, person_id))
            self.db.execute('UPDATE people SET bigfam_id=? WHERE id=?', (bigfam_id, person_id))
        return person_id

    def assign(self, face_ids, name, bigfam_id=None):
        face_ids = sorted(set(face_ids))
        if not face_ids:
            raise ValueError('Выберите группу или лица')
        with self.db:
            self._snapshot(face_ids, f'Назначено имя «{name.strip()}»')
            person_id = self.find_or_create_person(name, bigfam_id)
            self.db.executemany(
                'INSERT INTO face_people(face_id,person_id) VALUES(?,?) '
                "ON CONFLICT(face_id) DO UPDATE SET person_id=excluded.person_id,source='human',score=NULL,version=NULL",
                [(face_id, person_id) for face_id in face_ids])
            self.db.executemany('DELETE FROM face_exclusions WHERE face_id=?',
                                [(face_id,) for face_id in face_ids])
            # Мыльное лицо назвали — значит, человеку оно нужно; больше не прячем.
            face_quality.keep(self.db, face_ids)

    def exclude(self, face_ids):
        face_ids = sorted(set(face_ids))
        if not face_ids:
            raise ValueError('Выберите ошибочно сгруппированные лица')
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self._snapshot(face_ids, 'Лица исключены из автоматических групп')
            self.db.executemany('DELETE FROM face_people WHERE face_id=?',
                                [(face_id,) for face_id in face_ids])
            self.db.executemany(
                'INSERT INTO face_exclusions(face_id,created_at) VALUES(?,?) '
                'ON CONFLICT(face_id) DO UPDATE SET created_at=excluded.created_at',
                [(face_id, now) for face_id in face_ids])

    def exclude_path(self, path, folder=False):
        """Исключить разом все лица одного файла или всей папки (с вложенными)."""
        if folder:
            clause, params = albums.folder_clause(path)
            rows = self.db.execute(
                f'SELECT faces.id FROM faces JOIN photos ON photos.path=faces.path '
                f'WHERE 1=1{clause}', params).fetchall()
        else:
            rows = self.db.execute('SELECT id FROM faces WHERE path=?', (path,)).fetchall()
        face_ids = [row[0] for row in rows]
        if face_ids:
            self.exclude(face_ids)
        return len(face_ids)

    def video_people_hint(self, path):
        row = self.db.execute(
            'SELECT count FROM video_people_hints WHERE path=?', (path,)).fetchone()
        return row[0] if row else None

    def set_video_people_hint(self, path, count):
        """Сколько людей на самом деле в ролике — чтобы не плодить лишние грозди.

        Считаем заново только неназванные лица этого файла: названные и так
        закреплены за человеком, а мыльные и исключённые в группировку не идут.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            if count and count > 0:
                self.db.execute(
                    'INSERT INTO video_people_hints(path,count,created_at) VALUES(?,?,?) '
                    'ON CONFLICT(path) DO UPDATE SET count=excluded.count,'
                    'created_at=excluded.created_at', (path, int(count), now))
            else:
                self.db.execute('DELETE FROM video_people_hints WHERE path=?', (path,))
        with self.db:
            video_identities.mark_dirty(self.db)
        if count and count > 0:
            self.recluster_video(path, int(count))

    def recluster_video(self, path, count):
        if self.db.execute('SELECT 1 FROM faces WHERE path=? AND (track_start IS NOT NULL OR frame_time IS NOT NULL) LIMIT 1', (path,)).fetchone():
            # Hint is advisory. The worker enforces similarity and cannot-link.
            return 0
        blurry = self.blurry()
        rows = self.db.execute(
            'SELECT faces.id FROM faces '
            'LEFT JOIN face_people ON face_people.face_id=faces.id '
            'LEFT JOIN face_exclusions ON face_exclusions.face_id=faces.id '
            'WHERE faces.path=? AND face_people.face_id IS NULL '
            'AND face_exclusions.face_id IS NULL', (path,)).fetchall()
        face_ids = [row[0] for row in rows if row[0] not in blurry]
        if len(face_ids) < 2:
            return 0
        vectors, face_ids = self._vectors(face_ids)
        labels, probabilities = limited_linkage(vectors, count)
        highest = self.db.execute(
            'SELECT COALESCE(MAX(label),-1) FROM face_clusters').fetchone()[0]
        assigned = {face_id: (highest + 1 + int(label), float(probability), 'video-hint')
                    for face_id, label, probability in zip(face_ids, labels, probabilities)}
        self._store_labels(assigned)
        return len(assigned)

    def undo(self):
        row = self.db.execute(
            'SELECT id,description,before_json FROM label_history ORDER BY id DESC LIMIT 1').fetchone()
        if not row:
            return None
        states = json.loads(row[2])
        with self.db:
            video_identities.mark_dirty(self.db)
            for state in states:
                face_id = state['face_id']
                self.db.execute('DELETE FROM face_people WHERE face_id=?', (face_id,))
                self.db.execute('DELETE FROM face_exclusions WHERE face_id=?', (face_id,))
                if state['person_id'] is not None:
                    origin = state.get('origin') or ['human', None, None]
                    self.db.execute('INSERT INTO face_people(face_id,person_id,source,score,version) VALUES(?,?,?,?,?)',
                                    (face_id, state['person_id'], *origin))
                if state['excluded']:
                    self.db.execute('INSERT INTO face_exclusions(face_id,created_at) VALUES(?,?)',
                                    (face_id, datetime.now(timezone.utc).isoformat()))
            self.db.execute('DELETE FROM label_history WHERE id=?', (row[0],))
            self.db.execute('DELETE FROM people WHERE id NOT IN (SELECT person_id FROM face_people)')
        return row[1]


class MainWindow(QMainWindow):
    def __init__(self, store):
        super().__init__()
        self.store = store
        self.setWindowTitle('Локальная фототека — люди')
        self.resize(1200, 760)

        root = QWidget()
        layout = QVBoxLayout(root)
        help_text = QLabel(
            'Выберите одну или несколько групп слева, введите имя и назначьте его. '
            'Одинаковое имя объединяет группы. Ошибочные лица можно выделить справа и исключить.')
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        controls = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText('Имя человека, например Мама')
        self.assign_groups_button = QPushButton('Назначить имя выбранным группам')
        self.assign_faces_button = QPushButton('Назначить имя выбранным лицам')
        self.exclude_button = QPushButton('Это другой человек / исключить')
        self.undo_button = QPushButton('Отменить последнее действие')
        controls.addWidget(self.name_edit, 1)
        for button in (self.assign_groups_button, self.assign_faces_button,
                       self.exclude_button, self.undo_button):
            controls.addWidget(button)
        layout.addLayout(controls)

        splitter = QSplitter()
        self.group_list = QListWidget()
        self.group_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.face_list = QListWidget()
        self.face_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.face_list.setIconSize(QSize(144, 144))
        self.face_list.setGridSize(QSize(180, 190))
        self.face_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.face_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        splitter.addWidget(self.group_list)
        splitter.addWidget(self.face_list)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter, 1)
        self.status = QLabel()
        layout.addWidget(self.status)
        self.setCentralWidget(root)

        self.group_list.currentItemChanged.connect(self.show_group)
        self.face_list.itemDoubleClicked.connect(self.open_original)
        self.assign_groups_button.clicked.connect(self.assign_groups)
        self.assign_faces_button.clicked.connect(self.assign_faces)
        self.exclude_button.clicked.connect(self.exclude_faces)
        self.undo_button.clicked.connect(self.undo)
        self.refresh()

    def refresh(self):
        self.group_list.clear()
        groups = self.store.groups()
        for group in groups:
            item = QListWidgetItem(f"{group['title']} — {len(group['face_ids'])} лиц")
            item.setData(Qt.ItemDataRole.UserRole, group)
            self.group_list.addItem(item)
        named = sum(group['kind'] == 'person' for group in groups)
        self.status.setText(f'Лиц: {len(self.store.rows)} · Именованных людей: {named} · Разделов: {len(groups)}')
        if self.group_list.count():
            self.group_list.setCurrentRow(0)

    def show_group(self, current, previous=None):
        self.face_list.clear()
        if current is None:
            return
        group = current.data(Qt.ItemDataRole.UserRole)
        if group['name']:
            self.name_edit.setText(group['name'])
        for face_id in group['face_ids']:
            row = self.store.by_id[face_id]
            item = QListWidgetItem(QIcon(str(self.store.folder / row[3])), Path(row[1]).name)
            item.setToolTip(row[1])
            item.setData(Qt.ItemDataRole.UserRole, face_id)
            self.face_list.addItem(item)

    def selected_group_faces(self):
        return [face_id for item in self.group_list.selectedItems()
                for face_id in item.data(Qt.ItemDataRole.UserRole)['face_ids']]

    def selected_face_ids(self):
        return [item.data(Qt.ItemDataRole.UserRole) for item in self.face_list.selectedItems()]

    def apply(self, action):
        try:
            action()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, 'Не выполнено', str(exc))

    def assign_groups(self):
        self.apply(lambda: self.store.assign(self.selected_group_faces(), self.name_edit.text()))

    def assign_faces(self):
        self.apply(lambda: self.store.assign(self.selected_face_ids(), self.name_edit.text()))

    def exclude_faces(self):
        self.apply(lambda: self.store.exclude(self.selected_face_ids()))

    def undo(self):
        description = self.store.undo()
        if description is None:
            QMessageBox.information(self, 'Отмена', 'Нет действий для отмены')
        self.refresh()

    def open_original(self, item):
        face_id = item.data(Qt.ItemDataRole.UserRole)
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.store.by_id[face_id][1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--min-cluster-size', type=int, default=8)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    try:
        store = CatalogStore(args.data, args.min_cluster_size)
        window = MainWindow(store)
    except Exception as exc:
        QMessageBox.critical(None, 'Ошибка каталога', str(exc))
        return 1
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
