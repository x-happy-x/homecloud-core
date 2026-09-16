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

from authenticity_photos import ANIME_THRESHOLD
from prototype import cluster_embeddings, database


class CatalogStore:
    def __init__(self, folder, min_cluster_size=8, thread_safe=False, max_faces=0,
                 chunk=4000, assign_threshold=0.66):
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
        ''')
        if 'bigfam_id' not in {row[1] for row in self.db.execute('PRAGMA table_info(people)')}:
            self.db.execute('ALTER TABLE people ADD COLUMN bigfam_id TEXT')
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
        for label, blob in self.db.execute(
                'SELECT face_clusters.label,faces.embedding FROM face_clusters '
                'JOIN faces ON faces.id=face_clusters.face_id WHERE face_clusters.label>=0'):
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

    def ensure_labels(self):
        """Считаем метки только для новых лиц: старые уже лежат в каталоге."""
        pending = [row[0] for row in self.db.execute(
            'SELECT faces.id FROM faces LEFT JOIN face_clusters '
            'ON face_clusters.face_id=faces.id WHERE face_clusters.face_id IS NULL '
            'ORDER BY faces.id')]
        if not pending:
            return 0
        sums, counts = self._centroids()
        highest = max(sums) if sums else -1
        done = 0
        for offset in range(0, len(pending), self.chunk):
            batch = pending[offset:offset + self.chunk]
            vectors, batch = self._vectors(batch)
            if not batch:
                continue
            assigned = {}
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
                    vectors[leftovers], algorithm='hdbscan',
                    min_cluster_size=self.min_cluster_size)
                for position, index in enumerate(leftovers):
                    label = int(fresh[position])
                    if label < 0:
                        assigned[batch[index]] = (-1, 0.0, 'hdbscan')
                        continue
                    label += highest + 1
                    assigned[batch[index]] = (label, float(probabilities[position]), 'hdbscan')
                    if label in sums:
                        sums[label] += vectors[index]
                        counts[label] += 1
                    else:
                        sums[label] = vectors[index].copy()
                        counts[label] = 1
                highest = max(sums) if sums else highest
            else:
                for index in leftovers:
                    assigned[batch[index]] = (-1, 0.0, 'hdbscan')
            self._store_labels(assigned)
            done += len(batch)
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
        assigned = defaultdict(list)
        for row in self.db.execute(
                'SELECT face_people.face_id,people.id,people.name,people.bigfam_id FROM face_people '
                'JOIN people ON people.id=face_people.person_id'):
            assigned[(row[1], row[2], row[3])].append(row[0])
        excluded = {row[0] for row in self.db.execute('SELECT face_id FROM face_exclusions')}
        # Мультяшный или игровой персонаж, которого детектор принял за лицо —
        # не человек, группировать не о чем. Имя, если уже назначено кем-то
        # вручную, важнее любой автоматической догадки — до сюда не доходит.
        anime = {row[0] for row in self.db.execute(
            'SELECT face_id FROM face_authenticity WHERE anime_score>=?', (ANIME_THRESHOLD,))}
        assigned_ids = {face_id for members in assigned.values() for face_id in members}
        automatic = defaultdict(list)
        for face_id, label in self.auto_labels.items():
            if face_id not in assigned_ids and face_id not in excluded and face_id not in anime:
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
        if excluded:
            result.append({'key': 'excluded', 'title': 'Исключённые вручную', 'name': '',
                           'face_ids': sorted(excluded), 'kind': 'excluded'})
        pinned = dict(self.db.execute('SELECT group_key,face_id FROM group_avatars'))
        for group in result:
            chosen = pinned.get(group['key'])
            # выбранный кадр мог уехать в другую группу — тогда снова первое лицо
            group['avatar_pinned'] = chosen in group['face_ids']
            group['avatar_face'] = (chosen if group['avatar_pinned']
                                    else (group['face_ids'][0] if group['face_ids'] else None))
        return result

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
        state = [{'face_id': face_id, 'person_id': people.get(face_id),
                  'excluded': face_id in excluded} for face_id in face_ids]
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
                'ON CONFLICT(face_id) DO UPDATE SET person_id=excluded.person_id',
                [(face_id, person_id) for face_id in face_ids])
            self.db.executemany('DELETE FROM face_exclusions WHERE face_id=?',
                                [(face_id,) for face_id in face_ids])

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

    def undo(self):
        row = self.db.execute(
            'SELECT id,description,before_json FROM label_history ORDER BY id DESC LIMIT 1').fetchone()
        if not row:
            return None
        states = json.loads(row[2])
        with self.db:
            for state in states:
                face_id = state['face_id']
                self.db.execute('DELETE FROM face_people WHERE face_id=?', (face_id,))
                self.db.execute('DELETE FROM face_exclusions WHERE face_id=?', (face_id,))
                if state['person_id'] is not None:
                    self.db.execute('INSERT INTO face_people(face_id,person_id) VALUES(?,?)',
                                    (face_id, state['person_id']))
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
