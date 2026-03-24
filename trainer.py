#!/usr/bin/env python3
# vocab_trainer.py
import csv
import json
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QMenu,
)

def run_external_command_capture(template: str, *, text: str, word: str, translation: str, hint: str) -> Tuple[int, str, str]:
    """
    Executes a command template with placeholders:
      {text}, {word}, {translation}, {hint}

    Captures stdout/stderr and returns (exit_code, stdout, stderr).
    """
    if not template.strip():
        return 0, "", ""

    cmd_str = template.format(text=text, word=word, translation=translation, hint=hint)
    args = shlex.split(cmd_str)

    try:
        p = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError as e:
        return 127, "", str(e)
    except Exception as e:
        return 1, "", str(e)


def _resolve_audio_path(base_dir: str, wav_path: str) -> str:
    """Resolve a wav path. If relative, treat it as relative to the CSV location."""
    if not wav_path:
        return ""
    if os.path.isabs(wav_path):
        return wav_path
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, wav_path))
    return wav_path


# -----------------------------
# Data model
# -----------------------------

@dataclass
class VocabRow:
    word: str
    translation: str
    hint: str = ""
    tts_wav: str = ""  # optional path to a pre-rendered audio file


@dataclass
class WordStats:
    # Confidence in [0..100], higher means "known"
    confidence: int = 0
    seen: int = 0
    correct: int = 0
    wrong: int = 0
    last_seen_ts: float = 0.0


def stats_sidecar_path(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return base + ".stats.json"


def normalize_key(word: str, translation: str) -> str:
    # Stable key: exact strings, but stripped
    return f"{word.strip()}\t{translation.strip()}"


# -----------------------------
# CSV I/O
# -----------------------------

def detect_delimiter(sample: str) -> str:
    # Prefer tab if it exists, then semicolon, then comma.
    # This is reliable for European CSV exports.
    if "\t" in sample:
        return "\t"
    if ";" in sample:
        return ";"
    return ","


def load_lesson_csv(path: str) -> Tuple[List[VocabRow], str]:
    rows: List[VocabRow] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        # Read some bytes for delimiter detection
        pos = f.tell()
        sample = f.read(4096)
        f.seek(pos)

        delim = detect_delimiter(sample)

        reader = csv.reader(f, delimiter=delim)
        first_row = True
        for r in reader:
            if not r:
                continue

            # Optional header handling:
            # If the first row looks like a header ("word;translation;..."), skip it.
            if first_row:
                first_row = False
                r0 = (r[0] if len(r) > 0 else "").strip().casefold()
                r1 = (r[1] if len(r) > 1 else "").strip().casefold()
                if r0 in {"word", "deutsch"} and r1 in {"translation", "kroatisch"}:
                    continue
            word = (r[0] if len(r) > 0 else "").strip()
            trans = (r[1] if len(r) > 1 else "").strip()
            hint = (r[2] if len(r) > 2 else "").strip()
            wav = (r[3] if len(r) > 3 else "").strip()
            if not word and not trans and not hint:
                continue
            rows.append(VocabRow(word=word, translation=trans, hint=hint, tts_wav=wav))

    return rows, delim



def save_lesson_csv(path: str, rows: List[VocabRow], delimiter: str = ";") -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        for r in rows:
            # Keep backward compatibility with 3-column CSV files,
            # but support a 4th column (pre-rendered wave file path).
            row = [r.word, r.translation, r.hint]
            if getattr(r, "tts_wav", "").strip():
                row.append(r.tts_wav)
            writer.writerow(row)


# -----------------------------
# Stats I/O
# -----------------------------

def load_stats(path: str) -> Dict[str, WordStats]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, WordStats] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                out[k] = WordStats(
                    confidence=int(v.get("confidence", 0)),
                    seen=int(v.get("seen", 0)),
                    correct=int(v.get("correct", 0)),
                    wrong=int(v.get("wrong", 0)),
                    last_seen_ts=float(v.get("last_seen_ts", 0.0)),
                )
    return out


def save_stats(path: str, stats: Dict[str, WordStats]) -> None:
    raw = {k: asdict(v) for k, v in stats.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


# -----------------------------
# Training selection logic
# -----------------------------

def confidence_weight(conf: int) -> float:
    """
    Higher confidence => lower weight (seen less often).
    conf 0 => ~1.0
    conf 100 => ~0.01
    """
    conf = max(0, min(100, conf))
    # Smooth curve
    return max(0.01, (101 - conf) / 101.0)


def select_training_pool(
    rows: List[VocabRow],
    stats: Dict[str, WordStats],
    min_conf_to_hide: int,
    pool_size: int,
) -> List[Tuple[VocabRow, str, WordStats]]:
    """
    Returns a list of (row, key, stats) candidates, biased toward low confidence.
    Words with conf >= min_conf_to_hide are excluded (not repeated).
    """
    candidates: List[Tuple[VocabRow, str, WordStats]] = []
    for r in rows:
        key = normalize_key(r.word, r.translation)
        st = stats.get(key, WordStats())
        if st.confidence >= min_conf_to_hide:
            continue
        candidates.append((r, key, st))

    if not candidates:
        return []

    # Weighted sampling without replacement (approx):
    # Build weights, pick up to pool_size distinct by roulette.
    picked: List[Tuple[VocabRow, str, WordStats]] = []
    local = candidates[:]
    for _ in range(min(pool_size, len(local))):
        weights = [confidence_weight(st.confidence) for _, __, st in local]
        total = sum(weights)
        if total <= 0:
            break
        x = random.random() * total
        acc = 0.0
        idx = 0
        for i, w in enumerate(weights):
            acc += w
            if acc >= x:
                idx = i
                break
        picked.append(local.pop(idx))

    # Shuffle for variety
    random.shuffle(picked)
    return picked


def update_confidence(st: WordStats, was_correct: bool) -> WordStats:
    st.seen += 1
    st.last_seen_ts = time.time()
    if was_correct:
        st.correct += 1
        # Reward correct more if currently low, less if already high
        delta = 12 if st.confidence < 50 else 7 if st.confidence < 80 else 4
        st.confidence = min(100, st.confidence + delta)
    else:
        st.wrong += 1
        # Penalize wrong more if high (because it's surprising), but don't nuke to zero
        delta = 10 if st.confidence < 50 else 14 if st.confidence < 80 else 18
        st.confidence = max(0, st.confidence - delta)
    return st


# -----------------------------
# External command runner
# -----------------------------

def run_external_command(template: str, word: str, translation: str, hint: str) -> int:
    """
    Executes a command template with placeholders:
      {word}, {translation}, {hint}

    Runs silently (no stdout/stderr shown).
    Returns exit code.
    """
    if not template.strip():
        return 0

    cmd_str = template.format(word=word, translation=translation, hint=hint)
    args = shlex.split(cmd_str)

    try:
        p = subprocess.run(
            args,
            stdout=None,
            stderr=subprocess.DEVNULL,
        )
        return p.returncode
    except FileNotFoundError:
        return 127
    except Exception:
        return 1


def play_wave_file(path: str) -> int:
    """Play a wave/...

    Uses common Linux players ...
    Returns exit code.
    """
    if not path.strip():
        return 0
    if not os.path.exists(path):
        return 2

    candidates = [
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        ["aplay", path],
        ["paplay", path],
        ["play", path],
    ]

    for cmd in candidates:
        try:
            p = subprocess.run(cmd, stdout=None, stderr=subprocess.DEVNULL)
            if p.returncode == 0:
                return 0
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return 127



# -----------------------------
# Training dialog
# -----------------------------

class TrainingDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        lesson_rows: List[VocabRow],
        stats: Dict[str, WordStats],
        stats_path: str,
        min_conf_to_hide: int,
        pool_size: int,
        direction: str,
        tts_template: str,
        auto_tts: bool,
        mode: str = "training",  # "training" or "review"
    ):
        super().__init__(parent)
        self.setWindowTitle("Training")
        self.resize(560, 340)

        self.lesson_rows = lesson_rows
        self.stats = stats
        self.stats_path = stats_path
        self.min_conf_to_hide = min_conf_to_hide
        self.pool_size = pool_size
        self.direction = direction  # "Word→Translation" or "Translation→Word"
        self.tts_template = tts_template
        self.auto_tts = auto_tts

        # We'll use the directory of the stats file ...
        self.lesson_csv_dir = os.path.dirname(self.stats_path)
        self.mode = mode


        if self.mode == "review":
            # Admin review: go through all lesson rows in original order (no filtering, no shuffling)
            self.pool = []
            for r in lesson_rows:
                key = normalize_key(r.word, r.translation)
                st = stats.get(key, WordStats())
                self.pool.append((r, key, st))
        else:
            # Normal training: weighted pool, filtered by confidence
            self.pool = select_training_pool(
                lesson_rows, stats, min_conf_to_hide=min_conf_to_hide, pool_size=pool_size
            )

        self.index = 0

        # UI
        layout = QVBoxLayout(self)

        self.lbl_progress = QLabel("")
        self.lbl_question = QLabel("")
        self.lbl_question.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_question.setStyleSheet("font-size: 18px; font-weight: 600;")

        self.ed_answer = QLineEdit()
        self.ed_answer.setPlaceholderText("Type your answer and press Enter…")
        self.ed_answer.returnPressed.connect(self._on_check)


        if self.mode == "review":
            # Admin review doesn't require typing answers
            self.ed_answer.setPlaceholderText("Admin review: use Prev/Next, Speak, Play wave, Show solution…")
            # optional: disable enter-check behavior
            self.ed_answer.returnPressed.disconnect()



        btn_row = QHBoxLayout()
        self.btn_tts = QPushButton("Speak")
        self.btn_tts.clicked.connect(self._on_tts)
        self.btn_wave = QPushButton("Play wave")
        self.btn_wave.clicked.connect(self._on_wave)
        self.btn_show = QPushButton("Show solution")
        self.btn_show.clicked.connect(self._on_show_solution)

        self.btn_prev = QPushButton("Prev")
        self.btn_prev.clicked.connect(self._prev_card)

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self._next_card)

        self.btn_correct = QPushButton("I was correct")
        self.btn_correct.clicked.connect(lambda: self._mark(True))
        self.btn_wrong = QPushButton("I was wrong")
        self.btn_wrong.clicked.connect(lambda: self._mark(False))

        btn_row.addWidget(self.btn_tts)
        btn_row.addWidget(self.btn_wave)

        if self.mode == "review":
            btn_row.addWidget(self.btn_prev)
            btn_row.addWidget(self.btn_next)

        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_show)

        if self.mode != "review":
            btn_row.addWidget(self.btn_correct)
            btn_row.addWidget(self.btn_wrong)


        self.txt_feedback = QTextEdit()
        self.txt_feedback.setReadOnly(True)

        layout.addWidget(self.lbl_progress)
        layout.addWidget(self.lbl_question)
        layout.addWidget(self.ed_answer)
        layout.addLayout(btn_row)
        layout.addWidget(self.txt_feedback)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if not self.pool:
            self.lbl_progress.setText("No words to train (all above confidence threshold).")
            self.lbl_question.setText("")
            self.ed_answer.setEnabled(False)
            self.btn_tts.setEnabled(False)
            self.btn_show.setEnabled(False)
            self.btn_correct.setEnabled(False)
            self.btn_wrong.setEnabled(False)
        else:
            self._show_current()
            
    def _next_card(self) -> None:
        if not self.pool:
            return
        self.index = min(self.index + 1, len(self.pool) - 1)
        self._show_current()

    def _prev_card(self) -> None:
        if not self.pool:
            return
        self.index = max(self.index - 1, 0)
        self._show_current()

    def _current_pair(self) -> Optional[Tuple[VocabRow, str, WordStats]]:
        if 0 <= self.index < len(self.pool):
            return self.pool[self.index]
        return None

    def _question_answer(self, row: VocabRow) -> Tuple[str, str]:
        if self.direction == "Word→Translation":
            return (row.word, row.translation)
        else:
            return (row.translation, row.word)

    def _show_current(self) -> None:
        cur = self._current_pair()
        if not cur:
            self.lbl_progress.setText("Done.")
            self.lbl_question.setText("")
            self.ed_answer.setEnabled(False)
            return

        row, key, st = cur
        q, a = self._question_answer(row)

        if self.mode == "review":
            self.lbl_progress.setText(f"Review {self.index + 1}/{len(self.pool)}")
        else:
            self.lbl_progress.setText(
                f"Card {self.index + 1}/{len(self.pool)}   |   confidence: {st.confidence}   (hide ≥ {self.min_conf_to_hide})"
            )

        self.lbl_question.setText(q)
        self.ed_answer.clear()
        if self.mode != "review":
            self.ed_answer.setFocus()

        self.txt_feedback.setPlainText("")

        if self.auto_tts and self.tts_template.strip():
            self._on_tts()

    def _on_tts(self) -> None:
        cur = self._current_pair()
        if not cur:
            return
        row, _, __ = cur

        code = run_external_command(
            self.tts_template,
            word=row.word,
            translation=row.translation,
            hint=row.hint,
        )

        # No output spam, just show error if failed
        if code != 0:
            self.txt_feedback.setPlainText(f"TTS failed (exit={code}).")

    def _on_wave(self) -> None:
        cur = self._current_pair()
        if not cur:
            return
        row, _, __ = cur

        wav = getattr(row, "tts_wav", "") or ""
        if not wav.strip():
            self.txt_feedback.setPlainText("No wave file path in CSV (4th column).")
            return

        wav = _resolve_audio_path(self.lesson_csv_dir, wav)
        code = play_wave_file(wav)
        if code == 2:
            self.txt_feedback.setPlainText(f"Wave file not found:\n{wav}")
        elif code != 0:
            self.txt_feedback.setPlainText(f"Wave play failed (exit={code}).\nTried ffplay/aplay/paplay/play.")

    def _on_show_solution(self) -> None:
        cur = self._current_pair()
        if not cur:
            return
        row, _, st = cur
        q, a = self._question_answer(row)
        hint = f"\nHint: {row.hint}" if row.hint.strip() else ""
        self.txt_feedback.setPlainText(f"Q: {q}\nA: {a}{hint}\n\nCurrent confidence: {st.confidence}")

    def _on_check(self) -> None:
        cur = self._current_pair()
        if not cur:
            return
        row, key, st = cur
        _, a = self._question_answer(row)

        user = self.ed_answer.text().strip()
        expected = a.strip()

        # Simple check: case-insensitive exact match.
        # (You can extend this later with fuzzy matching.)
        ok = (user.casefold() == expected.casefold())
        self._mark(ok, auto=True, user_answer=user, expected=expected)

    def _mark(self, was_correct: bool, auto: bool = False, user_answer: str = "", expected: str = "") -> None:
        cur = self._current_pair()
        if not cur:
            return
        row, key, st = cur

        before = st.confidence
        st = update_confidence(st, was_correct)
        self.stats[key] = st
        save_stats(self.stats_path, self.stats)

        q, a = self._question_answer(row)
        hint = f"\nHint: {row.hint}" if row.hint.strip() else ""

        if auto:
            self.txt_feedback.setPlainText(
                f"{'✅ Correct' if was_correct else '❌ Wrong'}\n"
                f"Your answer: {user_answer}\nExpected: {expected}\n\n"
                f"Q: {q}\nA: {a}{hint}\n\n"
                f"Confidence: {before} → {st.confidence}"
            )
        else:
            self.txt_feedback.setPlainText(
                f"{'✅ Marked correct' if was_correct else '❌ Marked wrong'}\n\n"
                f"Q: {q}\nA: {a}{hint}\n\n"
                f"Confidence: {before} → {st.confidence}"
            )

        # Advance
        self.index += 1
        if self.index >= len(self.pool):
            # Optionally rebuild pool (in case you want continuous training).
            self.lbl_progress.setText("Training round finished.")
            self.lbl_question.setText("")
            self.ed_answer.setEnabled(False)
            return
        self._show_current()


# -----------------------------
# Main window
# -----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vocabulary Trainer (CSV + Confidence Stats)")
        self.resize(900, 520)

        self.current_csv_path: Optional[str] = None
        self.current_rows: List[VocabRow] = []
        self.current_stats: Dict[str, WordStats] = {}

        self.current_delimiter: str = ";"
        # Central UI
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Word", "Translation", "Hint", "Wave file", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        root.addWidget(self.table)

        # Controls
        controls = QHBoxLayout()

        self.btn_add = QPushButton("Add row")
        self.btn_add.clicked.connect(self.add_row)
        self.btn_del = QPushButton("Delete selected")
        self.btn_del.clicked.connect(self.delete_selected)

        controls.addWidget(self.btn_add)
        controls.addWidget(self.btn_del)
        controls.addStretch(1)

        self.btn_train = QPushButton("Start training")
        self.btn_train.clicked.connect(self.start_training)
        controls.addWidget(self.btn_train)
        
        self.btn_admin = QPushButton("Admin review")
        self.btn_admin.clicked.connect(self.start_admin_review)

        controls.addWidget(self.btn_train)
        controls.addWidget(self.btn_admin)

        

        root.addLayout(controls)

        # Training settings group
        settings_box = QGroupBox("Training & TTS settings")
        s_layout = QFormLayout(settings_box)

        self.spin_hide_conf = QSpinBox()
        self.spin_hide_conf.setRange(0, 100)
        self.spin_hide_conf.setValue(85)
        self.spin_hide_conf.setToolTip("Words with confidence ≥ this value are skipped.")

        self.spin_pool = QSpinBox()
        self.spin_pool.setRange(1, 10000)
        self.spin_pool.setValue(30)
        self.spin_pool.setToolTip("How many cards per training round.")

        self.combo_dir = QComboBox()
        self.combo_dir.addItems(["Word→Translation", "Translation→Word"])

        self.ed_cmd = QLineEdit()
        self.ed_cmd.setPlaceholderText(
            "External command template, e.g. espeak-ng -v de \"{word}\""
        )

        self.chk_auto_tts = QCheckBox("Auto speak each question")
        self.chk_auto_tts.setChecked(False)

        s_layout.addRow("Hide well-learned (confidence ≥):", self.spin_hide_conf)
        s_layout.addRow("Training pool size:", self.spin_pool)
        s_layout.addRow("Direction:", self.combo_dir)
        s_layout.addRow("External command (TTS) template:", self.ed_cmd)
        self.ed_translate_cmd = QLineEdit()
        self.ed_translate_cmd.setPlaceholderText(
            "External command template for translation, e.g. mytranslator --in \"{text}\""
        )
        
        self.ed_translate_cmd.setText("trans -brief -no-ansi -no-pager -e google -s de -t hr {word}")
        
        s_layout.addRow("External command (Translate) template:", self.ed_translate_cmd)
        s_layout.addRow("", self.chk_auto_tts)

        root.addWidget(settings_box)

        # Status
        self.setStatusBar(QStatusBar())

        # Toolbar / menu actions
        self._build_actions()

        # Start with an empty lesson
        self.refresh_table()

    def _show_text_dialog(self, title: str, text: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(700, 420)

        layout = QVBoxLayout(dlg)
        ed = QTextEdit()
        ed.setPlainText(text)
        layout.addWidget(ed)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_copy = QPushButton("Copy")
        btns.addButton(btn_copy, QDialogButtonBox.ButtonRole.ActionRole)

        def do_copy():
            QApplication.clipboard().setText(ed.toPlainText())

        btn_copy.clicked.connect(do_copy)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        dlg.exec()
        
    def _on_table_context_menu(self, pos) -> None:
        # Map to global position
        global_pos = self.table.viewport().mapToGlobal(pos)

        # Determine clicked row (optional)
        idx = self.table.indexAt(pos)
        clicked_row = idx.row() if idx.isValid() else -1

        menu = QMenu(self)

        act_translate = menu.addAction("Translate via external command…")
        act_copy_cell = menu.addAction("Copy selected cell(s)")

        chosen = menu.exec(global_pos)
        if chosen is None:
            return

        if chosen == act_copy_cell:
            txt = self._selected_cells_text()
            if txt:
                QApplication.clipboard().setText(txt)
                self.statusBar().showMessage("Copied selected cell(s) to clipboard.", 2500)
            return

        if chosen == act_translate:
            template = self.ed_translate_cmd.text().strip()
            if not template:
                QMessageBox.information(
                    self,
                    "No translate command",
                    "Please set the 'External command (Translate) template' first.",
                )
                return

            sel_text = self._selected_cells_text()
            if not sel_text:
                QMessageBox.information(self, "No selection", "Select one or more cells first.")
                return

            # Provide placeholders based on clicked row if available,
            # otherwise fall back to first selected row.
            row_for_data = clicked_row
            if row_for_data < 0:
                items = self.table.selectedItems()
                row_for_data = items[0].row() if items else 0

            word, translation, hint = self._row_data_for_index(row_for_data)

            code, out, err = run_external_command_capture(
                template,
                text=sel_text,
                word=word,
                translation=translation,
                hint=hint,
            )

            result = out.strip()
            if not result and err.strip():
                result = err.strip()

            if code != 0 and not result:
                result = f"Command failed (exit={code}) and returned no output."

            # Copy to clipboard “if possible” -> yes, do it by default:
            QApplication.clipboard().setText(result)

            # Show result in a textedit dialog:
            self._show_text_dialog("Translate result", result)

            self.statusBar().showMessage("Translate result copied to clipboard.", 3000)
            return
      
        
    def _selected_cells_text(self) -> str:
        items = self.table.selectedItems()
        if not items:
            return ""
        # Keep it simple: join selected cell texts line-by-line
        # (If you prefer row-wise formatting, you can change this.)
        return "\n".join((it.text() or "").strip() for it in items if (it.text() or "").strip())

    def _row_data_for_index(self, row_idx: int) -> Tuple[str, str, str]:
        word = self._cell_text(row_idx, 0)
        translation = self._cell_text(row_idx, 1)
        hint = self._cell_text(row_idx, 2)
        return word, translation, hint

    def _build_actions(self) -> None:
        tb = QToolBar("Main")
        self.addToolBar(tb)

        act_new = QAction("New", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self.new_lesson)

        act_open = QAction("Open CSV…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_csv)

        act_save = QAction("Save", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_csv)

        act_save_as = QAction("Save As…", self)
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_as.triggered.connect(self.save_csv_as)

        tb.addAction(act_new)
        tb.addAction(act_open)
        tb.addAction(act_save)
        tb.addAction(act_save_as)

    def new_lesson(self) -> None:
        if not self._maybe_save_changes():
            return
        self.current_csv_path = None
        self.current_rows = []
        self.current_stats = {}
        self.refresh_table()
        self.statusBar().showMessage("New lesson.", 3000)

    def open_csv(self) -> None:
        if not self._maybe_save_changes():
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open lesson CSV", "", "CSV files (*.csv);;All files (*.csv)"
        )
        if not path:
            return

        try:
            rows, delim = load_lesson_csv(path)
            self.current_delimiter = delim
        except Exception as e:
            QMessageBox.critical(self, "Open failed", f"Could not read CSV:\n{e}")
            return

        self.current_csv_path = path
        self.current_rows = rows

        # Load stats sidecar
        spath = stats_sidecar_path(path)
        self.current_stats = load_stats(spath)

        self.refresh_table()
        self.statusBar().showMessage(f"Loaded: {os.path.basename(path)}", 4000)

    def save_csv(self) -> None:
        if not self.current_csv_path:
            self.save_csv_as()
            return
        self._pull_table_to_rows()
        try:
            save_lesson_csv(self.current_csv_path, self.current_rows, delimiter=self.current_delimiter)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not write CSV:\n{e}")
            return
        self.statusBar().showMessage("Saved.", 2000)

    def save_csv_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save lesson CSV", "", "CSV files (*.csv);;All files (*.*)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        self._pull_table_to_rows()
        try:
            # Default to semicolon if new file
            if not self.current_delimiter:
                self.current_delimiter = ";"

            save_lesson_csv(path, self.current_rows, delimiter=self.current_delimiter)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not write CSV:\n{e}")
            return

        # If we previously had stats, keep them but write next to the new file
        self.current_csv_path = path
        save_stats(stats_sidecar_path(path), self.current_stats)

        self.statusBar().showMessage(f"Saved as: {os.path.basename(path)}", 4000)

    def add_row(self) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c in range(4):
            self.table.setItem(r, c, QTableWidgetItem(""))
        self.table.setCurrentCell(r, 0)

    def delete_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def refresh_table(self) -> None:
        self.table.setRowCount(0)
        for row in self.current_rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(row.word))
            self.table.setItem(r, 1, QTableWidgetItem(row.translation))
            self.table.setItem(r, 2, QTableWidgetItem(row.hint))
            self.table.setItem(r, 3, QTableWidgetItem(getattr(row, "tts_wav", "")))
            # confidence from stats sidecar
            key = normalize_key(row.word, row.translation)
            st = self.current_stats.get(key, WordStats())
            conf_item = QTableWidgetItem(str(st.confidence))
            conf_item.setFlags(conf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # read-only
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 4, conf_item)

    def _pull_table_to_rows(self) -> None:
        rows: List[VocabRow] = []
        for r in range(self.table.rowCount()):
            w = self._cell_text(r, 0)
            t = self._cell_text(r, 1)
            h = self._cell_text(r, 2)
            wav = self._cell_text(r, 3)
            if not w and not t and not h and not wav:
                continue
            rows.append(VocabRow(word=w, translation=t, hint=h, tts_wav=wav))
        self.current_rows = rows

        # Ensure stats keys exist for current rows (not required, but nice)
        for vr in self.current_rows:
            k = normalize_key(vr.word, vr.translation)
            if k not in self.current_stats:
                self.current_stats[k] = WordStats()

    def _cell_text(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return (item.text() if item else "").strip()

    def start_training(self) -> None:
        self._pull_table_to_rows()
        if not self.current_rows:
            QMessageBox.information(self, "No data", "The lesson is empty. Add some rows first.")
            return

        # If lesson isn't saved yet, stats would have nowhere stable to go
        if not self.current_csv_path:
            QMessageBox.information(self, "Save first", "Please save the CSV first so stats can be stored next to it.")
            self.save_csv_as()
            if not self.current_csv_path:
                return

        spath = stats_sidecar_path(self.current_csv_path)
        # Re-load to pick up external edits
        self.current_stats = load_stats(spath)
        # Ensure coverage
        for vr in self.current_rows:
            k = normalize_key(vr.word, vr.translation)
            self.current_stats.setdefault(k, WordStats())

        dlg = TrainingDialog(
            parent=self,
            lesson_rows=self.current_rows,
            stats=self.current_stats,
            stats_path=spath,
            min_conf_to_hide=self.spin_hide_conf.value(),
            pool_size=self.spin_pool.value(),
            direction=self.combo_dir.currentText(),
            tts_template=self.ed_cmd.text(),
            auto_tts=self.chk_auto_tts.isChecked(),
        )
        dlg.exec()

        # Re-save stats after training
        save_stats(spath, self.current_stats)
        self.refresh_table()
        self.statusBar().showMessage("Training finished. Stats saved.", 4000)

    def start_admin_review(self) -> None:
        self._pull_table_to_rows()
        if not self.current_rows:
            QMessageBox.information(self, "No data", "The lesson is empty. Add some rows first.")
            return

        # If lesson isn't saved yet, stats would have nowhere stable to go
        if not self.current_csv_path:
            QMessageBox.information(self, "Save first", "Please save the CSV first so stats can be stored next to it.")
            self.save_csv_as()
            if not self.current_csv_path:
                return

        spath = stats_sidecar_path(self.current_csv_path)
        self.current_stats = load_stats(spath)
        for vr in self.current_rows:
            k = normalize_key(vr.word, vr.translation)
            self.current_stats.setdefault(k, WordStats())

        dlg = TrainingDialog(
            parent=self,
            lesson_rows=self.current_rows,
            stats=self.current_stats,
            stats_path=spath,
            min_conf_to_hide=self.spin_hide_conf.value(),   # not used in review mode, kept for signature
            pool_size=self.spin_pool.value(),               # not used in review mode, kept for signature
            direction=self.combo_dir.currentText(),
            tts_template=self.ed_cmd.text(),
            auto_tts=self.chk_auto_tts.isChecked(),
            mode="review",                                  # <<< NEW
        )
        dlg.exec()

        # Save stats in case you still used mark buttons (optional)
        save_stats(spath, self.current_stats)
        self.statusBar().showMessage("Admin review finished.", 4000)




    def _maybe_save_changes(self) -> bool:
        # Simple heuristic: if table has anything, ask.
        # For a more robust solution, track a dirty flag on edits.
        if self.table.rowCount() == 0:
            return True

        resp = QMessageBox.question(
            self,
            "Save changes?",
            "Do you want to save your current lesson before continuing?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        )
        if resp == QMessageBox.StandardButton.Cancel:
            return False
        if resp == QMessageBox.StandardButton.Yes:
            self.save_csv()
        return True


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
