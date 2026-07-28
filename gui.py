import os
import sys
from pathlib import Path
import threading

from processing.realtime_vad import RealTimeVAD

from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt, pyqtSignal,  QObject
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


PDF_REPORT_PATH = "outputs/meeting_report.pdf"

SUPPORTED_AUDIO_FORMATS = (
    "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma"
)

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
}


class DropArea(QFrame):
    file_selected = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()

        self.setAcceptDrops(True)
        self.setObjectName("dropArea")
        self.setMinimumHeight(165)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(8)

        self.icon_label = QLabel("♫")
        self.icon_label.setObjectName("dropIcon")
        self.icon_label.setAlignment(Qt.AlignCenter)

        self.title_label = QLabel("Prevuci audio-snimak ovde")
        self.title_label.setObjectName("dropTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.subtitle_label = QLabel(
            "ili klikni da izabereš fajl sa računara"
        )
        self.subtitle_label.setObjectName("dropSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignCenter)

        self.file_label = QLabel(
            "Podržani formati: WAV, MP3, M4A, FLAC, OGG, AAC i WMA"
        )
        self.file_label.setObjectName("dropFile")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setWordWrap(True)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addWidget(self.file_label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.open_file_dialog()

    def open_file_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Izaberi audio-snimak",
            "",
            (
                f"Audio fajlovi ({SUPPORTED_AUDIO_FORMATS});;"
                "Svi fajlovi (*.*)"
            ),
        )

        if file_path:
            self.file_selected.emit(file_path)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragActive", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)

        urls = event.mimeData().urls()

        if urls:
            file_path = urls[0].toLocalFile()

            if file_path:
                self.file_selected.emit(file_path)

    def show_selected_file(self, file_path: str) -> None:
        self.icon_label.setText("✓")
        self.title_label.setText("Audio-snimak je izabran")
        self.subtitle_label.setText(Path(file_path).name)
        self.file_label.setText(file_path)

class RealtimeSignals(QObject):
    transcript_received = pyqtSignal(str)
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str)
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.audio_path = ""
        self.live_transcript_path = ""
        self.is_recording = False

        self.realtime_vad: RealTimeVAD | None = None
        self.realtime_thread: threading.Thread | None = None

        self.realtime_signals = RealtimeSignals()

        self.realtime_signals.transcript_received.connect(
            self.show_live_transcript
        )
        self.realtime_signals.recording_started.connect(
            self.recording_started
        )
        self.realtime_signals.recording_stopped.connect(
            self.recording_stopped
        )
        self.realtime_signals.error_occurred.connect(
            self.recording_error
        )
        self.process = QProcess(self)

        self.process.setProcessChannelMode(
            QProcess.MergedChannels
        )
        self.process.readyReadStandardOutput.connect(
            self.read_process_output
        )
        self.process.finished.connect(
            self.processing_finished
        )
        self.process.errorOccurred.connect(
            self.processing_error
        )

        self.setWindowTitle("Meeting Summarizer")
        self.setMinimumSize(980, 720)
        self.resize(1080, 790)

        self.setup_ui()
        self.apply_styles()

    def setup_ui(self) -> None:
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(44, 34, 44, 34)
        main_layout.setSpacing(22)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        logo = QLabel("M")
        logo.setObjectName("logo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(58, 58)

        title_container = QVBoxLayout()
        title_container.setSpacing(2)

        title = QLabel("Meeting Summarizer")
        title.setObjectName("mainTitle")

        subtitle = QLabel(
            "Offline transkripcija i automatsko sažimanje audio-snimaka"
        )
        subtitle.setObjectName("mainSubtitle")

        title_container.addWidget(title)
        title_container.addWidget(subtitle)

        header_layout.addWidget(logo)
        header_layout.addLayout(title_container)
        header_layout.addStretch()

        badge = QLabel("OFFLINE • PRIVATNO")
        badge.setObjectName("badge")
        badge.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(badge)
        main_layout.addLayout(header_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(22)

        left_card = QFrame()
        left_card.setObjectName("card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(26, 26, 26, 26)
        left_layout.setSpacing(18)

        step_label = QLabel("1. Izaberi ili snimi audio")
        step_label.setObjectName("sectionTitle")

        self.drop_area = DropArea()
        self.drop_area.file_selected.connect(
            self.set_audio_file
        )

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.choose_button = QPushButton("Izaberi fajl")
        self.choose_button.setObjectName("secondaryButton")
        self.choose_button.clicked.connect(
            self.drop_area.open_file_dialog
        )

        self.start_button = QPushButton("Pokreni obradu")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(
            self.start_processing
        )

        button_layout.addWidget(self.choose_button)
        button_layout.addWidget(self.start_button)
        recording_button_layout = QHBoxLayout()
        recording_button_layout.setSpacing(12)

        self.record_button = QPushButton("Pokreni snimanje")
        self.record_button.setObjectName("secondaryButton")
        self.record_button.clicked.connect(
            self.start_recording
        )

        self.stop_record_button = QPushButton("Zaustavi snimanje")
        self.stop_record_button.setObjectName("secondaryButton")
        self.stop_record_button.setEnabled(False)
        self.stop_record_button.clicked.connect(
            self.stop_recording
        )

        self.live_summary_button = QPushButton(
            "Sažmi live transkript"
        )
        self.live_summary_button.setObjectName("primaryButton")
        self.live_summary_button.setEnabled(False)
        self.live_summary_button.clicked.connect(
            self.start_live_summary
        )

        recording_button_layout.addWidget(self.record_button)
        recording_button_layout.addWidget(self.stop_record_button)

        live_summary_layout = QHBoxLayout()
        live_summary_layout.addWidget(self.live_summary_button)

        progress_title = QLabel("Status obrade")
        progress_title.setObjectName("sectionTitle")

        status_row = QHBoxLayout()

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDotIdle")

        self.status_label = QLabel(
            "Čeka se izbor audio-snimka"
        )
        self.status_label.setObjectName("statusLabel")

        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label)
        status_row.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        left_layout.addWidget(step_label)
        left_layout.addWidget(self.drop_area)
        left_layout.addLayout(button_layout)
        left_layout.addLayout(recording_button_layout)
        left_layout.addLayout(live_summary_layout)
        left_layout.addSpacing(4)
        left_layout.addWidget(progress_title)
        left_layout.addLayout(status_row)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch()

        right_card = QFrame()
        right_card.setObjectName("card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(26, 26, 26, 26)
        right_layout.setSpacing(14)

        log_header = QHBoxLayout()

        log_title = QLabel("Tok obrade")
        log_title.setObjectName("sectionTitle")

        self.clear_log_button = QPushButton("Obriši")
        self.clear_log_button.setObjectName("textButton")

        log_header.addWidget(log_title)
        log_header.addStretch()
        log_header.addWidget(self.clear_log_button)

        self.log_output = QPlainTextEdit()
        self.log_output.setObjectName("logOutput")
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText(
            "Ovde će biti prikazan napredak obrade..."
        )

        self.clear_log_button.clicked.connect(
            self.log_output.clear
        )

        result_title = QLabel("Rezultat")
        result_title.setObjectName("sectionTitle")

        result_buttons = QHBoxLayout()
        result_buttons.setSpacing(12)

        self.open_pdf_button = QPushButton("Otvori PDF")
        self.open_pdf_button.setObjectName("secondaryButton")
        self.open_pdf_button.setEnabled(False)
        self.open_pdf_button.clicked.connect(self.open_pdf)

        self.open_folder_button = QPushButton(
            "Otvori outputs folder"
        )
        self.open_folder_button.setObjectName("secondaryButton")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(
            self.open_output_folder
        )

        result_buttons.addWidget(self.open_pdf_button)
        result_buttons.addWidget(self.open_folder_button)

        right_layout.addLayout(log_header)
        right_layout.addWidget(self.log_output, 1)
        live_title = QLabel("Transkript uživo")
        live_title.setObjectName("sectionTitle")

        self.live_transcript_output = QPlainTextEdit()
        self.live_transcript_output.setReadOnly(True)
        self.live_transcript_output.setPlaceholderText(
            "Prepoznati govor će se prikazivati ovde..."
        )
        self.live_transcript_output.setMaximumHeight(180)

        right_layout.addWidget(live_title)
        right_layout.addWidget(self.live_transcript_output)
        right_layout.addWidget(result_title)
        right_layout.addLayout(result_buttons)

        content_layout.addWidget(left_card, 5)
        content_layout.addWidget(right_card, 6)

        main_layout.addLayout(content_layout, 1)

        footer_layout = QHBoxLayout()

        footer_text = QLabel(
            "Audio se obrađuje lokalno na računaru "
            "i ne šalje se na internet."
        )
        footer_text.setObjectName("footerText")

        footer_layout.addWidget(footer_text)
        footer_layout.addStretch()

        main_layout.addLayout(footer_layout)

    def apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget#centralWidget {
                background: #f6f4fb;
            }

            QLabel {
                color: #27243b;
            }

            QFrame#card {
                background: #ffffff;
                border: 1px solid #e7e1f2;
                border-radius: 20px;
            }

            QLabel#logo {
                background: #6c4fd3;
                color: white;
                border-radius: 18px;
                font-size: 28px;
                font-weight: 700;
            }

            QLabel#mainTitle {
                font-size: 27px;
                font-weight: 700;
                color: #25213a;
            }

            QLabel#mainSubtitle {
                font-size: 13px;
                color: #756f86;
            }

            QLabel#badge {
                background: #ece6ff;
                color: #6045bd;
                border-radius: 12px;
                padding: 8px 14px;
                font-size: 11px;
                font-weight: 700;
            }

            QLabel#sectionTitle {
                font-size: 15px;
                font-weight: 700;
                color: #332d4d;
            }

            QFrame#dropArea {
                background: #faf8ff;
                border: 2px dashed #c8b9ee;
                border-radius: 16px;
            }

            QFrame#dropArea:hover,
            QFrame#dropArea[dragActive="true"] {
                background: #f1ecff;
                border-color: #7256d8;
            }

            QLabel#dropIcon {
                color: #6c4fd3;
                font-size: 34px;
                font-weight: 700;
            }

            QLabel#dropTitle {
                color: #332d4d;
                font-size: 15px;
                font-weight: 700;
            }

            QLabel#dropSubtitle {
                color: #7e768f;
                font-size: 12px;
            }

            QLabel#dropFile {
                color: #9a93a8;
                font-size: 10px;
            }

            QPushButton {
                min-height: 42px;
                border-radius: 12px;
                padding: 0 18px;
                font-size: 13px;
                font-weight: 600;
            }

            QPushButton#primaryButton {
                background: #6c4fd3;
                color: white;
                border: none;
            }

            QPushButton#primaryButton:hover {
                background: #5e42c4;
            }

            QPushButton#primaryButton:pressed {
                background: #5036ae;
            }

            QPushButton#primaryButton:disabled {
                background: #cfc8df;
                color: #f8f7fb;
            }

            QPushButton#secondaryButton {
                background: #f5f1ff;
                color: #5d45b6;
                border: 1px solid #dcd2f4;
            }

            QPushButton#secondaryButton:hover {
                background: #ebe4ff;
                border-color: #bba9ea;
            }

            QPushButton#secondaryButton:disabled {
                background: #f2f0f5;
                color: #b5afc0;
                border-color: #e6e2eb;
            }

            QPushButton#textButton {
                min-height: 28px;
                background: transparent;
                color: #786d95;
                border: none;
                padding: 2px 8px;
            }

            QPushButton#textButton:hover {
                color: #5e42c4;
                background: #f2edff;
            }

            QLabel#statusDotIdle {
                color: #aaa4b6;
                font-size: 16px;
            }

            QLabel#statusDotRunning {
                color: #d39a32;
                font-size: 16px;
            }

            QLabel#statusDotSuccess {
                color: #3ca675;
                font-size: 16px;
            }

            QLabel#statusDotError {
                color: #d95a67;
                font-size: 16px;
            }

            QLabel#statusLabel {
                color: #665f76;
                font-size: 12px;
            }

            QProgressBar#progressBar {
                height: 11px;
                background: #eeeaf4;
                border: none;
                border-radius: 5px;
            }

            QProgressBar#progressBar::chunk {
                background: #6c4fd3;
                border-radius: 5px;
            }

            QPlainTextEdit#logOutput {
                background: #252236;
                color: #e8e3f4;
                border: none;
                border-radius: 14px;
                padding: 14px;
                font-family: Consolas;
                font-size: 11px;
                selection-background-color: #7358ce;
            }

            QLabel#footerText {
                color: #8c8599;
                font-size: 11px;
            }
            """
        )

    def start_recording(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(
                self,
                "Obrada je u toku",
                "Sačekaj da se trenutna obrada završi.",
            )
            return

        if self.is_recording:
            return

        self.audio_path = ""
        self.is_recording = True

        self.live_transcript_output.clear()
        self.log_output.appendPlainText(
            "\nUčitavanje real-time modela..."
        )

        self.choose_button.setEnabled(False)
        self.drop_area.setEnabled(False)
        self.start_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(False)

        self.set_status(
            "Učitavanje modela za real-time obradu...",
            "running",
        )

        self.realtime_thread = threading.Thread(
            target=self._start_realtime_worker,
            daemon=True,
        )
        self.realtime_thread.start()

    def _start_realtime_worker(self) -> None:
        try:
            self.realtime_vad = RealTimeVAD(
                device=1,
                input_sample_rate=44100,
                channels=2,
                on_transcript=(
                    self.realtime_signals
                    .transcript_received.emit
                ),
            )

            self.realtime_vad.start()

            self.realtime_signals.recording_started.emit()

        except Exception as error:
            self.realtime_signals.error_occurred.emit(
                str(error)
            )

    def stop_recording(self) -> None:
        if not self.is_recording:
            return

        self.stop_record_button.setEnabled(False)

        self.set_status(
            "Završavanje poslednjeg segmenta...",
            "running",
        )

        thread = threading.Thread(
            target=self._stop_realtime_worker,
            daemon=True,
        )
        thread.start()

    def _stop_realtime_worker(self) -> None:
        try:
            if self.realtime_vad is None:
                raise RuntimeError(
                    "Real-time obrada nije pokrenuta."
                )

            project_directory = Path(__file__).resolve().parent

            audio_output_path = (
                    project_directory
                    / "audio"
                    / "live_recording.wav"
            )

            transcript_output_path = (
                    project_directory
                    / "outputs"
                    / "transcript_live.txt"
            )

            audio_path, transcript_path = (
                self.realtime_vad.stop(
                    audio_output_path=str(
                        audio_output_path
                    ),
                    transcript_output_path=str(
                        transcript_output_path
                    ),
                )
            )

            self.realtime_signals.recording_stopped.emit(
                audio_path,
                transcript_path,
            )

        except Exception as error:
            self.realtime_signals.error_occurred.emit(
                str(error)
            )

    def recording_started(self) -> None:
        self.stop_record_button.setEnabled(True)

        self.set_status(
            "Real-time snimanje i transkripcija su u toku...",
            "running",
        )

        self.log_output.appendPlainText(
            "Real-time snimanje je pokrenuto. "
            "Govori u mikrofon..."
        )

    def show_live_transcript(self, text: str) -> None:
        self.live_transcript_output.appendPlainText(
            text
        )

        self.log_output.appendPlainText(
            f"Prepoznato: {text}"
        )

    def recording_stopped(
            self,
            audio_path: str,
            transcript_path: str,
    ) -> None:
        self.is_recording = False
        self.realtime_vad = None

        self.live_transcript_path = transcript_path


        self.choose_button.setEnabled(True)
        self.drop_area.setEnabled(True)
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)

        self.live_summary_button.setEnabled(
            os.path.isfile(self.live_transcript_path)
        )

        self.log_output.appendPlainText(
            "\nReal-time snimanje je završeno."
            f"\nAudio: {audio_path}"
            f"\nTranskript: {transcript_path}"
        )

        # Snimljeni WAV automatski postaje izabrani audio.
        self.set_audio_file(audio_path)

        QMessageBox.information(
            self,
            "Snimanje završeno",
            (
                "Snimanje i real-time transkripcija "
                "su završeni.\n\n"
                "Sada možeš da pokreneš kompletnu obradu."
            ),
        )

    def recording_error(self, message: str) -> None:
        self.is_recording = False
        self.realtime_vad = None

        self.choose_button.setEnabled(True)
        self.drop_area.setEnabled(True)
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)

        self.set_status(
            "Greška tokom real-time obrade",
            "error",
        )

        self.log_output.appendPlainText(
            f"\nGreška: {message}"
        )

        QMessageBox.critical(
            self,
            "Greška tokom snimanja",
            message,
        )
    def set_audio_file(self, file_path: str) -> None:
        if not os.path.isfile(file_path):
            QMessageBox.warning(
                self,
                "Neispravan fajl",
                "Izabrani fajl ne postoji.",
            )
            return

        extension = Path(file_path).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            QMessageBox.warning(
                self,
                "Nepodržan format",
                (
                    "Izaberi WAV, MP3, M4A, FLAC, "
                    "OGG, AAC ili WMA fajl."
                ),
            )
            return

        self.audio_path = file_path
        self.drop_area.show_selected_file(file_path)
        self.start_button.setEnabled(True)

        self.set_status(
            "Audio-snimak je spreman za obradu",
            "idle",
        )

        self.log_output.appendPlainText(
            f"Izabran fajl: {file_path}"
        )

    def start_live_summary(self) -> None:
        if not self.live_transcript_path:
            QMessageBox.warning(
                self,
                "Transkript nije pronađen",
                "Prvo pokreni i završi real-time snimanje.",
            )
            return

        if not os.path.isfile(self.live_transcript_path):
            QMessageBox.warning(
                self,
                "Transkript nije pronađen",
                (
                    "Live transkript ne postoji:\n"
                    f"{self.live_transcript_path}"
                ),
            )
            return

        if self.process.state() != QProcess.NotRunning:
            return

        self.start_pipeline_process(
            arguments=[
                "--transcript",
                self.live_transcript_path,
            ],
            log_message=(
                "Pokretanje sumarizacije live transkripta:\n"
                f"{self.live_transcript_path}\n"
            ),
        )

    def start_pipeline_process(
            self,
            arguments: list[str],
            log_message: str,
    ) -> None:
        self.set_controls_enabled(False)
        self.live_summary_button.setEnabled(False)

        self.open_pdf_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)

        self.log_output.clear()
        self.log_output.appendPlainText(log_message)

        self.set_status(
            "Obrada je u toku...",
            "running",
        )

        self.progress_bar.setRange(0, 0)

        python_executable = sys.executable

        project_directory = os.path.dirname(
            os.path.abspath(__file__)
        )

        main_script = os.path.join(
            project_directory,
            "main.py",
        )

        self.process.setWorkingDirectory(
            project_directory
        )

        environment = QProcessEnvironment.systemEnvironment()

        cuda_bin = (
            r"C:\Program Files\NVIDIA GPU Computing Toolkit"
            r"\CUDA\v12.4\bin"
        )

        current_path = environment.value("PATH")

        if os.path.isdir(cuda_bin) and cuda_bin not in current_path:
            environment.insert(
                "PATH",
                cuda_bin + os.pathsep + current_path,
            )

        environment.insert(
            "CUDA_PATH",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit"
            r"\CUDA\v12.4",
        )

        user_profile = os.environ.get("USERPROFILE", "")
        home_drive = os.environ.get("HOMEDRIVE", "")
        home_path = os.environ.get("HOMEPATH", "")

        if user_profile:
            environment.insert("USERPROFILE", user_profile)
            environment.insert("HOME", user_profile)

        if home_drive:
            environment.insert("HOMEDRIVE", home_drive)

        if home_path:
            environment.insert("HOMEPATH", home_path)

        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")

        self.process.setProcessEnvironment(environment)

        self.process.start(
            python_executable,
            [
                "-u",
                main_script,
                *arguments,
            ],
        )

    def start_processing(self) -> None:
        if not self.audio_path:
            return

        if self.process.state() != QProcess.NotRunning:
            return

        self.start_pipeline_process(
            arguments=[self.audio_path],
            log_message=(
                "Pokretanje kompletne audio-obrade za:\n"
                f"{self.audio_path}\n"
            ),
        )

    def read_process_output(self) -> None:
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode(
            "utf-8",
            errors="replace",
        )

        self.log_output.moveCursor(
            self.log_output.textCursor().End
        )
        self.log_output.insertPlainText(text)
        self.log_output.ensureCursorVisible()

        lowered = text.lower()

        if "[1/5]" in lowered:
            self.set_status(
                "Detekcija govora...",
                "running",
            )
        elif "[2/5]" in lowered:
            self.set_status(
                "Transkripcija audio-snimka...",
                "running",
            )
        elif "[3/5]" in lowered:
            self.set_status(
                "Čišćenje transkripta...",
                "running",
            )
        elif "[4/5]" in lowered:
            self.set_status(
                "Generisanje sažetka...",
                "running",
            )
        elif "[5/5]" in lowered:
            self.set_status(
                "Kreiranje PDF sažetka...",
                "running",
            )

    def processing_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        self.progress_bar.setRange(0, 100)
        self.set_controls_enabled(True)

        self.live_summary_button.setEnabled(
            bool(self.live_transcript_path)
            and os.path.isfile(self.live_transcript_path)
        )

        if (
            exit_status == QProcess.NormalExit
            and exit_code == 0
        ):
            self.progress_bar.setValue(100)

            self.set_status(
                "Obrada je uspešno završena",
                "success",
            )

            self.open_pdf_button.setEnabled(
                os.path.exists(PDF_REPORT_PATH)
            )
            self.open_folder_button.setEnabled(True)

            QMessageBox.information(
                self,
                "Obrada završena",
                (
                    "Audio-snimak je uspešno obrađen.\n\n"
                    f"PDF sažetak:\n{PDF_REPORT_PATH}"
                ),
            )
        else:
            self.progress_bar.setValue(0)

            self.set_status(
                "Obrada je prekinuta zbog greške",
                "error",
            )

            self.open_folder_button.setEnabled(True)

            self.log_output.appendPlainText(
                "\n\nProces je završen greškom."
                f"\nExit code: {exit_code}"
            )

            QMessageBox.critical(
                self,
                "Greška tokom obrade",
                (
                    "Pipeline je prekinut. "
                    "Detalji se nalaze u delu "
                    "'Tok obrade'."
                ),
            )

    def processing_error(
        self,
        process_error: QProcess.ProcessError,
    ) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.set_controls_enabled(True)

        self.live_summary_button.setEnabled(
            bool(self.live_transcript_path)
            and os.path.isfile(self.live_transcript_path)
        )

        self.set_status(
            "Proces nije mogao da se pokrene",
            "error",
        )

        self.log_output.appendPlainText(
            "\n\nQProcess greška: "
            + str(process_error)
        )

    def set_controls_enabled(self, enabled: bool) -> None:
        self.choose_button.setEnabled(
            enabled and not self.is_recording
        )
        self.drop_area.setEnabled(
            enabled and not self.is_recording
        )
        self.start_button.setEnabled(
            enabled
            and not self.is_recording
            and bool(self.audio_path)
        )
        self.record_button.setEnabled(
            enabled and not self.is_recording
        )
        self.stop_record_button.setEnabled(
            self.is_recording
        )

    def set_status(self, text: str, state: str) -> None:
        self.status_label.setText(text)

        state_objects = {
            "idle": "statusDotIdle",
            "running": "statusDotRunning",
            "success": "statusDotSuccess",
            "error": "statusDotError",
        }

        self.status_dot.setObjectName(
            state_objects.get(
                state,
                "statusDotIdle",
            )
        )

        self.status_dot.style().unpolish(
            self.status_dot
        )
        self.status_dot.style().polish(
            self.status_dot
        )

    def open_pdf(self) -> None:
        pdf_path = os.path.abspath(PDF_REPORT_PATH)

        if not os.path.exists(pdf_path):
            QMessageBox.warning(
                self,
                "PDF nije pronađen",
                f"Fajl ne postoji:\n{pdf_path}",
            )
            return

        os.startfile(pdf_path)

    def open_output_folder(self) -> None:
        output_directory = os.path.abspath(
            os.path.dirname(PDF_REPORT_PATH)
            or "outputs"
        )

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

        os.startfile(output_directory)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Meeting Summarizer")
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
