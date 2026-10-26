from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QCoreApplication, QProcess, QProcessEnvironment, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
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

from asrgui.paths import find_resource_root, workspace_paths


class M0Window(QMainWindow):
    def __init__(self, *, smoke_result: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("asrGUI M0")
        self.resize(980, 640)
        self.smoke_result = smoke_result
        self.smoke_finished = False
        self.stdout_buffer = ""
        self.worker_ready = False
        self.selected_audio: Path | None = None
        self.request_counter = 0

        central = QWidget()
        layout = QVBoxLayout(central)
        title = QLabel("asrGUI M0 离线转写验证")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(title)

        file_row = QHBoxLayout()
        self.file_label = QLabel("尚未选择音频")
        choose_button = QPushButton("选择音频")
        choose_button.clicked.connect(self.choose_audio)
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(choose_button)
        layout.addLayout(file_row)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("开始离线转写")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_transcription)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_transcription)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.cancel_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("worker 事件将显示在这里")
        layout.addWidget(self.log, 1)
        self.setCentralWidget(central)

        self.worker = QProcess(self)
        self.worker.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.worker.readyReadStandardOutput.connect(self.read_stdout)
        self.worker.readyReadStandardError.connect(self.read_stderr)
        self.worker.errorOccurred.connect(self.worker_error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.started.connect(self.worker_started)
        self.start_worker()

    def worker_command(self) -> tuple[str, list[str]]:
        app_dir = Path(QCoreApplication.applicationDirPath())
        worker = app_dir / "asr-worker"
        if getattr(sys, "frozen", False) and worker.is_file():
            return str(worker), ["--worker"]
        if getattr(sys, "frozen", False):
            return sys.executable, ["--worker"]
        return sys.executable, ["-m", "asrgui.m0.worker"]

    def start_worker(self) -> None:
        program, arguments = self.worker_command()
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("ASRGUI_DATA_ROOT", str(workspace_paths().data_root))
        if getattr(sys, "frozen", False):
            environment.insert(
                "ASRGUI_MODELS_ROOT", str(find_resource_root() / "models")
            )
        self.worker.setProcessEnvironment(environment)
        self.log.appendPlainText(f"启动 worker: {program}")
        self.worker.start(program, arguments)

    def next_request_id(self, prefix: str) -> str:
        self.request_counter += 1
        return f"{prefix}-{self.request_counter}"

    def send_request(
        self, request_id: str, request_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        request = {
            "id": request_id,
            "type": request_type,
            "payload": payload or {},
        }
        self.worker.write(
            (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
        )

    def worker_started(self) -> None:
        self.log.appendPlainText("worker 进程已启动，等待 ready 事件")

    def on_worker_ready(self) -> None:
        self.worker_ready = True
        self.start_button.setEnabled(self.selected_audio is not None)
        if self.smoke_result is not None:
            self.send_request("smoke-1", "smoke")

    def choose_audio(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择会议音频",
            str(workspace_paths().project_root / "Assets" / "audio"),
            "Audio (*.wav *.mp3 *.m4a *.aac *.flac);;All Files (*)",
        )
        if not filename:
            return
        self.selected_audio = Path(filename)
        self.file_label.setText(str(self.selected_audio))
        self.start_button.setEnabled(self.worker_ready)

    def start_transcription(self) -> None:
        if not self.selected_audio:
            return
        request_id = self.next_request_id("transcribe")
        self.progress.setValue(0)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.send_request(
            request_id,
            "transcribe",
            {"input": str(self.selected_audio), "chunk_seconds": 300},
        )

    def cancel_transcription(self) -> None:
        request_id = self.next_request_id("cancel")
        self.send_request(request_id, "cancel")
        self.cancel_button.setEnabled(False)

    def read_stdout(self) -> None:
        text = bytes(self.worker.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.stdout_buffer += text
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.log.appendPlainText(f"非协议输出: {line}")
                continue
            self.handle_event(event)

    def read_stderr(self) -> None:
        text = bytes(self.worker.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self.log.appendPlainText(text.rstrip())

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event")
        if event_type == "ready":
            self.on_worker_ready()
            return
        if event_type == "progress":
            if isinstance(event.get("percent"), (int, float)):
                self.progress.setValue(int(event["percent"]))
            message = event.get("message") or event.get("stage") or "处理中"
            self.log.appendPlainText(str(message))
            return
        if event_type == "cancel_requested":
            self.log.appendPlainText(str(event.get("message") or "已请求取消"))
            return
        if event_type == "completed":
            self.progress.setValue(100)
            self.start_button.setEnabled(self.selected_audio is not None)
            self.cancel_button.setEnabled(False)
            self.log.appendPlainText("任务完成")
            if event.get("id") == "smoke-1":
                self.finish_smoke(event.get("result") or {})
            return
        if event_type == "cancelled":
            self.start_button.setEnabled(self.selected_audio is not None)
            self.cancel_button.setEnabled(False)
            self.log.appendPlainText("任务已取消，可从检查点续跑")
            return
        if event_type == "failed":
            self.start_button.setEnabled(self.selected_audio is not None)
            self.cancel_button.setEnabled(False)
            message = event.get("message") or "worker 失败"
            self.log.appendPlainText(f"失败: {message}")
            if self.smoke_result is None:
                QMessageBox.critical(self, "asrGUI M0", message)

    def finish_smoke(self, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if self.smoke_result:
            self.smoke_result.parent.mkdir(parents=True, exist_ok=True)
            self.smoke_result.write_text(text, encoding="utf-8")
        print(text, end="")
        self.smoke_finished = True
        self.worker.write(b'{"id":"shutdown","type":"shutdown"}\n')
        QTimer.singleShot(100, QApplication.instance().quit)

    def worker_error(self, error: QProcess.ProcessError) -> None:
        self.log.appendPlainText(f"worker QProcess 错误: {error.name}")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

    def worker_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.worker_ready = False
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.log.appendPlainText(
            f"worker 已退出: exit_code={exit_code}, status={exit_status.name}"
        )
        if self.smoke_result is not None and not self.smoke_finished:
            QTimer.singleShot(0, lambda: QApplication.exit(1))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker.state() != QProcess.ProcessState.NotRunning:
            self.worker.write(b'{"id":"shutdown","type":"shutdown"}\n')
            self.worker.waitForFinished(1500)
            if self.worker.state() != QProcess.ProcessState.NotRunning:
                self.worker.kill()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asrgui-m0-ui")
    parser.add_argument("--smoke-result", type=Path)
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    window = M0Window(smoke_result=args.smoke_result)
    if args.smoke_result is None:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
