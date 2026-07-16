"""
Batch Processing Tab for Audiobook Generator
Allows processing multiple text files with progress tracking.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QFileDialog, QComboBox, QGroupBox, QTextEdit,
    QCheckBox, QMessageBox, QSizePolicy, QListWidget, QListWidgetItem,
    QMenu, QAction, QSplitter, QSpinBox, QFrame
)
from qtpy.QtCore import Qt, QThread, Signal, QTimer
from qtpy.QtGui import QPalette, QColor, QIcon


class BatchFile:
    """Represents a file in the batch processing queue."""
    
    def __init__(self, file_path: str, voice_path: Optional[str] = None):
        self.file_path = file_path
        self.voice_path = voice_path  # None = use default voice
        self.status = "queued"  # queued, processing, completed, failed
        self.current_chunk = 0
        self.total_chunks = 0
        self.error_message = None
        self.output_path = None
        self.processing_time = 0
        self.audio_duration = 0
        self.realtime_factor = 0
    
    @property
    def filename(self) -> str:
        return Path(self.file_path).name
    
    @property
    def progress_percent(self) -> float:
        if self.total_chunks == 0:
            return 0
        return (self.current_chunk / self.total_chunks) * 100
    
    @property
    def status_icon(self) -> str:
        icons = {
            "queued": "⏸",
            "processing": "⏳",
            "completed": "✓",
            "failed": "✗"
        }
        return icons.get(self.status, "?")


class BatchGenerationThread(QThread):
    """Thread for batch audiobook generation."""
    
    # Signals
    progress = Signal(dict)  # Overall progress updates
    file_progress = Signal(dict)  # Current file progress
    file_started = Signal(str)  # File path
    file_completed = Signal(str, dict)  # File path, result
    file_failed = Signal(str, str)  # File path, error message
    batch_completed = Signal(dict)  # Overall batch results
    
    def __init__(self, files: List[BatchFile], config, params: dict, 
                 max_workers_override: int = None):
        super().__init__()
        self.files = files
        self.config = config
        self.params = params
        self.max_workers_override = max_workers_override
        self.generator = None
        self._is_paused = False
        self._is_stopped = False
    
    def run(self):
        """Process all files in the batch."""
        from pocket_tts.audiobook.generator import AudiobookGenerator
        from pocket_tts.preprocessing.structure_detector import StructureDetector
        from pocket_tts.preprocessing.chunker import SmartChunker
        from pocket_tts.preprocessing.emotion_analyzer import EmotionAnalyzer
        from pocket_tts.preprocessing.parameter_mapper import ParameterMapper
        from pocket_tts.preprocessing.schema import BoundaryType
        
        start_time = time.time()
        completed = 0
        failed = 0
        
        for i, batch_file in enumerate(self.files):
            if self._is_stopped:
                break
            
            # Wait if paused
            while self._is_paused and not self._is_stopped:
                self.msleep(100)
            
            if self._is_stopped:
                break
            
            # Update status
            batch_file.status = "processing"
            batch_file.current_chunk = 0
            batch_file.total_chunks = 0
            
            self.file_started.emit(batch_file.file_path)
            self.progress.emit({
                'current_file': i + 1,
                'total_files': len(self.files),
                'current_filename': batch_file.filename,
                'status': 'processing'
            })
            
            try:
                result = self._process_file(batch_file)
                
                if result['success']:
                    batch_file.status = "completed"
                    batch_file.output_path = result.get('output_path')
                    batch_file.processing_time = result.get('processing_time', 0)
                    batch_file.audio_duration = result.get('audio_duration', 0)
                    batch_file.realtime_factor = result.get('realtime_factor', 0)
                    completed += 1
                    self.file_completed.emit(batch_file.file_path, result)
                else:
                    batch_file.status = "failed"
                    batch_file.error_message = result.get('reason', 'Unknown error')
                    failed += 1
                    self.file_failed.emit(batch_file.file_path, batch_file.error_message)
            
            except Exception as e:
                batch_file.status = "failed"
                batch_file.error_message = str(e)
                failed += 1
                self.file_failed.emit(batch_file.file_path, str(e))
        
        # Final results
        total_time = time.time() - start_time
        self.batch_completed.emit({
            'total_files': len(self.files),
            'completed': completed,
            'failed': failed,
            'total_time': total_time
        })
    
    def _process_file(self, batch_file: BatchFile) -> Dict[str, Any]:
        """Process a single file in the batch."""
        from pocket_tts.audiobook.generator import AudiobookGenerator
        from pocket_tts.preprocessing.structure_detector import StructureDetector
        from pocket_tts.preprocessing.chunker import SmartChunker
        from pocket_tts.preprocessing.emotion_analyzer import EmotionAnalyzer
        from pocket_tts.preprocessing.parameter_mapper import ParameterMapper
        from pocket_tts.preprocessing.schema import BoundaryType
        
        # Read text file
        with open(batch_file.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Get voice path
        voice_path = batch_file.voice_path or self.params.get('voice_path')
        if not voice_path:
            return {'success': False, 'reason': 'No voice specified'}
        
        # Preprocess text
        detector = StructureDetector()
        chunker = SmartChunker(
            mode=self.params.get('chunking_mode', 'smart'),
            min_words=self.params.get('min_words', 35),
            max_words=self.params.get('max_words', 175),
            respect_boundaries=True
        )
        analyzer = EmotionAnalyzer()
        
        # Get pause settings
        pi_enabled = self.params.get('pause_injection_enabled', False)
        if pi_enabled:
            sentence_pause_ms = 0
            paragraph_pause_ms = 0
            chapter_pause_ms = 0
        else:
            sentence_pause_ms = self.params.get('sentence_pause_ms', 500)
            paragraph_pause_ms = self.params.get('paragraph_pause_ms', 1000)
            chapter_pause_ms = self.params.get('chapter_pause_ms', 2000)
        
        boundary_pauses = {
            BoundaryType.SENTENCE_END: sentence_pause_ms,
            BoundaryType.PARAGRAPH_BREAK: paragraph_pause_ms,
            BoundaryType.CHAPTER_START: chapter_pause_ms
        }
        
        mapper = ParameterMapper(
            config=self.config,
            boundary_pauses=boundary_pauses,
            base_temperature=self.params.get('temperature', 0.8),
            base_eos_threshold=self.params.get('eos_threshold', 0.5),
            base_frames_after_eos=self.params.get('frames_after_eos', 30)
        )
        
        # Process text
        structure = detector.analyze(text)
        chunks = chunker.chunk(structure)
        
        if not chunks:
            return {'success': False, 'reason': 'No chunks generated from text'}
        
        # Analyze emotions
        texts_to_analyze = [chunk.text for chunk in chunks]
        emotion_results = analyzer.analyze_batch(texts_to_analyze)
        
        # Map emotions to parameters
        for chunk, emotion in zip(chunks, emotion_results):
            params = mapper.calculate_params(
                emotion=emotion['emotion'],
                punctuation=chunk.punctuation,
                boundary_type=chunk.boundary_type,
                word_count=chunk.word_count,
                emotion_scores=emotion['scores']
            )
            
            silence_duration_ms = mapper.calculate_silence_duration_ms(chunk.boundary_type)
            silence_duration_sec = silence_duration_ms / 1000.0
            
            chunk.tts_params = {
                'temperature': params.temperature,
                'eos_threshold': params.eos_threshold,
                'frames_after_eos': params.frames_after_eos,
                'speed_factor': params.speed_factor,
                'lsd_decode_steps': params.lsd_decode_steps
            }
            chunk.post_process = {'silence_duration': silence_duration_sec}
        
        batch_file.total_chunks = len(chunks)
        
        # Create generator
        generator = AudiobookGenerator(config=self.config)
        generator._pause_injection_enabled = pi_enabled
        generator._pause_durations = self.params.get('pause_durations', {})
        
        # Generate output path
        dataset_paths = AudiobookGenerator.generate_output_paths(batch_file.file_path, voice_path)
        output_path = str(dataset_paths['final_audio_path'])
        
        # Progress callback for this file
        def progress_callback(progress_data):
            batch_file.current_chunk = progress_data.get('current_chunk', 0)
            self.file_progress.emit({
                'file_path': batch_file.file_path,
                'current_chunk': batch_file.current_chunk,
                'total_chunks': batch_file.total_chunks,
                'elapsed_seconds': progress_data.get('elapsed_seconds', 0),
                'eta_seconds': progress_data.get('eta_seconds', 0)
            })
        
        # Generate audiobook
        result = generator.generate_audiobook(
            chunks=chunks,
            voice_path=voice_path,
            output_path=output_path,
            progress_callback=progress_callback,
            source_file=batch_file.file_path,
            save_dataset_chunks=True
        )
        
        return result
    
    def pause(self):
        self._is_paused = True
    
    def resume(self):
        self._is_paused = False
    
    def stop(self):
        self._is_stopped = True
        if self.generator:
            self.generator.cancel_generation()


class BatchTab(QWidget):
    """Batch Processing tab for processing multiple audiobook files."""
    
    def __init__(self, config=None, main_window=None):
        super().__init__()
        self.config = config
        self.main_window = main_window  # Reference to main window for getting parameters
        self.batch_files: List[BatchFile] = []
        self.generation_thread: Optional[BatchGenerationThread] = None
        self.start_time = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the batch processing UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Create main splitter for file list and details
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: File list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # File list group
        file_group = QGroupBox("Files to Process")
        file_layout = QVBoxLayout(file_group)
        
        # File list widget
        self.file_list_widget = QListWidget()
        self.file_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list_widget.customContextMenuRequested.connect(self.show_context_menu)
        file_layout.addWidget(self.file_list_widget)
        
        # File list buttons
        file_buttons_layout = QHBoxLayout()
        
        self.add_files_btn = QPushButton("Add Files")
        self.add_files_btn.clicked.connect(self.add_files)
        file_buttons_layout.addWidget(self.add_files_btn)
        
        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.clicked.connect(self.add_folder)
        file_buttons_layout.addWidget(self.add_folder_btn)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_selected)
        file_buttons_layout.addWidget(self.remove_btn)
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        file_buttons_layout.addWidget(self.clear_btn)
        
        file_layout.addLayout(file_buttons_layout)
        
        # Move buttons
        move_layout = QHBoxLayout()
        
        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_up)
        move_layout.addWidget(self.move_up_btn)
        
        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_down)
        move_layout.addWidget(self.move_down_btn)
        
        file_layout.addLayout(move_layout)
        
        left_layout.addWidget(file_group)
        
        # Voice settings
        voice_group = QGroupBox("Default Voice (for all files)")
        voice_layout = QHBoxLayout(voice_group)
        
        voice_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems([
            "alba", "brandon", "claire", "daniel",
            "evan", "freya", "grant", "lauren"
        ])
        self.voice_combo.setMinimumWidth(150)
        voice_layout.addWidget(self.voice_combo)
        
        self.custom_voice_btn = QPushButton("Custom WAV...")
        self.custom_voice_btn.clicked.connect(self.select_custom_voice)
        voice_layout.addWidget(self.custom_voice_btn)
        
        left_layout.addWidget(voice_group)
        
        splitter.addWidget(left_widget)
        
        # Right side: Progress and controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Overall progress
        progress_group = QGroupBox("Overall Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # File progress
        file_progress_layout = QHBoxLayout()
        file_progress_layout.addWidget(QLabel("File:"))
        self.file_progress_label = QLabel("0 / 0")
        file_progress_layout.addWidget(self.file_progress_label)
        progress_layout.addLayout(file_progress_layout)
        
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setValue(0)
        progress_layout.addWidget(self.overall_progress_bar)
        
        # Current file progress
        current_file_layout = QHBoxLayout()
        current_file_layout.addWidget(QLabel("Current:"))
        self.current_file_label = QLabel("None")
        current_file_layout.addWidget(self.current_file_label)
        progress_layout.addLayout(current_file_layout)
        
        self.current_progress_bar = QProgressBar()
        self.current_progress_bar.setValue(0)
        progress_layout.addWidget(self.current_progress_bar)
        
        # Stats
        stats_layout = QHBoxLayout()
        
        stats_layout.addWidget(QLabel("Elapsed:"))
        self.elapsed_label = QLabel("0:00")
        stats_layout.addWidget(self.elapsed_label)
        
        stats_layout.addWidget(QLabel("ETA:"))
        self.eta_label = QLabel("--:--")
        stats_layout.addWidget(self.eta_label)
        
        progress_layout.addLayout(stats_layout)
        
        right_layout.addWidget(progress_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Batch")
        self.start_btn.clicked.connect(self.start_batch)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px; }")
        controls_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_batch)
        self.pause_btn.setEnabled(False)
        controls_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_batch)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        controls_layout.addWidget(self.stop_btn)
        
        right_layout.addLayout(controls_layout)
        
        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: Consolas, monospace; }")
        log_layout.addWidget(self.log_text)
        
        right_layout.addWidget(log_group)
        
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
    
    def add_files(self):
        """Add text files to the batch."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Text Files",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if files:
            for file_path in files:
                # Check if already added
                if not any(f.file_path == file_path for f in self.batch_files):
                    batch_file = BatchFile(file_path)
                    self.batch_files.append(batch_file)
                    self._update_list_widget()
            
            self.log_message(f"Added {len(files)} file(s)")
    
    def add_folder(self):
        """Add all .txt files from a folder recursively."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            ""
        )
        
        if folder:
            count = 0
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith('.txt'):
                        file_path = os.path.join(root, file)
                        if not any(f.file_path == file_path for f in self.batch_files):
                            batch_file = BatchFile(file_path)
                            self.batch_files.append(batch_file)
                            count += 1
            
            self._update_list_widget()
            self.log_message(f"Added {count} file(s) from folder")
    
    def remove_selected(self):
        """Remove selected files from the batch."""
        selected = self.file_list_widget.currentRow()
        if selected >= 0 and selected < len(self.batch_files):
            removed = self.batch_files.pop(selected)
            self._update_list_widget()
            self.log_message(f"Removed: {removed.filename}")
    
    def clear_all(self):
        """Clear all files from the batch."""
        if self.batch_files:
            reply = QMessageBox.question(
                self,
                "Clear All",
                "Remove all files from the batch?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.batch_files.clear()
                self._update_list_widget()
                self.log_message("Cleared all files")
    
    def move_up(self):
        """Move selected file up in the list."""
        selected = self.file_list_widget.currentRow()
        if selected > 0:
            self.batch_files[selected], self.batch_files[selected - 1] = \
                self.batch_files[selected - 1], self.batch_files[selected]
            self._update_list_widget()
            self.file_list_widget.setCurrentRow(selected - 1)
    
    def move_down(self):
        """Move selected file down in the list."""
        selected = self.file_list_widget.currentRow()
        if selected >= 0 and selected < len(self.batch_files) - 1:
            self.batch_files[selected], self.batch_files[selected + 1] = \
                self.batch_files[selected + 1], self.batch_files[selected]
            self._update_list_widget()
            self.file_list_widget.setCurrentRow(selected + 1)
    
    def select_custom_voice(self):
        """Select a custom voice WAV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Voice WAV File",
            "",
            "WAV Files (*.wav);;All Files (*)"
        )
        
        if file_path:
            # Add custom voice to combo if not already there
            voice_name = f"Custom: {Path(file_path).name}"
            if self.voice_combo.findText(voice_name) == -1:
                self.voice_combo.addItem(voice_name, file_path)
            
            self.voice_combo.setCurrentText(voice_name)
            self.log_message(f"Selected custom voice: {Path(file_path).name}")
    
    def show_context_menu(self, pos):
        """Show context menu for file list."""
        item = self.file_list_widget.itemAt(pos)
        if not item:
            return
        
        row = self.file_list_widget.row(item)
        if row < 0 or row >= len(self.batch_files):
            return
        
        menu = QMenu(self)
        
        # Set voice submenu
        voice_menu = menu.addMenu("Set Voice")
        
        # Built-in voices
        for voice in ["alba", "brandon", "claire", "daniel", "evan", "freya", "grant", "lauren"]:
            action = voice_menu.addAction(voice)
            action.triggered.connect(lambda checked, v=voice, r=row: self.set_file_voice(r, v))
        
        # Custom voice
        voice_menu.addSeparator()
        custom_action = voice_menu.addAction("Custom WAV...")
        custom_action.triggered.connect(lambda checked, r=row: self.select_file_custom_voice(r))
        
        # Reset to default
        voice_menu.addSeparator()
        reset_action = voice_menu.addAction("Use Default Voice")
        reset_action.triggered.connect(lambda checked, r=row: self.set_file_voice(r, None))
        
        menu.exec_(self.file_list_widget.mapToGlobal(pos))
    
    def set_file_voice(self, row: int, voice: Optional[str]):
        """Set voice for a specific file."""
        if 0 <= row < len(self.batch_files):
            self.batch_files[row].voice_path = voice
            self._update_list_widget()
            self.log_message(f"Set voice for {self.batch_files[row].filename}: {voice or 'default'}")
    
    def select_file_custom_voice(self, row: int):
        """Select custom voice for a specific file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Voice WAV File",
            "",
            "WAV Files (*.wav);;All Files (*)"
        )
        
        if file_path and 0 <= row < len(self.batch_files):
            self.batch_files[row].voice_path = file_path
            self._update_list_widget()
            self.log_message(f"Set custom voice for {self.batch_files[row].filename}: {Path(file_path).name}")
    
    def _update_list_widget(self):
        """Update the list widget to reflect batch_files."""
        self.file_list_widget.clear()
        
        for batch_file in self.batch_files:
            # Build display text
            voice_text = ""
            if batch_file.voice_path:
                if os.path.isfile(batch_file.voice_path):
                    voice_text = f" [{Path(batch_file.voice_path).stem}]"
                else:
                    voice_text = f" [{batch_file.voice_path}]"
            else:
                voice_text = " [default]"
            
            status_text = ""
            if batch_file.status == "processing":
                status_text = f" - {batch_file.current_chunk}/{batch_file.total_chunks}"
            elif batch_file.status == "completed":
                status_text = f" ✓ ({batch_file.realtime_factor:.1f}x)"
            elif batch_file.status == "failed":
                status_text = f" ✗ {batch_file.error_message[:30]}"
            
            display_text = f"{batch_file.status_icon} {batch_file.filename}{voice_text}{status_text}"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, batch_file)
            self.file_list_widget.addItem(item)
    
    def start_batch(self):
        """Start batch processing."""
        if not self.batch_files:
            QMessageBox.warning(self, "No Files", "Please add files to process.")
            return
        
        if self.generation_thread and self.generation_thread.isRunning():
            QMessageBox.warning(self, "Already Running", "Batch processing is already in progress.")
            return
        
        # Get voice selection
        voice_selection = self.voice_combo.currentText()
        if voice_selection.startswith("Custom:"):
            voice_path = self.voice_combo.currentData()
            if voice_path is None:
                voice_path = voice_selection.replace("Custom: ", "")
        else:
            voice_path = voice_selection
        
        # Get parameters from main window if available
        params = self._get_parameters_from_main_window(voice_path)
        
        # Reset file statuses
        for batch_file in self.batch_files:
            batch_file.status = "queued"
            batch_file.current_chunk = 0
            batch_file.total_chunks = 0
            batch_file.error_message = None
        
        self._update_list_widget()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.overall_progress_bar.setValue(0)
        self.current_progress_bar.setValue(0)
        self.file_progress_label.setText(f"0 / {len(self.batch_files)}")
        self.current_file_label.setText("None")
        
        self.start_time = time.time()
        
        # Create and start thread
        self.generation_thread = BatchGenerationThread(
            files=self.batch_files,
            config=self.config,
            params=params
        )
        
        self.generation_thread.progress.connect(self.on_overall_progress)
        self.generation_thread.file_progress.connect(self.on_file_progress)
        self.generation_thread.file_started.connect(self.on_file_started)
        self.generation_thread.file_completed.connect(self.on_file_completed)
        self.generation_thread.file_failed.connect(self.on_file_failed)
        self.generation_thread.batch_completed.connect(self.on_batch_completed)
        
        self.generation_thread.start()
        
        self.log_message("Started batch processing")
    
    def _get_parameters_from_main_window(self, voice_path: str) -> dict:
        """Get parameters from main window's Generate tab."""
        params = {
            'voice_path': voice_path,
        }
        
        if self.main_window:
            # Get all parameters from main window
            params.update({
                'chunking_mode': self.main_window.chunking_mode_combo.currentText(),
                'min_words': self.main_window.min_words_spin.value(),
                'max_words': self.main_window.max_words_spin.value(),
                'temperature': self.main_window.temperature_spin.value(),
                'eos_threshold': self.main_window.eos_threshold_spin.value(),
                'frames_after_eos': self.main_window.frames_after_eos_spin.value(),
                'lsd_steps': self.main_window.lsd_steps_spin.value(),
                'pause_injection_enabled': self.main_window.pause_injection_check.isChecked(),
                'pause_durations': {
                    punct: spin.value() for punct, spin in self.main_window._pause_spinners.items()
                },
            })
            
            # Get pause durations (convert ms to frames)
            ms_to_frames = lambda ms: int((ms / 1000) * 24000)
            
            if params['pause_injection_enabled']:
                params['sentence_pause_ms'] = 0
                params['paragraph_pause_ms'] = 0
                params['chapter_pause_ms'] = 0
            else:
                params['sentence_pause_ms'] = self.main_window.sentence_pause_spin.value()
                params['paragraph_pause_ms'] = self.main_window.paragraph_pause_spin.value()
                params['chapter_pause_ms'] = self.main_window.chapter_pause_spin.value()
            
            # Update config with device settings
            if not hasattr(self.config, 'device') or not isinstance(self.config.device, dict):
                self.config.device = {}
            self.config.device['preferred'] = self.main_window.device_combo.currentText()
            
            # Update config with M4B settings
            if not hasattr(self.config, 'm4b'):
                self.config.m4b = {}
            self.config.m4b['enabled'] = self.main_window.m4b_enabled_check.isChecked()
            self.config.m4b['normalization_type'] = self.main_window.m4b_norm_combo.currentText()
            
            # Update config with ASR settings
            if not hasattr(self.config, 'asr_quality_control'):
                self.config.asr_quality_control = {}
            self.config.asr_quality_control['enabled'] = self.main_window.asr_enabled_check.isChecked()
            self.config.asr_quality_control['threshold'] = self.main_window.asr_threshold_spin.value()
            self.config.asr_quality_control['max_retries'] = self.main_window.asr_max_retries_spin.value()
            self.config.asr_quality_control['temp_decrement'] = self.main_window.asr_temp_decrement_spin.value()
        
        else:
            # Use default parameters if main window not available
            params.update({
                'chunking_mode': 'smart',
                'min_words': 35,
                'max_words': 175,
                'temperature': 0.8,
                'eos_threshold': 0.5,
                'frames_after_eos': 30,
                'lsd_steps': 10,
                'pause_injection_enabled': False,
                'pause_durations': {},
                'sentence_pause_ms': 500,
                'paragraph_pause_ms': 1000,
                'chapter_pause_ms': 2000,
            })
        
        return params
    
    def pause_batch(self):
        """Pause or resume batch processing."""
        if self.generation_thread:
            if self.generation_thread._is_paused:
                self.generation_thread.resume()
                self.pause_btn.setText("Pause")
                self.log_message("Resumed batch processing")
            else:
                self.generation_thread.pause()
                self.pause_btn.setText("Resume")
                self.log_message("Paused batch processing")
    
    def stop_batch(self):
        """Stop batch processing."""
        if self.generation_thread:
            reply = QMessageBox.question(
                self,
                "Stop Batch",
                "Stop batch processing? Current file will complete.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.generation_thread.stop()
                self.log_message("Stopping batch processing...")
    
    def on_overall_progress(self, data: dict):
        """Handle overall progress update."""
        current_file = data.get('current_file', 0)
        total_files = data.get('total_files', 0)
        
        self.file_progress_label.setText(f"{current_file} / {total_files}")
        self.overall_progress_bar.setValue(int((current_file / total_files) * 100) if total_files > 0 else 0)
    
    def on_file_progress(self, data: dict):
        """Handle current file progress update."""
        current_chunk = data.get('current_chunk', 0)
        total_chunks = data.get('total_chunks', 0)
        eta_seconds = data.get('eta_seconds', 0)
        
        self.current_progress_bar.setValue(int((current_chunk / total_chunks) * 100) if total_chunks > 0 else 0)
        self.current_file_label.setText(f"{current_chunk} / {total_chunks} chunks")
        
        # Update elapsed and ETA
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.elapsed_label.setText(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
        
        if eta_seconds > 0:
            self.eta_label.setText(f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}")
        
        # Update list widget
        self._update_list_widget()
    
    def on_file_started(self, file_path: str):
        """Handle file started."""
        self.log_message(f"Starting: {Path(file_path).name}")
        self._update_list_widget()
    
    def on_file_completed(self, file_path: str, result: dict):
        """Handle file completed."""
        filename = Path(file_path).name
        duration = result.get('audio_duration', 0)
        rtf = result.get('realtime_factor', 0)
        self.log_message(f"Completed: {filename} ({duration:.1f}s audio, {rtf:.1f}x realtime)")
        self._update_list_widget()
    
    def on_file_failed(self, file_path: str, error: str):
        """Handle file failed."""
        filename = Path(file_path).name
        self.log_message(f"Failed: {filename} - {error}")
        self._update_list_widget()
    
    def on_batch_completed(self, results: dict):
        """Handle batch completed."""
        total = results.get('total_files', 0)
        completed = results.get('completed', 0)
        failed = results.get('failed', 0)
        total_time = results.get('total_time', 0)
        
        self.log_message(f"Batch completed: {completed}/{total} files in {total_time:.1f}s")
        
        if failed > 0:
            self.log_message(f"Warning: {failed} file(s) failed")
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        
        self._update_list_widget()
    
    def log_message(self, message: str):
        """Add a message to the log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
