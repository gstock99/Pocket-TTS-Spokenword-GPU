# Pocket-TTS-Spokenword GPU

# GPU-Accelerated Emotion-Driven Audiobook Generator

<img width="1446" height="622" alt="pocket-tts-logo-v2-transparent" src="https://github.com/user-attachments/assets/637b5ed6-831f-4023-9b4c-741be21ab238" />

A GPU-accelerated version of Pocket-TTS-Spokenword-Danneausx / Kyutai's Pocket TTS releases that transforms plain text into emotionally expressive audiobooks. Uses NVIDIA CUDA for significantly faster generation with parallel processing.

**✨ Key Features:**

- **GPU Acceleration**: CUDA support for 6-8x realtime generation speed
- **Parallel Processing**: Multiple GPU workers for chunk-level parallelism
- **Emotion Analysis**: Automatic detection of emotions in text using DistilRoBERTa
- **Smart Chunking**: Intelligent text segmentation based on sentence structure
- **Expressive TTS**: Emotion-aware parameter control for natural voice variation
- **Batch Processing**: Process multiple files with per-file voice selection
- **Voice Cloning**: Custom voice support with emotion preservation
- **GUI Interface**: User-friendly desktop application for easy audiobook creation

Supports Python 3.12+ and requires NVIDIA GPU with CUDA support.

## Requirements

- **NVIDIA GPU** with at least 8 GB VRAM (recommended: 12 GB+)
- **CUDA Toolkit** 12.8+
- **Python** 3.12 or later
- **PyTorch** 2.11+ with CUDA support

## Installation

### Windows (Recommended)

1. Download and extract the release ZIP
2. Run `install.bat` - this will:
   - Set up Python environment
   - Install PyTorch with CUDA support
   - Download model weights from HuggingFace

bash
install.bat


**Important**: You may need to visit [kyutai/pocket-tts · Hugging Face](https://huggingface.co/kyutai/pocket-tts) to accept TOS before downloading the model.

### Manual Installation

bash
# Create virtual environment
python -m venv python\Scripts
python\Scripts\pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements_windows.txt

# Download models
python -c "from huggingface_hub import snapshot_download; snapshot_download('kyutai/pocket-tts')"


### Linux Installation

bash
# Clone repository
git clone https://github.com/gstock99/Pocket-TTS-Spokenword-GPU.git
cd Pocket-TTS-Spokenword-GPU

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA (Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements_windows.txt

# Download AI models
python -c "from huggingface_hub import snapshot_download; snapshot_download('kyutai/pocket-tts')"


**Note:** The `requirements_windows.txt` file works for Linux as well - the name is just a leftover from the original project.

### Launch GUI

bash
# Windows
launch_gui.py

# Or directly
python launch_gui.py

# Linux
python launch_gui.py


### GUI Features

#### Generate Audiobook Tab
- Select text file and voice
- Configure TTS parameters (temperature, chunking, pauses)
- Real-time progress tracking
- Automatic output to `Output/` folder

#### Batch Processing Tab
- Add multiple text files or entire folders
- Set different voices per file
- Overall and per-file progress tracking
- CPU priority control (default: Below Normal)
- Pause/Resume/Stop controls

#### Regenerate Chunks Tab
- Re-generate specific chunks with different settings
- Search and replace individual sections
- Quality control and debugging

### GPU Settings

The application automatically detects your GPU and configures:

- **Worker Count**: Calculated based on VRAM (~1.5 GB per worker)
  - 8 GB VRAM: ~4 workers
  - 12 GB VRAM: ~6 workers
  - 16 GB VRAM: ~9 workers
- **Device Selection**: Auto/CUDA/CPU selector
- **VRAM Management**: Automatic cleanup between files

### Performance Benchmarks

| GPU | Workers | Speed |
|-----|---------|-------|
| RTX 5070 (12 GB) | 6 | 6-8x realtime |
| RTX 4070 (12 GB) | 6 | 5-7x realtime |
| RTX 3080 (10 GB) | 5 | 4-6x realtime |

## Configuration

Configuration is stored in `pocket_tts/config/default_config.yaml`:

yaml
tts_core:
  temperature: 0.8
  eos_threshold: 0.5
  frames_after_eos: 30

chunking:
  mode: smart
  min_words: 35
  max_words: 175

device:
  auto: true
  preferred: cuda

parallel:
  max_workers: 6
  enabled: true


## Output Structure


Output/
  └── <BookTitle>/
      ├── <filename> [<voice>].wav
      ├── <filename> [<voice>].m4b  (if M4B enabled)
      └── TTS/
          ├── audio_chunks/
          │   ├── chunk_00000.wav
          │   └── ...
          ├── text_chunks/
          │   ├── chunk_00000.txt
          │   └── audiobook.chunks.json
          └── *.debug.log


## Voices

Built-in voices:
- **alba** - Female, clear and professional
- **brandon** - Male, deep and authoritative
- **claire** - Female, warm and friendly
- **daniel** - Male, neutral and versatile
- **evan** - Male, young and energetic
- **freya** - Female, soft and gentle
- **grant** - Male, mature and distinguished
- **lauren** - Female, bright and cheerful

Custom voices: Right-click a file in Batch Processing to select a custom WAV file.

## Troubleshooting

### Out of Memory
- Reduce worker count in settings
- Use shorter text chunks
- Close other GPU applications

### Low Speed
- Verify CUDA is enabled (check device selector)
- Update NVIDIA drivers
- Check GPU utilization in Task Manager

### No Audio Output
- Verify voice file exists
- Check output folder permissions
- Review debug log in output directory

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

Based on [Kyutai Pocket-TTS](https://github.com/kyutai-labs/pocket-tts).

## Authors

Manu Orsini*, Simon Rouard*, Gabriel De Marmiesse*, Václav Volhejn, Neil Zeghidour, Alexandre Défossez

*equal contribution

GPU acceleration and batch processing enhancements added for improved performance.
