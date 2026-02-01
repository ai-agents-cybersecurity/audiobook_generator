# Audiobook Generator - Quick Start Guide

## 1. Setup Environment

### Option A: Using Conda (Recommended)
```bash
cd audiobook_generator

# Create and activate conda environment
conda create -n audiobook-generator python=3.11 -y
conda activate audiobook-generator

# Install dependencies
pip install langchain langgraph langchain-ollama
pip install striprtf markdownify python-docx beautifulsoup4 lxml
pip install soundfile librosa numpy scipy
pip install faster-whisper
pip install typer rich tqdm pydantic

# Install qwen-tts from parent directory
pip install -e ..

# Install this package
pip install -e .
```

### Option B: Using the setup script
```bash
cd audiobook_generator
./setup.sh
```

## 2. Setup Ollama (for QA Agents)

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Pull the llama3 model (in another terminal)
ollama pull llama3
```

## 3. Generate Your Audiobook

### Basic Usage
```bash
# Generate audiobook from any supported file format
audiobook-generator generate mybook.rtf

# With custom output directory
audiobook-generator generate mybook.txt -o ./my_audiobook

# With different voice
audiobook-generator generate mybook.md -v Aiden

# High Performance (for Apple Silicon Max/Ultra)
# Uses 16 parallel workers and larger chunks for efficiency
audiobook-generator generate mybook.rtf --workers 16 --chunk-size 1000
```

### Analyze a Book First (No Audio Generation)
```bash
# See chapter structure and estimates
audiobook-generator info mybook.rtf
```

### Resume Interrupted Generation
```bash
# If generation was interrupted, resume from checkpoint
audiobook-generator resume ./audiobook_output/checkpoint.json
```

## 4. Test with Sample Book

```bash
# Test conversion logic only (no TTS required)
python test_conversion.py ../mybook/accelerando.rtf

# Full generation test
audiobook-generator generate ../mybook/accelerando.rtf -o ./test_output
```

## 5. Output Structure

After generation, you'll find:
```
audiobook_output/
├── audio/
│   ├── chapter_001_Lobsters.wav
│   ├── chapter_002_Troubadour.wav
│   └── ...
├── checkpoint.json    # Resume point
└── metadata.json      # Book info
```

## Available Voices

| Voice | Gender | Recommended For |
|-------|--------|-----------------|
| **Ryan** | Male | Audiobooks (default, best quality) |
| Aiden | Male | Casual narration |
| Vivian | Female | Multilingual content |
| Serena | Female | Multilingual content |

## Troubleshooting

### "ModuleNotFoundError"
Make sure you've activated the conda environment:
```bash
conda activate audiobook-generator
```

### "Ollama connection refused"
Start the Ollama service:
```bash
ollama serve
```

### Memory Issues on Apple Silicon
The generator is optimized for MPS. If you run into memory issues:
- Reduce workers: `--workers 4` (or 1)
- Reduce chunk size: `--chunk-size 300`
- Process one chapter at a time

### Low Quality Audio
- Try adjusting chunk size (300-600 chars works best)
- Check for unusual characters in your text
- Make sure language setting matches content

## Programmatic Usage

```python
from audiobook_generator.graph import run_audiobook_workflow

result = run_audiobook_workflow(
    input_file="mybook.rtf",
    output_dir="./output",
    voice="Ryan",
    language="English",
)

# Check results
for chapter in result.chapters:
    if chapter.audio_file:
        print(f"Chapter {chapter.number}: {chapter.audio_file}")
```

## Supported File Formats

- **RTF** - Rich Text Format (like your sample book)
- **TXT** - Plain text
- **MD** - Markdown
- **HTML** - Web pages
- **DOCX** - Microsoft Word documents

All formats are converted to clean markdown internally before processing.
