# Audiobook Generator

Convert books to audiobooks using Qwen3-TTS and LangGraph.

## Features

- **Multi-format Support**: RTF, TXT, Markdown, HTML, DOCX
- **Smart Chapter Detection**: Automatically detects chapter boundaries
- **Semantic Chunking**: Respects sentence and paragraph boundaries for natural speech
- **TTS Tagging Preprocess**: Adds Qwen3-TTS delivery tags before synthesis
- **Quality Verification**: Uses faster-whisper STT to verify audio quality
- **LangGraph Workflow**: Sophisticated pipeline with QA agents and retry logic
- **Resumption Support**: Checkpoint-based resumption for interrupted generations
- **Apple Silicon Optimized**: Works on MPS (M1/M2/M3 Macs)
- **Parallel Generation**: Utilizes multi-core CPUs/GPUs for faster generation

## Installation

### Using Conda (Recommended)

```bash
cd audiobook_generator
conda env create -f environment.yml
conda activate audiobook-generator
pip install -e .
```

### Manual Installation

```bash
pip install -e .
```

### Prerequisites

1. **Ollama** (for QA agents):
   ```bash
   # Install Ollama
   brew install ollama  # macOS

   # Start Ollama service
   ollama serve

   # Pull llama3 model
   ollama pull llama3
   ```

2. **Qwen3-TTS** models will be downloaded automatically on first run.

## Usage

### Basic Usage

```bash
# Generate audiobook from a file
audiobook-generator generate mybook.rtf

# Specify output directory and voice
audiobook-generator generate mybook.txt -o ./my_audiobook -v Ryan

# Resume from checkpoint
audiobook-generator generate mybook.txt --resume checkpoint.json

# High-performance parallel generation (Recommended for M1/M2/M3 Max/Ultra)
audiobook-generator generate mybook.rtf --workers 10 --chunk-size 1000
```

### Commands

```bash
# Generate audiobook
audiobook-generator generate <input_file> [options]

# Analyze file without generating audio
audiobook-generator info <input_file>

# Resume from checkpoint
audiobook-generator resume <checkpoint_file>

# List available voices
audiobook-generator voices
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output directory |
| `--voice` | `-v` | TTS voice (Ryan, Aiden, Vivian, Serena) |
| `--language` | `-l` | Language for TTS |
| `--chunk-size` | `-c` | Max characters per chunk (default: 500, recommended: 500-1000) |
| `--workers` | `-w` | Number of parallel generation workers (default: 1) |
| `--resume` | `-r` | Resume from checkpoint file |
| `--skip-qa` | | Skip QA verification steps |
| `--dry-run` | | Convert and split only, no audio |

### Available Voices

| Voice | Gender | Languages | Best For |
|-------|--------|-----------|----------|
| Ryan | Male | English | Audiobooks, narration (recommended) |
| Aiden | Male | English | Audiobooks, casual reading |
| Vivian | Female | Chinese, English | Multilingual content |
| Serena | Female | Chinese, English | Multilingual content |

## Workflow Architecture

The generator uses a LangGraph StateGraph with the following stages:

```
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│   Convert    │────▶│  Conversion QA    │────▶│     Split       │
│  (RTF→MD)    │     │ (LLM validation)  │     │ (chapters)      │
└──────────────┘     └───────────────────┘     └─────────────────┘
                              │ retry                    │
                              └──────────────────────────┘
                                                         │
                     ┌───────────────────────────────────┘
                     ▼
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│   Split QA   │────▶│      Clean        │────▶│     Chunk       │
│ (validation) │     │ (sanitize text)   │     │ (for TTS)       │
└──────────────┘     └───────────────────┘     └─────────────────┘
       │ retry                                          │
       └────────────────────────────────────────────────┘
                                                        │
                     ┌──────────────────────────────────┘
                     ▼
┌───────────────────┐
│  TTS Preprocess   │
│ (Qwen3 tags)      │
└───────────────────┘
                      │
                      ▼
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│   Generate   │────▶│     Verify        │────▶│    Audio QA     │
│ (Qwen3-TTS)  │     │ (faster-whisper)  │     │ (validation)    │
└──────────────┘     └───────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │    Complete     │
                                               │  (output files) │
                                               └─────────────────┘
```

## Output Structure

```
audiobook_output/
├── audio/
│   ├── chapter_001_Introduction.wav
│   ├── chapter_002_The_Beginning.wav
│   └── ...
├── checkpoint.json    # For resumption
└── metadata.json      # Book metadata
```

## Programmatic Usage

```python
from audiobook_generator.graph import run_audiobook_workflow

result = run_audiobook_workflow(
    input_file="mybook.rtf",
    output_dir="./output",
    voice="Ryan",
    language="English",
    language="English",
    chunk_size=500,
    workers=4,
)

print(f"Status: {result.stage}")
print(f"Chapters: {len(result.chapters)}")
```

## Qwen3-TTS Tag Configuration

The preprocessing step uses a strict allowlist of Qwen3-TTS tags stored in
`qwen3_tts_tags.json`. Add your approved tag literals or regex patterns there
before running the workflow. Tags not in the allowlist are stripped. See
`QWEN3_TTS_TAGS_PATH` in `.env.audiobook` to override the file location.

## Troubleshooting

### "Ollama not found"
Make sure Ollama is installed and running:
```bash
ollama serve
```

### "CUDA out of memory"
The generator is optimized for Apple Silicon (MPS). For CUDA, you may need to:
- Reduce chunk size: `--chunk-size 300`
- Use a smaller model variant

### "Low verification scores"
This usually indicates:
- Text with many proper nouns or technical terms
- Non-English text with English language setting
- Audio generation issues

Try adjusting the `--chunk-size` or checking your text for unusual characters.

## License

MIT License
