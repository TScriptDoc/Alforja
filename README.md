# Alforja (Andean Saddlebag) 👝

**Native Fedora Linux graphical manager and sandbox controller for `llama.cpp`.**

Built with **Python**, **GTK4**, and **Libadwaita**, Alforja follows the modern GNOME Human Interface Guidelines (HIG) and is specifically engineered to run large language models on consumer hardware under strict memory constraints (e.g., **Nvidia GTX 1650 with 4GB VRAM**).

---

## Key Features

### 1. The Halter: Granular Daemon Control
- **Full Lifecycle Control:** **Start**, **Stop** (`SIGTERM` with fallback to `SIGKILL`), **Pause** (`SIGSTOP`), and **Resume** (`SIGCONT`).
  - *The Pause feature sends `SIGSTOP` to freeze computation and preserve model weights in VRAM without unloading, instantly resuming when toggled.*
- **Real-Time Terminal Output:** Monospace log console streaming `stdout`/`stderr` non-blocking via `GLib.idle_add` with auto-scroll and clear controls.

### 2. Strict Offline Security & Network Sandbox
- **Physically Isolated:** Launches `llama-server` wrapped in Linux `unshare -r -n` namespaces, eliminating network access and loopback exfiltration.
- **Zero Telemetry:** Strictly offline. Zero analytics, zero background pings, zero auto-updates.

### 3. The Harness: Hardware-Tuned Configuration & Profiles
- **Persistent Profiles:** Automatically saves settings to `~/.config/alforja/config.json`. Includes named presets (*"GTX 1650 (Default 4GB Safe)"*, *"Low VRAM / 4K Context (KV Quantized)"*, *"CPU & GPU Balanced"*).
- **VRAM Optimizations for 4GB GPUs:**
  - **Flash Attention (`-fa`):** Reduces attention matrix memory footprint.
  - **KV Cache Quantization (`-ctk` & `-ctv`):** `q4_0` and `q8_0` cache types halve context VRAM requirements.
  - **Micro-Batch Size (`-ub`):** Smooths peak allocation spikes during prompt processing.
  - **Context Shift (`--ctx-shift`):** Prevents crashes when conversations exceed the context token limit.
- **Dynamic Flag Builder:** Add any custom `llama.cpp` CLI argument (e.g., `--timeout`, `--temp`, `--top-p`) via an interactive key-value editor.

### 4. The Pack: Models & Direct URL Downloader
- **Local Scanner:** Automatically maps `.gguf` weights and `.mmproj` vision adapters located in `~/.local/llama.cpp/models`.
- **One-Click Activation:** Set any model as the active target; automatically links matching vision projectors.
- **Live Downloader:** Direct URL downloader for `.gguf` and `.mmproj` files with smooth progress bars, real-time speed (MB/s), ETA calculations, and atomic `.part` verification.

---

## Installation & Running

### Requirements
- **OS:** Fedora Linux 40+ (or any Linux distribution with GTK4 / Libadwaita)
- **Dependencies:** `python3`, `python3-gobject`, `gtk4`, `libadwaita`, `util-linux` (`unshare`)
- **Backend:** Custom-compiled `llama-server` binary placed at:
  ```bash
  ~/.local/llama.cpp/bin/llama-server
  ```

### Running from Source
Clone the repository and run:
```bash
git clone https://github.com/tiago/Alforja.git
cd Alforja
python3 main.py
```

---

## Packaging

Alforja includes a unified packaging script to generate native `.rpm` and `.deb` installers:

```bash
python3 packaging/build-packages.py
```

The resulting packages will be generated in `dist/`:
- `dist/alforja-0.1.0-1.noarch.rpm` (Fedora / RHEL / openSUSE)
- `dist/alforja_0.1.0_all.deb` (Debian / Ubuntu / Linux Mint)

### Installing the RPM (Fedora)
```bash
sudo dnf install dist/alforja-0.1.0-1.noarch.rpm
```

### Installing the DEB (Debian/Ubuntu)
```bash
sudo apt install ./dist/alforja_0.1.0_all.deb
```

---

## Directory Structure

```text
Alforja/
├── main.py                     # Main application entry point & Libadwaita UI
├── agent.md                    # Project specification & architectural constraints
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── data/
│   ├── io.github.alforja.Alforja.desktop       # Desktop launcher
│   ├── io.github.alforja.Alforja.metainfo.xml  # AppStream metadata
│   └── icons/hicolor/scalable/apps/
│       └── io.github.alforja.Alforja.svg       # Native scalable application icon
├── packaging/
│   ├── build-packages.py       # Automated RPM & DEB packaging tool
│   ├── rpm/
│   │   └── alforja.spec        # RPM spec definition
│   └── deb/
│       └── DEBIAN/control      # Debian control definition
└── dist/                       # Built distribution packages
```

---

## License
MIT License. See [LICENSE](LICENSE) for full details.
