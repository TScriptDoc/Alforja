# Project Alforja - Master Instructions

## Role & Objective
You are an expert Linux desktop application developer specializing in Python, GTK4, and libadwaita. Your goal is to build "Alforja" (Andean saddlebag), a native Fedora Linux graphical manager for `llama.cpp`.

## Target Hardware & Environment
- OS: Fedora Linux (Wayland)
- GPU: Nvidia GTX 1650 (strict 4GB VRAM constraint)
- Backend: Custom compiled `llama-server` binary located at `~/.local/llama.cpp/bin/llama-server`

## Core Architecture & Security Rules
1. **Strictly Offline:** The application must NEVER connect to the internet in the background. Zero telemetry, zero auto-updates, zero silent API pings.
2. **Network Sandbox:** When Alforja launches the `llama-server` subprocess, it MUST wrap the command in `unshare -r -n` to physically isolate the inference engine from the network.
3. **No Bloat:** Rely on Python standard libraries (`subprocess`, `json`, `pathlib`, `urllib.request`, `os`, `signal`) wherever possible.

## Required Features
1. **Granular Daemon Control (The Halter):**
   - State management buttons in the UI: **Start**, **Stop** (sends SIGTERM, fallback to SIGKILL), **Pause** (sends SIGSTOP to freeze the process and VRAM state without unloading the model), and **Resume** (sends SIGCONT).
   - Uses `subprocess.Popen` to manage the daemon lifecycle, track the PID, and route stdout/stderr.
   - A real-time terminal log output view (`Gtk.TextView`) to monitor memory usage, prompt processing, and model hot-swapping.
2. **Comprehensive Server Configuration (The Harness):**
   - A dynamic UI capable of exposing *all* `llama-server` CLI arguments.
   - Core settings (Context Size `-c`, GPU Offloading `-ngl`, Port `--port`, Threads `-t`) must use visual sliders and spinbuttons (`Adw.SpinRow`, `Adw.ActionRow`).
   - An "Advanced/All Arguments" section that acts as a key-value builder for every other `llama.cpp` flag (e.g., `--timeout`, `--batch-size`, `--flash-attn`, `--ctx-shift`).
   - Save and load configuration states to a local JSON profile so the user does not have to rebuild their setup on every launch.
3. **Model & Router Management (The Pack):**
   - A UI to download `.gguf` and `.mmproj` files via direct URLs (network access is strictly scoped to this explicit action).
   - A local directory scanner mapping files in `~/.local/llama.cpp/models`.
   - A UI to generate and edit the `models.ini` file for Router Mode (assigning Vision and Text models for VRAM hot-swapping via `--models-preset`).

## UI Guidelines
- Follow GNOME Human Interface Guidelines (HIG).
- Use `Adw.Application`, `Adw.ApplicationWindow`, and `Adw.PreferencesGroup` for a clean, modern, native Linux aesthetic.
- Build a tabbed or side-pane navigation structure: Dashboard (Controls & Logs), Configuration (Arguments), and Storage (Model downloader/scanner).
- Alforja is strictly the manager/backend controller. Do not build a chat interface.

Please acknowledge these constraints and write the initial `main.py` boilerplate that sets up the GTK4 Libadwaita window, the HeaderBar with the Start/Stop/Pause controls, and the basic tabbed interface.