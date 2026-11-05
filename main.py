#!/usr/bin/env python3
"""
Alforja - Andean saddlebag
Native Fedora Linux graphical manager for llama.cpp.

Constraints & Architecture:
- Strictly Offline: Zero background network calls / telemetry.
- Network Sandbox: Wraps llama-server execution in `unshare -r -n`.
- Hardware-Aware: Optimized for Fedora Linux + Nvidia GTX 1650 (4GB VRAM constraint).
- Pure stdlib + GTK4 / Libadwaita.
"""

import collections
import enum
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Transparently forward to host Python if running inside a container missing GTK4/Adw
try:
    import gi
    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
except (ValueError, ModuleNotFoundError):
    if shutil.which("flatpak-spawn"):
        script_path = os.path.abspath(sys.argv[0])
        args = ["flatpak-spawn", "--host", "python3", script_path] + sys.argv[1:]
        os.execvp("flatpak-spawn", args)
    raise

import cairo
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk, Pango


DEFAULT_BINARY_PATH = Path.home() / ".local" / "llama.cpp" / "bin" / "llama-server"
DEFAULT_MODELS_DIR = Path.home() / ".local" / "llama.cpp" / "models"
CONFIG_DIR = Path.home() / ".config" / "alforja"
CONFIG_FILE = CONFIG_DIR / "config.json"
PROFILES_FILE = CONFIG_DIR / "profiles.json"


DEFAULT_PROFILES = {
    "GTX 1650 (Default 4GB Safe)": {
        "ctx_size": 2048,
        "gpu_layers": 24,
        "threads": max(1, (os.cpu_count() or 4) - 2),
        "port": 8080,
        "flash_attn": "auto",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "ubatch_size": 256,
        "batch_size": 512,
        "parallel": 1,
        "ctx_shift": True,
        "use_unshare": True,
        "custom_args": [],
    },
    "Low VRAM / 4K Context (KV Quantized)": {
        "ctx_size": 4096,
        "gpu_layers": 20,
        "threads": max(1, (os.cpu_count() or 4) - 2),
        "port": 8080,
        "flash_attn": "on",
        "cache_type_k": "q4_0",
        "cache_type_v": "q4_0",
        "ubatch_size": 128,
        "batch_size": 256,
        "parallel": 1,
        "ctx_shift": True,
        "use_unshare": True,
        "custom_args": [],
    },
    "CPU & GPU Balanced": {
        "ctx_size": 2048,
        "gpu_layers": 16,
        "threads": max(1, (os.cpu_count() or 4) - 1),
        "port": 8080,
        "flash_attn": "off",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "ubatch_size": 256,
        "batch_size": 512,
        "parallel": 1,
        "ctx_shift": False,
        "use_unshare": True,
        "custom_args": [],
    },
}


class ProcessState(enum.Enum):
    STOPPED = "Stopped"
    RUNNING = "Running"
    PAUSED = "Paused"


class DaemonManager:
    """Manages the llama-server subprocess lifecycle with network isolation."""

    def __init__(self, binary_path: Path = DEFAULT_BINARY_PATH):
        self.binary_path = binary_path
        self.process: subprocess.Popen | None = None
        self.state: ProcessState = ProcessState.STOPPED
        self.log_callback = None
        self.state_callback = None
        self.tps_callback = None
        self.use_unshare: bool = True
        self._reader_thread: threading.Thread | None = None

    def set_callbacks(self, log_callback, state_callback, tps_callback=None):
        self.log_callback = log_callback
        self.state_callback = state_callback
        self.tps_callback = tps_callback

    def _notify_state(self, new_state: ProcessState):
        self.state = new_state
        if self.state_callback:
            GLib.idle_add(self.state_callback, new_state)

    def _append_log(self, text: str):
        if self.log_callback:
            GLib.idle_add(self.log_callback, text)

    def start(self, args: list[str] | None = None, use_unshare: bool = False):
        if self.state != ProcessState.STOPPED:
            self._append_log("[Alforja] Server is already running or paused.\n")
            return

        if not self.binary_path.is_file():
            self._append_log(
                f"[Alforja Error] Binary not found at: {self.binary_path}\n"
            )
            return

        # Auto-cleanup any stale/lingering external llama-server processes to prevent VRAM or port collisions
        try:
            res = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True)
            if res.returncode == 0:
                pids = [int(p) for p in res.stdout.strip().split() if p.isdigit()]
                for pid in pids:
                    if not self.process or pid != self.process.pid:
                        self._append_log(f"[Alforja] Found lingering llama-server (PID {pid}). Releasing VRAM...\n")
                        try:
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(0.3)
                            os.kill(pid, signal.SIGKILL)
                        except Exception:
                            pass
        except Exception:
            pass

        self.use_unshare = use_unshare

        binary_and_args = [str(self.binary_path)]
        if args:
            binary_and_args.extend(args)

        if use_unshare:
            # Bring up the loopback interface inside the isolated namespace and exec llama-server
            inner_cmd = "ip link set lo up && exec " + " ".join(shlex.quote(c) for c in binary_and_args)
            cmd = ["unshare", "-r", "-n", "sh", "-c", inner_cmd]
        else:
            cmd = list(binary_and_args)

        if shutil.which("stdbuf"):
            cmd = ["stdbuf", "-oL", "-eL"] + cmd

        env = os.environ.copy()
        bin_dir = str(self.binary_path.parent)
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld}" if existing_ld else bin_dir

        try:
            self._append_log(f"[Alforja] Launching: {' '.join(cmd)}\n")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self._notify_state(ProcessState.RUNNING)
            mode_desc = "air-gapped namespace (unshare -n)" if use_unshare else "local loopback (127.0.0.1)"
            self._append_log(
                f"[Alforja] Server started ({mode_desc}, PID: {self.process.pid}).\n"
            )

            self._reader_thread = threading.Thread(
                target=self._stream_logs, daemon=True
            )
            self._reader_thread.start()

        except Exception as exc:
            self._append_log(f"[Alforja Error] Failed to start server: {exc}\n")
            self._notify_state(ProcessState.STOPPED)

    def _stream_logs(self):
        proc = self.process
        if not proc or not proc.stdout:
            return

        for line in proc.stdout:
            self._append_log(line)

            # Check for generation/prompt tokens per second
            if "tokens per second" in line and self.tps_callback:
                m = re.search(r"([\d\.]+)\s+tokens per second", line)
                if m:
                    try:
                        tps_val = float(m.group(1))
                        is_prompt = "prompt eval" in line
                        GLib.idle_add(self.tps_callback, tps_val, is_prompt)
                    except Exception:
                        pass

        proc.wait()
        retcode = proc.returncode
        self._append_log(f"\n[Alforja] Process exited with status code {retcode}\n")
        self.process = None
        self._notify_state(ProcessState.STOPPED)

    def pause(self):
        """Sends SIGSTOP to freeze execution and preserve VRAM allocation."""
        if self.state == ProcessState.RUNNING and self.process and self.process.pid:
            try:
                os.kill(self.process.pid, signal.SIGSTOP)
                self._notify_state(ProcessState.PAUSED)
                self._append_log(
                    f"[Alforja] Sent SIGSTOP to PID {self.process.pid} (Process & VRAM paused).\n"
                )
            except ProcessLookupError:
                self._notify_state(ProcessState.STOPPED)
            except Exception as exc:
                self._append_log(f"[Alforja Error] Failed to pause process: {exc}\n")

    def resume(self):
        """Sends SIGCONT to unfreeze execution."""
        if self.state == ProcessState.PAUSED and self.process and self.process.pid:
            try:
                os.kill(self.process.pid, signal.SIGCONT)
                self._notify_state(ProcessState.RUNNING)
                self._append_log(
                    f"[Alforja] Sent SIGCONT to PID {self.process.pid} (Process resumed).\n"
                )
            except ProcessLookupError:
                self._notify_state(ProcessState.STOPPED)
            except Exception as exc:
                self._append_log(f"[Alforja Error] Failed to resume process: {exc}\n")

    def stop(self):
        """Sends SIGTERM, falling back to SIGKILL if necessary."""
        if not self.process or not self.process.pid:
            self._notify_state(ProcessState.STOPPED)
            return

        pid = self.process.pid
        self._append_log(f"[Alforja] Stopping server (PID: {pid})...\n")

        if self.state == ProcessState.PAUSED:
            try:
                os.kill(pid, signal.SIGCONT)
            except Exception:
                pass

        try:
            self.process.terminate()

            def wait_and_kill():
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._append_log(
                        f"[Alforja] PID {pid} did not respond to SIGTERM. Sending SIGKILL...\n"
                    )
                    try:
                        self.process.kill()
                    except Exception:
                        pass

            threading.Thread(target=wait_and_kill, daemon=True).start()
        except ProcessLookupError:
            self._notify_state(ProcessState.STOPPED)
        except Exception as exc:
            self._append_log(f"[Alforja Error] Error while stopping process: {exc}\n")


class ModelDownloader:
    """Manages downloading .gguf and .mmproj files with progress and cancellation."""

    def __init__(self, dest_dir: Path = DEFAULT_MODELS_DIR):
        self.dest_dir = dest_dir
        self.cancel_event = threading.Event()
        self.is_downloading = False
        self._thread: threading.Thread | None = None

    def start_download(
        self,
        urls: list[str],
        progress_cb,
        log_cb,
        finished_cb,
    ):
        if self.is_downloading:
            return

        self.cancel_event.clear()
        self.is_downloading = True
        self._thread = threading.Thread(
            target=self._worker,
            args=(urls, progress_cb, log_cb, finished_cb),
            daemon=True,
        )
        self._thread.start()

    def cancel(self):
        if self.is_downloading:
            self.cancel_event.set()

    def _worker(self, urls: list[str], progress_cb, log_cb, finished_cb):
        def log(msg: str):
            GLib.idle_add(log_cb, msg + "\n")

        def update_progress(fraction: float, text: str):
            GLib.idle_add(progress_cb, fraction, text)

        self.dest_dir.mkdir(parents=True, exist_ok=True)
        total_items = len(urls)

        try:
            for idx, url in enumerate(urls, start=1):
                if self.cancel_event.is_set():
                    log("[Downloader] Download cancelled by user.")
                    GLib.idle_add(finished_cb, False, "Cancelled")
                    return

                clean_url = url.split("?")[0].rstrip("/")
                filename = clean_url.split("/")[-1]
                if not filename or not (filename.endswith(".gguf") or ".gguf" in filename):
                    filename = f"model_download_{int(time.time())}.gguf"

                dest_file = self.dest_dir / filename
                part_file = self.dest_dir / f"{filename}.part"

                log(f"[Downloader] Item {idx}/{total_items}: Connecting to {url}...")
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Alforja/1.0",
                        "Accept": "*/*",
                    },
                )

                with urllib.request.urlopen(req, timeout=30) as response:
                    content_len_hdr = response.headers.get("Content-Length")
                    total_bytes = int(content_len_hdr) if content_len_hdr else None

                    total_str = (
                        f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"
                        if total_bytes
                        else "Unknown size"
                    )
                    log(f"[Downloader] Target: {filename} ({total_str})")
                    log(f"[Downloader] Saving to: {dest_file}")

                    downloaded = 0
                    chunk_size = 1024 * 1024
                    start_time = time.time()
                    last_log_time = start_time

                    with open(part_file, "wb") as out_f:
                        while True:
                            if self.cancel_event.is_set():
                                log("[Downloader] Download cancelled. Cleaning up temporary files...")
                                part_file.unlink(missing_ok=True)
                                GLib.idle_add(finished_cb, False, "Cancelled")
                                return

                            chunk = response.read(chunk_size)
                            if not chunk:
                                break

                            out_f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()

                            file_fraction = (downloaded / total_bytes) if total_bytes else 0.0
                            overall_fraction = ((idx - 1) + file_fraction) / total_items

                            if now - last_log_time >= 0.5 or not chunk:
                                elapsed = now - start_time
                                speed = downloaded / elapsed if elapsed > 0 else 0
                                speed_mb = speed / (1024 * 1024)
                                downloaded_mb = downloaded / (1024 * 1024)

                                if total_bytes:
                                    total_mb = total_bytes / (1024 * 1024)
                                    pct = (downloaded / total_bytes) * 100.0
                                    eta_sec = int((total_bytes - downloaded) / speed) if speed > 0 else 0
                                    eta_m, eta_s = divmod(eta_sec, 60)
                                    status_text = (
                                        f"File {idx}/{total_items}: {pct:.1f}% "
                                        f"({downloaded_mb:.1f}/{total_mb:.1f} MB) - "
                                        f"{speed_mb:.1f} MB/s - ETA: {eta_m:02d}:{eta_s:02d}"
                                    )
                                    log_line = (
                                        f"[{time.strftime('%H:%M:%S')}] {pct:5.1f}% | "
                                        f"{downloaded_mb:.1f} MB / {total_mb:.1f} MB | "
                                        f"{speed_mb:.2f} MB/s | ETA: {eta_m:02d}m {eta_s:02d}s"
                                    )
                                else:
                                    status_text = f"File {idx}/{total_items}: {downloaded_mb:.1f} MB ({speed_mb:.1f} MB/s)"
                                    log_line = f"[{time.strftime('%H:%M:%S')}] {downloaded_mb:.1f} MB | {speed_mb:.2f} MB/s"

                                update_progress(overall_fraction, status_text)
                                log(log_line)
                                last_log_time = now

                    if part_file.exists():
                        part_file.replace(dest_file)

                log(f"[Downloader] Finished downloading: {filename}\n")

            update_progress(1.0, "All downloads completed successfully!")
            log("[Downloader] Download queue finished. Weights ready in models directory.")
            GLib.idle_add(finished_cb, True, "Complete")

        except Exception as exc:
            log(f"[Downloader Error] {exc}")
            if "part_file" in locals() and part_file.exists():
                part_file.unlink(missing_ok=True)
            GLib.idle_add(finished_cb, False, str(exc))

        finally:
            self.is_downloading = False


class AlforjaWindow(Adw.ApplicationWindow):
    """Main application window following GNOME Human Interface Guidelines."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title("Alforja")
        self.set_default_size(1000, 740)

        self.daemon = DaemonManager()
        self.daemon.set_callbacks(
            self.on_log_message,
            self.on_process_state_changed,
            self.on_tps_recorded,
        )

        self.downloader = ModelDownloader()
        self.active_model_path: Path | None = None
        self.active_mmproj_path: Path | None = None
        self.model_rows: list[Gtk.Widget] = []
        self.custom_arg_rows: list[tuple[Adw.ActionRow, Gtk.Entry, Gtk.Entry]] = []

        self.profiles: dict = {}
        self.active_profile_name = "GTX 1650 (Default 4GB Safe)"
        self._loading_config = False

        # Telemetry state
        self.telemetry_enabled = True
        self.tokens_history = collections.deque(maxlen=40)
        self.current_tps = 0.0
        self.prompt_tps = 0.0
        self._prev_cpu_times: tuple[float, float] | None = None
        self._telemetry_thread: threading.Thread | None = None
        self._stop_telemetry = threading.Event()

        self.setup_ui()
        self.load_profiles_and_config()
        self.init_models()
        self.update_ui_state(ProcessState.STOPPED)

        # Start background telemetry monitor
        self._start_telemetry_worker()

    def setup_ui(self):
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(self.main_box)

        # Header bar
        self.header_bar = Adw.HeaderBar()
        self.main_box.append(self.header_bar)

        # View Switcher in HeaderBar
        self.view_stack = Adw.ViewStack()
        self.view_switcher_title = Adw.ViewSwitcherTitle(
            stack=self.view_stack,
            title="Alforja",
            subtitle="llama.cpp Daemon Manager",
        )
        self.header_bar.set_title_widget(self.view_switcher_title)

        # HeaderBar Halter Controls (Left side)
        halter_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        halter_box.set_valign(Gtk.Align.CENTER)

        self.btn_start = Gtk.Button(
            icon_name="media-playback-start-symbolic",
            tooltip_text="Start llama-server",
        )
        self.btn_start.add_css_class("suggested-action")
        self.btn_start.connect("clicked", self.on_start_clicked)
        halter_box.append(self.btn_start)

        self.btn_pause = Gtk.Button(
            icon_name="media-playback-pause-symbolic",
            tooltip_text="Pause daemon (SIGSTOP - retains VRAM)",
        )
        self.btn_pause.connect("clicked", self.on_pause_clicked)
        halter_box.append(self.btn_pause)

        self.btn_stop = Gtk.Button(
            icon_name="media-playback-stop-symbolic",
            tooltip_text="Stop daemon (SIGTERM / SIGKILL)",
        )
        self.btn_stop.add_css_class("destructive-action")
        self.btn_stop.connect("clicked", self.on_stop_clicked)
        halter_box.append(self.btn_stop)

        # Chat Button: Join/Launch Alpaca inside the same sandbox bubble
        self.btn_alpaca = Gtk.Button(
            icon_name="com.jeffser.Alpaca",
            tooltip_text="Launch Alpaca Chat in Sandbox Bubble",
        )
        self.btn_alpaca.connect("clicked", self.on_launch_alpaca_clicked)
        self.btn_alpaca.set_sensitive(False)
        halter_box.append(self.btn_alpaca)

        self.header_bar.pack_start(halter_box)

        # HeaderBar Status Badge (Right side)
        self.status_badge = Gtk.Label(label="Stopped")
        self.status_badge.add_css_class("dim-label")
        self.status_badge.set_margin_end(6)
        self.header_bar.pack_end(self.status_badge)

        # ViewStack pages
        self.view_stack.set_vexpand(True)
        self.main_box.append(self.view_stack)

        self.build_dashboard_page()
        self.build_configuration_page()
        self.build_models_page()
        self.build_downloader_page()

        # Responsive ViewSwitcherBar for narrow viewports
        self.view_switcher_bar = Adw.ViewSwitcherBar(stack=self.view_stack)
        self.view_switcher_title.bind_property(
            "title-visible",
            self.view_switcher_bar,
            "reveal",
            GObject.BindingFlags.SYNC_CREATE,
        )
        self.main_box.append(self.view_switcher_bar)

    # =========================================================================
    # TAB 1: DASHBOARD (WITH LIVE HARDWARE TELEMETRY & TOKENS/S GRAPH)
    # =========================================================================
    def build_dashboard_page(self):
        scrolled_page = Gtk.ScrolledWindow()
        scrolled_page.set_vexpand(True)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        box.set_margin_top(12)
        box.set_margin_bottom(16)
        box.set_margin_start(16)
        box.set_margin_end(16)
        scrolled_page.set_child(box)

        # 1. System Status Group
        status_group = Adw.PreferencesGroup(title="Daemon and Sandbox Status")

        self.row_state = Adw.ActionRow(title="Daemon State", subtitle="Stopped")
        self.icon_state = Gtk.Image.new_from_icon_name("network-offline-symbolic")
        self.row_state.add_prefix(self.icon_state)

        self.btn_dash_alpaca = Gtk.Button(
            label="Open Alpaca in Bubble",
            icon_name="com.jeffser.Alpaca",
        )
        self.btn_dash_alpaca.set_valign(Gtk.Align.CENTER)
        self.btn_dash_alpaca.connect("clicked", self.on_launch_alpaca_clicked)
        self.btn_dash_alpaca.set_sensitive(False)
        self.row_state.add_suffix(self.btn_dash_alpaca)
        status_group.add(self.row_state)

        self.row_active_model = Adw.ActionRow(
            title="Active Model",
            subtitle="None selected",
        )
        self.row_active_model.add_prefix(
            Gtk.Image.new_from_icon_name("application-x-executable-symbolic")
        )
        status_group.add(self.row_active_model)

        self.row_sandbox = Adw.ActionRow(
            title="Sandbox Isolation",
            subtitle="Network namespace disabled (unshare -r -n)",
        )
        self.row_sandbox.add_prefix(
            Gtk.Image.new_from_icon_name("security-high-symbolic")
        )
        status_group.add(self.row_sandbox)

        box.append(status_group)

        # 2. Live Telemetry & Performance Widgets Group
        self.telemetry_group = Adw.PreferencesGroup(
            title="Live Hardware and Inference Telemetry",
            description="Real-time monitoring of GTX 1650 VRAM, temperature, compute load, and token generation speed.",
        )

        # Header suffix: Monitoring On/Off Switch to save CPU/GPU cycles
        telem_toggle_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        telem_toggle_box.set_valign(Gtk.Align.CENTER)

        lbl_toggle = Gtk.Label(label="Live Monitor")
        lbl_toggle.add_css_class("dim-label")
        telem_toggle_box.append(lbl_toggle)

        self.switch_telemetry = Gtk.Switch()
        self.switch_telemetry.set_active(True)
        self.switch_telemetry.set_valign(Gtk.Align.CENTER)
        self.switch_telemetry.connect("notify::active", self.on_telemetry_toggled)
        telem_toggle_box.append(self.switch_telemetry)

        self.telemetry_group.set_header_suffix(telem_toggle_box)

        # Row A: Nvidia GPU & VRAM (GTX 1650)
        self.row_gpu_vram = Adw.ActionRow(
            title="GPU Compute: 0% • 37°C",
            subtitle="VRAM Allocation: 0.0 / 4.0 GB (0%)",
        )
        self.row_gpu_vram.add_prefix(
            Gtk.Image.new_from_icon_name("video-display-symbolic")
        )

        # Dual progress bars container on the right of row
        vram_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        vram_box.set_valign(Gtk.Align.CENTER)
        vram_box.set_size_request(160, -1)

        self.prog_gpu = Gtk.ProgressBar()
        self.prog_gpu.set_fraction(0.0)
        self.prog_gpu.set_tooltip_text("GPU Core Compute Load")
        vram_box.append(self.prog_gpu)

        self.prog_vram = Gtk.ProgressBar()
        self.prog_vram.set_fraction(0.0)
        self.prog_vram.set_tooltip_text("VRAM Usage (4GB Maximum Constraint)")
        vram_box.append(self.prog_vram)

        self.row_gpu_vram.add_suffix(vram_box)
        self.telemetry_group.add(self.row_gpu_vram)

        # Row B: Host CPU & RAM
        self.row_cpu_ram = Adw.ActionRow(
            title="CPU Usage: 0%",
            subtitle="System RAM: 0.0 / 0.0 GB (0%)",
        )
        self.row_cpu_ram.add_prefix(
            Gtk.Image.new_from_icon_name("computer-symbolic")
        )

        ram_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        ram_box.set_valign(Gtk.Align.CENTER)
        ram_box.set_size_request(160, -1)

        self.prog_cpu = Gtk.ProgressBar()
        self.prog_cpu.set_fraction(0.0)
        self.prog_cpu.set_tooltip_text("Host CPU Core Utilization")
        ram_box.append(self.prog_cpu)

        self.prog_ram = Gtk.ProgressBar()
        self.prog_ram.set_fraction(0.0)
        self.prog_ram.set_tooltip_text("Host RAM Utilization")
        ram_box.append(self.prog_ram)

        self.row_cpu_ram.add_suffix(ram_box)
        self.telemetry_group.add(self.row_cpu_ram)

        # Row C: Generation Speed & Real-Time Sparkline Graph
        self.row_tokens = Adw.ActionRow(
            title="Inference Speed: 0.0 tokens/sec",
            subtitle="Prompt Processing: 0.0 t/s",
        )
        self.row_tokens.add_prefix(
            Gtk.Image.new_from_icon_name("utilities-system-monitor-symbolic")
            if Gtk.IconTheme.get_for_display(Gdk.Display.get_default()).has_icon("utilities-system-monitor-symbolic")
            else Gtk.Image.new_from_icon_name("system-run-symbolic")
        )

        # Graph Container inside Card
        graph_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        graph_box.set_margin_top(6)
        graph_box.set_margin_bottom(6)
        graph_box.set_margin_start(8)
        graph_box.set_margin_end(8)

        self.graph_tokens = Gtk.DrawingArea()
        self.graph_tokens.set_content_width(320)
        self.graph_tokens.set_content_height(84)
        self.graph_tokens.add_css_class("card")
        self.graph_tokens.set_draw_func(self.on_draw_tokens_graph)
        graph_box.append(self.graph_tokens)

        self.row_tokens.set_child(graph_box)
        self.telemetry_group.add(self.row_tokens)

        box.append(self.telemetry_group)

        # 3. Log Section
        log_group = Adw.PreferencesGroup(title="Server Log Console")

        btn_clear = Gtk.Button(
            icon_name="edit-clear-symbolic",
            tooltip_text="Clear console output",
        )
        btn_clear.add_css_class("flat")
        btn_clear.connect("clicked", lambda _: self.text_buffer.set_text(""))
        log_group.set_header_suffix(btn_clear)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_min_content_height(260)
        scrolled.add_css_class("card")

        self.text_view = Gtk.TextView()
        self.text_view.set_editable(False)
        self.text_view.set_cursor_visible(False)
        self.text_view.set_monospace(True)
        self.text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.text_view.set_left_margin(12)
        self.text_view.set_right_margin(12)
        self.text_view.set_top_margin(10)
        self.text_view.set_bottom_margin(10)
        self.text_buffer = self.text_view.get_buffer()

        scrolled.set_child(self.text_view)
        log_group.add(scrolled)

        box.append(log_group)

        page = self.view_stack.add_titled(scrolled_page, "dashboard", "Dashboard")
        page.set_icon_name("system-run-symbolic")

    # =========================================================================
    # TELEMETRY MONITORING & GRAPHING
    # =========================================================================
    def on_telemetry_toggled(self, switch, _param):
        self.telemetry_enabled = switch.get_active()
        if not self.telemetry_enabled:
            self.row_gpu_vram.set_title("GPU Compute: Monitoring Disabled (Resource Saver)")
            self.row_cpu_ram.set_title("CPU Usage: Monitoring Disabled")
            self.prog_gpu.set_fraction(0.0)
            self.prog_vram.set_fraction(0.0)
            self.prog_cpu.set_fraction(0.0)
            self.prog_ram.set_fraction(0.0)
        self.save_active_config()

    def on_tps_recorded(self, tps_val: float, is_prompt: bool):
        """Called when tokens/s regex is matched from daemon stdout."""
        if is_prompt:
            self.prompt_tps = tps_val
        else:
            self.current_tps = tps_val
            self.tokens_history.append(tps_val)
            self.graph_tokens.queue_draw()

        self.row_tokens.set_title(
            f"Generation Speed: {self.current_tps:.1f} tokens/sec"
        )
        self.row_tokens.set_subtitle(
            f"Prompt Processing: {self.prompt_tps:.1f} t/s • Peak Generation: {max(self.tokens_history or [0.0]):.1f} t/s"
        )
        return GLib.SOURCE_REMOVE

    def on_draw_tokens_graph(self, _area, cr, width, height):
        """Draws a smooth Cairo line chart and gradient fill for tokens/sec."""
        # Clean rounded background
        cr.set_source_rgba(0.10, 0.10, 0.13, 0.95)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        if not self.tokens_history:
            cr.set_source_rgba(0.55, 0.55, 0.60, 0.85)
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(12)
            cr.move_to(width / 2 - 95, height / 2 + 4)
            cr.show_text("Awaiting generation tokens/s...")
            return

        vals = list(self.tokens_history)
        max_val = max(vals) if vals else 30.0
        max_val = max(max_val, 10.0) * 1.20

        # Subtle grid dashed line
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.08)
        cr.set_dash([4.0, 4.0])
        cr.move_to(0, height * 0.5)
        cr.line_to(width, height * 0.5)
        cr.stroke()
        cr.set_dash([])

        step_x = width / max(1, len(vals) - 1) if len(vals) > 1 else width

        points = []
        for i, v in enumerate(vals):
            x = i * step_x
            y = height - ((v / max_val) * (height - 18)) - 8
            points.append((x, y))

        # Gradient area fill under curve
        cr.move_to(points[0][0], points[0][1])
        for x, y in points[1:]:
            cr.line_to(x, y)
        cr.line_to(points[-1][0], height)
        cr.line_to(points[0][0], height)
        cr.close_path()

        grad = cairo.LinearGradient(0, 0, 0, height)
        grad.add_color_stop_rgba(0.0, 0.21, 0.52, 0.95, 0.45)
        grad.add_color_stop_rgba(1.0, 0.21, 0.52, 0.95, 0.02)
        cr.set_source(grad)
        cr.fill()

        # Stroke curve
        cr.set_source_rgba(0.25, 0.58, 1.0, 1.0)
        cr.set_line_width(2.4)
        cr.move_to(points[0][0], points[0][1])
        for x, y in points[1:]:
            cr.line_to(x, y)
        cr.stroke()

        # Last point indicator
        last_x, last_y = points[-1]
        cr.arc(last_x, last_y, 4.0, 0, 2 * 3.14159)
        cr.set_source_rgba(0.25, 0.58, 1.0, 1.0)
        cr.fill()

        # Text overlay
        cr.set_source_rgba(0.95, 0.95, 0.98, 0.95)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(11)
        cr.move_to(10, 16)
        cr.show_text(f"Live: {vals[-1]:.1f} t/s  (Peak: {max(vals):.1f} t/s)")

    def _start_telemetry_worker(self):
        """Background thread polling system stats every 1.5s only when running."""
        def worker():
            while not self._stop_telemetry.is_set():
                if self.daemon.state == ProcessState.RUNNING and self.telemetry_enabled:
                    gpu_util, gpu_temp, vram_u, vram_t = self._poll_gpu()
                    cpu_pct, ram_u, ram_t = self._poll_cpu_ram()
                    GLib.idle_add(
                        self._apply_telemetry_update,
                        gpu_util,
                        gpu_temp,
                        vram_u,
                        vram_t,
                        cpu_pct,
                        ram_u,
                        ram_t,
                    )
                time.sleep(1.5)

        self._telemetry_thread = threading.Thread(target=worker, daemon=True)
        self._telemetry_thread.start()

    def _poll_gpu(self) -> tuple[float, float, float, float]:
        """Polls Nvidia GTX 1650 via nvidia-smi."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0 and res.stdout.strip():
                parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
                if len(parts) >= 4:
                    return parts[0], parts[1], parts[2], parts[3]
        except Exception:
            pass
        return 0.0, 0.0, 0.0, 4096.0

    def _poll_cpu_ram(self) -> tuple[float, float, float]:
        """Reads /proc/stat and /proc/meminfo with zero external bloat."""
        cpu_pct = 0.0
        try:
            with open("/proc/stat", "r") as f:
                fields = [float(x) for x in f.readline().split()[1:]]
            idle = fields[3] + fields[4]
            total = sum(fields)
            if self._prev_cpu_times:
                prev_idle, prev_total = self._prev_cpu_times
                d_idle = idle - prev_idle
                d_total = total - prev_total
                if d_total > 0:
                    cpu_pct = (1.0 - (d_idle / d_total)) * 100.0
            self._prev_cpu_times = (idle, total)
        except Exception:
            pass

        ram_u, ram_t = 0.0, 0.0
        try:
            mem = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    p = line.split(":")
                    if len(p) == 2:
                        mem[p[0].strip()] = int(p[1].split()[0])
            total_gb = mem.get("MemTotal", 0) / (1024 * 1024)
            avail_gb = mem.get("MemAvailable", 0) / (1024 * 1024)
            ram_t = total_gb
            ram_u = total_gb - avail_gb
        except Exception:
            pass

        return cpu_pct, ram_u, ram_t

    def _apply_telemetry_update(
        self, gpu_util, gpu_temp, vram_u, vram_t, cpu_pct, ram_u, ram_t
    ):
        if not self.telemetry_enabled or self.daemon.state != ProcessState.RUNNING:
            return GLib.SOURCE_REMOVE

        # Update GPU & VRAM
        vram_gb_u = vram_u / 1024.0
        vram_gb_t = vram_t / 1024.0
        vram_pct = (vram_u / vram_t) if vram_t > 0 else 0.0

        self.row_gpu_vram.set_title(f"GPU Compute: {gpu_util:.0f}% • {gpu_temp:.0f}°C")
        self.row_gpu_vram.set_subtitle(
            f"VRAM Allocation: {vram_gb_u:.2f} / {vram_gb_t:.1f} GB ({vram_pct*100:.1f}%)"
        )
        self.prog_gpu.set_fraction(min(1.0, max(0.0, gpu_util / 100.0)))
        self.prog_vram.set_fraction(min(1.0, max(0.0, vram_pct)))

        # Update CPU & RAM
        ram_pct = (ram_u / ram_t) if ram_t > 0 else 0.0
        self.row_cpu_ram.set_title(f"CPU Core Usage: {cpu_pct:.1f}%")
        self.row_cpu_ram.set_subtitle(
            f"System RAM: {ram_u:.1f} / {ram_t:.1f} GB ({ram_pct*100:.1f}%)"
        )
        self.prog_cpu.set_fraction(min(1.0, max(0.0, cpu_pct / 100.0)))
        self.prog_ram.set_fraction(min(1.0, max(0.0, ram_pct)))

        return GLib.SOURCE_REMOVE

    # =========================================================================
    # TAB 2: CONFIGURATION (THE HARNESS)
    # =========================================================================
    def build_configuration_page(self):
        pref_page = Adw.PreferencesPage()

        # Group 1: Configuration Profiles
        profile_group = Adw.PreferencesGroup(
            title="Configuration Profiles",
            description="Save and reload parameter presets for specific workflows.",
        )

        self.combo_profile = Adw.ComboRow(title="Active Profile")
        self.combo_profile.connect("notify::selected", self.on_profile_selected)
        profile_group.add(self.combo_profile)

        btn_save_profile = Gtk.Button(
            label="Save Profile Changes",
            icon_name="document-save-symbolic",
        )
        btn_save_profile.set_valign(Gtk.Align.CENTER)
        btn_save_profile.connect("clicked", self.on_save_profile_clicked)
        profile_group.set_header_suffix(btn_save_profile)

        pref_page.add(profile_group)

        # Group 2: Core Parameters
        core_group = Adw.PreferencesGroup(
            title="Core Harness Parameters",
            description="Fundamental server compute and VRAM allocation.",
        )

        self.row_config_model = Adw.ActionRow(
            title="Target Model",
            subtitle="None selected",
        )
        self.row_config_model.add_prefix(
            Gtk.Image.new_from_icon_name("application-x-executable-symbolic")
        )
        btn_goto_models = Gtk.Button(label="Browse Models")
        btn_goto_models.set_valign(Gtk.Align.CENTER)
        btn_goto_models.connect(
            "clicked", lambda _: self.view_stack.set_visible_child_name("models")
        )
        self.row_config_model.add_suffix(btn_goto_models)
        core_group.add(self.row_config_model)

        self.spin_ctx = Adw.SpinRow.new_with_range(512, 65536, 512)
        self.spin_ctx.set_title("Context Window Size (-c)")
        self.spin_ctx.set_subtitle("Tokens allocated for model memory.")
        self.spin_ctx.set_value(2048)
        self.spin_ctx.connect("notify::value", lambda *_: self.save_active_config())
        core_group.add(self.spin_ctx)

        self.spin_ngl = Adw.SpinRow.new_with_range(0, 128, 1)
        self.spin_ngl.set_title("GPU Layers to Offload (-ngl)")
        self.spin_ngl.set_subtitle("Layers offloaded to GTX 1650 (4GB VRAM constraint).")
        self.spin_ngl.set_value(24)
        self.spin_ngl.connect("notify::value", lambda *_: self.save_active_config())
        core_group.add(self.spin_ngl)

        self.spin_threads = Adw.SpinRow.new_with_range(1, 64, 1)
        self.spin_threads.set_title("CPU Threads (-t)")
        self.spin_threads.set_subtitle("Thread count for CPU compute kernels.")
        self.spin_threads.set_value(max(1, (os.cpu_count() or 4) - 2))
        self.spin_threads.connect("notify::value", lambda *_: self.save_active_config())
        core_group.add(self.spin_threads)

        self.entry_host = Adw.EntryRow(title="Host Interface (--host)")
        self.entry_host.set_text("127.0.0.1")
        self.entry_host.connect("changed", lambda *_: self.save_active_config())
        core_group.add(self.entry_host)

        self.spin_port = Adw.SpinRow.new_with_range(1024, 65535, 1)
        self.spin_port.set_title("Server Port (--port)")
        self.spin_port.set_subtitle("Local port binding.")
        self.spin_port.set_value(8080)
        self.spin_port.connect("notify::value", lambda *_: self.save_active_config())
        core_group.add(self.spin_port)

        pref_page.add(core_group)

        # Group 3: VRAM & Performance Optimization (Tailored to GTX 1650)
        opt_group = Adw.PreferencesGroup(
            title="VRAM and Performance Optimization",
            description="Tuning options for 4GB VRAM headroom and KV cache efficiency.",
        )

        self.combo_flash_attn = Adw.ComboRow(
            title="Flash Attention (-fa)",
            subtitle="Drastically reduces VRAM footprint of attention tensors.",
            model=Gtk.StringList.new(["auto", "on", "off"]),
        )
        self.combo_flash_attn.connect("notify::selected", lambda *_: self.save_active_config())
        opt_group.add(self.combo_flash_attn)

        self.combo_ctk = Adw.ComboRow(
            title="KV Cache Quantization K (-ctk)",
            subtitle="q4_0 / q8_0 halves context VRAM usage vs f16.",
            model=Gtk.StringList.new(["f16", "q8_0", "q4_0"]),
        )
        self.combo_ctk.connect("notify::selected", lambda *_: self.save_active_config())
        opt_group.add(self.combo_ctk)

        self.combo_ctv = Adw.ComboRow(
            title="KV Cache Quantization V (-ctv)",
            subtitle="Quantizes V tensors in context cache.",
            model=Gtk.StringList.new(["f16", "q8_0", "q4_0"]),
        )
        self.combo_ctv.connect("notify::selected", lambda *_: self.save_active_config())
        opt_group.add(self.combo_ctv)

        self.spin_ubatch = Adw.SpinRow.new_with_range(32, 2048, 32)
        self.spin_ubatch.set_title("Physical Micro-batch Size (-ub)")
        self.spin_ubatch.set_subtitle("Controls peak VRAM allocation spikes during prompt processing.")
        self.spin_ubatch.set_value(256)
        self.spin_ubatch.connect("notify::value", lambda *_: self.save_active_config())
        opt_group.add(self.spin_ubatch)

        self.spin_batch = Adw.SpinRow.new_with_range(64, 4096, 64)
        self.spin_batch.set_title("Logical Batch Size (-b)")
        self.spin_batch.set_subtitle("Maximum logical batch size.")
        self.spin_batch.set_value(512)
        self.spin_batch.connect("notify::value", lambda *_: self.save_active_config())
        opt_group.add(self.spin_batch)

        self.spin_parallel = Adw.SpinRow.new_with_range(1, 16, 1)
        self.spin_parallel.set_title("Parallel Slots (-np)")
        self.spin_parallel.set_subtitle("Number of simultaneous inference slots (keep 1 on 4GB VRAM).")
        self.spin_parallel.set_value(1)
        self.spin_parallel.connect("notify::value", lambda *_: self.save_active_config())
        opt_group.add(self.spin_parallel)

        self.switch_ctx_shift = Adw.SwitchRow(
            title="Context Shift (--context-shift)",
            subtitle="Discards older context tokens to keep chatting without crashing.",
        )
        self.switch_ctx_shift.set_active(True)
        self.switch_ctx_shift.connect("notify::active", lambda *_: self.save_active_config())
        opt_group.add(self.switch_ctx_shift)

        self.switch_unshare = Adw.SwitchRow(
            title="Air-gapped Network Namespace (unshare -n)",
            subtitle="Isolates server from all host networking. Keep OFF to allow local chat clients (Alpaca, Open WebUI) to connect via localhost.",
        )
        self.switch_unshare.set_active(False)
        self.switch_unshare.connect("notify::active", lambda *_: self.save_active_config())
        opt_group.add(self.switch_unshare)

        pref_page.add(opt_group)

        # Group 4: Dynamic Advanced Arguments Builder
        self.adv_group = Adw.PreferencesGroup(
            title="Advanced and Custom Arguments",
            description="Expose any arbitrary llama-server CLI flags (e.g. --timeout, --temp).",
        )

        btn_add_arg = Gtk.Button(
            label="Add Flag",
            icon_name="list-add-symbolic",
        )
        btn_add_arg.add_css_class("flat")
        btn_add_arg.connect("clicked", lambda _: self.add_custom_arg_row())
        self.adv_group.set_header_suffix(btn_add_arg)

        pref_page.add(self.adv_group)

        page = self.view_stack.add_titled(pref_page, "config", "Configuration")
        page.set_icon_name("preferences-system-symbolic")

    def add_custom_arg_row(self, flag: str = "", value: str = ""):
        row = Adw.ActionRow()

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_hexpand(True)
        box.set_valign(Gtk.Align.CENTER)

        entry_flag = Gtk.Entry(placeholder_text="Flag (e.g. --timeout)")
        entry_flag.set_text(flag)
        entry_flag.set_hexpand(True)
        entry_flag.connect("changed", lambda *_: self.save_active_config())
        box.append(entry_flag)

        entry_val = Gtk.Entry(placeholder_text="Value (optional)")
        entry_val.set_text(value)
        entry_val.set_hexpand(True)
        entry_val.connect("changed", lambda *_: self.save_active_config())
        box.append(entry_val)

        btn_remove = Gtk.Button(icon_name="user-trash-symbolic")
        btn_remove.add_css_class("flat")

        def remove_row(_):
            self.adv_group.remove(row)
            self.custom_arg_rows = [
                item for item in self.custom_arg_rows if item[0] != row
            ]
            self.save_active_config()

        btn_remove.connect("clicked", remove_row)
        box.append(btn_remove)

        row.set_child(box)
        self.adv_group.add(row)
        self.custom_arg_rows.append((row, entry_flag, entry_val))

    def clear_custom_args(self):
        for row, _, _ in self.custom_arg_rows:
            self.adv_group.remove(row)
        self.custom_arg_rows.clear()

    def get_custom_args(self) -> list[dict]:
        args = []
        for _, entry_flag, entry_val in self.custom_arg_rows:
            f = entry_flag.get_text().strip()
            v = entry_val.get_text().strip()
            if f:
                args.append({"flag": f, "value": v})
        return args

    # =========================================================================
    # TAB 3: MODELS STORAGE
    # =========================================================================
    def build_models_page(self):
        pref_page = Adw.PreferencesPage()

        self.models_group = Adw.PreferencesGroup(
            title="Local Models (~/.local/llama.cpp/models)",
            description="Stored GGUF models and vision projector adapters.",
        )

        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        btn_open_folder = Gtk.Button(
            icon_name="folder-symbolic",
            tooltip_text="Open models directory in file manager",
        )
        btn_open_folder.add_css_class("flat")
        btn_open_folder.connect("clicked", self.on_open_models_dir)
        btn_box.append(btn_open_folder)

        btn_refresh = Gtk.Button(
            icon_name="view-refresh-symbolic",
            tooltip_text="Refresh models list",
        )
        btn_refresh.add_css_class("flat")
        btn_refresh.connect("clicked", lambda _: self.refresh_models_list())
        btn_box.append(btn_refresh)

        self.models_group.set_header_suffix(btn_box)
        pref_page.add(self.models_group)

        page = self.view_stack.add_titled(pref_page, "models", "Models")
        page.set_icon_name("drive-harddisk-symbolic")

    def scan_models(self) -> list[tuple[str, float, Path]]:
        if not DEFAULT_MODELS_DIR.exists():
            return []
        results = []
        try:
            for item in sorted(DEFAULT_MODELS_DIR.iterdir()):
                if item.is_file() and item.name.endswith(".gguf"):
                    size_mb = item.stat().st_size / (1024 * 1024)
                    results.append((item.name, size_mb, item))
        except Exception as exc:
            print(f"Error scanning models: {exc}")
        return results

    def refresh_models_list(self):
        for row in self.model_rows:
            self.models_group.remove(row)
        self.model_rows.clear()

        models = self.scan_models()
        if not models:
            empty_row = Adw.ActionRow(
                title="No models found",
                subtitle=f"Use the Downloader tab or place .gguf files in {DEFAULT_MODELS_DIR}",
            )
            empty_row.add_prefix(
                Gtk.Image.new_from_icon_name("dialog-information-symbolic")
            )
            self.models_group.add(empty_row)
            self.model_rows.append(empty_row)
            return

        for name, size_mb, path in models:
            is_mmproj = ".mmproj" in name or "mmproj" in name.lower()
            size_str = (
                f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.1f} MB"
            )

            if is_mmproj:
                tag = "Vision Projector"
                sub_detail = f"{tag} • {size_str}"
            else:
                matched_proj = self.find_matching_mmproj(name)
                if matched_proj:
                    tag = "Base Model (Vision Capable)"
                    sub_detail = f"{tag} • {size_str} • 👁️ mmproj found"
                else:
                    tag = "Base Model (Text Only)"
                    sub_detail = f"{tag} • {size_str}"

            row = Adw.ActionRow(
                title=name,
                subtitle=sub_detail,
            )
            icon_name = (
                "video-display-symbolic"
                if is_mmproj
                else "application-x-executable-symbolic"
            )
            row.add_prefix(Gtk.Image.new_from_icon_name(icon_name))

            is_active = (
                self.active_model_path and self.active_model_path.resolve() == path.resolve()
            )

            actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            actions_box.set_valign(Gtk.Align.CENTER)

            if is_active:
                badge = Gtk.Label(label="Active")
                badge.add_css_class("accent")
                badge.set_valign(Gtk.Align.CENTER)
                badge.set_margin_end(6)
                actions_box.append(badge)
            elif is_mmproj:
                badge = Gtk.Label(label="Projector")
                badge.add_css_class("dim-label")
                badge.set_valign(Gtk.Align.CENTER)
                badge.set_margin_end(6)
                actions_box.append(badge)
            else:
                btn_activate = Gtk.Button(label="Activate")
                btn_activate.set_valign(Gtk.Align.CENTER)
                btn_activate.connect(
                    "clicked",
                    lambda _, p=path: self.on_model_activated(p),
                )
                actions_box.append(btn_activate)

            btn_delete = Gtk.Button(
                icon_name="user-trash-symbolic",
                tooltip_text=f"Delete {name} permanently",
            )
            btn_delete.add_css_class("flat")
            btn_delete.set_valign(Gtk.Align.CENTER)
            btn_delete.connect(
                "clicked",
                lambda _, p=path: self.on_delete_model_clicked(p),
            )
            actions_box.append(btn_delete)

            row.add_suffix(actions_box)
            self.models_group.add(row)
            self.model_rows.append(row)

    def on_delete_model_clicked(self, path: Path):
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
        size_str = f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.1f} MB"

        dialog = Adw.AlertDialog.new(
            f"Delete {path.name}?",
            f"Are you sure you want to permanently delete this file ({size_str}) from disk? This cannot be undone.",
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("delete", "Delete")
        dialog.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")

        def on_response(dlg, response):
            if response == "delete":
                self.do_delete_model(path)

        dialog.connect("response", on_response)
        dialog.present(self)

    def do_delete_model(self, path: Path):
        try:
            if self.daemon.state in (ProcessState.RUNNING, ProcessState.PAUSED):
                if self.active_model_path and self.active_model_path.resolve() == path.resolve():
                    self.daemon.stop()
                    self.on_log_message("[Alforja] Stopped server because active model is being deleted.\n")

            if self.active_model_path and self.active_model_path.resolve() == path.resolve():
                self.active_model_path = None
                self.active_mmproj_path = None
                self.row_active_model.set_subtitle("None selected")
                self.row_config_model.set_subtitle("None selected")

            if path.exists():
                path.unlink()
                self.on_log_message(f"[Alforja] Permanently deleted model: {path.name}\n")

            self.refresh_models_list()
            self.save_active_config()
        except Exception as exc:
            self.on_log_message(f"[Alforja Error] Failed to delete model {path.name}: {exc}\n")

    def on_model_activated(self, path: Path):
        was_running = self.daemon.state in (ProcessState.RUNNING, ProcessState.PAUSED)
        self.set_active_model(path)
        self.refresh_models_list()
        self.on_log_message(f"[Alforja] Active model set to: {path.name}\n")

        if was_running:
            self.on_log_message(
                f"[Alforja] Server is active. Automatically restarting to release VRAM and load {path.name}...\n"
            )
            self.daemon.stop()

            def restart_worker():
                # Wait until daemon process is fully dead and VRAM is freed
                for _ in range(30):
                    if not self.daemon.process:
                        break
                    time.sleep(0.1)
                time.sleep(0.5)
                GLib.idle_add(lambda: self.on_start_clicked(None))

            threading.Thread(target=restart_worker, daemon=True).start()

    def on_open_models_dir(self, _button):
        DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        Gio.AppInfo.launch_default_for_uri(
            DEFAULT_MODELS_DIR.as_uri(),
            None,
        )

    # =========================================================================
    # TAB 4: DOWNLOADER
    # =========================================================================
    def build_downloader_page(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(16)
        box.set_margin_end(16)

        dl_group = Adw.PreferencesGroup(
            title="Direct URL Model Downloader",
            description="Download .gguf models directly into ~/.local/llama.cpp/models",
        )

        self.entry_model_url = Adw.EntryRow(title="Model URL (.gguf)")
        dl_group.add(self.entry_model_url)

        self.entry_mmproj_url = Adw.EntryRow(
            title="Vision Projector URL (Optional .mmproj.gguf)"
        )
        dl_group.add(self.entry_mmproj_url)

        box.append(dl_group)

        # Progress and Action Controls
        ctrl_group = Adw.PreferencesGroup(title="Download Status")

        ctrl_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        ctrl_box.set_margin_top(6)
        ctrl_box.set_margin_bottom(6)

        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_show_text(True)
        self.progress_bar.set_text("Ready")
        self.progress_bar.set_fraction(0.0)
        ctrl_box.append(self.progress_bar)

        btn_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        btn_bar.set_halign(Gtk.Align.END)

        self.btn_cancel_dl = Gtk.Button(
            label="Cancel",
            icon_name="process-stop-symbolic",
        )
        self.btn_cancel_dl.add_css_class("destructive-action")
        self.btn_cancel_dl.set_sensitive(False)
        self.btn_cancel_dl.connect("clicked", self.on_cancel_download_clicked)
        btn_bar.append(self.btn_cancel_dl)

        self.btn_start_dl = Gtk.Button(
            label="Start Download",
            icon_name="folder-download-symbolic",
        )
        self.btn_start_dl.add_css_class("suggested-action")
        self.btn_start_dl.connect("clicked", self.on_start_download_clicked)
        btn_bar.append(self.btn_start_dl)

        ctrl_box.append(btn_bar)
        ctrl_group.add(ctrl_box)
        box.append(ctrl_group)

        # Download Console Group
        console_group = Adw.PreferencesGroup(title="Download Output Console")

        btn_clear_dl = Gtk.Button(
            icon_name="edit-clear-symbolic",
            tooltip_text="Clear console output",
        )
        btn_clear_dl.add_css_class("flat")
        btn_clear_dl.connect("clicked", lambda _: self.dl_text_buffer.set_text(""))
        console_group.set_header_suffix(btn_clear_dl)

        dl_scrolled = Gtk.ScrolledWindow()
        dl_scrolled.set_vexpand(True)
        dl_scrolled.set_min_content_height(240)
        dl_scrolled.add_css_class("card")

        self.dl_text_view = Gtk.TextView()
        self.dl_text_view.set_editable(False)
        self.dl_text_view.set_cursor_visible(False)
        self.dl_text_view.set_monospace(True)
        self.dl_text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.dl_text_view.set_left_margin(12)
        self.dl_text_view.set_right_margin(12)
        self.dl_text_view.set_top_margin(10)
        self.dl_text_view.set_bottom_margin(10)
        self.dl_text_buffer = self.dl_text_view.get_buffer()

        dl_scrolled.set_child(self.dl_text_view)
        console_group.add(dl_scrolled)

        box.append(console_group)

        page = self.view_stack.add_titled(box, "downloader", "Downloader")
        page.set_icon_name("folder-download-symbolic")

    def on_start_download_clicked(self, _button):
        urls = []
        model_url = self.entry_model_url.get_text().strip()
        mmproj_url = self.entry_mmproj_url.get_text().strip()

        if model_url:
            urls.append(model_url)
        if mmproj_url:
            urls.append(mmproj_url)

        if not urls:
            self.on_dl_log("[Downloader] Error: Please enter at least a Model URL (.gguf).\n")
            return

        self.btn_start_dl.set_sensitive(False)
        self.btn_cancel_dl.set_sensitive(True)
        self.entry_model_url.set_sensitive(False)
        self.entry_mmproj_url.set_sensitive(False)
        self.progress_bar.set_fraction(0.0)
        self.progress_bar.set_text("Connecting...")

        self.on_dl_log("[Downloader] Starting download job...\n")

        self.downloader.start_download(
            urls=urls,
            progress_cb=self.on_dl_progress,
            log_cb=self.on_dl_log,
            finished_cb=self.on_dl_finished,
        )

    def on_cancel_download_clicked(self, _button):
        self.downloader.cancel()
        self.btn_cancel_dl.set_sensitive(False)

    def on_dl_progress(self, fraction: float, status_text: str):
        self.progress_bar.set_fraction(fraction)
        self.progress_bar.set_text(status_text)
        return GLib.SOURCE_REMOVE

    def on_dl_log(self, text: str):
        end_iter = self.dl_text_buffer.get_end_iter()
        self.dl_text_buffer.insert(end_iter, text)
        mark = self.dl_text_buffer.get_insert()
        self.dl_text_view.scroll_to_mark(mark, 0.0, False, 0.0, 1.0)
        return GLib.SOURCE_REMOVE

    def on_dl_finished(self, success: bool, message: str):
        self.btn_start_dl.set_sensitive(True)
        self.btn_cancel_dl.set_sensitive(False)
        self.entry_model_url.set_sensitive(True)
        self.entry_mmproj_url.set_sensitive(True)

        if success:
            self.progress_bar.set_fraction(1.0)
            self.progress_bar.set_text("Download complete!")
            self.entry_model_url.set_text("")
            self.entry_mmproj_url.set_text("")
        else:
            self.progress_bar.set_text(f"Failed: {message}")

        self.refresh_models_list()
        return GLib.SOURCE_REMOVE

    # =========================================================================
    # DAEMON STATE & SIGNALS
    # =========================================================================
    def on_log_message(self, message: str):
        end_iter = self.text_buffer.get_end_iter()
        self.text_buffer.insert(end_iter, message)
        mark = self.text_buffer.get_insert()
        self.text_view.scroll_to_mark(mark, 0.0, False, 0.0, 1.0)
        return GLib.SOURCE_REMOVE

    def on_process_state_changed(self, new_state: ProcessState):
        self.update_ui_state(new_state)
        return GLib.SOURCE_REMOVE

    def update_ui_state(self, state: ProcessState):
        if state == ProcessState.STOPPED:
            self.status_badge.set_label("Stopped")
            self.status_badge.remove_css_class("accent")
            self.status_badge.remove_css_class("warning")
            self.status_badge.add_css_class("dim-label")

            self.btn_start.set_sensitive(True)
            self.btn_pause.set_sensitive(False)
            self.btn_stop.set_sensitive(False)
            self.btn_alpaca.set_sensitive(False)
            self.btn_dash_alpaca.set_sensitive(False)
            self.btn_pause.set_icon_name("media-playback-pause-symbolic")
            self.btn_pause.set_tooltip_text("Pause daemon (SIGSTOP)")

            self.row_state.set_subtitle("Stopped")
            self.icon_state.set_from_icon_name("network-offline-symbolic")

            self.row_gpu_vram.set_title("GPU Compute: Idle (Server Stopped)")
            self.row_cpu_ram.set_title("CPU Usage: Idle")
            self.prog_gpu.set_fraction(0.0)
            self.prog_vram.set_fraction(0.0)
            self.prog_cpu.set_fraction(0.0)

        elif state == ProcessState.RUNNING:
            self.status_badge.set_label("Running")
            self.status_badge.remove_css_class("dim-label")
            self.status_badge.remove_css_class("warning")
            self.status_badge.add_css_class("accent")

            self.btn_start.set_sensitive(False)
            self.btn_pause.set_sensitive(True)
            self.btn_stop.set_sensitive(True)
            self.btn_alpaca.set_sensitive(True)
            self.btn_dash_alpaca.set_sensitive(True)
            self.btn_pause.set_icon_name("media-playback-pause-symbolic")
            self.btn_pause.set_tooltip_text("Pause daemon (SIGSTOP)")

            pid = self.daemon.process.pid if self.daemon.process else "N/A"
            self.row_state.set_subtitle(f"Running (PID: {pid})")
            self.icon_state.set_from_icon_name("network-idle-symbolic")

        elif state == ProcessState.PAUSED:
            self.status_badge.set_label("Paused")
            self.status_badge.remove_css_class("accent")
            self.status_badge.remove_css_class("dim-label")
            self.status_badge.add_css_class("warning")

            self.btn_start.set_sensitive(False)
            self.btn_pause.set_sensitive(True)
            self.btn_stop.set_sensitive(True)
            self.btn_alpaca.set_sensitive(False)
            self.btn_dash_alpaca.set_sensitive(False)
            self.btn_pause.set_icon_name("media-playback-start-symbolic")
            self.btn_pause.set_tooltip_text("Resume daemon (SIGCONT)")

            pid = self.daemon.process.pid if self.daemon.process else "N/A"
            self.row_state.set_subtitle(f"Paused (SIGSTOP frozen, PID: {pid})")
            self.icon_state.set_from_icon_name("media-playback-pause-symbolic")

            self.row_gpu_vram.set_title("GPU Compute: Paused (VRAM Retained in Background)")

    def on_start_clicked(self, _button):
        if not self.active_model_path or not self.active_model_path.exists():
            models = self.scan_models()
            if models:
                self.set_active_model(models[0][2])
            else:
                self.on_log_message(
                    "[Alforja Error] No .gguf model found! Please download or place a model in ~/.local/llama.cpp/models first.\n"
                )
                return

        cfg = self.get_ui_config()
        host = cfg.get("host", "127.0.0.1").strip() or "127.0.0.1"
        use_unshare = cfg.get("use_unshare", False)

        # Build comprehensive aliases so clients like Alpaca never fail on model name mismatches
        model_fname = self.active_model_path.name
        stem_name = re.sub(r"(\.[qQ]\d[^.]*|\.gguf)$", "", model_fname)
        raw_aliases = [
            "default",
            model_fname,
            stem_name,
            stem_name.lower(),
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o",
        ]
        alias_str = ",".join(dict.fromkeys(a.strip() for a in raw_aliases if a and a.strip()))

        args = [
            "-m", str(self.active_model_path),
            "--host", host,
            "--port", str(cfg["port"]),
            "-c", str(cfg["ctx_size"]),
            "-ngl", str(cfg["gpu_layers"]),
            "-t", str(cfg["threads"]),
            "-fa", cfg["flash_attn"],
            "-ctk", cfg["cache_type_k"],
            "-ctv", cfg["cache_type_v"],
            "-ub", str(cfg["ubatch_size"]),
            "-b", str(cfg["batch_size"]),
            "-np", str(cfg["parallel"]),
            "--alias", alias_str,
        ]

        if cfg["ctx_shift"]:
            args.append("--context-shift")

        if self.active_mmproj_path and self.active_mmproj_path.exists():
            args.extend(["--mmproj", str(self.active_mmproj_path)])
            self.on_log_message(f"[Alforja] 👁️ Multimodal vision enabled with projector: {self.active_mmproj_path.name}\n")
        else:
            self.on_log_message("[Alforja] ℹ️ Running in text-only mode (no vision projector linked).\n")

        for item in cfg["custom_args"]:
            flag = item.get("flag", "").strip()
            val = item.get("value", "").strip()
            if flag:
                args.append(flag)
                if val:
                    args.append(val)

        # Update dashboard sandbox indicator
        if use_unshare:
            self.row_sandbox.set_subtitle("Air-gapped namespace (unshare -n) - Host network blocked")
        else:
            self.row_sandbox.set_subtitle(f"Local Loopback ({host}:{cfg['port']}) - Offline & accessible to Alpaca")

        self.save_active_config()
        self.daemon.start(args, use_unshare=use_unshare)

    def on_pause_clicked(self, _button):
        if self.daemon.state == ProcessState.RUNNING:
            self.daemon.pause()
        elif self.daemon.state == ProcessState.PAUSED:
            self.daemon.resume()

    def on_stop_clicked(self, _button):
        self.daemon.stop()

    def on_launch_alpaca_clicked(self, _button):
        """Launches Alpaca Flatpak joining the daemon's air-gapped sandbox bubble."""
        if self.daemon.state != ProcessState.RUNNING:
            self.on_log_message("[Alforja] Server must be running before launching Alpaca.\n")
            return

        pid = self.daemon.process.pid if self.daemon.process else None
        if not pid:
            return

        if self.daemon.use_unshare:
            cmd = [
                "nsenter",
                "-t", str(pid),
                "-U", "-n",
                "--preserve-credentials",
                "flatpak", "run",
                "--filesystem=host",
                "com.jeffser.Alpaca",
            ]
            desc = "inside the shared air-gapped sandbox bubble"
        else:
            cmd = [
                "flatpak", "run",
                "--filesystem=host",
                "com.jeffser.Alpaca",
            ]
            desc = "on local loopback (127.0.0.1)"

        try:
            subprocess.Popen(cmd)
            self.on_log_message(f"[Alforja] Launched Alpaca {desc} (joining sandbox PID {pid}).\n")
        except Exception as exc:
            self.on_log_message(f"[Alforja Error] Failed to launch Alpaca: {exc}\n")

    # =========================================================================
    # CONFIGURATION & PROFILES IMPLEMENTATION
    # =========================================================================
    def load_profiles_and_config(self):
        self._loading_config = True
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        self.profiles = dict(DEFAULT_PROFILES)
        if PROFILES_FILE.exists():
            try:
                with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                    saved_profiles = json.load(f)
                    if isinstance(saved_profiles, dict):
                        self.profiles.update(saved_profiles)
            except Exception as exc:
                print(f"[Config] Error reading profiles: {exc}")

        self.update_profiles_combo()

        active_cfg = None
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    active_cfg = json.load(f)
            except Exception as exc:
                print(f"[Config] Error reading active config: {exc}")

        if not active_cfg:
            active_cfg = self.profiles.get(
                self.active_profile_name, DEFAULT_PROFILES["GTX 1650 (Default 4GB Safe)"]
            )

        self.apply_config_to_ui(active_cfg)
        self._loading_config = False

    def update_profiles_combo(self):
        profile_names = list(self.profiles.keys())
        self.profile_string_list = Gtk.StringList.new(profile_names)
        self.combo_profile.set_model(self.profile_string_list)

        if self.active_profile_name in profile_names:
            idx = profile_names.index(self.active_profile_name)
            self.combo_profile.set_selected(idx)

    def apply_config_to_ui(self, cfg: dict):
        self._loading_config = True

        self.spin_ctx.set_value(cfg.get("ctx_size", 2048))
        self.spin_ngl.set_value(cfg.get("gpu_layers", 24))
        self.spin_threads.set_value(cfg.get("threads", 4))
        self.entry_host.set_text(cfg.get("host", "127.0.0.1"))
        self.spin_port.set_value(cfg.get("port", 8080))

        fa_val = cfg.get("flash_attn", "auto")
        self.set_combo_value(self.combo_flash_attn, ["auto", "on", "off"], fa_val)

        ctk_val = cfg.get("cache_type_k", "f16")
        self.set_combo_value(self.combo_ctk, ["f16", "q8_0", "q4_0"], ctk_val)

        ctv_val = cfg.get("cache_type_v", "f16")
        self.set_combo_value(self.combo_ctv, ["f16", "q8_0", "q4_0"], ctv_val)

        self.spin_ubatch.set_value(cfg.get("ubatch_size", 256))
        self.spin_batch.set_value(cfg.get("batch_size", 512))
        self.spin_parallel.set_value(cfg.get("parallel", 1))
        self.switch_ctx_shift.set_active(cfg.get("ctx_shift", True))
        self.switch_unshare.set_active(cfg.get("use_unshare", False))

        self.telemetry_enabled = cfg.get("telemetry_enabled", True)
        self.switch_telemetry.set_active(self.telemetry_enabled)

        self.clear_custom_args()
        for item in cfg.get("custom_args", []):
            if isinstance(item, dict):
                self.add_custom_arg_row(item.get("flag", ""), item.get("value", ""))

        saved_model = cfg.get("active_model")
        if saved_model:
            model_p = Path(saved_model)
            if model_p.exists():
                self.set_active_model(model_p)

        self._loading_config = False

    def set_combo_value(self, combo: Adw.ComboRow, options: list[str], val: str):
        if val in options:
            combo.set_selected(options.index(val))
        else:
            combo.set_selected(0)

    def get_ui_config(self) -> dict:
        fa_model = self.combo_flash_attn.get_model()
        ctk_model = self.combo_ctk.get_model()
        ctv_model = self.combo_ctv.get_model()

        fa_val = fa_model.get_string(self.combo_flash_attn.get_selected()) if fa_model else "auto"
        ctk_val = ctk_model.get_string(self.combo_ctk.get_selected()) if ctk_model else "f16"
        ctv_val = ctv_model.get_string(self.combo_ctv.get_selected()) if ctv_model else "f16"

        return {
            "ctx_size": int(self.spin_ctx.get_value()),
            "gpu_layers": int(self.spin_ngl.get_value()),
            "threads": int(self.spin_threads.get_value()),
            "host": self.entry_host.get_text().strip() or "127.0.0.1",
            "port": int(self.spin_port.get_value()),
            "flash_attn": fa_val,
            "cache_type_k": ctk_val,
            "cache_type_v": ctv_val,
            "ubatch_size": int(self.spin_ubatch.get_value()),
            "batch_size": int(self.spin_batch.get_value()),
            "parallel": int(self.spin_parallel.get_value()),
            "ctx_shift": self.switch_ctx_shift.get_active(),
            "use_unshare": self.switch_unshare.get_active(),
            "telemetry_enabled": self.switch_telemetry.get_active(),
            "active_model": str(self.active_model_path) if self.active_model_path else "",
            "custom_args": self.get_custom_args(),
        }

    def save_active_config(self):
        if self._loading_config:
            return
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cfg = self.get_ui_config()
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception as exc:
            print(f"[Config] Error saving config: {exc}")

    def on_profile_selected(self, _combo, _param):
        if self._loading_config:
            return
        idx = self.combo_profile.get_selected()
        model = self.combo_profile.get_model()
        if not model or idx < 0:
            return
        profile_name = model.get_string(idx)
        if profile_name in self.profiles:
            self.active_profile_name = profile_name
            self.apply_config_to_ui(self.profiles[profile_name])
            self.save_active_config()
            self.on_log_message(f"[Alforja] Loaded profile: {profile_name}\n")

    def on_save_profile_clicked(self, _button):
        cfg = self.get_ui_config()
        self.profiles[self.active_profile_name] = cfg
        try:
            with open(PROFILES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, indent=2)
            self.save_active_config()
            self.on_log_message(f"[Alforja] Saved profile: {self.active_profile_name}\n")
        except Exception as exc:
            self.on_log_message(f"[Alforja Error] Failed saving profile: {exc}\n")

    def init_models(self):
        models = self.scan_models()
        if models and not self.active_model_path:
            for name, _, path in models:
                if ".mmproj" not in name:
                    self.set_active_model(path)
                    break
            if not self.active_model_path:
                self.set_active_model(models[0][2])
        self.refresh_models_list()

    def find_matching_mmproj(self, model_name: str) -> Path | None:
        """Finds matching multimodal vision projector for a model filename."""
        if not DEFAULT_MODELS_DIR.exists():
            return None
        clean = re.sub(r"(\.[qQ]\d[^.]*|\.gguf)$", "", model_name).lower()
        parts = re.split(r"[-_.\s]+", clean)
        family = parts[0] if parts else ""
        if not family or len(family) < 3:
            return None

        best_match = None
        best_score = 0
        for item in DEFAULT_MODELS_DIR.glob("*.gguf"):
            item_lower = item.name.lower()
            if "mmproj" not in item_lower:
                continue
            if family not in item_lower:
                continue
            score = sum(1 for p in parts if p in item_lower)
            if score > best_score:
                best_score = score
                best_match = item
        return best_match

    def set_active_model(self, model_path: Path):
        self.active_model_path = model_path
        name = model_path.name

        self.active_mmproj_path = self.find_matching_mmproj(name)
        if self.active_mmproj_path and self.active_mmproj_path.exists():
            sub_text = f"{name} (👁️ Vision Projector: {self.active_mmproj_path.name})"
        else:
            sub_text = f"{name} (Text Only)"

        self.row_active_model.set_subtitle(sub_text)
        self.row_config_model.set_subtitle(sub_text)
        self.save_active_config()


class AlforjaApp(Adw.Application):
    def __init__(self):
        super().__init__(
            application_id="io.github.alforja.Alforja",
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )

    def do_activate(self):
        win = self.props.active_window
        if not win:
            win = AlforjaWindow(application=self)
        win.present()


def main():
    app = AlforjaApp()
    return app.run(None)


if __name__ == "__main__":
    main()
