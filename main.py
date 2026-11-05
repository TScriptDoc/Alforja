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

import enum
import json
import os
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

from gi.repository import Adw, Gio, GLib, GObject, Gtk, Pango


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
        self._reader_thread: threading.Thread | None = None

    def set_callbacks(self, log_callback, state_callback):
        self.log_callback = log_callback
        self.state_callback = state_callback

    def _notify_state(self, new_state: ProcessState):
        self.state = new_state
        if self.state_callback:
            GLib.idle_add(self.state_callback, new_state)

    def _append_log(self, text: str):
        if self.log_callback:
            GLib.idle_add(self.log_callback, text)

    def start(self, args: list[str] | None = None):
        if self.state != ProcessState.STOPPED:
            self._append_log("[Alforja] Server is already running or paused.\n")
            return

        if not self.binary_path.is_file():
            self._append_log(
                f"[Alforja Error] Binary not found at: {self.binary_path}\n"
            )
            return

        # Network sandbox: 'unshare -r -n' physically isolates daemon from network
        cmd = ["unshare", "-r", "-n", str(self.binary_path)]
        if args:
            cmd.extend(args)

        # Ensure LD_LIBRARY_PATH discovers libllama and libggml shared libs
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
            self._append_log(
                f"[Alforja] Server started in network sandbox (PID: {self.process.pid}).\n"
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

        # If paused, send SIGCONT first so it can process SIGTERM
        if self.state == ProcessState.PAUSED:
            try:
                os.kill(pid, signal.SIGCONT)
            except Exception:
                pass

        try:
            self.process.terminate()  # SIGTERM

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
                    chunk_size = 1024 * 1024  # 1 MB buffer
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

                    # Atomically promote completed file
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
        self.set_default_size(980, 720)

        self.daemon = DaemonManager()
        self.daemon.set_callbacks(self.on_log_message, self.on_process_state_changed)

        self.downloader = ModelDownloader()
        self.active_model_path: Path | None = None
        self.active_mmproj_path: Path | None = None
        self.model_rows: list[Gtk.Widget] = []
        self.custom_arg_rows: list[tuple[Adw.ActionRow, Gtk.Entry, Gtk.Entry]] = []

        self.profiles: dict = {}
        self.active_profile_name = "GTX 1650 (Default 4GB Safe)"
        self._loading_config = False

        self.setup_ui()
        self.load_profiles_and_config()
        self.init_models()
        self.update_ui_state(ProcessState.STOPPED)

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
    # CONFIGURATION PERSISTENCE & PROFILES
    # =========================================================================
    def load_profiles_and_config(self):
        """Loads profiles and active config from ~/.config/alforja/."""
        self._loading_config = True
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load profiles
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

        # 2. Load active configuration
        active_cfg = None
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    active_cfg = json.load(f)
            except Exception as exc:
                print(f"[Config] Error reading active config: {exc}")

        if not active_cfg:
            active_cfg = self.profiles.get(self.active_profile_name, DEFAULT_PROFILES["GTX 1650 (Default 4GB Safe)"])

        self.apply_config_to_ui(active_cfg)
        self._loading_config = False

    def update_profiles_combo(self):
        """Updates the profiles dropdown model."""
        profile_names = list(self.profiles.keys())
        self.profile_string_list = Gtk.StringList.new(profile_names)
        self.combo_profile.set_model(self.profile_string_list)

        if self.active_profile_name in profile_names:
            idx = profile_names.index(self.active_profile_name)
            self.combo_profile.set_selected(idx)

    def apply_config_to_ui(self, cfg: dict):
        """Applies configuration dictionary values to UI controls."""
        self._loading_config = True

        self.spin_ctx.set_value(cfg.get("ctx_size", 2048))
        self.spin_ngl.set_value(cfg.get("gpu_layers", 24))
        self.spin_threads.set_value(cfg.get("threads", 4))
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

        # Restore custom key-value arguments
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
        """Collects current configuration from all UI widgets."""
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
            "port": int(self.spin_port.get_value()),
            "flash_attn": fa_val,
            "cache_type_k": ctk_val,
            "cache_type_v": ctv_val,
            "ubatch_size": int(self.spin_ubatch.get_value()),
            "batch_size": int(self.spin_batch.get_value()),
            "parallel": int(self.spin_parallel.get_value()),
            "ctx_shift": self.switch_ctx_shift.get_active(),
            "active_model": str(self.active_model_path) if self.active_model_path else "",
            "custom_args": self.get_custom_args(),
        }

    def save_active_config(self):
        """Auto-saves current UI configuration to config.json."""
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
        """Saves current settings into the active profile or a new one."""
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
        """Scans models and selects the first available if none is active."""
        models = self.scan_models()
        if models and not self.active_model_path:
            for name, _, path in models:
                if ".mmproj" not in name:
                    self.set_active_model(path)
                    break
            if not self.active_model_path:
                self.set_active_model(models[0][2])
        self.refresh_models_list()

    def set_active_model(self, model_path: Path):
        self.active_model_path = model_path
        name = model_path.name
        self.row_active_model.set_subtitle(name)
        self.row_config_model.set_subtitle(name)

        # Detect matching vision projector
        base_prefix = name.split(".")[0]
        self.active_mmproj_path = None
        for item in DEFAULT_MODELS_DIR.glob("*.gguf"):
            if ".mmproj" in item.name and base_prefix.lower() in item.name.lower():
                self.active_mmproj_path = item
                break

        self.save_active_config()

    # =========================================================================
    # TAB 1: DASHBOARD
    # =========================================================================
    def build_dashboard_page(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(16)
        box.set_margin_end(16)

        # Status Group
        pref_group = Adw.PreferencesGroup(title="System and Runtime Status")

        self.row_state = Adw.ActionRow(title="Daemon State", subtitle="Stopped")
        self.icon_state = Gtk.Image.new_from_icon_name("network-offline-symbolic")
        self.row_state.add_prefix(self.icon_state)
        pref_group.add(self.row_state)

        self.row_active_model = Adw.ActionRow(
            title="Active Model",
            subtitle="None selected",
        )
        self.row_active_model.add_prefix(
            Gtk.Image.new_from_icon_name("application-x-executable-symbolic")
        )
        pref_group.add(self.row_active_model)

        self.row_sandbox = Adw.ActionRow(
            title="Sandbox Isolation",
            subtitle="Network namespace disabled (unshare -r -n)",
        )
        self.row_sandbox.add_prefix(
            Gtk.Image.new_from_icon_name("security-high-symbolic")
        )
        pref_group.add(self.row_sandbox)

        self.row_vram = Adw.ActionRow(
            title="GPU and VRAM Profile",
            subtitle="Nvidia GTX 1650 (4GB Limit Constraint)",
        )
        self.row_vram.add_prefix(
            Gtk.Image.new_from_icon_name("video-display-symbolic")
        )
        pref_group.add(self.row_vram)

        box.append(pref_group)

        # Log Section
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
        scrolled.set_min_content_height(280)
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

        page = self.view_stack.add_titled(box, "dashboard", "Dashboard")
        page.set_icon_name("system-run-symbolic")

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

        # Context Size (-c)
        self.spin_ctx = Adw.SpinRow.new_with_range(512, 65536, 512)
        self.spin_ctx.set_title("Context Window Size (-c)")
        self.spin_ctx.set_subtitle("Tokens allocated for model memory.")
        self.spin_ctx.set_value(2048)
        self.spin_ctx.connect("notify::value", lambda *_: self.save_active_config())
        core_group.add(self.spin_ctx)

        # GPU Offloading (-ngl)
        self.spin_ngl = Adw.SpinRow.new_with_range(0, 128, 1)
        self.spin_ngl.set_title("GPU Layers to Offload (-ngl)")
        self.spin_ngl.set_subtitle("Layers offloaded to GTX 1650 (4GB VRAM constraint).")
        self.spin_ngl.set_value(24)
        self.spin_ngl.connect("notify::value", lambda *_: self.save_active_config())
        core_group.add(self.spin_ngl)

        # Threads (-t)
        self.spin_threads = Adw.SpinRow.new_with_range(1, 64, 1)
        self.spin_threads.set_title("CPU Threads (-t)")
        self.spin_threads.set_subtitle("Thread count for CPU compute kernels.")
        self.spin_threads.set_value(max(1, (os.cpu_count() or 4) - 2))
        self.spin_threads.connect("notify::value", lambda *_: self.save_active_config())
        core_group.add(self.spin_threads)

        # Port (--port)
        self.spin_port = Adw.SpinRow.new_with_range(1024, 65535, 1)
        self.spin_port.set_title("Server Port (--port)")
        self.spin_port.set_subtitle("Local port binding (isolated within namespace).")
        self.spin_port.set_value(8080)
        self.spin_port.connect("notify::value", lambda *_: self.save_active_config())
        core_group.add(self.spin_port)

        pref_page.add(core_group)

        # Group 3: VRAM & Performance Optimization (Tailored to GTX 1650)
        opt_group = Adw.PreferencesGroup(
            title="VRAM and Performance Optimization",
            description="Tuning options for 4GB VRAM headroom and KV cache efficiency.",
        )

        # Flash Attention (-fa)
        self.combo_flash_attn = Adw.ComboRow(
            title="Flash Attention (-fa)",
            subtitle="Drastically reduces VRAM footprint of attention tensors.",
            model=Gtk.StringList.new(["auto", "on", "off"]),
        )
        self.combo_flash_attn.connect("notify::selected", lambda *_: self.save_active_config())
        opt_group.add(self.combo_flash_attn)

        # KV Cache Type K (-ctk)
        self.combo_ctk = Adw.ComboRow(
            title="KV Cache Quantization K (-ctk)",
            subtitle="q4_0 / q8_0 halves context VRAM usage vs f16.",
            model=Gtk.StringList.new(["f16", "q8_0", "q4_0"]),
        )
        self.combo_ctk.connect("notify::selected", lambda *_: self.save_active_config())
        opt_group.add(self.combo_ctk)

        # KV Cache Type V (-ctv)
        self.combo_ctv = Adw.ComboRow(
            title="KV Cache Quantization V (-ctv)",
            subtitle="Quantizes V tensors in context cache.",
            model=Gtk.StringList.new(["f16", "q8_0", "q4_0"]),
        )
        self.combo_ctv.connect("notify::selected", lambda *_: self.save_active_config())
        opt_group.add(self.combo_ctv)

        # Micro-batch size (-ub)
        self.spin_ubatch = Adw.SpinRow.new_with_range(32, 2048, 32)
        self.spin_ubatch.set_title("Physical Micro-batch Size (-ub)")
        self.spin_ubatch.set_subtitle("Controls peak VRAM allocation spikes during prompt processing.")
        self.spin_ubatch.set_value(256)
        self.spin_ubatch.connect("notify::value", lambda *_: self.save_active_config())
        opt_group.add(self.spin_ubatch)

        # Logical batch size (-b)
        self.spin_batch = Adw.SpinRow.new_with_range(64, 4096, 64)
        self.spin_batch.set_title("Logical Batch Size (-b)")
        self.spin_batch.set_subtitle("Maximum logical batch size.")
        self.spin_batch.set_value(512)
        self.spin_batch.connect("notify::value", lambda *_: self.save_active_config())
        opt_group.add(self.spin_batch)

        # Parallel decoding slots (-np)
        self.spin_parallel = Adw.SpinRow.new_with_range(1, 16, 1)
        self.spin_parallel.set_title("Parallel Slots (-np)")
        self.spin_parallel.set_subtitle("Number of simultaneous inference slots (keep 1 on 4GB VRAM).")
        self.spin_parallel.set_value(1)
        self.spin_parallel.connect("notify::value", lambda *_: self.save_active_config())
        opt_group.add(self.spin_parallel)

        # Context Shift (--ctx-shift)
        self.switch_ctx_shift = Adw.SwitchRow(
            title="Context Shift (--ctx-shift)",
            subtitle="Discards older context tokens to keep chatting without crashing.",
        )
        self.switch_ctx_shift.set_active(True)
        self.switch_ctx_shift.connect("notify::active", lambda *_: self.save_active_config())
        opt_group.add(self.switch_ctx_shift)

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
        """Adds a key-value row for arbitrary CLI flags."""
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
        """Scans ~/.local/llama.cpp/models for .gguf files."""
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
        """Rebuilds the rows in the models group."""
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
            is_mmproj = ".mmproj" in name
            tag = "Vision Projector" if is_mmproj else "Base Model"
            size_str = (
                f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.1f} MB"
            )

            row = Adw.ActionRow(
                title=name,
                subtitle=f"{tag} • {size_str}",
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

            if is_active:
                badge = Gtk.Label(label="Active")
                badge.add_css_class("accent")
                badge.set_valign(Gtk.Align.CENTER)
                badge.set_margin_end(6)
                row.add_suffix(badge)
            else:
                btn_activate = Gtk.Button(label="Activate")
                btn_activate.set_valign(Gtk.Align.CENTER)
                btn_activate.connect(
                    "clicked",
                    lambda _, p=path: self.on_model_activated(p),
                )
                row.add_suffix(btn_activate)

            self.models_group.add(row)
            self.model_rows.append(row)

    def on_model_activated(self, path: Path):
        self.set_active_model(path)
        self.refresh_models_list()
        self.on_log_message(f"[Alforja] Active model set to: {path.name}\n")

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
            self.btn_pause.set_icon_name("media-playback-pause-symbolic")
            self.btn_pause.set_tooltip_text("Pause daemon (SIGSTOP)")

            self.row_state.set_subtitle("Stopped")
            self.icon_state.set_from_icon_name("network-offline-symbolic")

        elif state == ProcessState.RUNNING:
            self.status_badge.set_label("Running")
            self.status_badge.remove_css_class("dim-label")
            self.status_badge.remove_css_class("warning")
            self.status_badge.add_css_class("accent")

            self.btn_start.set_sensitive(False)
            self.btn_pause.set_sensitive(True)
            self.btn_stop.set_sensitive(True)
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
            self.btn_pause.set_icon_name("media-playback-start-symbolic")
            self.btn_pause.set_tooltip_text("Resume daemon (SIGCONT)")

            pid = self.daemon.process.pid if self.daemon.process else "N/A"
            self.row_state.set_subtitle(f"Paused (SIGSTOP frozen, PID: {pid})")
            self.icon_state.set_from_icon_name("media-playback-pause-symbolic")

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
        args = [
            "-m", str(self.active_model_path),
            "-c", str(cfg["ctx_size"]),
            "-ngl", str(cfg["gpu_layers"]),
            "-t", str(cfg["threads"]),
            "--port", str(cfg["port"]),
            "-fa", cfg["flash_attn"],
            "-ctk", cfg["cache_type_k"],
            "-ctv", cfg["cache_type_v"],
            "-ub", str(cfg["ubatch_size"]),
            "-b", str(cfg["batch_size"]),
            "-np", str(cfg["parallel"]),
        ]

        if cfg["ctx_shift"]:
            args.append("--ctx-shift")

        if self.active_mmproj_path and self.active_mmproj_path.exists():
            args.extend(["--mmproj", str(self.active_mmproj_path)])

        for item in cfg["custom_args"]:
            flag = item.get("flag", "").strip()
            val = item.get("value", "").strip()
            if flag:
                args.append(flag)
                if val:
                    args.append(val)

        self.save_active_config()
        self.daemon.start(args)

    def on_pause_clicked(self, _button):
        if self.daemon.state == ProcessState.RUNNING:
            self.daemon.pause()
        elif self.daemon.state == ProcessState.PAUSED:
            self.daemon.resume()

    def on_stop_clicked(self, _button):
        self.daemon.stop()


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
