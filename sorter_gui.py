"""Window for the people sorter: pick a folder, press Start.

This is a front end for :mod:`people_sorter`. It runs that program as a
separate process, so the sorting rules, folder layout and report are exactly
the ones described there, and a run can be paused or stopped at any moment. Photos are
always copied; the originals are never moved or changed.

Usage::

    python sorter_gui.py
"""

from __future__ import annotations

import codecs
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Final

import config
import people_sorter
import quality_rules as rules

WINDOW_TITLE: Final[str] = "People Sorter"
SORTER_SCRIPT: Final[Path] = config.PROJECT_ROOT / "people_sorter.py"

#: Suffix of the output folder suggested next to the chosen input folder.
OUTPUT_SUFFIX: Final[str] = "_sorted"
DEFAULT_SAMPLE_SIZE: Final[int] = 300

#: How often the window collects output from the running sorter.
_POLL_MS: Final[int] = 100
#: Grace period after Stop before the sorter is ended forcibly.
_STOP_TIMEOUT_MS: Final[int] = 5000
#: Free space wanted on top of the input size before a full run starts.
_SPACE_MARGIN: Final[float] = 1.1

#: Pausing suspends the sorter process, which Windows has no signal for.
_CAN_PAUSE: Final[bool] = hasattr(signal, "SIGSTOP")

#: A tqdm progress line, e.g. ``Analysing:  12%|#2   | 37/300 [00:20<02:10, ...]``.
_PROGRESS_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<phase>[A-Za-z ]+):\s*\d+%\|.*\|\s*(?P<done>\d+)/(?P<total>\d+)"
    r"\s*\[[^<\]]*(?:<(?P<left>[^,\]]+))?"
)

#: Start-up chatter from the model libraries, kept out of the message box.
_NOISE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[WIE]\d{4} |WARNING: All log messages before absl|INFO: Created TensorFlow Lite)"
)

#: Advanced settings: (command line flag, parser destination, label).
_ADVANCED: Final[tuple[tuple[str, str, str], ...]] = (
    ("--yaw", "yaw", "Head turn that makes a photo bad (degrees)"),
    ("--pitch-down", "pitch_down", "Looking-down angle that makes a photo bad (degrees)"),
    ("--eye-openness", "eye_openness", "Eyes count as closed at or below (0-1)"),
    ("--similar-threshold", "similar_threshold", "Similarity needed to group photos (0-1)"),
)


def suggested_output(input_dir: Path) -> Path:
    """Return the output folder offered for ``input_dir``: a sibling of it."""
    return input_dir.parent / f"{input_dir.name}{OUTPUT_SUFFIX}"


def existing_results(output_dir: Path) -> list[Path]:
    """Return the result folders an earlier run left in ``output_dir``."""
    return [output_dir / name for name in people_sorter.MANAGED_DIRS
            if (output_dir / name).exists()]


def folder_size(folder: Path) -> int:
    """Return the total size in bytes of the supported images under ``folder``."""
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if Path(name).suffix.lower() in config.SUPPORTED_EXTENSIONS:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
    return total


def free_space(folder: Path) -> int:
    """Return the free bytes on the disk that will hold ``folder``."""
    probe = folder.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def open_folder(folder: Path) -> None:
    """Show ``folder`` in the system file browser."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    elif os.name == "nt":
        os.startfile(str(folder))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(folder)])


class SorterWindow:
    """The main window and the sorter process it controls."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.process: subprocess.Popen[bytes] | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.stopping = False
        self.paused = False
        self.last_output: Path | None = None
        #: The output path last filled in automatically, to tell it from a
        #: folder the user chose themselves.
        self.auto_output = ""
        self.status_before_pause = ""

        defaults = people_sorter.build_parser()
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.sample_var = tk.BooleanVar(value=False)
        self.sample_size_var = tk.StringVar(value=str(DEFAULT_SAMPLE_SIZE))
        self.candid_var = tk.BooleanVar(value=False)
        self.similar_var = tk.BooleanVar(value=True)
        self.sheets_var = tk.BooleanVar(value=True)
        #: One tick box per class, grouped by the command line flag it feeds.
        self.class_vars: dict[str, dict[str, tk.BooleanVar]] = {
            flag: {name: tk.BooleanVar(value=True) for name in names}
            for flag, names in (
                ("--verdicts", rules.VERDICTS),
                ("--people", rules.PEOPLE_CATEGORIES),
                ("--other", people_sorter.OTHER_CATEGORIES),
            )
        }
        self.fresh_var = tk.BooleanVar(value=False)
        self.advanced_vars = {
            flag: tk.StringVar(value=str(defaults.get_default(dest)))
            for flag, dest, _label in _ADVANCED
        }
        self.status_var = tk.StringVar(value="Choose the folder of photos to sort.")

        root.title(WINDOW_TITLE)
        root.minsize(720, 680)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Photos folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.input_var).grid(
            row=0, column=1, sticky="ew", padx=8)
        self.input_button = ttk.Button(frame, text="Browse…", command=self.choose_input)
        self.input_button.grid(row=0, column=2)

        ttk.Label(frame, text="Save sorted copies in").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.output_var).grid(
            row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        self.output_button = ttk.Button(frame, text="Browse…", command=self.choose_output)
        self.output_button.grid(row=1, column=2, pady=(8, 0))

        options = ttk.LabelFrame(frame, text="Options", padding=10)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        sample = ttk.Frame(options)
        sample.pack(anchor="w")
        ttk.Checkbutton(sample, text="Test on a random sample of",
                        variable=self.sample_var).pack(side="left")
        ttk.Spinbox(sample, from_=1, to=100000, width=7,
                    textvariable=self.sample_size_var).pack(side="left", padx=6)
        ttk.Label(sample, text="photos first").pack(side="left")
        ttk.Checkbutton(
            options, variable=self.candid_var,
            text="Wedding candids: send failing candid photos to review instead of bad",
        ).pack(anchor="w", pady=(6, 0))

        folders = ttk.LabelFrame(frame, text="Folders to create", padding=10)
        folders.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        extras = {
            people_sorter.SIMILAR_DIRNAME: self.similar_var,
            people_sorter.SHEETS_DIRNAME: self.sheets_var,
        }
        for row, (title, boxes) in enumerate((
            ("Quality", self.class_vars["--verdicts"]),
            ("People in the photo", self.class_vars["--people"]),
            ("Other", {**self.class_vars["--other"], **extras}),
        )):
            ttk.Label(folders, text=title).grid(row=row, column=0, sticky="w", pady=2)
            for column, (name, variable) in enumerate(boxes.items(), start=1):
                ttk.Checkbutton(folders, text=name, variable=variable).grid(
                    row=row, column=column, sticky="w", padx=(12, 0), pady=2)
        ttk.Label(
            folders, foreground="gray",
            text=f"A photo of people needs its quality and its people type ticked. "
                 f"Every other photo is copied to '{people_sorter.OTHERS_DIRNAME}'.",
        ).grid(row=3, column=0, columnspan=5, sticky="w", pady=(6, 0))

        self.advanced_button = ttk.Button(frame, text="Show advanced settings",
                                          command=self.toggle_advanced)
        self.advanced_button.grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.advanced = ttk.LabelFrame(frame, text="Advanced", padding=10)
        for row, (flag, _dest, label) in enumerate(_ADVANCED):
            ttk.Label(self.advanced, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(self.advanced, width=8, textvariable=self.advanced_vars[flag]).grid(
                row=row, column=1, sticky="w", padx=8, pady=2)
        ttk.Checkbutton(
            self.advanced, variable=self.fresh_var,
            text="Analyse every photo again (ignore saved analysis)",
        ).grid(row=len(_ADVANCED), column=0, columnspan=2, sticky="w", pady=(6, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        self.start_button = ttk.Button(buttons, text="Start", command=self.start)
        self.start_button.pack(side="left")
        self.pause_button = ttk.Button(buttons, text="Pause", command=self.toggle_pause,
                                       state="disabled")
        self.pause_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.open_button = ttk.Button(buttons, text="Open output folder",
                                      command=self.open_output, state="disabled")
        self.open_button.pack(side="right")

        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Label(frame, textvariable=self.status_var).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(4, 0))

        log_frame = ttk.Frame(frame)
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        frame.rowconfigure(9, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled",
                           font=("Menlo", 11) if sys.platform == "darwin" else "TkFixedFont")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

    def toggle_advanced(self) -> None:
        """Show or hide the advanced settings."""
        if self.advanced.winfo_ismapped():
            self.advanced.grid_remove()
            self.advanced_button.configure(text="Show advanced settings")
        else:
            self.advanced.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0))
            self.advanced_button.configure(text="Hide advanced settings")

    # ------------------------------------------------------------------ #
    # Folders
    # ------------------------------------------------------------------ #

    def choose_input(self) -> None:
        """Ask for the photos folder and suggest an output folder beside it."""
        chosen = filedialog.askdirectory(title="Choose the folder of photos",
                                         initialdir=self.input_var.get() or None)
        if not chosen:
            return
        self.input_var.set(chosen)
        if self.output_var.get().strip() in ("", self.auto_output):
            self.auto_output = str(suggested_output(Path(chosen)))
            self.output_var.set(self.auto_output)
        self.status_var.set("Ready. Press Start.")

    def choose_output(self) -> None:
        """Ask where the sorted copies should be saved."""
        chosen = filedialog.askdirectory(title="Choose where to save the sorted copies",
                                         initialdir=self.output_var.get() or None)
        if chosen:
            self.output_var.set(chosen)

    def open_output(self) -> None:
        """Show the last run's output folder."""
        if self.last_output and self.last_output.is_dir():
            open_folder(self.last_output)

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #

    def build_command(self, input_dir: Path, output_dir: Path, *, clean: bool) -> list[str]:
        """Return the people sorter command line for the current settings.

        Raises:
            ValueError: If a setting is not a valid number, or no class is ticked.
        """
        command = [sys.executable, "-u", str(SORTER_SCRIPT),
                   "--input", str(input_dir), "--output", str(output_dir),
                   "--transfer", "copy"]
        if clean:
            command.append("--clean")
        if self.sample_var.get():
            try:
                size = int(self.sample_size_var.get())
            except ValueError:
                size = 0
            if size < 1:
                raise ValueError("The sample size must be a whole number of photos.")
            command += ["--limit", str(size), "--shuffle"]
        if self.candid_var.get():
            command.append("--candid-lenient")
        ticked = {flag: [name for name, variable in boxes.items() if variable.get()]
                  for flag, boxes in self.class_vars.items()}
        if not ((ticked["--verdicts"] and ticked["--people"]) or ticked["--other"]):
            raise ValueError(
                "Tick at least one quality together with one people type, or one of "
                f"{' / '.join(people_sorter.OTHER_CATEGORIES)}.")
        for flag, names in ticked.items():
            command += [flag, *names]
        if not self.similar_var.get():
            command.append("--no-similar")
        if not self.sheets_var.get():
            command.append("--no-sheets")
        if self.fresh_var.get():
            command.append("--fresh")
        for flag, _dest, label in _ADVANCED:
            try:
                value = float(self.advanced_vars[flag].get())
            except ValueError:
                raise ValueError(f"'{label}' must be a number.") from None
            command += [flag, str(value)]
        return command

    def start(self) -> None:
        """Check the settings and launch the sorter."""
        if self.process is not None:
            return
        input_text = self.input_var.get().strip()
        output_text = self.output_var.get().strip()
        if not input_text or not Path(input_text).is_dir():
            messagebox.showerror(WINDOW_TITLE, "Choose the folder that holds the photos.")
            return
        if not output_text:
            messagebox.showerror(WINDOW_TITLE, "Choose where to save the sorted copies.")
            return
        input_dir, output_dir = Path(input_text).resolve(), Path(output_text).resolve()
        if input_dir == output_dir:
            messagebox.showerror(
                WINDOW_TITLE, "The sorted copies need a different folder from the photos.")
            return

        clean = False
        if existing_results(output_dir):
            if people_sorter.read_state(output_dir).get("transfer") == "move":
                messagebox.showerror(
                    WINDOW_TITLE,
                    f"{output_dir}\n\nholds original photos moved there by an earlier "
                    "run, so it will not be replaced. Choose another folder for the "
                    "sorted copies.")
                return
            if not messagebox.askyesno(
                    WINDOW_TITLE,
                    f"{output_dir}\n\nalready contains sorted results. Replace them?\n\n"
                    "Only the sorted copies are deleted. Your original photos and the "
                    "saved analysis are kept. Choose No to pick another folder."):
                return
            clean = True

        try:
            command = self.build_command(input_dir, output_dir, clean=clean)
        except ValueError as error:
            messagebox.showerror(WINDOW_TITLE, str(error))
            return
        if not self.sample_var.get() and not self._enough_space(input_dir, output_dir):
            return

        self._clear_log()
        try:
            self.process = subprocess.Popen(
                command, cwd=config.PROJECT_ROOT, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except OSError as error:
            messagebox.showerror(WINDOW_TITLE, f"Could not start the sorter: {error}")
            return
        self.stopping = False
        self.paused = False
        self.last_output = output_dir
        self._set_running(True)
        self.status_var.set("Starting…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(15)
        threading.Thread(target=self._read_output, args=(self.process,),
                         daemon=True).start()
        self.root.after(_POLL_MS, self._poll)

    def _enough_space(self, input_dir: Path, output_dir: Path) -> bool:
        """Warn when the disk may be too small for a copy of every photo."""
        needed = int(folder_size(input_dir) * _SPACE_MARGIN)
        free = free_space(output_dir)
        if free >= needed:
            return True
        gigabyte = 1024 ** 3
        return messagebox.askyesno(
            WINDOW_TITLE,
            f"Copying these photos needs about {needed / gigabyte:.1f} GB, but only "
            f"{free / gigabyte:.1f} GB is free where the copies will be saved.\n\n"
            "Start anyway?")

    def _read_output(self, process: subprocess.Popen[bytes]) -> None:
        """Forward the sorter's output to the window, one line at a time.

        Runs on a background thread. Progress bars redraw with a carriage
        return, so that counts as a line break too.
        """
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            *complete, pending = re.split(r"[\r\n]", pending + decoder.decode(chunk))
            for line in complete:
                self.lines.put(line)
        if pending:
            self.lines.put(pending)
        process.wait()
        self.lines.put(None)

    def _poll(self) -> None:
        """Show whatever the sorter has printed since the last poll."""
        finished = False
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                finished = True
            elif line.strip():
                self._handle_line(line)
        if finished:
            self._finish()
        else:
            self.root.after(_POLL_MS, self._poll)

    def _handle_line(self, line: str) -> None:
        match = _PROGRESS_RE.match(line.strip())
        if not match:
            if not _NOISE_RE.match(line.strip()):
                self._append_log(line.rstrip())
            return
        done, total = int(match["done"]), int(match["total"])
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.stop()
            self.progress.configure(mode="determinate")
        self.progress.configure(maximum=max(total, 1), value=done)
        status = f"{match['phase'].strip()} photos: {done} of {total}"
        left = (match["left"] or "").strip()
        if left and "?" not in left and done < total:
            status += f"  ({left} left)"
        if self.paused:
            # Output printed just before the pause must not hide the notice.
            self.status_before_pause = status
        else:
            self.status_var.set(status)

    def toggle_pause(self) -> None:
        """Freeze the running sorter, or let a frozen one carry on.

        The process is suspended where it stands, so nothing is lost and
        Resume continues with the very next photo.
        """
        if self.process is None or self.stopping or not _CAN_PAUSE:
            return
        self.paused = not self.paused
        self.process.send_signal(signal.SIGSTOP if self.paused else signal.SIGCONT)
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        indeterminate = str(self.progress.cget("mode")) == "indeterminate"
        if self.paused:
            self.status_before_pause = self.status_var.get()
            self.status_var.set("Paused. Press Resume to continue.")
            if indeterminate:
                self.progress.stop()
        else:
            self.status_var.set(self.status_before_pause)
            if indeterminate:
                self.progress.start(15)

    def stop(self) -> None:
        """Ask the running sorter to stop."""
        if self.process is None or self.stopping:
            return
        if self.paused:
            self.toggle_pause()  # a suspended process cannot act on the request
        self.stopping = True
        self.stop_button.configure(state="disabled")
        self.pause_button.configure(state="disabled")
        self.status_var.set("Stopping…")
        process = self.process
        if os.name == "nt":
            process.terminate()
        else:
            process.send_signal(signal.SIGINT)
            self.root.after(_STOP_TIMEOUT_MS, lambda: self._force_stop(process))

    @staticmethod
    def _force_stop(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.kill()

    def _finish(self) -> None:
        """Report how the run ended and unlock the window."""
        assert self.process is not None
        code = self.process.returncode
        self.process = None
        self.paused = False
        self.pause_button.configure(text="Pause")
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self._set_running(False)
        has_output = bool(self.last_output and self.last_output.is_dir())
        self.open_button.configure(state="normal" if has_output else "disabled")
        if self.stopping:
            self.progress.configure(value=0)
            self.status_var.set(
                "Stopped. Press Start to continue; photos already analysed are not redone.")
        elif code == 0:
            self.progress.configure(maximum=1, value=1)
            self.status_var.set(f"Finished. Sorted copies are in {self.last_output}")
        else:
            self.progress.configure(value=0)
            self.status_var.set("The sorter stopped with an error. See the messages below.")

    def on_close(self) -> None:
        """Close the window, ending a run in progress if the user agrees."""
        if self.process is not None:
            if not messagebox.askyesno(
                    WINDOW_TITLE, "Sorting is still running. Stop it and close?"):
                return
            self.process.kill()
        self.root.destroy()

    # ------------------------------------------------------------------ #
    # Widgets
    # ------------------------------------------------------------------ #

    def _set_running(self, running: bool) -> None:
        idle_state = "disabled" if running else "normal"
        for button in (self.start_button, self.input_button, self.output_button):
            button.configure(state=idle_state)
        self.stop_button.configure(state="normal" if running else "disabled")
        self.pause_button.configure(
            state="normal" if running and _CAN_PAUSE else "disabled")
        if running:
            self.open_button.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def main() -> int:
    """Program entry point."""
    root = tk.Tk()
    SorterWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
