#!/usr/bin/env python3
"""
Discovery UI
=============
Rich-based terminal UI for the discovery pipeline.

Provides live-updating display with:
- Overall progress bar and metrics
- Per-worker status lines with spinners
- Stage progress visualization
- Results feed

Usage:
    from brandkit.ui import DiscoveryUI

    with DiscoveryUI(target=50) as ui:
        ui.update_stage("generation", "Generating names...")
        ui.update_progress(found=10, generated=100)
        ui.add_worker("domain", 1, "Voltara")
        ui.update_worker("domain", 1, status="checking", details=".com")
        ui.complete_worker("domain", 1, success=True)
        ui.add_result("Voltara", success=True, domain=".com")
"""

import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum

from settings import get_setting

# Check if Rich is available
try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
    from rich.text import Text
    from rich.layout import Layout
    from rich.style import Style
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class WorkerStatus(Enum):
    """Status of a worker."""
    IDLE = "idle"
    QUEUED = "queued"
    WORKING = "working"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class WorkerState:
    """State of a single worker."""
    worker_id: int
    name: Optional[str] = None
    status: WorkerStatus = WorkerStatus.IDLE
    details: str = ""
    tld_status: Dict[str, str] = field(default_factory=dict)  # tld -> status (pending/checking/ok/fail)
    started_at: Optional[float] = None


@dataclass
class StageState:
    """State of a pipeline stage."""
    name: str
    display_name: str
    total: int = 0
    completed: int = 0
    passed: int = 0
    active: bool = False


@dataclass
class ResultEntry:
    """A result entry for the results feed."""
    name: str
    success: bool
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


class DiscoveryUI:
    """
    Rich-based live UI for discovery pipeline.

    Example:
        with DiscoveryUI(target=50, parallel=True) as ui:
            ui.update_stage("generation")
            for name in names:
                ui.add_worker("domain", 1, name)
                # ... do work ...
                ui.update_worker("domain", 1, status="success")
                ui.add_result(name, success=True)
    """

    # Stage definitions
    STAGES = [
        ("generation", "Generating names"),
        ("quality", "Quality filtering"),
        ("similarity", "Similarity check"),
        ("domain", "Domain availability"),
        ("trademark", "Trademark check"),
    ]

    # Spinner frames
    SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    # TLD display order
    TLDS = ["com", "de", "eu", "io", "co"]

    def __init__(self,
                 target: int = 0,
                 batch_size: int = 100,
                 method: str = "blend",
                 profile: str = "camping_rv",
                 parallel: bool = False,
                 max_workers: int = 10,
                 quiet: bool = False,
                 accumulate_mode: bool = False):
        """
        Initialize discovery UI.

        Args:
            target: Target number of valid names (0 = single batch)
            batch_size: Names per batch
            method: Generation method
            profile: Nice class profile
            parallel: Whether parallel mode is enabled
            max_workers: Max concurrent workers
            quiet: Suppress UI (use simple output)
            accumulate_mode: Use accumulate-then-validate display
        """
        self.target = target
        self.batch_size = batch_size
        self.method = method
        self.profile = profile
        self.parallel = parallel
        self.max_workers = max_workers
        self.quiet = quiet
        self.accumulate_mode = accumulate_mode

        # State
        self.found = 0
        self.generated = 0
        self.round = 0
        self.current_stage = 0
        self.start_time = time.time()

        # Accumulation mode state
        self.accumulation_target = 0
        self.accumulation_count = 0
        self.accumulation_generated = 0
        self.accumulation_pass_rate = 0.0
        self.accumulation_quality = ""
        self.in_accumulation = False
        self.in_validation = False
        self.validation_stages = {}  # stage -> (passed, total)

        # Workers state (by stage)
        self.domain_workers: Dict[int, WorkerState] = {}
        self.trademark_workers: Dict[int, WorkerState] = {}

        # Stage states
        self.stages: Dict[str, StageState] = {
            key: StageState(name=key, display_name=display)
            for key, display in self.STAGES
        }

        # Results feed (last N results)
        self.results: List[ResultEntry] = []
        self.max_results = get_setting("ui.max_results")
        if self.max_results is None:
            raise ValueError("ui.max_results must be set in app.yaml")

        # Threading
        self._lock = threading.Lock()
        self._spinner_idx = 0

        # Rich components
        self.console = Console() if RICH_AVAILABLE else None
        self.live: Optional[Live] = None

        # Check if we should use Rich UI
        self.use_rich = (
            RICH_AVAILABLE and
            not quiet and
            sys.stdout.isatty()
        )

    def __enter__(self):
        """Start the live display."""
        if self.use_rich:
            self.live = Live(
                self._render(),
                console=self.console,
                refresh_per_second=10,
                transient=False
            )
            self.live.__enter__()
        else:
            self._print_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live display."""
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)
        return False

    def _print_header(self):
        """Print simple header for non-Rich mode."""
        print("Brandkit Discovery Pipeline")
        print("=" * 60)
        if self.target:
            print(f"Target:   {self.target} valid names")
        print(f"Batch:    {self.batch_size} names using '{self.method}'")
        print(f"Profile:  {self.profile}")
        if self.parallel:
            print(f"Parallel: ENABLED (max {self.max_workers} workers)")
        print("=" * 60)

    def _get_spinner(self) -> str:
        """Get current spinner frame."""
        self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNER_FRAMES)
        return self.SPINNER_FRAMES[self._spinner_idx]

    def _format_elapsed(self) -> str:
        """Format elapsed time."""
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def _format_tld_status(self, tld_status: Dict[str, str]) -> Text:
        """Format TLD status display."""
        text = Text()
        for tld in self.TLDS:
            status = tld_status.get(tld, "pending")
            if status == "pending":
                text.append(f".{tld} ", style="dim")
                text.append("○ ", style="dim")
            elif status == "checking":
                text.append(f".{tld} ", style="yellow")
                text.append("⋯ ", style="yellow")
            elif status == "ok":
                text.append(f".{tld} ", style="green")
                text.append("✓ ", style="green bold")
            elif status == "fail":
                text.append(f".{tld} ", style="red")
                text.append("✗ ", style="red")
        return text

    def _render_header(self) -> Panel:
        """Render the header panel."""
        # Progress percentage
        if self.target > 0:
            pct = min(100, (self.found / self.target) * 100)
            progress_text = f"{self.found}/{self.target}"
        else:
            pct = 0
            progress_text = str(self.found)

        # Build header content
        header = Table.grid(padding=(0, 2))
        header.add_column(justify="left")
        header.add_column(justify="left")
        header.add_column(justify="left")
        header.add_column(justify="right")

        header.add_row(
            Text("🎯 Target: ", style="dim") + Text(str(self.target) if self.target else "batch", style="bold"),
            Text("✓ Found: ", style="dim") + Text(progress_text, style="bold green"),
            Text("🔄 Round: ", style="dim") + Text(str(self.round), style="bold"),
            Text("⏱ ", style="dim") + Text(self._format_elapsed(), style="bold cyan"),
        )

        # Progress bar
        if self.target > 0:
            bar_width = 50
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_bar = Text(f"[{bar}] {pct:.0f}%", style="green" if pct >= 100 else "blue")
        else:
            progress_bar = Text(f"Generated: {self.generated}", style="dim")

        content = Group(header, Text(""), progress_bar)

        return Panel(
            content,
            title="[bold]Brandkit Discovery[/bold]",
            border_style="blue",
            box=box.ROUNDED
        )

    def _render_workers(self, stage: str, workers: Dict[int, WorkerState]) -> Optional[Panel]:
        """Render a workers panel."""
        if not workers and stage not in ("domain", "trademark"):
            return None

        stage_state = self.stages.get(stage)
        if not stage_state or not stage_state.active:
            return None

        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Spinner", width=2)
        table.add_column("Name", width=14)
        table.add_column("Status", min_width=30)

        # Sort workers by ID
        sorted_workers = sorted(workers.values(), key=lambda w: w.worker_id)

        active_count = 0
        for worker in sorted_workers[:self.max_workers]:
            if worker.status == WorkerStatus.IDLE:
                table.add_row(
                    Text("○", style="dim"),
                    Text("(idle)", style="dim"),
                    Text("")
                )
            elif worker.status == WorkerStatus.QUEUED:
                table.add_row(
                    Text("○", style="dim"),
                    Text(worker.name or "", style="dim"),
                    Text("(queued)", style="dim")
                )
            elif worker.status == WorkerStatus.WORKING:
                active_count += 1
                spinner = Text(self._get_spinner(), style="cyan bold")
                name_text = Text(worker.name or "", style="bold")

                if stage == "domain":
                    status = self._format_tld_status(worker.tld_status)
                else:
                    status = Text(worker.details, style="yellow")

                table.add_row(spinner, name_text, status)
            elif worker.status == WorkerStatus.SUCCESS:
                table.add_row(
                    Text("✓", style="green bold"),
                    Text(worker.name or "", style="green"),
                    Text(worker.details, style="green dim")
                )
            elif worker.status == WorkerStatus.FAILED:
                table.add_row(
                    Text("✗", style="red bold"),
                    Text(worker.name or "", style="red"),
                    Text(worker.details, style="red dim")
                )

        # Show idle count if there are idle workers
        idle_count = sum(1 for w in workers.values() if w.status == WorkerStatus.IDLE)
        if idle_count > 0 and len(sorted_workers) < self.max_workers:
            remaining = self.max_workers - len(sorted_workers)
            if remaining > 0:
                table.add_row(
                    Text("○", style="dim"),
                    Text(f"({remaining} idle)", style="dim"),
                    Text("")
                )

        stage_name = self.stages[stage].display_name if stage in self.stages else stage
        title = f"[bold]{stage_name}[/bold]"
        if active_count > 0:
            title += f" ({active_count} active)"

        return Panel(table, title=title, border_style="cyan", box=box.ROUNDED)

    def _render_results(self) -> Optional[Panel]:
        """Render the results panel."""
        if not self.results:
            return None

        text = Text()
        for i, result in enumerate(self.results[-self.max_results:]):
            if i > 0:
                text.append("  ")

            if result.success:
                text.append("✓ ", style="green bold")
                text.append(result.name, style="green")
                if result.detail:
                    text.append(f" ({result.detail})", style="green dim")
            else:
                text.append("✗ ", style="red bold")
                text.append(result.name, style="red")
                if result.detail:
                    text.append(f" ({result.detail})", style="red dim")

        return Panel(text, title="[bold]Results[/bold]", border_style="green", box=box.ROUNDED)

    def _render_stage_progress(self) -> Panel:
        """Render current stage progress."""
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Stage", width=3)
        table.add_column("Name", width=20)
        table.add_column("Progress", min_width=20)

        for i, (key, display) in enumerate(self.STAGES, 1):
            stage = self.stages[key]

            if i < self.current_stage:
                # Completed
                icon = Text("✓", style="green bold")
                name = Text(display, style="green")
                if stage.total > 0:
                    progress = Text(f"{stage.passed}/{stage.total} passed", style="green dim")
                else:
                    progress = Text("done", style="green dim")
            elif i == self.current_stage:
                # Active
                icon = Text(self._get_spinner(), style="cyan bold")
                name = Text(display, style="cyan bold")
                if stage.total > 0:
                    progress = Text(f"{stage.completed}/{stage.total}", style="cyan")
                else:
                    progress = Text("in progress...", style="cyan")
            else:
                # Pending
                icon = Text("○", style="dim")
                name = Text(display, style="dim")
                progress = Text("", style="dim")

            table.add_row(icon, name, progress)

        return Panel(table, title=f"[bold]Round {self.round}[/bold]", border_style="yellow", box=box.ROUNDED)

    def refresh(self):
        """Force a refresh of the display."""
        if self.live:
            self.live.update(self._render())

    # === Public API ===

    def start_round(self, round_num: int):
        """Start a new round."""
        with self._lock:
            self.round = round_num
            self.current_stage = 0
            # Reset stage states
            for stage in self.stages.values():
                stage.total = 0
                stage.completed = 0
                stage.passed = 0
                stage.active = False
            # Clear workers
            self.domain_workers.clear()
            self.trademark_workers.clear()

        if not self.use_rich:
            print(f"\n{'─' * 60}")
            print(f"ROUND {round_num}" + (f" (have {self.found}/{self.target})" if self.target else ""))
            print(f"{'─' * 60}")

        self.refresh()

    def update_stage(self, stage: str, total: int = 0, completed: int = 0, passed: int = 0):
        """Update stage progress."""
        stage_idx = next((i for i, (k, _) in enumerate(self.STAGES, 1) if k == stage), 0)

        with self._lock:
            self.current_stage = stage_idx
            if stage in self.stages:
                self.stages[stage].total = total
                self.stages[stage].completed = completed
                self.stages[stage].passed = passed
                self.stages[stage].active = True

        if not self.use_rich:
            stage_name = self.stages[stage].display_name if stage in self.stages else stage
            print(f"\n[{stage_idx}/5] {stage_name}...")

        self.refresh()

    def complete_stage(self, stage: str, passed: int, total: int):
        """Mark a stage as complete."""
        with self._lock:
            if stage in self.stages:
                self.stages[stage].completed = total
                self.stages[stage].passed = passed
                self.stages[stage].active = False

        if not self.use_rich:
            print(f"      Passed: {passed}, Filtered: {total - passed}")

        self.refresh()

    def update_progress(self, found: int = None, generated: int = None):
        """Update overall progress."""
        with self._lock:
            if found is not None:
                self.found = found
            if generated is not None:
                self.generated = generated
        self.refresh()

    def update_batch_size(self, batch_size: int):
        """Update batch size (for adaptive sizing)."""
        with self._lock:
            self.batch_size = batch_size
        if not self.use_rich:
            print(f"      [adaptive] batch size adjusted to {batch_size}")
        self.refresh()

    def add_worker(self, stage: str, worker_id: int, name: str):
        """Add/update a worker."""
        workers = self.domain_workers if stage == "domain" else self.trademark_workers

        with self._lock:
            workers[worker_id] = WorkerState(
                worker_id=worker_id,
                name=name,
                status=WorkerStatus.WORKING,
                started_at=time.time()
            )
        self.refresh()

    def update_worker(self, stage: str, worker_id: int,
                      status: str = None, details: str = None,
                      tld: str = None, tld_status: str = None):
        """Update worker status."""
        workers = self.domain_workers if stage == "domain" else self.trademark_workers

        with self._lock:
            if worker_id in workers:
                worker = workers[worker_id]
                if status:
                    worker.status = WorkerStatus(status) if isinstance(status, str) else status
                if details:
                    worker.details = details
                if tld and tld_status:
                    worker.tld_status[tld] = tld_status
        self.refresh()

    def complete_worker(self, stage: str, worker_id: int, success: bool, details: str = ""):
        """Mark a worker as complete."""
        workers = self.domain_workers if stage == "domain" else self.trademark_workers

        with self._lock:
            if worker_id in workers:
                worker = workers[worker_id]
                worker.status = WorkerStatus.SUCCESS if success else WorkerStatus.FAILED
                worker.details = details
        self.refresh()

    def remove_worker(self, stage: str, worker_id: int):
        """Remove a worker."""
        workers = self.domain_workers if stage == "domain" else self.trademark_workers

        with self._lock:
            if worker_id in workers:
                del workers[worker_id]
        self.refresh()

    def add_result(self, name: str, success: bool, detail: str = ""):
        """Add a result to the results feed."""
        with self._lock:
            self.results.append(ResultEntry(
                name=name,
                success=success,
                detail=detail
            ))
            # Keep only last N results
            if len(self.results) > self.max_results * 2:
                self.results = self.results[-self.max_results:]

        if not self.use_rich:
            status = "✓" if success else "✗"
            detail_str = f" ({detail})" if detail else ""
            print(f"      {name:<15} {status}{detail_str}")

        self.refresh()

    # === Accumulation Mode API ===

    def start_accumulation(self, round_num: int, target: int, quality: str):
        """Start accumulation phase for a validation round."""
        with self._lock:
            self.round = round_num
            self.accumulation_target = target
            self.accumulation_count = 0
            self.accumulation_generated = 0
            self.accumulation_pass_rate = 0.0
            self.accumulation_quality = quality
            self.in_accumulation = True
            self.in_validation = False
            self.validation_stages = {}
            # Clear workers
            self.domain_workers.clear()
            self.trademark_workers.clear()

        if not self.use_rich:
            print(f"\n{'─' * 60}")
            print(f"ROUND {round_num} - ACCUMULATING {quality.upper()} CANDIDATES")
            print(f"{'─' * 60}")

        self.refresh()

    def update_accumulation(self, accumulated: int, target: int, generated: int, pass_rate: float):
        """Update accumulation progress."""
        with self._lock:
            self.accumulation_count = accumulated
            self.accumulation_target = target
            self.accumulation_generated = generated
            self.accumulation_pass_rate = pass_rate

        if not self.use_rich:
            # Only print every 500 generated to avoid spam
            if generated % 500 == 0 or accumulated == target:
                print(f"      Accumulated: {accumulated}/{target} ({pass_rate:.1f}% pass rate, {generated} generated)")

        self.refresh()

    def start_validation(self, candidate_count: int):
        """Start validation phase."""
        with self._lock:
            self.in_accumulation = False
            self.in_validation = True
            self.validation_stages = {}

        if not self.use_rich:
            print(f"\nVALIDATING {candidate_count} candidates...")

        self.refresh()

    def update_validation_stage(self, stage: str, passed: int, total: int):
        """Update validation stage progress."""
        with self._lock:
            self.validation_stages[stage] = (passed, total)

        if not self.use_rich:
            print(f"      {stage}: {passed}/{total} passed")

        self.refresh()

    def _render_accumulation(self) -> Panel:
        """Render accumulation progress panel."""
        pct = (self.accumulation_count / self.accumulation_target * 100) if self.accumulation_target > 0 else 0

        # Progress bar
        bar_width = 40
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        content = Text()
        content.append(f"Accumulating ", style="dim")
        content.append(f"{self.accumulation_quality.upper()}", style="bold cyan")
        content.append(f" candidates\n\n", style="dim")
        content.append(f"[{bar}] ", style="cyan")
        content.append(f"{self.accumulation_count}/{self.accumulation_target}\n\n", style="bold")
        content.append(f"Generated: ", style="dim")
        content.append(f"{self.accumulation_generated:,}", style="bold")
        content.append(f"  │  Pass rate: ", style="dim")
        content.append(f"{self.accumulation_pass_rate:.2f}%", style="bold yellow" if self.accumulation_pass_rate < 1 else "bold green")

        return Panel(content, title=f"[bold]Phase 1: Accumulate[/bold]", border_style="cyan", box=box.ROUNDED)

    def _render_validation(self) -> Panel:
        """Render validation progress panel."""
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Stage", width=3)
        table.add_column("Name", width=15)
        table.add_column("Progress", min_width=15)

        stage_order = ["similarity", "domain", "trademark"]
        stage_names = {"similarity": "Similarity", "domain": "Domain", "trademark": "Trademark"}

        for stage in stage_order:
            if stage in self.validation_stages:
                passed, total = self.validation_stages[stage]
                icon = Text("✓", style="green bold")
                name = Text(stage_names[stage], style="green")
                progress = Text(f"{passed}/{total} passed", style="green dim")
            else:
                icon = Text("○", style="dim")
                name = Text(stage_names[stage], style="dim")
                progress = Text("", style="dim")

            table.add_row(icon, name, progress)

        return Panel(table, title=f"[bold]Phase 2: Validate[/bold]", border_style="yellow", box=box.ROUNDED)

    def _render(self) -> Group:
        """Render the complete UI."""
        with self._lock:
            components = [self._render_header()]

            if self.accumulate_mode:
                # Accumulate mode rendering
                if self.in_accumulation:
                    components.append(self._render_accumulation())
                elif self.in_validation:
                    components.append(self._render_validation())
            else:
                # Round-based mode rendering
                if self.round > 0:
                    components.append(self._render_stage_progress())

            # Worker panels (for parallel mode)
            if self.parallel and self.in_validation:
                domain_panel = self._render_workers("domain", self.domain_workers)
                if domain_panel:
                    components.append(domain_panel)

                trademark_panel = self._render_workers("trademark", self.trademark_workers)
                if trademark_panel:
                    components.append(trademark_panel)

            # Results
            results_panel = self._render_results()
            if results_panel:
                components.append(results_panel)

            return Group(*components)

    def print_summary(self, viable_names: List[tuple]):
        """Print final summary."""
        if self.live:
            self.live.stop()

        self.console.print()
        self.console.print("=" * 60, style="bold")
        self.console.print("DISCOVERY COMPLETE", style="bold green")
        self.console.print("=" * 60, style="bold")
        self.console.print()
        self.console.print(f"Rounds:    {self.round}")
        self.console.print(f"Generated: {self.generated}")
        self.console.print(f"Found:     {self.found}", style="bold green")
        self.console.print(f"Time:      {self._format_elapsed()}")

        if viable_names:
            self.console.print()
            self.console.print(f"VIABLE NAMES ({len(viable_names)}):", style="bold")

            table = Table(box=box.SIMPLE)
            table.add_column("Name", style="bold")
            table.add_column("Score", justify="right")
            table.add_column("Quality")
            table.add_column("Domains")

            for name_str, score, quality, domains in sorted(viable_names, key=lambda x: -x[1]):
                com_flag = "* " if "com" in domains else "  "
                domain_str = com_flag + ", ".join(domains[:3])

                quality_style = {
                    "excellent": "bold green",
                    "good": "green",
                    "acceptable": "yellow",
                    "poor": "red"
                }.get(quality, "")

                table.add_row(
                    name_str,
                    f"{score:.2f}",
                    Text(quality, style=quality_style),
                    domain_str
                )

            self.console.print(table)
            self.console.print("\n* = .com available", style="dim")

    def log(self, message: str, style: str = None):
        """Log a message (for debugging/info)."""
        if self.use_rich:
            self.console.log(message, style=style)
        else:
            print(message)


# === Simple fallback for non-TTY ===

class SimpleUI:
    """Simple non-interactive UI fallback."""

    def __init__(self, **kwargs):
        self.target = kwargs.get('target', 0)
        self.found = 0
        self.generated = 0
        self.round = 0
        self.accumulate_mode = kwargs.get('accumulate_mode', False)
        self._last_accumulation_print = 0

    def __enter__(self):
        print("Brandkit Discovery Pipeline")
        print("=" * 60)
        return self

    def __exit__(self, *args):
        return False

    def start_round(self, round_num: int):
        self.round = round_num
        print(f"\n{'─' * 60}")
        print(f"ROUND {round_num}")

    def update_stage(self, stage: str, **kwargs):
        print(f"\n[{stage}] Processing...")

    def complete_stage(self, stage: str, passed: int, total: int):
        print(f"      Passed: {passed}/{total}")

    def update_progress(self, found: int = None, generated: int = None):
        if found is not None:
            self.found = found
        if generated is not None:
            self.generated = generated

    def update_batch_size(self, batch_size: int):
        print(f"      [adaptive] batch size adjusted to {batch_size}")

    def add_worker(self, *args, **kwargs):
        pass

    def update_worker(self, *args, **kwargs):
        pass

    def complete_worker(self, *args, **kwargs):
        pass

    def remove_worker(self, *args, **kwargs):
        pass

    def add_result(self, name: str, success: bool, detail: str = ""):
        status = "CLEAR" if success else "CONFLICT"
        print(f"      {name:<15} {status}")

    def print_summary(self, viable_names):
        print(f"\n{'=' * 60}")
        print("DISCOVERY COMPLETE")
        print(f"Generated: {self.generated}")
        print(f"Found: {self.found}")
        if viable_names:
            print(f"\nVIABLE NAMES ({len(viable_names)}):")
            for name_str, score, quality, domains in sorted(viable_names, key=lambda x: -x[1]):
                com_flag = "* " if "com" in domains else "  "
                print(f"  {com_flag}{name_str:<15} {score:.2f} ({quality})")

    def refresh(self):
        pass

    def log(self, message: str, **kwargs):
        print(message)

    # === Accumulation Mode API ===

    def start_accumulation(self, round_num: int, target: int, quality: str):
        self.round = round_num
        self._last_accumulation_print = 0
        print(f"\n{'─' * 60}")
        print(f"ROUND {round_num} - ACCUMULATING {quality.upper()} CANDIDATES (target: {target})")
        print(f"{'─' * 60}")

    def update_accumulation(self, accumulated: int, target: int, generated: int, pass_rate: float):
        # Only print every 500 generated to avoid spam
        if generated - self._last_accumulation_print >= 500 or accumulated >= target:
            print(f"      Accumulated: {accumulated}/{target} ({pass_rate:.2f}% pass rate, {generated:,} generated)")
            self._last_accumulation_print = generated

    def start_validation(self, candidate_count: int):
        print(f"\nVALIDATING {candidate_count} candidates...")

    def update_validation_stage(self, stage: str, passed: int, total: int):
        print(f"      {stage}: {passed}/{total} passed")


def get_ui(parallel: bool = False, quiet: bool = False, **kwargs) -> DiscoveryUI:
    """Get appropriate UI based on environment."""
    if quiet or not sys.stdout.isatty() or not RICH_AVAILABLE:
        return SimpleUI(**kwargs)
    return DiscoveryUI(parallel=parallel, quiet=quiet, **kwargs)


__all__ = [
    'DiscoveryUI',
    'SimpleUI',
    'get_ui',
    'WorkerStatus',
    'RICH_AVAILABLE',
]
