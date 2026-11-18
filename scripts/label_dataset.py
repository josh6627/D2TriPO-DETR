#!/usr/bin/env python3
"""Local desktop labeler. Run in the d2tripo environment; no GPU/server needed."""

import argparse
import copy
import json
import math
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk

from PIL import Image, ImageTk

try:
    from .label_project import Project, grasp_rectangle, validate, edge_parts, edge_description, RELATION_NAMES
except ImportError:
    from label_project import Project, grasp_rectangle, validate, edge_parts, edge_description, RELATION_NAMES


MAX_RENDER_PIXELS = 8_000_000
MAX_RENDER_EDGE = 8192
PREVIEW_INTERVAL_MS = 33
LOGGER = logging.getLogger("d2tripo.labeler")


def bounded_zoom(size, requested):
    """Limit display allocations independently of original annotation coordinates."""
    width, height = size
    return min(
        max(0.25, min(5, requested)),
        math.sqrt(MAX_RENDER_PIXELS / (width * height)),
        MAX_RENDER_EDGE / max(width, height),
    )


def configure_diagnostics(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=2)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("START pid=%s python=%s", os.getpid(), sys.executable)


class Labeler:
    def __init__(self, root, project, font_size=16):
        self.root, self.project = root, project
        self.font_size = font_size
        # Retain the interpreter's native fonts and ttk theme; only increase size.
        self.font_family = tkfont.nametofont("TkDefaultFont", root=root).actual(
            "family"
        )
        for name in (
            "TkDefaultFont",
            "TkTextFont",
            "TkMenuFont",
            "TkHeadingFont",
            "TkCaptionFont",
        ):
            tkfont.nametofont(name, root=root).configure(size=font_size)
        root.option_add("*Listbox.font", "TkTextFont")
        style = ttk.Style(root)
        style.configure(".", font="TkDefaultFont")
        style.configure("TButton", padding=(8, 5))
        self.heading_font = (self.font_family, font_size + 2, "bold")
        self.index, self.selected, self.scale = 0, None, 1.0
        self.pending, self.history, self.resizing = [], [], None
        self._photo_key, self._image_item = None, None
        self._showing_error = False
        self._preview_item = self._preview_kind = self._preview_after = None
        self._preview_request = self._preview_options = None
        self._preview_last = 0.0
        self._cursor = ""
        self._test_color = None
        root.title("D2TriPO-DETR | Dataset labeler")
        root.geometry(
            f"{min(1500, root.winfo_screenwidth()-60)}x{min(1000, root.winfo_screenheight()-80)}"
        )
        root.report_callback_exception = self.report_error
        self.mode = tk.StringVar(value="select")
        self.class_name = tk.StringVar(value=project.classes[0])
        self.class_search, self.class_matches = tk.StringVar(), tk.StringVar()
        self.difficulty = tk.StringVar(value="e")
        self.reviewed = tk.BooleanVar()
        self.relationship_confirmation = tk.StringVar(value="Confirm relationships")
        self.require_grasps = tk.BooleanVar(value=project.require_grasps)
        self.status, self.instruction = tk.StringVar(), tk.StringVar()
        self.zoom_text = tk.StringVar(value="100%")
        self.use_for_test = tk.BooleanVar()
        self.test_status = tk.StringVar()

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="< Previous", command=lambda: self.navigate(-1)).pack(
            side="left"
        )
        ttk.Button(top, text="Next >", command=lambda: self.navigate(1)).pack(
            side="left"
        )
        self.image_choice = ttk.Combobox(top, state="readonly", width=24)
        self.image_choice.pack(side="left", padx=8)
        self.image_choice.bind(
            "<<ComboboxSelected>>", lambda e: self.load(self.image_choice.current())
        )
        ttk.Button(top, text="Next unfinished", command=self.next_unfinished).pack(
            side="left"
        )
        actions = ttk.Frame(root, padding=(8, 0, 8, 8))
        actions.pack(fill="x")
        ttk.Button(actions, text="Mark complete", command=self.complete).pack(
            side="left"
        )
        self.confirm_button = ttk.Button(
            actions,
            textvariable=self.relationship_confirmation,
            command=self.confirm_relationships,
        )
        self.confirm_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Export completed...", command=self.export).pack(
            side="right"
        )
        split_controls = ttk.Frame(root, padding=(8, 0, 8, 8))
        split_controls.pack(fill="x")
        self.test_checkbox = ttk.Checkbutton(
            split_controls,
            text="Use this image for TEST",
            variable=self.use_for_test,
            command=self.toggle_test_image,
        )
        self.test_checkbox.pack(side="left")
        split_button = ttk.Button(
            split_controls, text="Split ratios...", command=self.edit_split_ratios
        )
        split_button.pack(side="right")
        if project.fixed_splits is not None:
            self.test_checkbox.state(["disabled"])
            split_button.state(["disabled"])
        self.test_status_label = ttk.Label(
            root, textvariable=self.test_status, padding=(8, 0, 8, 8), wraplength=1300
        )
        self.test_status_label.pack(fill="x")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body)
        sidebar.pack(side="right", fill="y")
        self.side_canvas = tk.Canvas(sidebar, width=520, highlightthickness=0)
        side_scroll = ttk.Scrollbar(
            sidebar, orient="vertical", command=self.side_canvas.yview
        )
        side_scroll.pack(side="right", fill="y")
        self.side_canvas.pack(side="left", fill="both", expand=True)
        self.side_canvas.configure(yscrollcommand=side_scroll.set)
        side = ttk.Frame(self.side_canvas, padding=12)
        side_window = self.side_canvas.create_window(0, 0, window=side, anchor="nw")
        side.bind(
            "<Configure>",
            lambda e: self.side_canvas.configure(
                scrollregion=self.side_canvas.bbox("all")
            ),
        )
        self.side_canvas.bind(
            "<Configure>",
            lambda e: self.side_canvas.itemconfigure(side_window, width=e.width),
        )
        ttk.Checkbutton(
            side,
            text="Require grasps for completion",
            variable=self.require_grasps,
            command=self.change_grasp_mode,
        ).pack(anchor="w", pady=(0, 10))
        ttk.Label(side, text="1. Object boxes", font=self.heading_font).pack(anchor="w")
        ttk.Label(side, text="Search object classes:").pack(anchor="w", pady=(8, 0))
        search_row = ttk.Frame(side)
        search_row.pack(fill="x", pady=5)
        self.search_entry = ttk.Entry(search_row, textvariable=self.class_search)
        self.search_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(
            search_row, text="Clear", command=lambda: self.class_search.set("")
        ).pack(side="right", padx=(5, 0))
        self.classes = ttk.Combobox(
            side,
            textvariable=self.class_name,
            values=project.classes,
            state="readonly",
            width=28,
        )
        self.classes.pack(fill="x", pady=5)
        ttk.Label(side, textvariable=self.class_matches, wraplength=480).pack(
            anchor="w"
        )
        self.class_search.trace_add("write", self.filter_classes)
        self.search_entry.bind("<Return>", self.focus_class_picker)
        self.search_entry.bind("<Down>", self.focus_class_picker)
        self.filter_classes()
        ttk.Button(side, text="Add class...", command=self.add_class).pack(
            fill="x", pady=(6, 0)
        )
        ttk.Button(
            side, text="Apply class to selected object", command=self.reclassify
        ).pack(fill="x", pady=4)
        for text, mode in [
            ("Select / resize object [S]", "select"),
            ("New box (drag corners) [B]", "box"),
            ("Redraw selected box (drag) [R]", "rebox"),
            ("New grasp (3 clicks) [G]", "grasp"),
        ]:
            ttk.Radiobutton(
                side,
                text=text,
                variable=self.mode,
                value=mode,
                command=lambda: self.switch_tool(self.mode.get()),
            ).pack(anchor="w", pady=2)
        self.objects = tk.Listbox(side, height=6, exportselection=False)
        self.objects.pack(fill="x", pady=8)
        self.objects.bind("<<ListboxSelect>>", self.select_object)
        ttk.Button(
            side,
            text="Delete selected object + its grasps/edges",
            command=self.delete_object,
        ).pack(fill="x")
        ttk.Label(
            side, text="2. Grasps for selected object", font=self.heading_font
        ).pack(anchor="w", pady=(18, 6))
        ttk.Label(
            side,
            text="Click jaw contact A, contact B, then a point\nto set rectangle height.",
        ).pack(anchor="w")
        row = ttk.Frame(side)
        row.pack(fill="x", pady=6)
        ttk.Radiobutton(row, text="Easy", value="e", variable=self.difficulty).pack(
            side="left"
        )
        ttk.Radiobutton(row, text="Hard", value="h", variable=self.difficulty).pack(
            side="left", padx=16
        )
        self.grasps = tk.Listbox(side, height=4, exportselection=False)
        self.grasps.pack(fill="x", pady=5)
        ttk.Button(side, text="Delete selected grasp", command=self.delete_grasp).pack(
            fill="x"
        )
        ttk.Label(
            side, text="3. Directed relationships", font=self.heading_font
        ).pack(anchor="w", pady=(18, 6))
        ttk.Label(
            side,
            text="Select the obstructing / upper object (arrow tail).\n"
                 "Then choose the affected target (arrowhead).",
        ).pack(anchor="w")
        self.relation_type = ttk.Combobox(
            side, state="readonly", values=[f"{kind}: {name}" for kind, name in RELATION_NAMES.items()]
        )
        self.relation_type.current(0)
        self.relation_type.pack(fill="x", pady=6)
        ttk.Label(
            side, text="1: above / occludes target (purple solid arrow).\n"
                       "2: restricts grasping target (orange dashed arrow).\n"
                       "Example: #2 -> #3 means #2 obstructs grasping #3.\n"
                       "Removing #2 first may clear access to #3.\n"
                       "This is a dependency, not a guaranteed pick order.\n"
                       "Nearby alone is not obstruction. Class 1 takes priority.",
            wraplength=480,
        ).pack(anchor="w")
        ttk.Label(side, text="Target: affected object (arrowhead)").pack(anchor="w", pady=(6, 0))
        self.below = ttk.Combobox(side, state="readonly", width=28)
        self.below.pack(fill="x", pady=6)
        ttk.Button(
            side, text="Add: selected object -> target object", command=self.add_edge
        ).pack(fill="x")
        self.edges = tk.Listbox(side, height=4, exportselection=False)
        self.edges.pack(fill="x", pady=5)
        ttk.Button(
            side, text="Delete selected relationship", command=self.delete_edge
        ).pack(fill="x")
        ttk.Checkbutton(
            side,
            text="Relationships reviewed (also if none)",
            variable=self.reviewed,
            command=self.review,
        ).pack(anchor="w", pady=10)
        ttk.Button(side, text="Undo last edit (Ctrl+Z)", command=self.undo).pack(
            fill="x"
        )
        ttk.Label(
            side,
            text="Edits save automatically. Escape cancels\nan unfinished box or grasp.",
        ).pack(anchor="w", pady=10)
        # Scope wheel handling to sidebar widgets, including labels and buttons.
        # A binding on the canvas alone does not receive its children's events.
        self.setup_sidebar_scroll(sidebar)

        drawing = ttk.Frame(body)
        drawing.pack(side="left", fill="both", expand=True)
        zoom = ttk.Frame(drawing, padding=8)
        zoom.pack(fill="x")
        ttk.Button(zoom, text="-", width=3, command=lambda: self.zoom(0.8)).pack(
            side="left"
        )
        ttk.Label(zoom, textvariable=self.zoom_text, width=6, anchor="center").pack(
            side="left"
        )
        ttk.Button(zoom, text="+", width=3, command=lambda: self.zoom(1.25)).pack(
            side="left"
        )
        ttk.Button(zoom, text="Fit image", command=self.fit_image).pack(
            side="left", padx=8
        )
        ttk.Label(
            drawing, textvariable=self.instruction, wraplength=650, padding=(8, 0, 8, 8)
        ).pack(anchor="w")
        viewport = ttk.Frame(drawing)
        viewport.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(viewport, background="#20252b", highlightthickness=0)
        horizontal = ttk.Scrollbar(
            viewport, orient="horizontal", command=self.canvas.xview
        )
        vertical = ttk.Scrollbar(viewport, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(
            xscrollcommand=horizontal.set, yscrollcommand=vertical.set
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        root.bind("<Escape>", lambda e: self.cancel())
        root.bind("<Control-z>", lambda e: self.undo())
        root.bind("<Control-s>", lambda e: self.project.save())
        root.bind("<KeyPress>", self.tool_shortcut)
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1300).pack(
            fill="x"
        )
        self.load(
            next(
                (
                    i
                    for i, n in enumerate(project.files)
                    if not project.sample(n)["complete"]
                ),
                0,
            )
        )
        root.protocol("WM_DELETE_WINDOW", self.close)

    def switch_tool(self, mode):
        self.mode.set(mode)
        self.cancel()

    def setup_sidebar_scroll(self, sidebar):
        # Tk interprets a bindtag starting with '.' as a widget path. Prefix the
        # canvas path so this is a unique class tag, not a nonexistent window.
        self._sidebar_wheel_tag = f"D2TriPOSidebarWheel:{self.side_canvas}"
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.side_canvas.bind_class(self._sidebar_wheel_tag, sequence, self.scroll_sidebar)
        self.bind_sidebar_scroll(sidebar)
        LOGGER.info("SIDEBAR_SCROLL enabled revision=2026-11-18-r2")

    def bind_sidebar_scroll(self, widget):
        tags = widget.bindtags()
        if self._sidebar_wheel_tag not in tags:
            widget.bindtags((tags[0], self._sidebar_wheel_tag, *tags[1:]))
        for child in widget.winfo_children():
            self.bind_sidebar_scroll(child)

    def scroll_sidebar(self, event):
        """Scroll the sidebar without refreshing annotations or uploading images."""
        # Combobox popdowns are Tcl-created widgets; leave grabs/modal UI alone.
        if self.root.tk.call("grab", "current", self.root._w):
            return
        number = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if number in (4, 5):
            units = -3 if number == 4 else 3
        elif delta:
            units = (-1 if delta > 0 else 1) * max(1, int(abs(delta) / 120)) * 3
        else:
            return
        # Lists keep native scrolling until their boundary, then scroll the panel.
        if event.widget.winfo_class() == "Listbox":
            first, last = event.widget.yview()
            if (units < 0 and first > 0) or (units > 0 and last < 1):
                return
        first, last = self.side_canvas.yview()
        if (units < 0 and first > 0) or (units > 0 and last < 1):
            self.side_canvas.yview_scroll(units, "units")
        return "break"

    @property
    def sample(self):
        return self.project.sample(self.project.files[self.index])

    def report_error(self, kind, value, traceback):
        LOGGER.error("CALLBACK_ERROR", exc_info=(kind, value, traceback))
        if self._showing_error:
            return
        self._showing_error = True
        try:
            messagebox.showerror("Cannot apply change", str(value), parent=self.root)
        finally:
            self._showing_error = False

    def filter_classes(self, *args):
        query = self.class_search.get().strip().casefold()
        matches = [name for name in self.project.classes if query in name.casefold()]
        self.classes["values"] = matches
        if self.class_name.get() not in matches:
            self.class_name.set(matches[0] if matches else "")
        self.class_matches.set(
            f"{len(matches)} matching classes"
            if matches
            else "No matches — clear search or add a class."
        )

    def focus_class_picker(self, event=None):
        if self.classes["values"]:
            self.classes.focus_set()
            self.classes.event_generate("<Down>")
        return "break"

    def selected_class_id(self):
        if self.class_name.get() not in self.project.classes:
            raise ValueError(
                "Choose an object class; clear or change the search to see matches."
            )
        return self.project.classes.index(self.class_name.get())

    def tool_shortcut(self, event):
        # Preserve typing, dropdown navigation, and modified shortcuts such as Ctrl+S.
        # Tk creates combobox popdowns internally, without Python widget objects.
        # focus_get()/grab_current() try to resolve those objects and raise KeyError.
        if event.state & (0x4 | 0x8 | 0x40 | 0x80) or self.root.tk.call(
            "grab", "current", self.root._w
        ):
            return
        focus = self.root.tk.call("focus", "-displayof", self.root._w)
        if focus and self.root.tk.call("winfo", "class", focus) in (
            "Entry",
            "TEntry",
            "Text",
            "TCombobox",
            "Spinbox",
            "TSpinbox",
        ):
            return
        mode = {"s": "select", "b": "box", "r": "rebox", "g": "grasp"}.get(
            event.keysym.lower()
        )
        if mode is not None:
            self.switch_tool(mode)
            return "break"

    def load(self, index):
        self.clear_preview()
        self.index, self.selected, self.pending, self.history = index, None, [], []
        self.resizing = None
        with Image.open(self.project.images / self.project.files[index]) as image:
            self.image = image.convert("RGB")
        self.scale = bounded_zoom(self.image.size, self.scale)
        self.zoom_text.set(f"{self.scale:.0%}")
        LOGGER.info(
            "LOAD image=%s size=%s scale=%.3f",
            self.project.files[index],
            self.image.size,
            self.scale,
        )
        self.refresh()

    def navigate(self, delta):
        self.load(max(0, min(len(self.project.files) - 1, self.index + delta)))

    def next_unfinished(self):
        for offset in range(1, len(self.project.files) + 1):
            index = (self.index + offset) % len(self.project.files)
            if not self.project.sample(self.project.files[index])["complete"]:
                self.load(index)
                return
        messagebox.showinfo("Finished", "All images are marked complete.")

    def change(self, operation):
        before = copy.deepcopy(self.sample)
        try:
            operation()
            validate(self.sample, self.project.classes, self.image.size)
        except Exception:
            self.project.data["samples"][self.project.files[self.index]] = before
            raise
        self.history.append(before)
        del self.history[:-100]
        self.sample["complete"] = False
        self.project.save()
        self.pending = []
        self.refresh()

    def undo(self):
        if self.resizing:
            self.cancel()
            return
        if self.history:
            self.project.data["samples"][
                self.project.files[self.index]
            ] = self.history.pop()
            self.project.save()
            if not any(o["id"] == self.selected for o in self.sample["objects"]):
                self.selected = None
            self.pending = []
            self.refresh()

    def cancel(self):
        self.pending = []
        self.resizing = None
        self.set_cursor("")
        self.refresh()

    def zoom(self, factor):
        self.resizing = None
        self.scale = bounded_zoom(self.image.size, self.scale * factor)
        self.zoom_text.set(f"{self.scale:.0%}")
        LOGGER.info(
            "ZOOM image=%s scale=%.3f", self.project.files[self.index], self.scale
        )
        self.draw()

    def fit_image(self):
        self.root.update_idletasks()
        factor = min(
            max(1, self.canvas.winfo_width() - 24) / self.image.width,
            max(1, self.canvas.winfo_height() - 24) / self.image.height,
        )
        self.zoom(factor / self.scale)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def refresh(self):
        # A delete can invalidate selection before change() redraws the canvas.
        # Clear it before drawing handles or handling subsequent mouse movement.
        if not any(o["id"] == self.selected for o in self.sample["objects"]):
            self.selected = None
            self.resizing = None
            self.set_cursor("")
        test_names = set(self.project.data["test_images"])
        is_test = self.project.files[self.index] in test_names
        if self.use_for_test.get() != is_test:
            self.use_for_test.set(is_test)
        choices = tuple(
            f'{"[TEST] " if n in test_names else ""}'
            f'{"✓" if self.project.sample(n)["complete"] else "·"} {n}'
            for n in self.project.files
        )
        if tuple(self.image_choice["values"]) != choices:
            self.image_choice["values"] = choices
        if self.image_choice.current() != self.index:
            self.image_choice.current(self.index)
        self.reviewed.set(self.sample["relations_reviewed"])
        self.relationship_confirmation.set(
            "Relationships confirmed"
            if self.sample["relations_reviewed"]
            else "Confirm relationships"
        )
        self.objects.delete(0, "end")
        for i, obj in enumerate(self.sample["objects"]):
            self.objects.insert(
                "end", f'#{obj["id"]} · {self.project.classes[obj["class_id"]]}'
            )
            if obj["id"] == self.selected:
                self.objects.selection_set(i)
        self.grasp_indices = [
            i
            for i, g in enumerate(self.sample["grasps"])
            if g["object_id"] == self.selected
        ]
        self.grasps.delete(0, "end")
        for i in self.grasp_indices:
            self.grasps.insert(
                "end", f'Grasp {i+1} · {self.sample["grasps"][i]["difficulty"]}'
            )
        self.below["values"] = [
            f'#{o["id"]} {self.project.classes[o["class_id"]]}'
            for o in self.sample["objects"]
        ]
        self.below.set("")
        self.edges.delete(0, "end")
        for edge in self.sample["edges"]:
            self.edges.insert("end", edge_description(edge))
        count = sum(self.project.sample(n)["complete"] for n in self.project.files)
        progress = self.project.test_progress()
        difference = progress["target"] - progress["selected"]
        if not count:
            feedback, color = (
                "Complete images to calculate the export target.",
                "#333333",
            )
        elif difference > 0:
            feedback, color = (
                f"Choose {difference} more completed image(s) for test.",
                "#8a4b00",
            )
        elif difference < 0:
            feedback, color = (
                f"{-difference} over target; uncheck that many completed test images.",
                "#a02020",
            )
        else:
            feedback, color = "TEST TARGET REACHED.", "#176327"
        ratios = self.project.data["split_fractions"]
        test_text = (
            f'Test: {progress["selected"]}/{count} completed images '
            f'({progress["percentage"]:.1f}%) | Target: {progress["target"]} images '
            f'({ratios["test"]:.0%}) | {feedback}\n'
            f'Split: {ratios["train"]:.0%} train / {ratios["val"]:.0%} validation / '
            f'{ratios["test"]:.0%} test | {progress["planned"]} marked for test in the whole project.'
        )
        if self.project.fixed_splits is not None:
            split = self.project.fixed_splits[self.project.files[self.index]]
            split_counts = {s: sum(v == s for v in self.project.fixed_splits.values())
                            for s in ("train", "val", "test")}
            test_text = (
                f'Original split: {split.upper()} | Splits fixed on export: '
                f'{split_counts["train"]} train / {split_counts["val"]} validation / '
                f'{split_counts["test"]} test | {count}/{len(self.project.files)} reviewed and complete.'
            )
            color = "#333333"
        if self.test_status.get() != test_text:
            self.test_status.set(test_text)
        if self._test_color != color:
            self.test_status_label.configure(foreground=color)
            self._test_color = color
        self.status.set(
            f"{self.index+1}/{len(self.project.files)} images | "
            f'{count} complete | Edits saved automatically   |   {len(self.sample["objects"])} objects   |   '
            f'{len(self.sample["grasps"])} grasps   |   {len(self.sample["edges"])} relationships'
        )
        hints = {
            "select": "Click a box to select it; drag any side or square handle to resize.",
            "box": "Drag two corners around an object.",
            "rebox": "Select an object, then drag its replacement box.",
            "grasp": "Select an object; click jaw A, jaw B, then rectangle height.",
        }
        self.instruction.set(hints[self.mode.get()])
        self.draw()

    def draw(self):
        c, s = self.canvas, self.scale
        c.delete("annotation")
        self.clear_preview()
        size = (max(1, int(self.image.width * s)), max(1, int(self.image.height * s)))
        key = (id(self.image), size)
        if self._photo_key != key:
            # Release the old server-side image before allocating a zoomed replacement.
            if self._image_item is not None:
                c.delete(self._image_item)
            self.photo = None
            with self.image.resize(size) as display:
                self.photo = ImageTk.PhotoImage(display, master=self.root)
            self._image_item = c.create_image(0, 0, anchor="nw", image=self.photo)
            self._photo_key = key
            LOGGER.info(
                "RENDER image=%s display=%s", self.project.files[self.index], size
            )
            c.configure(scrollregion=(0, 0, *size))
        for obj in self.sample["objects"]:
            color = "#ffd166" if obj["id"] == self.selected else "#52d9ff"
            c.create_rectangle(
                *(v * s for v in obj["box"]), outline=color, width=3, tags="annotation"
            )
            c.create_text(
                obj["box"][0] * s + 3,
                obj["box"][1] * s + 3,
                anchor="nw",
                fill=color,
                text=f'#{obj["id"]} {self.project.classes[obj["class_id"]]}',
                font=(self.font_family, self.font_size, "bold"),
                tags="annotation",
            )
        for grasp in self.sample["grasps"]:
            color = "#a5ff90" if grasp["object_id"] == self.selected else "#6ca36a"
            points = grasp["points"]
            c.create_polygon(
                *(v * s for p in points for v in p),
                outline=color,
                fill="",
                width=2,
                tags="annotation",
            )
            # Colored jaw faces are edges 1–4 and 2–3; opening runs between them.
            for a, b in ((0, 3), (1, 2)):
                c.create_line(
                    *(v * s for p in (points[a], points[b]) for v in p),
                    fill="#ff8080",
                    width=4,
                    tags="annotation",
                )
        centers = {
            o["id"]: (
                (o["box"][0] + o["box"][2]) * s / 2,
                (o["box"][1] + o["box"][3]) * s / 2,
            )
            for o in self.sample["objects"]
        }
        for edge in self.sample["edges"]:
            a, b, kind = edge_parts(edge)
            c.create_line(
                *centers[a],
                *centers[b],
                arrow="last",
                fill="#e0aaff" if kind == 1 else "#ffad42",
                dash=() if kind == 1 else (7, 4),
                width=2,
                tags="annotation",
            )
        for x, y in self.pending:
            c.create_oval(
                x * s - 4,
                y * s - 4,
                x * s + 4,
                y * s + 4,
                fill="#ff8080",
                tags="annotation",
            )
        if self.mode.get() == "select" and self.selected is not None:
            x1, y1, x2, y2 = self.require_object()["box"]
            for x, y in (
                (x1, (y1 + y2) / 2),
                (x2, (y1 + y2) / 2),
                ((x1 + x2) / 2, y1),
                ((x1 + x2) / 2, y2),
            ):
                c.create_rectangle(
                    x * s - 5,
                    y * s - 5,
                    x * s + 5,
                    y * s + 5,
                    fill="#ffd166",
                    outline="#20252b",
                    width=1,
                    tags="annotation",
                )

    def box_side(self, event, obj):
        # Hit tolerance stays eight screen pixels at every zoom level.
        x, y = (
            self.canvas.canvasx(event.x) / self.scale,
            self.canvas.canvasy(event.y) / self.scale,
        )
        x1, y1, x2, y2 = obj["box"]
        tolerance = 8 / self.scale
        candidates = []
        if y1 - tolerance <= y <= y2 + tolerance:
            candidates.extend([(abs(x - x1), 0), (abs(x - x2), 2)])
        if x1 - tolerance <= x <= x2 + tolerance:
            candidates.extend([(abs(y - y1), 1), (abs(y - y2), 3)])
        if candidates:
            distance, side = min(candidates)
            if distance <= tolerance:
                return side
        return None

    def resized_box(self, event):
        box = self.resizing["box"].copy()
        side = self.resizing["side"]
        axis = side % 2
        value = box[side] + self.point(event)[axis] - self.resizing["start"][axis]
        value = max(0, min(self.image.size[axis], value))
        # Keep the opposite side fixed and prevent crossing or zero-area boxes.
        box[side] = (
            min(value, box[side + 2] - 1) if side < 2 else max(value, box[side - 2] + 1)
        )
        return box

    def point(self, event):
        return [
            max(0, min(self.image.width, self.canvas.canvasx(event.x) / self.scale)),
            max(0, min(self.image.height, self.canvas.canvasy(event.y) / self.scale)),
        ]

    def require_object(self):
        obj = next(
            (o for o in self.sample["objects"] if o["id"] == self.selected), None
        )
        if obj is None:
            raise ValueError("Select an object first.")
        return obj

    def select_object(self, event=None):
        selection = self.objects.curselection()
        if selection:
            self.resizing = None
            obj = self.sample["objects"][selection[0]]
            self.selected = obj["id"]
            LOGGER.info(
                "SELECT image=%s object=%s",
                self.project.files[self.index],
                self.selected,
            )
            self.class_search.set("")
            self.class_name.set(self.project.classes[obj["class_id"]])
            self.pending = []
            self.refresh()

    def press(self, event):
        self.canvas.focus_set()
        point = self.point(event)
        mode = self.mode.get()
        if mode == "select":
            if self.selected is not None:
                obj = self.require_object()
                side = self.box_side(event, obj)
                if side is not None:
                    self.begin_resize(obj, side, point)
                    return
            hits = [
                o
                for o in self.sample["objects"]
                if o["box"][0] <= point[0] <= o["box"][2]
                and o["box"][1] <= point[1] <= o["box"][3]
            ]
            # Repeated clicks cycle overlapping boxes.
            if hits:
                ids = [o["id"] for o in hits]
                self.selected = (
                    ids[(ids.index(self.selected) + 1) % len(ids)]
                    if self.selected in ids
                    else ids[-1]
                )
                LOGGER.info(
                    "SELECT image=%s object=%s",
                    self.project.files[self.index],
                    self.selected,
                )
                self.class_search.set("")
                self.class_name.set(
                    self.project.classes[self.require_object()["class_id"]]
                )
                self.refresh()
                obj = self.require_object()
                side = self.box_side(event, obj)
                if side is not None:
                    self.begin_resize(obj, side, point)
        elif mode in ("box", "rebox"):
            if mode == "rebox":
                self.require_object()
            self.pending = [point]
        elif mode == "grasp":
            self.require_object()
            self.pending.append(point)
            if len(self.pending) == 3:
                a, b, p = self.pending
                length = math.dist(a, b)
                if length < 1:
                    self.pending = []
                    raise ValueError("Place the jaw contacts at least one pixel apart.")
                height = (
                    abs((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
                    / length
                )
                try:
                    points = grasp_rectangle(a, b, height)
                finally:
                    self.pending = []
                self.change(
                    lambda: self.sample["grasps"].append(
                        {
                            "object_id": self.selected,
                            "points": points,
                            "difficulty": self.difficulty.get(),
                        }
                    )
                )
            self.draw()

    def set_cursor(self, cursor):
        if self._cursor != cursor:
            self.canvas.configure(cursor=cursor)
            self._cursor = cursor

    def clear_preview(self):
        if self._preview_after is not None:
            self.root.after_cancel(self._preview_after)
            self._preview_after = None
        if self._preview_item is not None:
            self.canvas.delete(self._preview_item)
        self._preview_item = self._preview_kind = self._preview_options = None
        self._preview_request = None
        self._preview_last = 0.0

    def queue_preview(self, kind, coordinates, **options):
        # Keep only the latest pointer position and at most one scheduled frame.
        self._preview_request = (kind, tuple(coordinates), options)
        if self._preview_after is not None:
            return
        elapsed = (time.monotonic()-self._preview_last)*1000
        if elapsed >= PREVIEW_INTERVAL_MS:
            self.flush_preview()
        else:
            self._preview_after = self.root.after(
                max(1, math.ceil(PREVIEW_INTERVAL_MS-elapsed)), self.flush_preview
            )

    def flush_preview(self):
        self._preview_after = None
        request, self._preview_request = self._preview_request, None
        if request is None:
            return
        kind, coordinates, options = request
        if self._preview_item is None or kind != self._preview_kind:
            if self._preview_item is not None:
                self.canvas.delete(self._preview_item)
            self._preview_item = getattr(self.canvas, 'create_'+kind)(
                *coordinates, tags='preview', **options
            )
            self._preview_kind = kind
        else:
            self.canvas.coords(self._preview_item, *coordinates)
            if options != self._preview_options:
                self.canvas.itemconfigure(self._preview_item, **options)
        self._preview_options = options
        self._preview_last = time.monotonic()

    def begin_resize(self, obj, side, point):
        self.clear_preview()
        self.resizing = {'id': obj['id'], 'side': side,
                         'box': obj['box'].copy(), 'start': point}
        LOGGER.info('RESIZE_BEGIN image=%s object=%s side=%s box=%s',
                    self.project.files[self.index], obj['id'], side, obj['box'])

    def drag(self, event):
        if self.resizing:
            self.queue_preview('rectangle',
                               (v*self.scale for v in self.resized_box(event)),
                               outline='white', width=2)
        elif self.mode.get() in ('box', 'rebox') and self.pending:
            self.queue_preview('rectangle',
                               (v*self.scale for p in (self.pending[0], self.point(event)) for v in p),
                               outline='white', width=2)

    def motion(self, event):
        side = (self.box_side(event, self.require_object())
                if self.mode.get() == 'select' and self.selected is not None else None)
        self.set_cursor(('sb_h_double_arrow' if side % 2 == 0 else 'sb_v_double_arrow')
                        if side is not None else '')
        if self.mode.get() == 'grasp' and self.pending:
            point = self.point(event)
            if len(self.pending) == 1:
                self.queue_preview('line',
                                   (v*self.scale for p in (self.pending[0], point) for v in p),
                                   fill='white', width=2)
            else:
                a, b = self.pending
                length = math.dist(a, b)
                if length < 1:
                    self.clear_preview()
                    return
                height = abs((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0]))/length
                if height >= .5:
                    points = grasp_rectangle(a, b, height)
                    self.queue_preview('polygon',
                                       (v*self.scale for p in points for v in p),
                                       outline='white', fill='', width=2)
                else:
                    self.clear_preview()

    def release(self, event):
        if self.resizing:
            self.clear_preview()
            box = self.resized_box(event)
            original = self.resizing
            self.resizing = None
            self.set_cursor("")
            if box != original["box"]:
                obj = next(
                    o for o in self.sample["objects"] if o["id"] == original["id"]
                )
                LOGGER.info(
                    "RESIZE image=%s object=%s box=%s",
                    self.project.files[self.index],
                    obj["id"],
                    box,
                )
                self.change(lambda: obj.update(box=box))
            else:
                self.draw()
            return
        if self.mode.get() not in ("box", "rebox") or not self.pending:
            return
        self.clear_preview()
        a, b = self.pending[0], self.point(event)
        box = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
        self.pending = []
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            self.draw()
            return
        if self.mode.get() == "rebox":
            self.change(lambda: self.require_object().update(box=box))
        else:
            new_id = max([o["id"] for o in self.sample["objects"]], default=0) + 1

            def add():
                self.sample["objects"].append(
                    {"id": new_id, "class_id": self.selected_class_id(), "box": box}
                )
                self.sample["relations_reviewed"] = False

            self.change(add)
            self.selected = new_id
            self.refresh()

    def add_class(self):
        name = simpledialog.askstring(
            "Add class", "New class name (IDs 0–30 stay fixed):"
        )
        if name is not None:
            index = self.project.add_class(name)
            self.class_search.set("")
            self.class_name.set(self.project.classes[index])
            self.refresh()

    def reclassify(self):
        obj = self.require_object()
        self.change(lambda: obj.update(class_id=self.selected_class_id()))

    def delete_object(self):
        obj = self.require_object()

        def remove():
            self.sample["objects"].remove(obj)
            self.sample["grasps"] = [
                g for g in self.sample["grasps"] if g["object_id"] != obj["id"]
            ]
            self.sample["edges"] = [
                e for e in self.sample["edges"] if obj["id"] not in e[:2]
            ]
            self.sample["relations_reviewed"] = False

        self.change(remove)

    def delete_grasp(self):
        selection = self.grasps.curselection()
        if selection:
            index = self.grasp_indices[selection[0]]
            self.change(lambda: self.sample["grasps"].pop(index))

    def add_edge(self):
        self.require_object()
        if self.below.current() < 0:
            raise ValueError("Choose the affected target object.")
        below = self.sample["objects"][self.below.current()]["id"]
        kind = self.relation_type.current() + 1
        if kind not in RELATION_NAMES:
            raise ValueError("Choose a relationship type.")
        existing = next((i for i, edge in enumerate(self.sample["edges"])
                         if edge_parts(edge)[:2] == (self.selected, below)), None)
        if existing is not None:
            previous_kind = edge_parts(self.sample["edges"][existing])[2]
            if previous_kind == kind:
                return
            if previous_kind == 1:
                raise ValueError("Overlap/stacking takes priority. Delete that relationship first if it is incorrect.")

        def add():
            edge = [self.selected, below, kind]
            if existing is None:
                self.sample["edges"].append(edge)
            else:
                self.sample["edges"][existing] = edge
            self.sample["relations_reviewed"] = False

        self.change(add)

    def delete_edge(self):
        selection = self.edges.curselection()
        if selection:
            def remove():
                self.sample["edges"].pop(selection[0])
                self.sample["relations_reviewed"] = False
            self.change(remove)

    def review(self):
        self.change(lambda: self.sample.update(relations_reviewed=self.reviewed.get()))

    def confirm_relationships(self):
        if not self.sample["relations_reviewed"]:
            self.reviewed.set(True)
            self.review()

    def change_grasp_mode(self):
        self.project.set_require_grasps(self.require_grasps.get())
        self.history = []
        self.refresh()

    def complete(self):
        if self.pending:
            raise ValueError(
                "Finish the current annotation or press Escape before completing."
            )
        validate(
            self.sample,
            self.project.classes,
            self.image.size,
            complete=True,
            require_grasps=self.project.require_grasps,
        )
        self.sample["complete"] = True
        self.project.save()
        self.refresh()

    def toggle_test_image(self):
        self.project.set_test_image(
            self.project.files[self.index], self.use_for_test.get()
        )
        self.refresh()

    def edit_split_ratios(self):
        fractions = self.project.data["split_fractions"]
        value = simpledialog.askstring(
            "Split ratios",
            "Train, validation, test percentages (sum to 100):",
            initialvalue=", ".join(
                f"{100*fractions[s]:g}" for s in ("train", "val", "test")
            ),
            parent=self.root,
        )
        if value is None:
            return
        try:
            train, val, test = map(float, value.split(","))
        except ValueError as exc:
            raise ValueError(
                "Enter three percentages, for example: 70, 10, 20"
            ) from exc
        if (
            not math.isfinite(train)
            or train <= 0
            or not math.isclose(train + val + test, 100, abs_tol=1e-6)
        ):
            raise ValueError("All three percentages must be positive and sum to 100.")
        self.project.set_split_fractions(val / 100, test / 100)
        self.refresh()

    def export(self):
        parent = filedialog.askdirectory(
            title="Choose parent folder for a NEW exported dataset"
        )
        if not parent:
            return
        name = simpledialog.askstring(
            "Dataset folder", "New folder name:", initialvalue="topdown_labeled"
        )
        if not name:
            return
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Enter a single new folder name.")
        output = self.project.export(Path(parent) / name)
        manifest = json.loads((output / "manifest.json").read_text())
        counts = manifest["split_counts"]
        messagebox.showinfo(
            "Export complete",
            f'{output}\n\n{counts["train"]} training / '
            f'{counts["val"]} validation / {counts["test"]} test images.\n'
            "Your chosen test images are held out.\nSee scripts/README_labeling.md for training.",
        )

    def close(self):
        self.clear_preview()
        self.project.save()
        LOGGER.info("CLOSE")
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/topdown_renders"))
    parser.add_argument(
        "--project", type=Path, default=Path("data/topdown_labels/project.json")
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["density_map.png"],
        help="Filename to omit; repeatable",
    )
    parser.add_argument(
        "--export", type=Path, help="Export completed labels without opening the GUI"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Validation fraction (default: 0.10 or saved setting)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="Manually selected test fraction (default: 0.20 or saved setting)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--font-size", type=int, default=16, help="UI text size in points (default: 16)"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Diagnostic log (default: labeler.log beside project.json)",
    )
    parser.add_argument(
        "--no-grasps",
        action="store_const",
        const=False,
        default=None,
        dest="require_grasps",
        help="Allow completion without grasps; persist boxes/relationships mode",
    )
    args = parser.parse_args()
    if args.font_size < 10:
        parser.error("--font-size must be at least 10")
    project = Project(
        args.images, args.project, args.exclude, require_grasps=args.require_grasps
    )
    if args.val_fraction is not None or args.test_fraction is not None:
        fractions = project.data["split_fractions"]
        project.set_split_fractions(
            fractions["val"] if args.val_fraction is None else args.val_fraction,
            fractions["test"] if args.test_fraction is None else args.test_fraction,
        )
    if args.export:
        print(project.export(args.export, seed=args.seed))
    else:
        root = tk.Tk()
        configure_diagnostics(args.log_file or args.project.with_name("labeler.log"))
        Labeler(root, project, font_size=args.font_size)
        root.mainloop()


if __name__ == "__main__":
    main()
