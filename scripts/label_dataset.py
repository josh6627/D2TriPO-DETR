#!/usr/bin/env python3
"""Local desktop labeler. Run in the d2tripo environment; no GPU/server needed."""
import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk

from PIL import Image, ImageTk

try:
    from .label_project import Project, grasp_rectangle, validate
except ImportError:
    from label_project import Project, grasp_rectangle, validate


READABLE_FONT_FAMILIES = ('DejaVu Sans', 'Noto Sans', 'Liberation Sans', 'Arial', 'Inter')


def prefer_smooth_font_renderer(root):
    """Use system Tk when a Linux environment only exposes bitmap X11 fonts.

    Some Conda Tk builds lack Xft. Changing the requested family cannot fix
    their rendering. The labeler needs only Tk/Pillow, so it can use system
    Python independently of the model's training environment.
    """
    if sys.platform != 'linux':
        return
    available = {name.casefold() for name in tkfont.families(root)}
    if any(name.casefold() in available for name in READABLE_FONT_FAMILIES):
        return
    candidate = Path('/usr/bin/python3')
    if not candidate.is_file() or candidate.resolve() == Path(sys.executable).resolve():
        return
    probe = (
        'import tkinter as tk; from tkinter import font; from PIL import Image, ImageTk; '
        'root=tk.Tk(); root.withdraw(); '
        'available={name.casefold() for name in font.families(root)}; '
        f'found=any(name.casefold() in available for name in {READABLE_FONT_FAMILIES!r}); '
        'root.destroy(); raise SystemExit(0 if found else 1)'
    )
    try:
        result = subprocess.run([str(candidate), '-c', probe], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return
    if result.returncode == 0:
        root.destroy()
        os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])


class Labeler:
    def __init__(self, root, project, font_size=14):
        self.root, self.project = root, project
        self.font_size = font_size
        self._configure_theme()
        self.heading_font = (self.font_family, font_size+2, 'bold')
        self.index, self.selected, self.scale = 0, None, 1.0
        self.pending, self.history, self.resizing = [], [], None
        root.title('D2TriPO-DETR | Annotation workspace')
        root.geometry(f'{min(1500, root.winfo_screenwidth()-60)}x{min(1000, root.winfo_screenheight()-80)}')
        root.configure(background=self.colors['background'])
        root.report_callback_exception = lambda kind, value, tb: messagebox.showerror('Cannot apply change', str(value))
        self.mode = tk.StringVar(value='select')
        self.class_name = tk.StringVar(value=project.classes[0])
        self.class_search, self.class_matches = tk.StringVar(), tk.StringVar()
        self.difficulty = tk.StringVar(value='e')
        self.reviewed = tk.BooleanVar()
        self.relationship_confirmation = tk.StringVar(value='Confirm relationships')
        self.require_grasps = tk.BooleanVar(value=project.require_grasps)
        self.status, self.instruction = tk.StringVar(), tk.StringVar()
        self.progress_text, self.selection_text = tk.StringVar(), tk.StringVar()
        self.image_state, self.zoom_text = tk.StringVar(), tk.StringVar(value='100%')

        header = ttk.Frame(root, padding=(24, 18))
        header.pack(fill='x')
        brand = ttk.Frame(header)
        brand.pack(side='left')
        ttk.Label(brand, text='Annotation workspace', style='Title.TLabel').pack(anchor='w')
        ttk.Label(brand, text='D2TriPO-DETR  /  '+project.images.name, style='Muted.TLabel').pack(anchor='w', pady=(3, 0))
        ttk.Button(header, text='Export dataset', command=self.export).pack(side='right')
        progress = ttk.Frame(header, padding=(0, 0, 24, 0))
        progress.pack(side='right')
        ttk.Label(progress, textvariable=self.progress_text, style='Muted.TLabel').pack(anchor='e', pady=(0, 6))
        self.progress_bar = ttk.Progressbar(progress, length=200, maximum=len(project.files))
        self.progress_bar.pack()

        navigation = ttk.Frame(root, style='Surface.TFrame', padding=(18, 12))
        navigation.pack(fill='x', padx=16)
        ttk.Button(navigation, text='<', width=3, command=lambda: self.navigate(-1)).pack(side='left')
        self.image_choice = ttk.Combobox(navigation, state='readonly', width=17)
        self.image_choice.pack(side='left', padx=8)
        self.image_choice.bind('<<ComboboxSelected>>', lambda e: self.load(self.image_choice.current()))
        ttk.Button(navigation, text='>', width=3, command=lambda: self.navigate(1)).pack(side='left')
        ttk.Button(navigation, text='Next unfinished', command=self.next_unfinished).pack(side='left', padx=(12, 0))
        ttk.Button(navigation, text='Mark complete', style='Primary.TButton', command=self.complete).pack(side='right')
        self.confirm_button = ttk.Button(navigation, textvariable=self.relationship_confirmation,
                                         command=self.confirm_relationships)
        self.confirm_button.pack(side='right', padx=10)

        body = ttk.Frame(root, padding=(16, 14, 16, 0))
        body.pack(fill='both', expand=True)
        sidebar = ttk.Frame(body, style='Surface.TFrame', padding=14, width=450)
        sidebar.pack(side='right', fill='y', padx=(14, 0))
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text='INSPECTOR', style='Eyebrow.TLabel').pack(anchor='w')
        ttk.Label(sidebar, textvariable=self.selection_text, style='Surface.TLabel',
                  wraplength=405, font=self.heading_font).pack(anchor='w', pady=(6, 16))
        self.inspector = ttk.Notebook(sidebar)
        self.inspector.pack(fill='both', expand=True)
        self.object_panel = self._inspector_tab('Objects')
        self.relation_panel = self._inspector_tab('Relations')
        self.grasp_panel = self._inspector_tab('Grasps')

        objects = self.object_panel
        self._section(objects, 'Object class', 'Search by name, then draw a box or apply the class.')
        search_row = ttk.Frame(objects, style='Surface.TFrame')
        search_row.pack(fill='x', pady=(12, 8))
        self.search_entry = ttk.Entry(search_row, textvariable=self.class_search)
        self.search_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(search_row, text='Clear', style='Quiet.TButton', command=lambda: self.class_search.set('')).pack(side='right', padx=(6, 0))
        self.classes = ttk.Combobox(objects, textvariable=self.class_name, values=project.classes, state='readonly', width=22)
        self.classes.pack(fill='x')
        ttk.Label(objects, textvariable=self.class_matches, style='Hint.TLabel', wraplength=370).pack(anchor='w', pady=(6, 12))
        self.class_search.trace_add('write', self.filter_classes)
        self.search_entry.bind('<Return>', self.focus_class_picker)
        self.search_entry.bind('<Down>', self.focus_class_picker)
        self.filter_classes()
        class_actions = ttk.Frame(objects, style='Surface.TFrame')
        class_actions.pack(fill='x')
        ttk.Button(class_actions, text='Add class', command=self.add_class).pack(side='left')
        ttk.Button(class_actions, text='Apply to selected', command=self.reclassify).pack(side='right')
        ttk.Separator(objects).pack(fill='x', pady=20)
        self._section(objects, 'Objects in this image', 'Select a row or click a box on the image.')
        self.objects = self._listbox(objects, height=7)
        self.objects.bind('<<ListboxSelect>>', self.select_object)
        ttk.Button(objects, text='Delete selected object', style='Danger.TButton', command=self.delete_object).pack(fill='x', pady=(8, 0))
        ttk.Label(objects, text='Deleting an object also removes its grasps and relationships.',
                  style='Hint.TLabel', wraplength=370).pack(anchor='w', pady=8)

        relations = self.relation_panel
        self._section(relations, 'Pick order', 'The selected object is above the lower object and should be picked first.')
        ttk.Label(relations, text='Lower object', style='Surface.TLabel').pack(anchor='w', pady=(16, 8))
        self.below = ttk.Combobox(relations, state='readonly', width=22)
        self.below.pack(fill='x')
        ttk.Button(relations, text='Add selected above lower', command=self.add_edge).pack(fill='x', pady=10)
        self.edges = self._listbox(relations, height=6)
        ttk.Button(relations, text='Remove relationship', style='Danger.TButton', command=self.delete_edge).pack(fill='x', pady=8)
        ttk.Separator(relations).pack(fill='x', pady=16)
        ttk.Checkbutton(relations, text='Relationships reviewed', variable=self.reviewed,
                        command=self.review).pack(anchor='w')
        ttk.Label(relations, text='Confirm review even if no objects overlap. You can also use the button beside Mark complete.',
                  style='Hint.TLabel', wraplength=370).pack(anchor='w', pady=10)

        grasps = self.grasp_panel
        self._section(grasps, 'Grasp annotations', 'Select an object. Click jaw contact A, contact B, then set the rectangle height.')
        ttk.Checkbutton(grasps, text='Require grasps for completion', variable=self.require_grasps,
                        command=self.change_grasp_mode).pack(anchor='w', pady=16)
        row = ttk.Frame(grasps, style='Surface.TFrame')
        row.pack(fill='x', pady=(0, 12))
        ttk.Radiobutton(row, text='Easy', value='e', variable=self.difficulty).pack(side='left')
        ttk.Radiobutton(row, text='Hard', value='h', variable=self.difficulty).pack(side='left', padx=20)
        self.grasps = self._listbox(grasps, height=6)
        ttk.Button(grasps, text='Remove grasp', style='Danger.TButton', command=self.delete_grasp).pack(fill='x', pady=8)
        ttk.Label(grasps, text='Red edges show the jaw faces. Disable the requirement to label boxes and relationships only.',
                  style='Hint.TLabel', wraplength=370).pack(anchor='w', pady=8)

        drawing = ttk.Frame(body)
        drawing.pack(side='left', fill='both', expand=True)
        tools = ttk.Frame(drawing, style='Surface.TFrame', padding=8)
        tools.pack(fill='x', pady=(0, 10))
        for text, mode in [('Select  S', 'select'), ('Box  B', 'box'), ('Redraw  R', 'rebox'), ('Grasp  G', 'grasp')]:
            ttk.Radiobutton(tools, text=text, style='Tool.TRadiobutton', variable=self.mode, value=mode,
                            command=lambda: self.switch_tool(self.mode.get())).pack(side='left', padx=3)
        ttk.Button(tools, text='Undo', style='Quiet.TButton', command=self.undo).pack(side='right')
        viewport = ttk.Frame(drawing)
        viewport.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(viewport, background='#0b1018', highlightthickness=1, highlightbackground=self.colors['border'])
        horizontal = ttk.Scrollbar(viewport, orient='horizontal', command=self.canvas.xview)
        vertical = ttk.Scrollbar(viewport, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        canvas_footer = ttk.Frame(drawing, padding=(0, 10))
        canvas_footer.pack(fill='x')
        ttk.Label(canvas_footer, textvariable=self.image_state, style='Badge.TLabel', padding=(10, 5)).pack(side='left')
        ttk.Button(canvas_footer, text='+', width=3, command=lambda: self.zoom(1.25)).pack(side='right')
        ttk.Label(canvas_footer, textvariable=self.zoom_text, style='Muted.TLabel', width=6, anchor='center').pack(side='right')
        ttk.Button(canvas_footer, text='-', width=3, command=lambda: self.zoom(.8)).pack(side='right')
        ttk.Button(canvas_footer, text='Fit image', command=self.fit_image).pack(side='right', padx=(0, 8))
        ttk.Label(drawing, textvariable=self.instruction, style='Muted.TLabel', wraplength=650).pack(anchor='w', pady=(0, 6))
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        root.bind('<Escape>', lambda e: self.cancel())
        root.bind('<Control-z>', lambda e: self.undo())
        root.bind('<Control-s>', lambda e: self.project.save())
        root.bind('<KeyPress>', self.tool_shortcut)
        ttk.Label(root, textvariable=self.status, style='Status.TLabel', padding=(24, 10), wraplength=1300).pack(fill='x')
        self.load(next((i for i, n in enumerate(project.files) if not project.sample(n)['complete']), 0))
        root.protocol('WM_DELETE_WINDOW', self.close)

    def _configure_theme(self):
        self.colors = {'background': '#111720', 'surface': '#1a2330', 'input': '#111b28',
                       'text': '#f1f5fa', 'muted': '#c3d0df', 'accent': '#5ee0b5', 'border': '#324255'}
        c = self.colors
        families = {name.casefold(): name for name in tkfont.families(self.root)}
        self.font_family = next((families[f.casefold()] for f in READABLE_FONT_FAMILIES if f.casefold() in families),
                                tkfont.nametofont('TkDefaultFont', root=self.root).actual('family'))
        for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont', 'TkCaptionFont'):
            tkfont.nametofont(name, root=self.root).configure(family=self.font_family, size=self.font_size)
        self.root.option_add('*Listbox.font', 'TkTextFont')
        self.root.option_add('*TCombobox*Listbox.background', c['input'])
        self.root.option_add('*TCombobox*Listbox.foreground', c['text'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', '#315c60')
        self.root.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font='TkDefaultFont', background=c['background'], foreground=c['text'],
                        bordercolor=c['border'], lightcolor=c['border'], darkcolor=c['border'])
        style.configure('TFrame', background=c['background'])
        style.configure('Surface.TFrame', background=c['surface'])
        style.configure('TLabel', background=c['background'], foreground=c['text'])
        style.configure('Surface.TLabel', background=c['surface'])
        style.configure('Muted.TLabel', foreground=c['muted'])
        style.configure('Hint.TLabel', foreground=c['muted'], background=c['surface'], font=(self.font_family, self.font_size))
        style.configure('Title.TLabel', font=(self.font_family, self.font_size+8, 'bold'))
        style.configure('Eyebrow.TLabel', foreground=c['accent'], background=c['surface'], font=(self.font_family, self.font_size, 'bold'))
        style.configure('Badge.TLabel', background='#233e3b', foreground=c['accent'], font=(self.font_family, self.font_size, 'bold'))
        style.configure('Status.TLabel', foreground=c['muted'], font=(self.font_family, self.font_size))
        style.configure('TButton', background='#28374a', foreground=c['text'], padding=(12, 9), relief='flat', borderwidth=0)
        style.map('TButton', background=[('active', '#354a61'), ('pressed', '#405873')])
        style.configure('Primary.TButton', background=c['accent'], foreground='#102a25', font=(self.font_family, self.font_size, 'bold'))
        style.map('Primary.TButton', background=[('active', '#8ceccc'), ('pressed', '#40b994')])
        style.configure('Quiet.TButton', background=c['surface'])
        style.configure('Danger.TButton', background=c['surface'], foreground='#f5a5ab')
        style.configure('TEntry', fieldbackground=c['input'], foreground=c['text'], insertcolor=c['text'], padding=9, borderwidth=1)
        style.configure('TCombobox', fieldbackground=c['input'], background='#28374a', foreground=c['text'], arrowcolor=c['muted'], padding=8)
        style.map('TCombobox', fieldbackground=[('readonly', c['input'])], foreground=[('readonly', c['text'])], selectbackground=[('readonly', c['input'])], selectforeground=[('readonly', c['text'])])
        style.configure('TCheckbutton', background=c['surface'], padding=(0, 4))
        style.map('TCheckbutton', background=[('active', c['surface'])])
        style.configure('TRadiobutton', background=c['surface'])
        style.map('TRadiobutton', background=[('active', c['surface'])])
        style.layout('Tool.TRadiobutton', [('Radiobutton.padding', {'sticky': 'nswe', 'children': [('Radiobutton.label', {'sticky': 'nswe'})]})])
        style.configure('Tool.TRadiobutton', background=c['surface'], foreground=c['muted'], padding=(12, 10))
        style.map('Tool.TRadiobutton', background=[('selected', '#274d46'), ('active', '#2b394b')], foreground=[('selected', c['accent']), ('active', c['text'])])
        style.configure('TNotebook', background=c['surface'], borderwidth=0, tabmargins=(0, 0, 0, 10),
                        lightcolor=c['surface'], darkcolor=c['surface'], bordercolor=c['surface'])
        style.configure('TNotebook.Tab', background=c['surface'], foreground=c['muted'], padding=(12, 10), borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', '#293c4b')], foreground=[('selected', c['text'])])
        style.configure('TScrollbar', background='#324255', troughcolor=c['background'], arrowcolor=c['muted'], borderwidth=0)
        style.configure('TProgressbar', background=c['accent'], troughcolor='#293848', borderwidth=0, lightcolor=c['accent'], darkcolor=c['accent'])
        style.configure('TSeparator', background=c['border'])

    def _inspector_tab(self, title):
        page = ttk.Frame(self.inspector, style='Surface.TFrame')
        self.inspector.add(page, text=title)
        canvas = tk.Canvas(page, background=self.colors['surface'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(page, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        content = ttk.Frame(canvas, style='Surface.TFrame', padding=(2, 14, 8, 4))
        window = canvas.create_window(0, 0, window=content, anchor='nw')
        content.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window, width=e.width))
        return content

    def _section(self, parent, title, note):
        ttk.Label(parent, text=title, style='Surface.TLabel', font=self.heading_font).pack(anchor='w')
        ttk.Label(parent, text=note, style='Hint.TLabel', wraplength=370).pack(anchor='w', pady=(6, 0))

    def _listbox(self, parent, height):
        frame = ttk.Frame(parent, style='Surface.TFrame')
        frame.pack(fill='x', pady=(12, 0))
        result = tk.Listbox(frame, height=height, exportselection=False, background=self.colors['input'],
                            foreground=self.colors['text'], selectbackground='#315c60', selectforeground='white',
                            highlightthickness=1, highlightbackground=self.colors['border'], borderwidth=0,
                            activestyle='none', relief='flat')
        scroll = ttk.Scrollbar(frame, orient='vertical', command=result.yview)
        scroll.pack(side='right', fill='y')
        result.pack(side='left', fill='x', expand=True)
        result.configure(yscrollcommand=scroll.set)
        return result

    def switch_tool(self, mode):
        self.mode.set(mode)
        target = self.grasp_panel if mode == 'grasp' else self.object_panel
        self.inspector.select(target.master.master)
        self.cancel()

    @property
    def sample(self):
        return self.project.sample(self.project.files[self.index])

    def filter_classes(self, *args):
        query = self.class_search.get().strip().casefold()
        matches = [name for name in self.project.classes if query in name.casefold()]
        self.classes['values'] = matches
        if self.class_name.get() not in matches:
            self.class_name.set(matches[0] if matches else '')
        self.class_matches.set(f'{len(matches)} matching classes' if matches else 'No matches — clear search or add a class.')

    def focus_class_picker(self, event=None):
        if self.classes['values']:
            self.classes.focus_set()
            self.classes.event_generate('<Down>')
        return 'break'

    def selected_class_id(self):
        if self.class_name.get() not in self.project.classes:
            raise ValueError('Choose an object class; clear or change the search to see matches.')
        return self.project.classes.index(self.class_name.get())

    def tool_shortcut(self, event):
        # Preserve typing, dropdown navigation, and modified shortcuts such as Ctrl+S.
        # Tk creates combobox popdowns internally, without Python widget objects.
        # focus_get()/grab_current() try to resolve those objects and raise KeyError.
        if (event.state & (0x4 | 0x8 | 0x40 | 0x80)
                or self.root.tk.call('grab', 'current', self.root._w)):
            return
        focus = self.root.tk.call('focus', '-displayof', self.root._w)
        if focus and self.root.tk.call('winfo', 'class', focus) in (
                'Entry', 'TEntry', 'Text', 'TCombobox', 'Spinbox', 'TSpinbox'):
            return
        mode = {'s': 'select', 'b': 'box', 'r': 'rebox', 'g': 'grasp'}.get(event.keysym.lower())
        if mode is not None:
            self.switch_tool(mode)
            return 'break'

    def load(self, index):
        self.index, self.selected, self.pending, self.history = index, None, [], []
        self.resizing = None
        with Image.open(self.project.images / self.project.files[index]) as image:
            self.image = image.convert('RGB')
        self.refresh()

    def navigate(self, delta):
        self.load(max(0, min(len(self.project.files)-1, self.index+delta)))

    def next_unfinished(self):
        for offset in range(1, len(self.project.files)+1):
            index = (self.index+offset) % len(self.project.files)
            if not self.project.sample(self.project.files[index])['complete']:
                self.load(index)
                return
        messagebox.showinfo('Finished', 'All images are marked complete.')

    def change(self, operation):
        before = copy.deepcopy(self.sample)
        try:
            operation()
            validate(self.sample, self.project.classes, self.image.size)
        except Exception:
            self.project.data['samples'][self.project.files[self.index]] = before
            raise
        self.history.append(before)
        self.sample['complete'] = False
        self.project.save()
        self.pending = []
        self.refresh()

    def undo(self):
        if self.resizing:
            self.cancel()
            return
        if self.history:
            self.project.data['samples'][self.project.files[self.index]] = self.history.pop()
            self.project.save()
            if not any(o['id'] == self.selected for o in self.sample['objects']):
                self.selected = None
            self.pending = []
            self.refresh()

    def cancel(self):
        self.pending = []
        self.resizing = None
        self.canvas.configure(cursor='')
        self.refresh()

    def zoom(self, factor):
        self.resizing = None
        self.scale = max(0.25, min(5, self.scale*factor))
        self.zoom_text.set(f'{self.scale:.0%}')
        self.draw()

    def fit_image(self):
        self.root.update_idletasks()
        factor = min(max(1, self.canvas.winfo_width()-24)/self.image.width,
                     max(1, self.canvas.winfo_height()-24)/self.image.height)
        self.zoom(factor/self.scale)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def refresh(self):
        # A delete can invalidate selection before change() redraws the canvas.
        # Clear it before drawing handles or handling subsequent mouse movement.
        if not any(o['id'] == self.selected for o in self.sample['objects']):
            self.selected = None
            self.resizing = None
            self.canvas.configure(cursor='')
        self.image_choice['values'] = [f'{"✓" if self.project.sample(n)["complete"] else "·"} {n}' for n in self.project.files]
        self.image_choice.current(self.index)
        self.reviewed.set(self.sample['relations_reviewed'])
        self.relationship_confirmation.set('Relationships confirmed' if self.sample['relations_reviewed']
                                           else 'Confirm relationships')
        self.objects.delete(0, 'end')
        for i, obj in enumerate(self.sample['objects']):
            self.objects.insert('end', f'#{obj["id"]} · {self.project.classes[obj["class_id"]]}')
            if obj['id'] == self.selected:
                self.objects.selection_set(i)
        self.grasp_indices = [i for i, g in enumerate(self.sample['grasps']) if g['object_id'] == self.selected]
        self.grasps.delete(0, 'end')
        for i in self.grasp_indices:
            self.grasps.insert('end', f'Grasp {i+1} · {self.sample["grasps"][i]["difficulty"]}')
        self.below['values'] = [f'#{o["id"]} {self.project.classes[o["class_id"]]}' for o in self.sample['objects']]
        self.below.set('')
        self.edges.delete(0, 'end')
        for a, b in self.sample['edges']:
            self.edges.insert('end', f'#{a} → #{b}')
        count = sum(self.project.sample(n)['complete'] for n in self.project.files)
        self.progress_text.set(f'{count} / {len(self.project.files)} images complete')
        self.progress_bar['value'] = count
        self.image_state.set('COMPLETE' if self.sample['complete'] else 'IN PROGRESS')
        selected = next((o for o in self.sample['objects'] if o['id'] == self.selected), None)
        self.selection_text.set(f'#{selected["id"]}  {self.project.classes[selected["class_id"]]}'
                                if selected else 'No object selected')
        self.status.set(f'Edits saved automatically   |   {len(self.sample["objects"])} objects   |   '
                        f'{len(self.sample["grasps"])} grasps   |   {len(self.sample["edges"])} relationships')
        hints = {'select': 'Click a box to select it; drag any side or square handle to resize.', 'box': 'Drag two corners around an object.',
                 'rebox': 'Select an object, then drag its replacement box.',
                 'grasp': 'Select an object; click jaw A, jaw B, then rectangle height.'}
        self.instruction.set(hints[self.mode.get()])
        self.draw()

    def draw(self):
        c, s = self.canvas, self.scale
        c.delete('all')
        self.photo = ImageTk.PhotoImage(self.image.resize((int(self.image.width*s), int(self.image.height*s))))
        c.create_image(0, 0, anchor='nw', image=self.photo)
        c.configure(scrollregion=(0, 0, self.image.width*s, self.image.height*s))
        for obj in self.sample['objects']:
            color = '#ffd166' if obj['id'] == self.selected else '#52d9ff'
            c.create_rectangle(*(v*s for v in obj['box']), outline=color, width=3)
            c.create_text(obj['box'][0]*s+3, obj['box'][1]*s+3, anchor='nw', fill=color,
                          text=f'#{obj["id"]} {self.project.classes[obj["class_id"]]}', font=(self.font_family, self.font_size, 'bold'))
        for grasp in self.sample['grasps']:
            color = '#a5ff90' if grasp['object_id'] == self.selected else '#6ca36a'
            points = grasp['points']
            c.create_polygon(*(v*s for p in points for v in p), outline=color, fill='', width=2)
            # Colored jaw faces are edges 1–4 and 2–3; opening runs between them.
            for a, b in ((0, 3), (1, 2)):
                c.create_line(*(v*s for p in (points[a], points[b]) for v in p), fill='#ff8080', width=4)
        centers = {o['id']: ((o['box'][0]+o['box'][2])*s/2, (o['box'][1]+o['box'][3])*s/2)
                   for o in self.sample['objects']}
        for a, b in self.sample['edges']:
            c.create_line(*centers[a], *centers[b], arrow='last', fill='#e0aaff', width=2)
        for x, y in self.pending:
            c.create_oval(x*s-4, y*s-4, x*s+4, y*s+4, fill='#ff8080')
        if self.mode.get() == 'select' and self.selected is not None:
            x1, y1, x2, y2 = self.require_object()['box']
            for x, y in ((x1, (y1+y2)/2), (x2, (y1+y2)/2), ((x1+x2)/2, y1), ((x1+x2)/2, y2)):
                c.create_rectangle(x*s-5, y*s-5, x*s+5, y*s+5,
                                   fill='#ffd166', outline='#20252b', width=1)

    def box_side(self, event, obj):
        # Hit tolerance stays eight screen pixels at every zoom level.
        x, y = self.canvas.canvasx(event.x)/self.scale, self.canvas.canvasy(event.y)/self.scale
        x1, y1, x2, y2 = obj['box']
        tolerance = 8/self.scale
        candidates = []
        if y1-tolerance <= y <= y2+tolerance:
            candidates.extend([(abs(x-x1), 0), (abs(x-x2), 2)])
        if x1-tolerance <= x <= x2+tolerance:
            candidates.extend([(abs(y-y1), 1), (abs(y-y2), 3)])
        if candidates:
            distance, side = min(candidates)
            if distance <= tolerance:
                return side
        return None

    def resized_box(self, event):
        box = self.resizing['box'].copy()
        side = self.resizing['side']
        axis = side % 2
        value = box[side] + self.point(event)[axis] - self.resizing['start'][axis]
        value = max(0, min(self.image.size[axis], value))
        # Keep the opposite side fixed and prevent crossing or zero-area boxes.
        box[side] = min(value, box[side+2]-1) if side < 2 else max(value, box[side-2]+1)
        return box

    def point(self, event):
        return [max(0, min(self.image.width, self.canvas.canvasx(event.x)/self.scale)),
                max(0, min(self.image.height, self.canvas.canvasy(event.y)/self.scale))]

    def require_object(self):
        obj = next((o for o in self.sample['objects'] if o['id'] == self.selected), None)
        if obj is None:
            raise ValueError('Select an object first.')
        return obj

    def select_object(self, event=None):
        selection = self.objects.curselection()
        if selection:
            self.resizing = None
            obj = self.sample['objects'][selection[0]]
            self.selected = obj['id']
            self.class_search.set('')
            self.class_name.set(self.project.classes[obj['class_id']])
            self.pending = []
            self.refresh()

    def press(self, event):
        self.canvas.focus_set()
        point = self.point(event)
        mode = self.mode.get()
        if mode == 'select':
            if self.selected is not None:
                obj = self.require_object()
                side = self.box_side(event, obj)
                if side is not None:
                    self.resizing = {'id': obj['id'], 'side': side, 'box': obj['box'].copy(), 'start': point}
                    return
            hits = [o for o in self.sample['objects'] if o['box'][0] <= point[0] <= o['box'][2]
                    and o['box'][1] <= point[1] <= o['box'][3]]
            # Repeated clicks cycle overlapping boxes.
            if hits:
                ids = [o['id'] for o in hits]
                self.selected = ids[(ids.index(self.selected)+1) % len(ids)] if self.selected in ids else ids[-1]
                self.class_search.set('')
                self.class_name.set(self.project.classes[self.require_object()['class_id']])
                self.refresh()
                obj = self.require_object()
                side = self.box_side(event, obj)
                if side is not None:
                    self.resizing = {'id': obj['id'], 'side': side, 'box': obj['box'].copy(), 'start': point}
        elif mode in ('box', 'rebox'):
            if mode == 'rebox':
                self.require_object()
            self.pending = [point]
        elif mode == 'grasp':
            self.require_object()
            self.pending.append(point)
            if len(self.pending) == 3:
                a, b, p = self.pending
                length = math.dist(a, b)
                if length < 1:
                    self.pending = []
                    raise ValueError('Place the jaw contacts at least one pixel apart.')
                height = abs((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])) / length
                try:
                    points = grasp_rectangle(a, b, height)
                finally:
                    self.pending = []
                self.change(lambda: self.sample['grasps'].append({
                    'object_id': self.selected, 'points': points, 'difficulty': self.difficulty.get()}))
            self.draw()

    def drag(self, event):
        if self.resizing:
            self.canvas.delete('preview')
            self.canvas.create_rectangle(*(v*self.scale for v in self.resized_box(event)),
                                         outline='white', width=3, dash=(5, 3), tags='preview')
            return
        if self.mode.get() in ('box', 'rebox') and self.pending:
            self.canvas.delete('preview')
            self.canvas.create_rectangle(*(v*self.scale for p in (self.pending[0], self.point(event)) for v in p),
                                         outline='white', dash=(5, 3), tags='preview')

    def motion(self, event):
        side = self.box_side(event, self.require_object()) if self.mode.get() == 'select' and self.selected is not None else None
        self.canvas.configure(cursor=('sb_h_double_arrow' if side % 2 == 0 else 'sb_v_double_arrow')
                              if side is not None else '')
        if self.mode.get() == 'grasp' and self.pending:
            self.canvas.delete('preview')
            point = self.point(event)
            if len(self.pending) == 1:
                self.canvas.create_line(*(v*self.scale for p in (self.pending[0], point) for v in p),
                                        fill='white', tags='preview')
            else:
                a, b = self.pending
                length = math.dist(a, b)
                if length < 1:
                    return
                height = abs((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])) / length
                if height >= 0.5:
                    points = grasp_rectangle(a, b, height)
                    self.canvas.create_polygon(*(v*self.scale for p in points for v in p), outline='white',
                                               fill='', tags='preview')

    def release(self, event):
        if self.resizing:
            box = self.resized_box(event)
            original = self.resizing
            self.resizing = None
            self.canvas.configure(cursor='')
            if box != original['box']:
                obj = next(o for o in self.sample['objects'] if o['id'] == original['id'])
                self.change(lambda: obj.update(box=box))
            else:
                self.draw()
            return
        if self.mode.get() not in ('box', 'rebox') or not self.pending:
            return
        a, b = self.pending[0], self.point(event)
        box = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
        self.pending = []
        if box[2]-box[0] < 1 or box[3]-box[1] < 1:
            self.draw()
            return
        if self.mode.get() == 'rebox':
            self.change(lambda: self.require_object().update(box=box))
        else:
            new_id = max([o['id'] for o in self.sample['objects']], default=0)+1
            def add():
                self.sample['objects'].append({
                    'id': new_id, 'class_id': self.selected_class_id(), 'box': box})
                self.sample['relations_reviewed'] = False
            self.change(add)
            self.selected = new_id
            self.refresh()

    def add_class(self):
        name = simpledialog.askstring('Add class', 'New class name (IDs 0–30 stay fixed):')
        if name is not None:
            index = self.project.add_class(name)
            self.class_search.set('')
            self.class_name.set(self.project.classes[index])
            self.refresh()

    def reclassify(self):
        obj = self.require_object()
        self.change(lambda: obj.update(class_id=self.selected_class_id()))

    def delete_object(self):
        obj = self.require_object()
        def remove():
            self.sample['objects'].remove(obj)
            self.sample['grasps'] = [g for g in self.sample['grasps'] if g['object_id'] != obj['id']]
            self.sample['edges'] = [e for e in self.sample['edges'] if obj['id'] not in e]
            self.sample['relations_reviewed'] = False
        self.change(remove)

    def delete_grasp(self):
        selection = self.grasps.curselection()
        if selection:
            index = self.grasp_indices[selection[0]]
            self.change(lambda: self.sample['grasps'].pop(index))

    def add_edge(self):
        self.require_object()
        if self.below.current() < 0:
            raise ValueError('Choose the lower object.')
        below = self.sample['objects'][self.below.current()]['id']
        edge = [self.selected, below]
        if edge not in self.sample['edges']:
            self.change(lambda: self.sample['edges'].append(edge))

    def delete_edge(self):
        selection = self.edges.curselection()
        if selection:
            self.change(lambda: self.sample['edges'].pop(selection[0]))

    def review(self):
        self.change(lambda: self.sample.update(relations_reviewed=self.reviewed.get()))

    def confirm_relationships(self):
        if not self.sample['relations_reviewed']:
            self.reviewed.set(True)
            self.review()

    def change_grasp_mode(self):
        self.project.set_require_grasps(self.require_grasps.get())
        self.history = []
        self.refresh()

    def complete(self):
        if self.pending:
            raise ValueError('Finish the current annotation or press Escape before completing.')
        validate(self.sample, self.project.classes, self.image.size, complete=True,
                 require_grasps=self.project.require_grasps)
        self.sample['complete'] = True
        self.project.save()
        self.refresh()

    def export(self):
        parent = filedialog.askdirectory(title='Choose parent folder for a NEW exported dataset')
        if not parent:
            return
        name = simpledialog.askstring('Dataset folder', 'New folder name:', initialvalue='topdown_labeled')
        if not name:
            return
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Enter a single new folder name.')
        output = self.project.export(Path(parent) / name)
        messagebox.showinfo('Export complete', f'{output}\n\n80/20 train/validation split; classes.json records class IDs.\nSee scripts/README_labeling.md for training.')

    def close(self):
        self.project.save()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('data/topdown_renders'))
    parser.add_argument('--project', type=Path, default=Path('data/topdown_labels/project.json'))
    parser.add_argument('--exclude', action='append', default=['density_map.png'], help='Filename to omit; repeatable')
    parser.add_argument('--export', type=Path, help='Export completed labels without opening the GUI')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--font-size', type=int, default=14, help='UI text size in points (default: 14)')
    parser.add_argument('--no-grasps', action='store_const', const=False, default=None,
                        dest='require_grasps', help='Allow completion without grasps; persist boxes/relationships mode')
    args = parser.parse_args()
    if args.font_size < 10:
        parser.error('--font-size must be at least 10')
    project = Project(args.images, args.project, args.exclude, require_grasps=args.require_grasps)
    if args.export:
        print(project.export(args.export, args.val_fraction, args.seed))
    else:
        root = tk.Tk()
        prefer_smooth_font_renderer(root)
        Labeler(root, project, font_size=args.font_size)
        root.mainloop()


if __name__ == '__main__':
    main()
