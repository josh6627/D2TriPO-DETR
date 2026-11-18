# Labeler desktop freeze investigation — November 6–7, 2026

## Recurrence after Next image, November 18 at 12:16 PM EST

The user reported a crash immediately after clicking Next image. In boot
`21e5fc78b530499d88d5490e359473a4`, kernel `7.0.0-42-generic` logged
`GT0: GUC: Engine reset failed on 0:0 (rcs0)` at **12:16:30.365997 PM EST**,
followed by forcewake timeouts. No OOM-kill entries were found in that boot.
The next boot began at 12:19:56 PM. The current boot runs `7.0.0-43-generic`;
both `7.0.0-38-generic` and the failing `-42` remain installed.

The final labeler entries are LOAD/RENDER of `000361.png` at 12:16:20.981 /
12:16:21.006, displayed at 1420×1158. There is no subsequent callback error or
normal CLOSE entry. An earlier review-validation exception at 12:12:23 was
followed by continued labeling and is not evidence of the eventual failure.
The saved annotations for the preceding image `000360.png` contain no edges;
`000361.png` has two class-1 edges and no class-2 edges. This does not establish
a scroll or dashed-proximity-arrow trigger. Current labeler source exactly
matches the preserved direction-clarification snapshot.

Preserved the project, available rotated labeler logs, exact application source,
previous-boot kernel journal and annotation audit under
`results/labeler_freeze_20261118_121630/`. All 486 annotations validate, including
350 completed images and the unchanged 43-class registry. The source project
was read, not rewritten.

The evidence confirms recurrence of the Intel graphics failure, but does not
identify the application operation or kernel defect that triggers it. No new
application patch, driver change or reboot was performed. The next controlled
comparison is the same labeler on the currently booted `-43`, keeping app code
unchanged. If it recurs, a one-time boot of installed `-38` is another comparison;
neither has been established as a fix. Avoid attributing stability to an app
change while simultaneously changing the kernel.

## Proximity direction clarification, November 18, 2026

Clarified sidebar instructions and relationship-list text: `#2 → #3` means
`#2 obstructs grasping #3`; the selected source is the arrow tail and the
affected target is the arrowhead. Removing the source may clear access to the
target, without certifying a full pick sequence. Export metadata and labeling
guides now state the same rule. Stored edge orientation and arrow rendering
remain unchanged. A code snapshot and isolated UI patch are saved as
`direction_clarification_label_dataset.py` and `direction_clarification.patch`
in `results/labeler_sidebar_scroll_2026-11-18/` for future crash comparisons.
Twenty-nine targeted tests passed, including a selected-object #2 / target #3
edit through export verifying class 2 only in the source-row/target-column
entry. No window-based stability test or new crash diagnosis was performed.

## Sidebar startup error and correction, November 18, 2026

After the scrolling change, the user reported startup failing in `bind_class`
with `TclError: bad window path name` for the canvas path suffixed with
`.SidebarWheel`. The new tag began with `.`, which Tk interprets as a widget
path rather than an arbitrary class tag. This identifies a Python/Tk startup
regression introduced by the scrolling patch; it supplies no evidence of a
recurrence of the earlier kernel/GPU freezes.

Corrected registration prefixes the canvas path with `D2TriPOSidebarWheel:`.
The startup marker is now `SIDEBAR_SCROLL enabled revision=2026-11-18-r2`.
Added a regression test that exercises registration using the reported canvas
path and checks all three wheel bindings use a non-window class tag. Eleven
tests passed; eight window tests still skipped without a display, so actual
window startup has not been verified here. The initial failed source remains
in the original scrolling snapshots; `corrected_label_dataset.py` and
`startup_tag_fix.patch` in that same artifact folder preserve this correction.

## Sidebar scrolling change, November 18, 2026

The user reported that wheel scrolling did not work over sidebar labels and
asked to preserve a record in case crashes return. No new crash was reported
in this request. Inspection found no sidebar wheel bindings.

The local, uncommitted change in `scripts/label_dataset.py` adds one scoped
bindtag to existing sidebar widgets for `<MouseWheel>` and Linux buttons 4/5.
The callback moves only the sidebar viewport. It does not call annotation
redraw, image upload, save, or schedule callbacks. Listboxes retain native
scrolling while they can move, handing off to the panel at a boundary; closed
comboboxes scroll the panel without changing their value. Modal grabs and
combobox popdowns are left to Tk using raw Tcl grab inspection, avoiding Python
lookup of Tcl-created widget paths. No global wheel binding is installed.

An exact local working-tree baseline, updated source, isolated patch, regression
test source and SHA-256 metadata are saved in
`results/labeler_sidebar_scroll_2026-11-18/`. The baseline already includes the
earlier November 18 proximity/typed-edge UI (including dashed orange arrows)
and imported fixed-split support. The patch isolates scrolling from those
earlier changes. These artifacts are ignored local files, not a Git commit.
Startup now logs `SIDEBAR_SCROLL enabled revision=2026-11-18` once. Keep the
project-adjacent `labeler.log` and rotated backups if a crash recurs; for the
combined project these are under
`data/combined_datasets/combined_real_cluttered_lab/`.

Validation: seven new display-free tests cover wheel direction, widget routing,
boundaries, modal grabs, tag scope/idempotence and 500 wheel events without
redraw/save/scheduled callbacks. Together with startup/font and render-budget
checks, 10 tests passed; eight window-based tests skipped because a display was
unavailable. This does not establish physical GPU/driver stability. If crashes
return, correlate the startup marker and last logged operation with the kernel
evidence described below, comparing the saved scrolling patch and earlier
proximity rendering changes rather than assuming a cause.

## Resize recurrence and targeted rendering patch, November 7 at 6:04 PM EST

The user reported another full-system freeze while resizing. Boot
`c797710ddd674fd29413b5c9b2a3c181` logged the same Intel GuC engine-reset failure
at **6:04:39 PM EST**, followed by forcewake timeouts. The latest surviving
labeler entry is an image load at 6:04:15 PM; no resize completion is recorded
after that load. Evidence and a project backup are in
`results/labeler_freeze_20261107_180439/`.

Code review identified unthrottled delete/recreate operations for each pointer
event, repeated cursor configurations, and unnecessary test-status rewrites.
The targeted patch retains all UI and export functionality but:

- Reuses one preview canvas item and coalesces motion into at most one pending
  callback, with roughly 33 ms between frames and solid rather than dashed lines.
- Commits the exact release-event position instead of the last preview position.
- Cancels queued frames on redraw/cancel, navigation, zoom, and application close.
- Avoids unchanged cursor, image-dropdown, scroll-region, and test-status updates.
- Logs resize beginnings as well as commits for future correlation.

This is a rendering-load mitigation. It has not yet been established as a fix
for the physical Intel-driver failure. No driver or kernel configuration changed.

## Recurrence with the classic UI, November 7 at 5:42 PM EST

Boot `0bb87921f4d249e8a06a5b29beb1d7a8` was running Linux
`7.0.0-42-generic`. At **5:42:58 PM EST**, the Intel kernel driver again recorded
`GT0: GUC: Engine reset failed on 0:0 (rcs0)` and forcewake timeouts. The next
boot started at 5:43:56 PM. No OOM-kill entry was found in the checked boot.

The labeler log confirms Conda Python was used, so the automatic renderer switch
was no longer involved. The final pre-freeze logged action was a resize on
`synth_frame_000288_cam0.png` at 5:42:34 PM, at 100% zoom. An earlier callback
error at 5:34:30 PM was an ordinary validation error from attempting to complete
an image with no objects; labeling continued afterward. No callback traceback
was recorded at the time of the system failure.

Restoring the classic UI did not prevent recurrence. This strengthens the
evidence for a driver/display-path failure; it does not establish the exact
application operation that triggers the driver bug. The proposed comparison
with kernel `7.0.0-38-generic` has not occurred in the captured failing boot.

The project parsed successfully: all 125 sample annotations validated, with
79 completed images and 16 saved test selections. A project backup, labeler log,
and filtered kernel evidence were saved in
`results/labeler_freeze_20261107_174258/`. No graphics, kernel, or boot settings
were modified.

## UI rollback for comparison

At the user's request, the classic native-theme UI with one scrollable sidebar
was restored, with 16-point text by default. Automatic switching from Conda
Python/Tk to system Python/Tk was removed. Search, shortcuts, resize handles,
annotation fixes, background caching, rendering limits, and diagnostic logging
remain. All 19 labeling tests pass on the software virtual display. Whether this
rollback prevents the actual laptop graphics failure has not yet been tested.

## Recurrence after the rendering fix, 2:22 PM EST

The next reported freeze occurred in boot
`e1fe1c64d2b14519957051b24c13077a`. At **2:22:32 PM EST**, the kernel again logged
`GT0: GUC: Engine reset failed on 0:0 (rcs0)` followed by repeated forcewake
timeouts through 2:22:42 PM. The machine restarted at 2:23:49 PM.

The diagnostic log confirms the optimized labeler was running under system
Python from 2:05:44 PM. Images stayed at 1920×1080 / 100% zoom. Selections and
resize commits did not trigger background reuploads; the final logged resize
was at 2:22:17 PM on `synth_frame_002283_cam0.png`. No callback exception was
recorded. Thus the cache and memory changes did not prevent the system failure.
The failure is reproducible at the Intel graphics-driver level, although the
specific triggering operation remains unknown.

The lab project JSON was reread and validated without editing it. A copy was
saved beside it as `project.freeze-backup-20261106-142232.json`. The labeler log
and filtered kernel evidence were copied to
`results/labeler_freeze_20261106_142232/`.

A controlled next test is booting the already installed **7.0.0-38-generic**
kernel instead of **7.0.0-42-generic**. Its kernel image, initramfs, and module
directory are present. Select its normal entry under GRUB's **Advanced options
for Ubuntu**, then confirm with `uname -r`. Selecting an older installed kernel
is supported by the [Ubuntu GRUB submenu documentation](https://help.ubuntu.com/community/Grub2/Submenus).
This tests whether the failure depends on the current kernel; it is not an
established fix. No boot defaults, graphics configuration, packages, or kernel
parameters were changed. No reboot was initiated.

Do not assume `LIBGL_ALWAYS_SOFTWARE=1` will bypass the problem: the labeler
draws through Tk/X11, whose display path can still use the failing Intel driver.

## Initial investigation

The strongest evidence is a failed Intel graphics-engine reset in the kernel
journal during one of the two sessions reported to freeze. The exact trigger
is not established. The labeler's unnecessary image uploads were a plausible
source of display load, and have been removed; this is a mitigation, not proof
that the underlying graphics-driver problem is resolved.

## Observations

The user reported freezes while selecting/resizing boxes on the same image set,
on different images. The newer `cluttered-sim-scenes` input has 75 RGB images at
1920×1080. The original top-down renders are 512×512.

The boot identified by `bf59796ad27849e9af52eca8ae16cc41` ran from 1:03 PM until
1:31 PM EST. At **1:30:25 PM EST**, its kernel journal recorded:

```text
i915 0000:00:02.0: [drm] *ERROR* GT0: GUC: Engine reset failed on 0:0 (rcs0) because 0x00000000
i915 0000:00:02.0: [drm] *ERROR* vdbox0: timed out waiting for forcewake ack request.
```

105 matching graphics-failure messages were found in that boot, including
repeated video-engine forcewake timeouts until 1:31:27 PM. These are kernel
Intel `i915` errors, and support a graphics/display hang rather than just a
Python callback exception. They do not identify which application triggered it.

The next boot (`14bf3f426d144ca29899b0b32f044ae3`) had no matching engine-reset,
forcewake, or GPU-hang entries; its last recorded messages were at 1:56:05 PM.
A hard reset can prevent final messages from reaching persistent storage, so
there is insufficient evidence to assign both freezes the same cause.

No kernel OOM-kill messages were found in the three preceding boots. At the
post-reboot inspection, 25 GiB of RAM was available, swap usage was zero, and
156 GiB of disk space was available. These post-reboot figures do not establish
memory usage at the moment of a freeze.

Hardware/software observed: Intel Raptor Lake-P Iris Xe integrated graphics,
NVIDIA RTX 4060 Laptop GPU, Linux `7.0.0-42-generic`. NVIDIA startup firmware/power
warnings were also present; no NVIDIA Xid failure was identified near the
reported events. No driver, kernel, power-management, or display-session settings
were changed during this investigation.

## Labeler changes

Previously every `draw()` deleted the canvas background and resized/uploaded a
new Tk image, including on selections and after completed resizes. At 500% zoom,
a 1920×1080 source became 9600×5400: 51.84 million display pixels, about 198 MiB
per four-byte pixel buffer, with additional Pillow/Tk/display allocations.

The labeler now:

- Retains the background canvas item and Tk image across annotation redraws.
  Only loading another image or changing the displayed dimensions uploads pixels.
- Releases the previous background before allocating a replacement.
- Caps displayed images at 8 million pixels and 8,192 pixels per side.
  Full-resolution source data and annotation coordinates remain intact.
- Retains at most 100 undo snapshots per image.
- Writes bounded rotating diagnostics beside the project, including image loads,
  selections, resize commits, zoom changes, and callback tracebacks.
- Prevents nested exception dialogs from recursively opening more dialogs.

Regression tests include 300 redraws of a 1920×1080 background without an
additional image upload, zoom and image-load cache invalidation, bounded display
sizes, and the existing editing/training/export checks. GUI tests run on a
software-only virtual X display to avoid stressing the actual laptop display.
These tests cannot reproduce or rule out a physical graphics-driver hang.

## If it happens again

Keep the project-adjacent `labeler.log` and its rotated backups. After reboot,
compare its last action with the previous boot's kernel journal:

```bash
journalctl --list-boots
journalctl -k -b -1 --no-pager --grep='i915|GPU|forcewake|Engine reset|oom-kill|Out of memory|Killed process'
```

`-1` means the previous boot; choose the affected boot ID if more reboots have
occurred. A recurring Intel engine-reset failure warrants investigation of the
machine's Intel graphics/kernel stack. The current evidence does not justify
claiming that changing a particular driver option will fix it.
