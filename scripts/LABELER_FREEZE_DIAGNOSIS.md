# Labeler desktop freeze investigation — November 6–7, 2026

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
