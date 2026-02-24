import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""  # prevent cv2 Qt conflict with vispy

import argparse
import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
from vispy import app, scene
from vispy.color import Colormap
import dv_processing as dv


class EventVisualizer:
    """3D visualization of event camera data with optional 2D preview, image saving, and GIF export."""

    def __init__(self, file_path=None, live=False, slice_ms=50, time_scale=1e-6,
                 max_events=100000, z_scale=1000, export_gif=None, show_axes=False,
                 show_preview=True, save_images=None):
        self.file_path = file_path
        self.live = live
        self.show_axes = show_axes
        self.slice_ms = slice_ms
        self.time_scale = time_scale
        self.max_events = max_events
        self.z_scale = z_scale
        self.export_gif = export_gif
        self.show_preview = show_preview
        self.save_images = save_images

        self.first_ts_us = None
        self.frames = []
        self.frame_count = 0
        self.cmap = Colormap([(0, 0, 0, 1), (0, 0, 1, 1)])

        self._init_camera()
        self._init_canvas()
        self._init_preview()
        self._init_slicer()
        self._init_timer()

    def _init_camera(self):
        if self.live:
            self.camera = dv.io.CameraCapture()
        else:
            self.camera = dv.io.MonoCameraRecording(self.file_path)
        self.resolution = self.camera.getEventResolution()

    def _init_canvas(self):
        self.canvas = scene.SceneCanvas(keys='interactive', show=True,
                                        bgcolor='white', title='3D Event Volume')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'arcball'
        self.view.camera.fov = 45
        self.view.camera.distance = 1000

        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.view.add(self.scatter)

        if self.show_axes:
            self._draw_axes()

    def _draw_axes(self):
        w, h = self.resolution[0], self.resolution[1]
        axes = [
            (np.array([[0, 0, 0], [w, 0, 0]]), 'red', 'X'),
            (np.array([[0, 0, 0], [0, h, 0]]), 'green', 'Y'),
            (np.array([[0, 0, 0], [0, 0, self.z_scale]]), 'blue', 'Time'),
        ]
        for pos, color, label in axes:
            scene.visuals.Line(pos=pos, color=color, width=2, parent=self.view.scene)
            scene.visuals.Text(label, color=color, font_size=12,
                               pos=pos[1] + 20, parent=self.view.scene)

    def _init_preview(self):
        if not self.show_preview and not self.save_images:
            self.dv_visualizer = None
            return
        self.dv_visualizer = dv.visualization.EventVisualizer(self.resolution)
        if self.save_images:
            Path(self.save_images).mkdir(parents=True, exist_ok=True)
        if self.show_preview:
            w, h = self.resolution[0], self.resolution[1]
            self.preview_canvas = scene.SceneCanvas(keys='interactive', show=True,
                                                    size=(w, h), title='2D Event Preview')
            self.preview_view = self.preview_canvas.central_widget.add_view()
            self.preview_view.camera = 'panzoom'
            self.preview_view.camera.set_range(x=(0, w), y=(0, h), margin=0)
            self.preview_image = scene.visuals.Image(parent=self.preview_view.scene)

    def _init_slicer(self):
        self.slicer = dv.EventStreamSlicer()
        self.slicer.doEveryTimeInterval(
            datetime.timedelta(milliseconds=self.slice_ms),
            self._handle_slice
        )

    def _init_timer(self):
        self.timer = app.Timer()
        self.timer.connect(self._update)
        self.timer.start(0.05)

    def _handle_slice(self, event_slice):
        events = event_slice.numpy()
        if len(events) == 0:
            return

        if self.first_ts_us is None:
            self.first_ts_us = events['timestamp'][0]

        # 2D event frame preview / save
        if self.dv_visualizer is not None:
            img = self.dv_visualizer.generateImage(event_slice)
            if self.show_preview:
                rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                self.preview_image.set_data(rgb)
                self.preview_canvas.update()
            if self.save_images:
                out_path = Path(self.save_images) / f"frame1_{self.frame_count:06d}.png"
                cv.imwrite(str(out_path), img)
            self.frame_count += 1

        # 3D scatter plot
        if len(events) > self.max_events:
            idx = np.random.choice(len(events), self.max_events, replace=False)
            events = events[idx]

        x = events['x']
        y = events['y']
        t = (events['timestamp'] - self.first_ts_us) * self.time_scale
        t = t - t.min()
        z = t / (t.max() + 1e-6) * self.z_scale
        p = events['polarity'].astype(np.float32)

        points = np.stack([x, y, z], axis=1)
        colors = self.cmap.map(p)
        self.scatter.set_data(points, edge_width=0, face_color=colors, size=2)

        if self.export_gif:
            frame = self.canvas.render(size=(800, 600))
            self.frames.append(frame)

    def _update(self, _):
        if self.camera.isRunning():
            events = self.camera.getNextEventBatch()
            if events is not None:
                self.slicer.accept(events)
        elif self.export_gif and self.frames:
            self._save_gif()

    def _save_gif(self):
        import imageio.v3 as iio
        output_path = Path(self.export_gif)
        iio.imwrite(str(output_path), np.stack(self.frames), loop=0)
        print(f"GIF saved to {output_path}")
        self.frames.clear()
        self.export_gif = None

    def run(self):
        app.run()


def parse_args():
    parser = argparse.ArgumentParser(description="3D Event Camera Visualizer")
    parser.add_argument("file_path", nargs='?', default=None,
                        help="Path to .aedat4 file (omit for live camera)")
    parser.add_argument("--live", action="store_true",
                        help="Use live camera capture instead of a file")
    parser.add_argument("--slice-ms", type=int, default=50,
                        help="Event slice duration in ms (default: 50)")
    parser.add_argument("--time-scale", type=float, default=1e-6,
                        help="Timestamp scaling factor (default: 1e-6)")
    parser.add_argument("--max-events", type=int, default=100000,
                        help="Max events per slice (default: 100000)")
    parser.add_argument("--z-scale", type=float, default=1000,
                        help="Z-axis (time) scale (default: 1000)")
    parser.add_argument("--show-axes", action="store_true",
                        help="Draw X, Y, Time axes in the scene")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable 2D event frame preview window")
    parser.add_argument("--save-images", type=str, default=None,
                        help="Directory to save event frame images (optional)")
    parser.add_argument("--export-gif", type=str, default=None,
                        help="Path to save GIF of 3D view (optional)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not args.live and args.file_path is None:
        raise SystemExit("Error: provide a file_path or use --live for camera capture")
    viz = EventVisualizer(
        file_path=args.file_path,
        live=args.live,
        slice_ms=args.slice_ms,
        time_scale=args.time_scale,
        max_events=args.max_events,
        z_scale=args.z_scale,
        show_axes=args.show_axes,
        show_preview=not args.no_preview,
        save_images=args.save_images,
        export_gif=args.export_gif,
    )
    viz.run()
