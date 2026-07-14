from pathlib import Path
import socket
import threading
import time

try:
    from flask import Flask, Response, jsonify, render_template, send_file
    from werkzeug.serving import make_server, WSGIRequestHandler
    FLASK_IMPORT_ERROR = None
except (ModuleNotFoundError, ImportError) as exc:
    Flask = None
    Response = None
    jsonify = None
    render_template = None
    send_file = None
    make_server = None
    WSGIRequestHandler = object
    FLASK_IMPORT_ERROR = exc


class LowLatencyRequestHandler(WSGIRequestHandler):
    """Keep each MJPEG connection's kernel send buffer tiny so latency can't
    accumulate: once the buffer fills, the stream generator blocks on yield and
    then emits the *newest* frame, dropping stale ones instead of queuing a
    growing backlog. TCP_NODELAY avoids Nagle batching so frames go out at once.
    """

    def setup(self):
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024)
        except OSError:
            pass

from .image_utils import build_camera_placeholder_svg

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = PACKAGE_ROOT / 'templates'
STATIC_DIR = PACKAGE_ROOT / 'static'


class FlaskServerThread(threading.Thread):
    def __init__(self, app, host, port):
        super().__init__(daemon=True)
        self._server = make_server(
            host, port, app, threaded=True,
            request_handler=LowLatencyRequestHandler,
        )

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        self._server.shutdown()


def create_app( state, page_title, 
                battery_topic, image_topic, control_topic, storage_path,
                refresh_interval_ms, image_refresh_interval_ms,
                header_logo_path, telechips_logo_path, topst_logo_path, 
                image_display_width, image_display_height,
                debug_image, opencv_grayscale_topic, opencv_blur_topic,
                opencv_edge_topic, graph_snapshot_provider=None ):
    
    app = Flask( __name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR),)
    app.json.sort_keys = False

    @app.get('/')
    def index():
        return render_template(
            'index.html',
            page_title=page_title,
            battery_topic=battery_topic,
            image_topic=image_topic,
            control_topic=control_topic,
            storage_path=storage_path,
            refresh_interval_ms=refresh_interval_ms,
            image_refresh_interval_ms=image_refresh_interval_ms,
            placeholder_url='/api/frame/placeholder',
            header_logo_url='/assets/header-logo',
            telechips_logo_url='/assets/telechips-logo',
            topst_logo_url='/assets/topst-logo',
            debug_image=debug_image,
            opencv_grayscale_topic=opencv_grayscale_topic,
            opencv_blur_topic=opencv_blur_topic,
            opencv_edge_topic=opencv_edge_topic,
        )

    @app.get('/api/status')
    def api_status():
        return jsonify(state.snapshot())

    @app.get('/api/graph')
    def api_graph():
        if graph_snapshot_provider is None:
            return jsonify({'nodes': [], 'edges': []})
        return jsonify(graph_snapshot_provider())

    @app.get('/api/frame')
    def api_frame():
        frame_bytes = state.get_latest_frame()
        if frame_bytes is None:
            return Response(
                build_camera_placeholder_svg(image_display_width, image_display_height, image_topic),
                mimetype='image/svg+xml',
            )

        response = Response(frame_bytes, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    @app.get('/api/frame/placeholder')
    def api_frame_placeholder():
        return Response(
            build_camera_placeholder_svg(image_display_width, image_display_height, image_topic),
            mimetype='image/svg+xml',
        )

    def mjpeg_stream(frame_seq_provider):
        # Pushes each new JPEG frame exactly once as an MJPEG (multipart) stream.
        # Avoids per-frame HTTP polling so the browser stays in sync at the
        # publish rate (~30 fps) with minimal latency.
        boundary = b'--frame\r\n'
        last_seq = None
        idle_frames = 0
        while True:
            frame_bytes, seq = frame_seq_provider()
            if frame_bytes is None or seq == last_seq:
                # Wait briefly for the next frame instead of busy-spinning.
                time.sleep(0.005)
                idle_frames += 1
                # Periodic keep-alive comment so proxies keep the socket open.
                if idle_frames >= 600:
                    idle_frames = 0
                    yield b'--frame\r\nContent-Type: text/plain\r\n\r\n\r\n'
                continue
            last_seq = seq
            idle_frames = 0
            yield (
                boundary
                + b'Content-Type: image/jpeg\r\n'
                + b'Content-Length: ' + str(len(frame_bytes)).encode('ascii') + b'\r\n\r\n'
                + frame_bytes
                + b'\r\n'
            )

    def make_stream_response(frame_seq_provider):
        return Response(
            mjpeg_stream(frame_seq_provider),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                'Pragma': 'no-cache',
                'Connection': 'close',
                'X-Accel-Buffering': 'no',
            },
        )

    @app.get('/api/stream')
    def api_stream():
        return make_stream_response(state.get_latest_frame_seq)

    @app.get('/api/stream/grayscale')
    def api_stream_grayscale():
        return make_stream_response(lambda: state.get_debug_frame_seq('grayscale'))

    @app.get('/api/stream/blur')
    def api_stream_blur():
        return make_stream_response(lambda: state.get_debug_frame_seq('blur'))

    @app.get('/api/stream/edge')
    def api_stream_edge():
        return make_stream_response(lambda: state.get_debug_frame_seq('edge'))

    @app.get('/api/frame/grayscale')
    def api_frame_grayscale():
        frame_bytes = state.get_debug_frame('grayscale')
        if frame_bytes is None:
            return Response(
                build_camera_placeholder_svg(
                    image_display_width, image_display_height, opencv_grayscale_topic
                ),
                mimetype='image/svg+xml',
            )

        response = Response(frame_bytes, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    @app.get('/api/frame/blur')
    def api_frame_blur():
        frame_bytes = state.get_debug_frame('blur')
        if frame_bytes is None:
            return Response(
                build_camera_placeholder_svg(
                    image_display_width, image_display_height, opencv_blur_topic
                ),
                mimetype='image/svg+xml',
            )

        response = Response(frame_bytes, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    @app.get('/api/frame/edge')
    def api_frame_edge():
        frame_bytes = state.get_debug_frame('edge')
        if frame_bytes is None:
            return Response(
                build_camera_placeholder_svg(
                    image_display_width, image_display_height, opencv_edge_topic
                ),
                mimetype='image/svg+xml',
            )

        response = Response(frame_bytes, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    @app.get('/assets/header-logo')
    def header_logo():
        return send_file(str(header_logo_path), mimetype='image/png', max_age=0)

    @app.get('/assets/telechips-logo')
    def telechips_logo():
        return send_file(str(telechips_logo_path), mimetype='image/png', max_age=0)

    @app.get('/assets/topst-logo')
    def topst_logo():
        return send_file(str(topst_logo_path), mimetype='image/png', max_age=0)

    return app
