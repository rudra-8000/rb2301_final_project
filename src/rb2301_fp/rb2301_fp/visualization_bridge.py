"""One-way JSON snapshots over an SSH-forwarded localhost HTTP connection.

Only the common Marker fields used by this project cross the connection. ROS
serialization, graph discovery and robot commands never cross this bridge.
The robot-side server uses Python's standard library and is Python 3.8 compatible.
"""
import argparse
import gzip
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
import socket
import threading
import time
import uuid
import zlib

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024


def encode_marker(marker):
    """Use a small explicit schema shared by Foxy and Jazzy, not ROS CDR."""
    p, q = marker.pose.position, marker.pose.orientation
    s, c = marker.scale, marker.color
    return dict(ns=marker.ns, id=marker.id, type=marker.type, action=marker.action,
                frame=marker.header.frame_id,
                pose=[p.x, p.y, p.z, q.x, q.y, q.z, q.w],
                scale=[s.x, s.y, s.z], color=[c.r, c.g, c.b, c.a],
                lifetime=[marker.lifetime.sec, marker.lifetime.nanosec],
                points=[[p.x, p.y, p.z] for p in marker.points],
                colors=[[c.r, c.g, c.b, c.a] for c in marker.colors])


def decode_scene(snapshot, stamp):
    """Rebuild local ROS messages and use the local clock for RViz."""
    if snapshot.get('version') != 1:
        raise ValueError('Unsupported visualization snapshot version')
    entries = snapshot['markers']
    if not isinstance(entries, list) or len(entries) > 128:
        raise ValueError('Invalid marker count')
    scene = MarkerArray()
    for entry in entries:
        marker = Marker()
        marker.header.frame_id = entry['frame']
        marker.header.stamp = stamp
        marker.ns, marker.id = entry['ns'], entry['id']
        marker.type, marker.action = entry['type'], entry['action']
        pose = entry['pose']
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, pose[:3])
        (marker.pose.orientation.x, marker.pose.orientation.y,
         marker.pose.orientation.z, marker.pose.orientation.w) = map(float, pose[3:])
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, entry['scale'])
        marker.color = ColorRGBA(**dict(zip(('r', 'g', 'b', 'a'), map(float, entry['color']))))
        marker.lifetime.sec, marker.lifetime.nanosec = entry['lifetime']
        if len(entry['points']) > 60000 or len(entry['colors']) > 60000:
            raise ValueError('Too many points in visualization snapshot')
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                         for p in entry['points']]
        marker.colors = [ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=float(c[3]))
                         for c in entry['colors']]
        scene.markers.append(marker)
    return scene


class VisualizationSnapshotServer:
    """Cache the latest frame; HTTP clients never block the controller thread."""

    def __init__(self, port=8765):
        self._latest = None
        self._stream = uuid.uuid4().hex
        self._sequence = 0
        self._closed = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_GET(self):
                if self.path != '/snapshot':
                    self.send_error(404)
                    return
                frame = owner._latest  # Immutable byte pair, replaced atomically.
                if frame is None or owner._closed:
                    self.send_error(503, 'Waiting for the first visualization frame')
                    return
                payload, compressed = frame
                use_gzip = 'gzip' in self.headers.get('Accept-Encoding', '')
                if use_gzip:
                    payload = compressed
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                if use_gzip:
                    self.send_header('Content-Encoding', 'gzip')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        # Reachable only through a local connection, normally an SSH -L tunnel.
        self._server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.port = self._server.server_port
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True)
        self._thread.start()

    def publish(self, scene):
        self._sequence += 1
        snapshot = dict(version=1, stream=self._stream, sequence=self._sequence,
                        markers=[encode_marker(marker) for marker in scene.markers])
        payload = json.dumps(snapshot, separators=(',', ':'), allow_nan=False).encode('utf-8')
        if len(payload) > MAX_SNAPSHOT_BYTES:
            raise ValueError('Visualization snapshot exceeds 8 MiB')
        self._latest = (payload, gzip.compress(payload, compresslevel=1))

    def close(self):
        self._closed = True
        if self._thread.is_alive():
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=1.0)


class VisualizationRelay(Node):
    """Download in one worker; refresh RViz independently of network latency."""

    def __init__(self, port=8765, timeout=3.0, stale_after=3.0, poll_rate=5.0):
        if not all(math.isfinite(v) and v > 0 for v in (timeout, stale_after, poll_rate)):
            raise ValueError('timeout, stale_after and poll_rate must be finite and positive')
        super().__init__('fp_visualization_relay')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(
            MarkerArray, '/obstaclecourse/visualization', qos)
        self.url = 'http://127.0.0.1:{}/snapshot'.format(port)
        self._port = port
        self._timeout = timeout
        self._stale_after = stale_after
        self._poll_period = 1.0 / poll_rate
        self._frame = None
        self._last_error = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connection = None
        self._last_warning = -math.inf
        self._connected = False
        self._visible = False
        self._worker = threading.Thread(target=self._download_loop, daemon=True)
        self._worker.start()
        self.timer = self.create_timer(0.1, self.poll_snapshot)
        self.get_logger().info('Waiting for visualization at ' + self.url)

    def _download_once(self, connection):
        connection.request('GET', '/snapshot', headers={'Accept-Encoding': 'gzip'})
        with connection.getresponse() as response:
            if response.status != 200:
                raise HTTPException('HTTP {}: {}'.format(response.status, response.reason))
            payload = response.read(MAX_SNAPSHOT_BYTES + 1)
            compressed = response.getheader('Content-Encoding') == 'gzip'
        if len(payload) > MAX_SNAPSHOT_BYTES:
            raise ValueError('Visualization snapshot exceeds 8 MiB')
        if compressed:
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as response:
                payload = response.read(MAX_SNAPSHOT_BYTES + 1)
            if len(payload) > MAX_SNAPSHOT_BYTES:
                raise ValueError('Visualization snapshot exceeds 8 MiB')
        return json.loads(payload)

    def _download_loop(self):
        last_key = None
        connection = None
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    if connection is None:
                        connection = HTTPConnection('127.0.0.1', self._port, timeout=self._timeout)
                        self._connection = connection
                    snapshot = self._download_once(connection)
                    key = (snapshot['stream'], snapshot['sequence'])
                    if key != last_key:
                        scene = decode_scene(snapshot, Marker().header.stamp)
                        with self._lock:
                            self._frame = (scene, time.monotonic())
                        last_key = key
                    with self._lock:
                        self._last_error = None
                except (OSError, HTTPException, EOFError, zlib.error, ValueError, KeyError,
                        TypeError, IndexError, AssertionError) as error:
                    with self._lock:
                        self._last_error = str(error)
                    if connection is not None:
                        connection.close()
                    connection = None
                    self._connection = None
                # One request at a time; slow links never accumulate requests.
                self._stop.wait(max(0.05, self._poll_period - (time.monotonic() - started)))
        finally:
            if connection is not None:
                connection.close()

    def poll_snapshot(self):
        """Never wait for HTTP in a ROS callback. Hold brief gaps, clear stale data."""
        now = time.monotonic()
        with self._lock:
            frame, error = self._frame, self._last_error
        if frame is not None and now - frame[1] < self._stale_after:
            scene = frame[0]
            stamp = self.get_clock().now().to_msg()
            for marker in scene.markers:
                marker.header.stamp = stamp
            self.publisher.publish(scene)
            self._visible = True
            if not self._connected:
                self.get_logger().info('Receiving visualization snapshots')
            self._connected = True
            return
        if self._visible:
            self.publisher.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
            self._visible = False
        self._connected = False
        if now - self._last_warning >= 5.0:
            reason = 'waiting for the first snapshot' if frame is None else (
                'no new snapshot for {:.1f}s'.format(now - frame[1]))
            if error:
                reason += '; ' + error
            self.get_logger().warning('Visualization unavailable: ' + reason)
            self._last_warning = now

    def destroy_node(self):
        self._stop.set()
        connection = self._connection
        if connection is not None and connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._worker.join(timeout=self._timeout + 0.5)
        return super().destroy_node()


def main(args=None):
    from rclpy.executors import ExternalShutdownException

    parser = argparse.ArgumentParser(description='Relay SSH visualization snapshots into local RViz')
    parser.add_argument('--port', type=int, default=8765, help='Local SSH forwarding port')
    parser.add_argument('--timeout', type=float, default=3.0, help='HTTP socket timeout, seconds')
    parser.add_argument('--stale-after', type=float, default=3.0,
                        help='Clear markers after this many seconds without a new snapshot')
    parser.add_argument('--poll-rate', type=float, default=5.0, help='Maximum HTTP requests per second')
    cli, ros_args = parser.parse_known_args(args)
    # Applied before rclpy.init: this CLI never joins the physical robot network.
    os.environ['ROS_DOMAIN_ID'] = os.environ.get('FP_RVIZ_DOMAIN_ID', '166')
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    for name in ('ROS_STATIC_PEERS', 'ROS_DISCOVERY_SERVER', 'ROS_SUPER_CLIENT'):
        os.environ.pop(name, None)
    rclpy.init(args=ros_args)
    node = VisualizationRelay(cli.port, timeout=cli.timeout,
                              stale_after=cli.stale_after, poll_rate=cli.poll_rate)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
