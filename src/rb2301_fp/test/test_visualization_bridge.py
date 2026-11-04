"""Check JSON/HTTP transport, local ROS publishing and SSH process isolation."""
import ast
import gzip
from http.client import BadStatusLine, HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import zlib

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp.obstacle_course import ObstacleCourseNode
from rb2301_fp.visualization_bridge import VisualizationRelay, VisualizationSnapshotServer


class VisualizationBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=['--ros-args', '-p', 'visualization_transport:=ssh',
                         '-p', 'visualization_port:=0'], domain_id=168)

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.source = ObstacleCourseNode(is_simulation=True, calibrate=True)
        self.source.timer.cancel()
        self.source._rviz_timer.cancel()
        self.relay = VisualizationRelay(self.source._rviz_server.port, poll_rate=20.0)
        self.relay.timer.cancel()
        self.reader = Node('fp_snapshot_test_reader')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.reader)
        self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def tearDown(self):
        self.executor.shutdown()
        self.reader.destroy_node()
        self.relay.destroy_node()
        self.source.destroy_node()

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(predicate(), 'Timed out waiting for the HTTP worker')

    def test_snapshot_reaches_late_local_subscriber_with_geometry_and_colors(self):
        self.source.calibrate = False
        self.source.pose = np.array([2.0, -1.0, 90.0])
        self.source._ob = np.array([[3.0, 0.0], [3.0, 0.1]])
        self.source._segmentation = [self.source._ob.copy()]
        self.source._rviz_predicted_trajectory = np.array([
            [2.0, -1.0, 1.57, 0.1, 0.0], [2.0, -0.9, 1.57, 0.1, 0.0]])
        self.source.publish_rviz_visualization()
        self.wait_for(lambda: self.relay._frame is not None)
        # Robot-side SSH mode has no DDS visualization publisher at all.
        self.assertEqual(self.source.count_publishers('/obstaclecourse/visualization'), 1)
        self.assertFalse(hasattr(self.source, '_rviz_publisher'))
        self.assertEqual(self.source._rviz_server._server.server_address[0], '127.0.0.1')
        self.relay.poll_snapshot()
        received = []
        self.reader.create_subscription(MarkerArray, '/obstaclecourse/visualization',
                                        received.append, self.qos)
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
        self.assertTrue(received, 'Local transient-local relay did not publish')
        scene = {m.ns: m for m in received[-1].markers}
        self.assertEqual(len(scene), 8)
        self.assertEqual(scene['predicted_trajectory'].action, Marker.ADD)
        self.assertEqual(scene['reference_goal'].action, Marker.DELETE)
        self.assertEqual(scene['robot_heading'].points[-1].y, -0.5)
        self.assertEqual(scene['obstacles'].points[0].x, 3.0)
        self.assertEqual(len(scene['segmentation'].colors), 2)
        self.assertAlmostEqual(scene['segmentation'].colors[0].g, 0.4, places=6)
        self.assertTrue(all(m.header.frame_id == 'fp_visualization' for m in scene.values()))
        self.assertTrue(all(m.lifetime.sec == 1 for m in scene.values()))

    def test_short_gap_is_refreshed_but_frozen_snapshot_expires(self):
        self.relay._stale_after = 0.25
        self.source.pose = np.zeros(3)
        self.source._ob = np.array([[1.0, 0.0]])
        with mock.patch.object(self.relay.publisher, 'publish') as publish:
            self.source.publish_rviz_visualization()
            self.wait_for(lambda: self.relay._frame is not None)
            self.relay.poll_snapshot()
            self.relay.poll_snapshot()
            self.assertEqual(publish.call_count, 2)  # Keep RViz alive during a short gap.
            previous_time = self.relay._frame[1]
            time.sleep(0.3)  # Server responds, but its sequence has not changed.
            self.relay.poll_snapshot()
            self.assertEqual(publish.call_args.args[0].markers[0].action, Marker.DELETEALL)
            self.assertEqual(publish.call_count, 3)
            self.relay.poll_snapshot()
            self.assertEqual(publish.call_count, 3)  # Clear only once.
            self.source._ob = np.empty((0, 2))
            self.source.publish_rviz_visualization()
            self.wait_for(lambda: self.relay._frame[1] != previous_time)
            self.relay.poll_snapshot()
            self.assertEqual(publish.call_count, 4)
            scene = {m.ns: m for m in publish.call_args.args[0].markers}
            self.assertEqual(scene['obstacles'].action, Marker.DELETE)

    def test_waiting_disconnect_and_server_restart_do_not_crash(self):
        with mock.patch.object(self.relay.publisher, 'publish') as publish:
            self.relay.poll_snapshot()  # No pose yet: HTTP 503, no ROS message.
            self.assertEqual(publish.call_count, 0)
            self.source.pose = np.zeros(3)
            self.source.publish_rviz_visualization()
            self.wait_for(lambda: self.relay._frame is not None)
            self.relay.poll_snapshot()
            self.assertEqual(publish.call_count, 1)
            previous_time = self.relay._frame[1]
            scene = publish.call_args.args[0]
            port = self.source._rviz_server.port
            self.source._rviz_server.close()
            self.wait_for(lambda: self.relay._last_error is not None)
            self.relay.poll_snapshot()  # Hold the last frame through a short disconnection.
            restarted = VisualizationSnapshotServer(port)
            try:
                restarted.publish(scene)
                self.wait_for(lambda: self.relay._frame[1] != previous_time)
                self.relay.poll_snapshot()
                self.assertEqual(publish.call_count, 3)
            finally:
                restarted.close()

    def test_bridge_module_parses_with_foxy_python38(self):
        module = Path(__file__).resolve().parents[1] / 'rb2301_fp/visualization_bridge.py'
        ast.parse(module.read_text(), feature_version=(3, 8))

    def test_http_protocol_error_is_recoverable(self):
        for error in (BadStatusLine('closed'), EOFError('truncated gzip'), zlib.error('corrupt gzip')):
            with mock.patch.object(self.relay, '_download_once', side_effect=error):
                self.wait_for(lambda: self.relay._last_error == str(error))
            previous = self.relay._frame
            self.source.pose = np.zeros(3)
            self.source.publish_rviz_visualization()
            self.wait_for(lambda: self.relay._frame is not previous)
            self.relay.poll_snapshot()
            self.assertTrue(self.relay._connected)

    def test_server_compresses_snapshots_and_reuses_http_connection(self):
        self.source.pose = np.zeros(3)
        self.source.publish_rviz_visualization()
        connection = HTTPConnection('127.0.0.1', self.source._rviz_server.port, timeout=2.0)
        try:
            connection.request('GET', '/snapshot', headers={'Accept-Encoding': 'gzip'})
            with connection.getresponse() as response:
                self.assertEqual(response.getheader('Content-Encoding'), 'gzip')
                self.assertFalse(response.will_close)
                snapshot = json.loads(gzip.decompress(response.read()))
            self.assertEqual(snapshot['version'], 1)
            first_socket = connection.sock
            connection.request('GET', '/snapshot')
            with connection.getresponse() as response:
                json.loads(response.read())
            self.assertIs(connection.sock, first_socket)
        finally:
            connection.close()

    def test_slow_legacy_server_does_not_block_ros_callbacks(self):
        self.source.pose = np.zeros(3)
        self.source.publish_rviz_visualization()
        payload = self.source._rviz_server._latest[0]
        entered = threading.Event()

        class SlowHandler(BaseHTTPRequestHandler):
            # HTTP/1.0 and uncompressed: exercise compatibility with the old robot code.
            def do_GET(self):
                entered.set()
                time.sleep(0.6)  # Longer than the old 0.3-second timeout.
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), SlowHandler)
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={'poll_interval': 0.1}, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.relay.destroy_node()
        self.relay = VisualizationRelay(server.server_port, timeout=2.0)
        self.executor.add_node(self.relay)
        heartbeats = []
        self.reader.create_timer(0.02, lambda: heartbeats.append(time.monotonic()))
        self.assertTrue(entered.wait(timeout=2.0))
        deadline = time.monotonic() + 0.45
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)
        self.assertGreaterEqual(len(heartbeats), 10, 'HTTP download blocked the ROS executor')
        self.wait_for(lambda: self.relay._frame is not None)
        self.relay.poll_snapshot()
        self.assertTrue(self.relay._connected)


class SshLauncherTests(unittest.TestCase):
    def test_launcher_isolates_local_children_and_cleans_up_owned_tunnel(self):
        launcher = Path(__file__).resolve().parents[3] / 'rviz_fp.sh'
        if not launcher.exists():
            self.skipTest('Workspace-level rviz_fp.sh is not available')
        with tempfile.TemporaryDirectory(prefix='fp-ssh-launcher-') as directory:
            root = Path(directory)
            shutil.copy2(launcher, root / 'rviz_fp.sh')
            (root / 'install').mkdir()
            (root / 'install/setup.bash').write_text('# Test workspace\n')
            commands = root / 'commands'
            commands.mkdir()
            prefix = root / 'prefix'
            relay_dir = prefix / 'lib/rb2301_fp'
            relay_dir.mkdir(parents=True)

            def executable(path, source):
                path.write_text('#!/usr/bin/python3\n' + source)
                path.chmod(0o755)

            executable(commands / 'ros2', 'print({!r})\n'.format(str(prefix)))
            executable(commands / 'ssh',
                'import json, os, sys\n'
                'with open(os.environ["FP_LAUNCH_TEST"] + "/ssh.jsonl", "a") as f:\n'
                '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n')
            executable(relay_dir / 'visualization_relay',
                'import json, os, time\n'
                'with open(os.environ["FP_LAUNCH_TEST"] + "/relay.json", "w") as f:\n'
                '    json.dump(dict(os.environ), f)\n'
                'while True: time.sleep(0.1)\n')
            executable(commands / 'rviz2',
                'import json, os, pathlib, time\n'
                'root = pathlib.Path(os.environ["FP_LAUNCH_TEST"])\n'
                'for _ in range(100):\n'
                '    if (root / "relay.json").exists(): break\n'
                '    time.sleep(0.01)\n'
                '(root / "rviz.json").write_text(json.dumps(dict(os.environ)))\n')
            env = dict(os.environ, PATH=str(commands) + ':' + os.environ['PATH'],
                       FP_LAUNCH_TEST=str(root), ROS_DOMAIN_ID='14',
                       ROS_AUTOMATIC_DISCOVERY_RANGE='SUBNET',
                       ROS_STATIC_PEERS='192.168.1.214', ROS_DISCOVERY_SERVER='192.168.1.214:11811')
            env.pop('FP_RVIZ_DOMAIN_ID', None)
            subprocess.run(['bash', str(root / 'rviz_fp.sh'), '--ssh', 'bingda@test-robot'],
                           env=env, check=True, timeout=10)
            for name in ('relay', 'rviz'):
                child = json.loads((root / (name + '.json')).read_text())
                self.assertEqual(child['ROS_DOMAIN_ID'], '166')
                self.assertEqual(child['ROS_AUTOMATIC_DISCOVERY_RANGE'], 'LOCALHOST')
                self.assertNotIn('ROS_STATIC_PEERS', child)
                self.assertNotIn('ROS_DISCOVERY_SERVER', child)
            ssh = [json.loads(line) for line in (root / 'ssh.jsonl').read_text().splitlines()]
            self.assertIn('-M', ssh[0])
            self.assertIn('127.0.0.1:8765:127.0.0.1:8765', ssh[0])
            self.assertIn('exit', ssh[-1])


if __name__ == '__main__':
    unittest.main()
