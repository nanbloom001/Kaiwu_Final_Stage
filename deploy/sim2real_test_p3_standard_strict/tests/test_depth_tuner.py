#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "realsense_depth_tuner", ROOT / "tools" / "realsense_depth_tuner.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DepthTunerTest(unittest.TestCase):
    def test_policy_invalid_mask_and_central_fraction(self):
        depth = np.full((3, 6), 1000, dtype=np.uint16)
        depth[:, 2:4] = 0
        depth[0, 0] = 5000
        mask = MODULE.compute_invalid_mask(depth, 0.001)
        stats = MODULE.compute_invalid_stats(mask)
        self.assertAlmostEqual(stats["whole"], 7 / 18)
        self.assertEqual(stats["central_third"], 1.0)

    def test_invalid_pixels_are_magenta(self):
        depth = np.asarray([[0, 1000, 5000]], dtype=np.uint16)
        image, invalid = MODULE.render_depth(depth, 0.001, 0.1, 5.0)
        np.testing.assert_array_equal(image[0, 0], [255, 0, 255])
        np.testing.assert_array_equal(image[0, 2], [255, 0, 255])
        self.assertFalse(invalid[0, 1])

    def test_filter_effect_distinguishes_recovered_and_lost(self):
        raw = np.asarray([[True, False, True]], dtype=bool)
        filtered = np.asarray([[False, True, True]], dtype=bool)
        image = MODULE.render_invalid_changes(raw, filtered)
        np.testing.assert_array_equal(image[0, 0], [0, 220, 0])
        np.testing.assert_array_equal(image[0, 1], [0, 0, 255])
        np.testing.assert_array_equal(image[0, 2], [255, 255, 255])

    def test_stream_gap_with_sensor_gap_points_to_camera_or_usb(self):
        events = MODULE.classify_stream_discontinuity(10, 14, 220.0, 200.0, 120.0)
        self.assertEqual(events[0]["kind"], "FRAME_JUMP")
        self.assertEqual(events[0]["dropped_frames"], 3)
        self.assertEqual(events[1]["kind"], "STREAM_GAP")
        self.assertEqual(events[1]["suspected_cause"], "camera_or_usb_stall")

    def test_stream_gap_with_normal_sensor_clock_points_to_host(self):
        events = MODULE.classify_stream_discontinuity(10, 11, 220.0, 33.3, 120.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "STREAM_GAP")
        self.assertEqual(
            events[0]["suspected_cause"], "host_processing_or_delivery_stall"
        )

    def test_parameter_reset_uses_sdk_default(self):
        parameter = MODULE.NumericParameter(
            "test", "Test", 0.0, 10.0, 1.0, 7.0, default_value=3.0
        )
        parameter.set_desired(9.0)
        self.assertEqual(parameter.reset(), 3.0)
        self.assertEqual(parameter.desired_value(), 3.0)

    def test_hardware_parameter_waits_for_debounce(self):
        class Target:
            def __init__(self):
                self.value = 1.0
                self.set_calls = []

            def set_option(self, _option, value):
                self.value = value
                self.set_calls.append(value)

            def get_option(self, _option):
                return self.value

        class Logger:
            def __init__(self):
                self.events = []

            def event(self, kind, **fields):
                self.events.append((kind, fields))

        target = Target()
        logger = Logger()
        parameter = MODULE.NumericParameter(
            "laser", "Laser", 0.0, 360.0, 30.0, 0.0,
            default_value=150.0, target=target, option=object(),
        )
        parameter.set_desired(360.0)
        with parameter.lock:
            parameter.desired_changed_at = 100.0
        self.assertFalse(parameter.apply(logger, settle_s=0.4, now_s=100.2))
        self.assertEqual(target.set_calls, [])
        self.assertTrue(parameter.apply(logger, settle_s=0.4, now_s=100.5))
        self.assertEqual(target.set_calls, [360.0])

    def test_parameter_cap_clamps_desired_but_preserves_device_maximum(self):
        parameter = MODULE.NumericParameter(
            "laser", "Laser", 0.0, 360.0, 30.0, 330.0, default_value=150.0
        )
        parameter.cap_maximum(240.0)
        self.assertEqual(parameter.maximum, 240.0)
        self.assertEqual(parameter.device_maximum, 360.0)
        self.assertEqual(parameter.desired_value(), 240.0)

    def test_competition_camera_options_are_exposed(self):
        keys = {spec[0] for spec in MODULE.COMPETITION_SENSOR_OPTION_SPECS}
        self.assertTrue({
            "frames_queue", "auto_exposure_limit_toggle", "auto_exposure_limit",
            "auto_gain_limit_toggle", "auto_gain_limit", "global_time",
            "error_polling",
        }.issubset(keys))
        self.assertFalse({"hdr", "inter_cam_sync", "output_trigger"} & keys)
        restart_required = {spec[0] for spec in MODULE.COMPETITION_SENSOR_OPTION_SPECS if spec[3]}
        self.assertEqual(restart_required, {
            "auto_exposure_limit_toggle", "auto_exposure_limit",
            "auto_gain_limit_toggle", "auto_gain_limit",
        })

    def test_frame_metadata_readback_is_non_uvc(self):
        class Frame:
            def supports_frame_metadata(self, metadata):
                return metadata in {
                    MODULE.rs.frame_metadata_value.actual_exposure,
                    MODULE.rs.frame_metadata_value.gain_level,
                }

            def get_frame_metadata(self, metadata):
                if metadata == MODULE.rs.frame_metadata_value.actual_exposure:
                    return 8500
                return 64

        values = MODULE.read_frame_metadata(Frame())
        self.assertEqual(values["actual_exposure_us"], 8500.0)
        self.assertEqual(values["actual_gain"], 64.0)
        self.assertNotIn("actual_laser", values)

    def test_absolute_frame_age_uses_synchronized_timestamp_domains(self):
        for domain in (
            str(MODULE.rs.timestamp_domain.global_time),
            str(MODULE.rs.timestamp_domain.system_time),
        ):
            self.assertAlmostEqual(
                MODULE.absolute_frame_age_ms(1_000_000.0, domain, 1_000_087.5),
                87.5,
            )

    def test_absolute_frame_age_rejects_hardware_clock_and_bad_offsets(self):
        hardware = str(MODULE.rs.timestamp_domain.hardware_clock)
        self.assertTrue(np.isnan(
            MODULE.absolute_frame_age_ms(1_000.0, hardware, 1_050.0)
        ))
        global_time = str(MODULE.rs.timestamp_domain.global_time)
        self.assertTrue(np.isnan(
            MODULE.absolute_frame_age_ms(1_000.0, global_time, 100_000.0)
        ))
        self.assertTrue(np.isnan(
            MODULE.absolute_frame_age_ms(1_010.0, global_time, 1_000.0)
        ))

    def test_display_latency_csv_keeps_frame_and_filter_context(self):
        required = {
            "frame_number", "frame_timestamp_ms", "timestamp_domain",
            "frame_age_receive_ms", "frame_age_ready_ms",
            "frame_age_gui_submit_ms", "gui_schedule_ms", "gui_convert_ms",
            "frames_queue_size", "temporal_enabled", "temporal_alpha",
            "temporal_delta", "temporal_persistence",
        }
        self.assertTrue(required.issubset(MODULE.DiagnosticLogger.DISPLAY_METRIC_FIELDS))

    def test_rejected_option_remains_unapplied_and_is_rate_limited(self):
        class Target:
            def __init__(self):
                self.calls = 0

            def set_option(self, _option, _value):
                self.calls += 1
                raise RuntimeError("stale device node")

        class Logger:
            def event(self, *_args, **_kwargs):
                pass

        parameter = MODULE.NumericParameter(
            "laser", "Laser", 0.0, 360.0, 30.0, 0.0,
            target=Target(), option=object(),
        )
        parameter.set_desired(90.0)
        with parameter.lock:
            parameter.desired_changed_at = 100.0
        self.assertFalse(parameter.apply(Logger(), now_s=101.0, retry_s=2.0))
        self.assertTrue(np.isnan(parameter.applied_value()))
        self.assertFalse(parameter.apply(Logger(), now_s=102.0, retry_s=2.0))
        self.assertEqual(parameter.target.calls, 1)
        self.assertFalse(parameter.apply(Logger(), now_s=103.1, retry_s=2.0))
        self.assertEqual(parameter.target.calls, 2)

    def test_restart_required_option_is_not_retried_in_capture_loop(self):
        class Target:
            def __init__(self):
                self.calls = 0

            def set_option(self, _option, _value):
                self.calls += 1

            def get_option(self, _option):
                return 1.0

        class Logger:
            def event(self, *_args, **_kwargs):
                pass

        target = Target()
        parameter = MODULE.NumericParameter(
            "auto_gain_limit", "Gain limit", 0.0, 248.0, 1.0, 248.0,
            target=target, option=object(), restart_required=True,
        )
        parameter.set_desired(100.0)
        tuner = MODULE.DepthTuner.__new__(MODULE.DepthTuner)
        tuner.args = SimpleNamespace(camera_debounce_ms=400.0)
        tuner.logger = Logger()
        tuner.filter_lock = threading.Lock()
        tuner.filter_enabled = {"spatial": False, "temporal": False, "hole_fill": False}
        tuner.filter_options = {}
        tuner.sensor_options = {"auto_gain_limit": parameter}
        tuner.sensor_controls_ready = threading.Event()
        tuner.sensor_controls_ready.set()

        tuner._apply_options(force=True)

        self.assertEqual(target.calls, 0)
        self.assertEqual(parameter.desired_value(), 100.0)

    def test_manual_restart_skips_pipeline_reopen_when_all_controls_reject(self):
        class Target:
            def set_option(self, _option, _value):
                raise RuntimeError("firmware rejected control")

            def get_option(self, _option):
                return 248.0

        class Pipeline:
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1

        class Logger:
            def __init__(self):
                self.events = []

            def event(self, kind, **fields):
                self.events.append((kind, fields))

        parameter = MODULE.NumericParameter(
            "auto_gain_limit", "Gain limit", 0.0, 248.0, 1.0, 248.0,
            target=Target(), option=object(), restart_required=True,
        )
        parameter.set_desired(100.0)
        tuner = MODULE.DepthTuner.__new__(MODULE.DepthTuner)
        tuner.args = SimpleNamespace(camera_debounce_ms=400.0)
        tuner.logger = Logger()
        tuner.pipeline = Pipeline()
        tuner.filter_lock = threading.Lock()
        tuner.filter_enabled = {"spatial": False, "temporal": False, "hole_fill": False}
        tuner.filter_options = {}
        tuner.sensor_options = {"auto_gain_limit": parameter}
        tuner.restart_requested = threading.Event()
        tuner.restart_requested.set()
        tuner.rebind_requested = threading.Event()
        tuner.restart_in_progress = threading.Event()
        tuner.sensor_controls_ready = threading.Event()
        tuner.control_state_lock = threading.Lock()
        tuner.control_state = "RESTART_REQUESTED"

        self.assertTrue(tuner._restart_stream("manual_restart", preserve_all=True))

        self.assertEqual(tuner.pipeline.stop_calls, 0)
        self.assertEqual(tuner.control_status(), "ONLINE")
        self.assertTrue(tuner.sensor_controls_ready.is_set())
        self.assertFalse(tuner.restart_requested.is_set())
        self.assertEqual(tuner._pending_restart_option_keys(), set())
        self.assertIn("STREAM_RESTART_SKIPPED", [kind for kind, _ in tuner.logger.events])

    def test_restart_request_is_ignored_while_device_is_offline(self):
        class Logger:
            def __init__(self):
                self.events = []

            def event(self, kind, **fields):
                self.events.append((kind, fields))

        tuner = MODULE.DepthTuner.__new__(MODULE.DepthTuner)
        tuner.logger = Logger()
        tuner.running = threading.Event()
        tuner.running.set()
        tuner.restart_requested = threading.Event()
        tuner.sensor_controls_ready = threading.Event()
        tuner.control_state_lock = threading.Lock()
        tuner.control_state = "WAITING_FOR_DEVICE"

        tuner.request_stream_restart()

        self.assertFalse(tuner.restart_requested.is_set())
        self.assertEqual(tuner.logger.events[-1][0], "STREAM_RESTART_REQUEST_IGNORED")
        self.assertEqual(
            tuner.logger.events[-1][1]["reason"], "camera_controls_not_online"
        )

    def test_frame_metadata_can_confirm_write_when_uvc_readback_failed(self):
        parameter = MODULE.NumericParameter(
            "laser", "Laser", 0.0, 360.0, 30.0, 0.0
        )
        parameter.set_desired(90.0)
        parameter.invalidate_applied()
        self.assertTrue(parameter.confirm_from_frame_metadata(90.0))
        self.assertEqual(parameter.applied_value(), 90.0)
        self.assertFalse(parameter.confirm_from_frame_metadata(90.0))
        parameter.set_desired(120.0)
        self.assertFalse(parameter.confirm_from_frame_metadata(90.0))

    def test_rebind_resets_unsafe_value_but_preserves_safe_value(self):
        class Range:
            min = 0.0
            max = 360.0
            step = 30.0
            default = 150.0

        class Target:
            def get_option_range(self, _option):
                return Range()

            def get_option(self, _option):
                return 90.0

        unsafe = MODULE.NumericParameter(
            "laser", "Laser", 0.0, 360.0, 30.0, 150.0, option=object()
        )
        unsafe.set_desired(240.0)
        self.assertTrue(unsafe.rebind(Target(), preserve_desired=False))
        self.assertEqual(unsafe.desired_value(), 90.0)
        self.assertEqual(unsafe.applied_value(), 90.0)

        safe = MODULE.NumericParameter(
            "frames_queue", "Queue", 0.0, 360.0, 30.0, 150.0, option=object()
        )
        safe.set_desired(240.0)
        self.assertTrue(safe.rebind(Target(), preserve_desired=True))
        self.assertEqual(safe.desired_value(), 240.0)
        self.assertEqual(safe.applied_value(), 90.0)

    def test_automatic_reenumeration_rebinds_and_only_reapplies_safe_options(self):
        queue_option = object()
        laser_option = object()

        class Range:
            def __init__(self, minimum, maximum, step, default):
                self.min = minimum
                self.max = maximum
                self.step = step
                self.default = default

        class Sensor:
            def __init__(self):
                self.values = {queue_option: 4.0, laser_option: 90.0}

            def get_option_range(self, option):
                if option is queue_option:
                    return Range(0.0, 32.0, 1.0, 16.0)
                return Range(0.0, 360.0, 30.0, 150.0)

            def get_option(self, option):
                return self.values[option]

            def set_option(self, option, value):
                self.values[option] = value

            def set_notifications_callback(self, _callback):
                pass

            def get_depth_scale(self):
                return 0.001

        sensor = Sensor()

        class Device:
            def first_depth_sensor(self):
                return sensor

            def get_info(self, key):
                if key == MODULE.rs.camera_info.serial_number:
                    return "SERIAL"
                if key == MODULE.rs.camera_info.usb_type_descriptor:
                    return "2.1"
                raise AssertionError(key)

        class Profile:
            def get_device(self):
                return Device()

        class Pipeline:
            def __init__(self):
                self.stop_calls = 0
                self.start_calls = 0

            def stop(self):
                self.stop_calls += 1

            def start(self, _config):
                self.start_calls += 1
                return Profile()

        class Logger:
            def __init__(self):
                self.kinds = []

            def event(self, kind, **_fields):
                self.kinds.append(kind)

        tuner = MODULE.DepthTuner.__new__(MODULE.DepthTuner)
        tuner.args = SimpleNamespace(
            width=480, height=270, fps=30, camera_debounce_ms=400.0
        )
        tuner.logger = Logger()
        tuner.pipeline = Pipeline()
        tuner.profile = None
        tuner.depth_sensor = None
        tuner.depth_scale = 0.001
        tuner.device_serial = "SERIAL"
        tuner.usb_type = "2.1"
        tuner.device_generation = 1
        tuner.device_lost_at = 1.0
        tuner.sensor_offset_min_ms = 1.0
        tuner.restart_requested = threading.Event()
        tuner.rebind_requested = threading.Event()
        tuner.rebind_requested.set()
        tuner.restart_in_progress = threading.Event()
        tuner.sensor_controls_ready = threading.Event()
        tuner.control_state_lock = threading.Lock()
        tuner.control_state = "RECONNECT_PENDING"
        tuner.filter_lock = threading.Lock()
        tuner.filter_enabled = {"spatial": False, "temporal": False, "hole_fill": False}
        tuner.filter_options = {}
        queue = MODULE.NumericParameter(
            "frames_queue", "Queue", 0.0, 32.0, 1.0, 16.0,
            target=sensor, option=queue_option,
        )
        queue.set_desired(2.0)
        laser = MODULE.NumericParameter(
            "laser", "Laser", 0.0, 360.0, 30.0, 150.0,
            target=sensor, option=laser_option,
        )
        laser.set_desired(240.0)
        tuner.sensor_options = {"frames_queue": queue, "laser": laser}

        tuner._restart_stream("device_reenumerated", preserve_all=False)

        self.assertEqual(queue.desired_value(), 2.0)
        self.assertEqual(queue.applied_value(), 2.0)
        self.assertEqual(laser.desired_value(), 90.0)
        self.assertEqual(laser.applied_value(), 90.0)
        self.assertEqual(tuner.device_generation, 2)
        self.assertEqual(tuner.control_status(), "ONLINE")
        self.assertTrue(tuner.sensor_controls_ready.is_set())
        self.assertFalse(tuner.rebind_requested.is_set())
        self.assertIn("DEVICE_CONTROL_REBOUND", tuner.logger.kinds)


if __name__ == "__main__":
    unittest.main()
