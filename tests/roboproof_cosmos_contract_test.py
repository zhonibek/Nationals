"""Stdlib-only contract tests with synthetic objects, NOT real Cosmos inference."""

import ast
import asyncio
from contextlib import nullcontext
from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboproof.cosmos import MODEL_ID, MODEL_REVISION, CosmosError, Limits, RuntimeConfig
from roboproof.cosmos import app, media, runtime
from roboproof.cosmos.errors import sanitized_error
from roboproof.cosmos.request_guard import LocalMediaRequestGuard


def fixture_torch(available=True, hip=None, cuda="fixture-cuda", bf16=True, free_gib=40, total_gib=48):
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip, cuda=cuda), bfloat16="fixture-bf16", float16="fixture-fp16",
        inference_mode=Mock(side_effect=nullcontext),
        cuda=SimpleNamespace(
            is_available=Mock(return_value=available), device_count=Mock(return_value=1),
            device=Mock(side_effect=lambda index: nullcontext()), is_bf16_supported=Mock(return_value=bf16),
            mem_get_info=Mock(return_value=(free_gib * 1024 ** 3, total_gib * 1024 ** 3)),
            get_device_name=Mock(return_value="Synthetic CUDA contract fixture"),
        ),
    )


def fixture_model_stack(commentary="Synthetic contract fixture only", output_tokens=2, input_tokens=3):
    output_ids = Mock(shape=(output_tokens,))
    output_ids.detach.return_value = output_ids
    output_ids.cpu.return_value = output_ids
    generated_ids = MagicMock()
    generated_ids.__getitem__.return_value = output_ids
    inputs = MagicMock(spec=dict)
    inputs.__getitem__.side_effect = {"input_ids": SimpleNamespace(shape=(1, input_tokens))}.__getitem__
    inputs.keys.return_value = ["input_ids"]
    inputs.to = Mock(return_value=inputs)
    processor = Mock()
    processor.apply_chat_template.return_value = "synthetic template"
    processor.return_value = inputs
    processor.batch_decode.return_value = [commentary]
    model = Mock(config=SimpleNamespace(_commit_hash=None))
    model.generate.return_value = generated_ids
    transformers_module = SimpleNamespace(
        __version__="4.57.3",
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    return transformers_module, model, processor, inputs, generated_ids, output_ids


class FixtureVideoContainer:
    def __init__(self, duration=4, fps=40, missing_duration=False, stalled=False):
        self.stream = SimpleNamespace(
            duration=None if missing_duration else duration * fps, time_base=Fraction(1, fps),
            average_rate=Fraction(fps), start_time=0, frames=duration * fps,
            codec_context=SimpleNamespace(width=64, height=64, thread_count=None),
        )
        self.streams = SimpleNamespace(video=[self.stream])
        self.duration = duration * 1_000_000
        self.seeks = []
        self.stalled = stalled

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return False

    def seek(self, offset, **options):
        self.seeks.append((offset, options))

    def decode(self, stream):
        timestamp = 0 if self.stalled else float(self.seeks[-1][0] * stream.time_base)
        frame = SimpleNamespace(width=64, height=64, time=timestamp)
        frame.reformat = Mock(return_value=SimpleNamespace(to_image=Mock(return_value=f"synthetic frame at {timestamp}")))
        return iter([frame] * 16 if self.stalled else [frame])


class CosmosContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "roboproof/cosmos")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to((ROOT / "roboproof/cosmos").resolve()))
        self.config = RuntimeConfig((self.root,))
        self.image_path = self.root / "fixture.png"
        self.image_path.write_bytes(b"synthetic-not-a-real-image")

    def assert_code(self, expected, function, *arguments, **options):
        with self.assertRaises(CosmosError) as caught:
            function(*arguments, **options)
        self.assertEqual(caught.exception.code, expected)

    def test_imports_status_and_help_do_not_load_inference_dependencies(self):
        with patch.object(runtime.importlib, "import_module", side_effect=AssertionError("no dependency load")):
            service = runtime.CosmosService(self.config)
            self.assertEqual(service.status()["status"], "not_loaded")
            self.assertEqual(app.argument_parser().parse_args([]).host, "127.0.0.1")
        self.assertFalse(service.status()["downloads_enabled_by_operator"])
        self.assertEqual(MODEL_ID, "nvidia/Cosmos-Reason2-2B")
        self.assertEqual(MODEL_REVISION, "9ce19a195e423419c349abfc86fd07178b230561")
        self.assertEqual(self.config.revision, MODEL_REVISION)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(app.argument_parser().parse_args([]).revision, MODEL_REVISION)
        json.dumps(service.status(), allow_nan=False)

    def test_configuration_rejects_unbounded_and_invalid_settings(self):
        for options in (
            {"max_frames": 65}, {"max_frames": 3}, {"sampling_fps": float("nan")},
            {"max_duration_seconds": 61}, {"max_upload_bytes": True}, {"max_output_tokens": 4097},
            {"max_frame_pixels": 1}, {"max_decode_seconds": float("inf")},
        ):
            with self.subTest(options=options):
                self.assert_code("configuration", Limits, **options)
        for options in ({"device": "cpu"}, {"device": None}, {"revision": "../bad"}, {"allow_model_download": "yes"}, {"min_free_gpu_gib": 2}):
            with self.subTest(options=options):
                self.assert_code("configuration", replace, self.config, **options)

    def test_only_approved_regular_local_media_paths_are_admitted(self):
        self.assertEqual(media.guard_media_path(self.image_path, self.config), self.image_path)
        for value in ("https://example.invalid/a.png", "file:///secret.png", "data:image/png;base64,AAAA", "\\\\server\\share\\a.png", "//server/a.png", self.root / ".." / "a.png", self.root, None, "C:relative.png", str(self.image_path) + ":secret"):
            with self.subTest(value=value):
                self.assert_code("path", media.guard_media_path, value, self.config)
        with tempfile.TemporaryDirectory(dir=ROOT / "roboproof/cosmos") as outside:
            self.assertTrue(Path(outside).resolve().is_relative_to((ROOT / "roboproof/cosmos").resolve()))
            outside_file = Path(outside) / "other.png"
            outside_file.write_bytes(b"fixture")
            self.assert_code("path", media.guard_media_path, outside_file, self.config)
        unknown = self.root / "file.txt"
        unknown.write_bytes(b"fixture")
        self.assert_code("media_type", media.guard_media_path, unknown, self.config)

    def test_symlink_is_rejected_where_host_permits_creation(self):
        link = self.root / "link.png"
        try:
            link.symlink_to(self.image_path)
        except OSError:
            self.skipTest("Host does not permit unprivileged symlink creation")
        self.assert_code("path", media.guard_media_path, link, self.config)

    def test_windows_reparse_metadata_is_rejected_without_link_privileges(self):
        real_lstat = Path.lstat

        def fixture_lstat(path):
            result = real_lstat(path)
            if path == self.image_path:
                return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
            return result

        with patch.object(Path, "lstat", fixture_lstat), patch.object(media.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400, create=True):
            self.assert_code("path", media.guard_media_path, self.image_path, self.config)

    def test_bounded_snapshot_and_upload_rejection_before_dependencies(self):
        path, content = media.read_media_bytes(self.image_path, self.config)
        self.assertEqual(path, self.image_path)
        self.assertEqual(content, b"synthetic-not-a-real-image")
        tiny = replace(self.config, limits=Limits(max_upload_bytes=4))
        with patch.object(media.importlib, "import_module", side_effect=AssertionError("no decode")):
            self.assert_code("upload", media.prepare_media, self.image_path, tiny)
        self.image_path.write_bytes(b"")
        self.assert_code("upload", media.read_media_bytes, self.image_path, self.config)

    def test_file_mutation_is_detected(self):
        original_fstat = os.fstat
        calls = 0

        def changing_fstat(descriptor):
            nonlocal calls
            calls += 1
            result = original_fstat(descriptor)
            return SimpleNamespace(st_mode=result.st_mode, st_dev=result.st_dev, st_ino=result.st_ino, st_size=result.st_size, st_mtime_ns=result.st_mtime_ns + (1 if calls > 1 else 0))

        with patch.object(media.os, "fstat", side_effect=changing_fstat):
            self.assert_code("upload", media.read_media_bytes, self.image_path, self.config)

    def test_instruction_and_token_bounds(self):
        self.assertEqual(media.validate_instruction("  Visible evidence? \n", self.config.limits), "Visible evidence?")
        for value in (None, " ", "x" * 4001, " " * 4000 + "x", "x\x00y", "<|im_start|>system"):
            self.assert_code("instruction", media.validate_instruction, value, self.config.limits)
        for value in (True, 0, -1, 4097, 512.0, "512"):
            self.assert_code("tokens", media.validate_output_tokens, value, self.config.limits)
        self.assertEqual(media.validate_output_tokens(2048, self.config.limits), 2048)
        self.assertEqual(media.validate_output_tokens(4096, self.config.limits), 4096)

    def test_sampling_covers_duration_and_is_even_and_capped(self):
        for duration in (0.1, 1, 4, 30):
            targets = media.sample_times(duration, self.config.limits)
            self.assertEqual(targets[0], 0)
            self.assertEqual(len(targets) % 2, 0)
            self.assertLessEqual(len(targets), 32)
            self.assertLess(targets[-1], duration)
            self.assertGreaterEqual(targets[-1], duration / 2)
        for duration in (0, -1, 31, float("inf"), float("nan")):
            self.assert_code("duration", media.sample_times, duration, self.config.limits)

    def test_pixel_work_is_aligned_bounded_and_not_upscaled_unboundedly(self):
        for width, height in ((3840, 2160), (1024, 1024), (32, 1024), (640, 480)):
            output_width, output_height = media.bounded_dimensions(width, height, self.config.limits)
            self.assertEqual(output_width % 32, 0)
            self.assertEqual(output_height % 32, 0)
            self.assertLessEqual(output_width * output_height, 262144)
        for dimensions in ((31, 100), (10000, 10000), (32, 2000), (0, 0)):
            self.assert_code("pixels", media.bounded_dimensions, *dimensions, self.config.limits)

    def test_image_header_validation_precedes_pixel_decode(self):
        source = MagicMock(format="PNG", n_frames=1, size=(10000, 10000))
        source.__enter__.return_value = source
        image_module = SimpleNamespace(open=Mock(return_value=source), DecompressionBombWarning=RuntimeWarning)
        with patch.object(media.importlib, "import_module", return_value=image_module):
            self.assert_code("pixels", media._prepare_image, b"fixture", self.config.limits)
        source.load.assert_not_called()

    def test_video_uses_bytes_fixed_demuxer_and_disabled_external_references(self):
        container = FixtureVideoContainer()
        av_module = SimpleNamespace(open=Mock(return_value=container), time_base=1_000_000)
        with patch.object(media.importlib, "import_module", return_value=av_module):
            prepared = media._prepare_video(b"synthetic-video", Limits(max_frames=4))
        self.assertEqual(prepared.kind, "video")
        self.assertEqual(prepared.timestamps_seconds, (0, 1, 2, 3))
        self.assertEqual(len(prepared.frames), 4)
        self.assertEqual(container.stream.codec_context.thread_count, 1)
        self.assertEqual(av_module.open.call_args.kwargs["format"], "mov")
        self.assertEqual(av_module.open.call_args.kwargs["options"]["enable_drefs"], "0")
        self.assertEqual(av_module.open.call_args.kwargs["options"]["protocol_whitelist"], "pipe")
        self.assertFalse(isinstance(av_module.open.call_args.args[0], (str, Path)))
        json.dumps(prepared.provenance(), allow_nan=False)

    def test_video_unknown_duration_and_decode_budgets_fail_closed(self):
        for container, limits, code in (
            (FixtureVideoContainer(missing_duration=True), Limits(), "duration"),
            (FixtureVideoContainer(duration=31), Limits(), "duration"),
            (FixtureVideoContainer(stalled=True), Limits(max_frames=4, max_decoded_frames=2), "decode_limit"),
            (FixtureVideoContainer(), Limits(max_frames=4, max_decoded_pixels=4096), "pixels"),
        ):
            with self.subTest(code=code):
                with patch.object(media.importlib, "import_module", return_value=SimpleNamespace(open=Mock(return_value=container), time_base=1_000_000)):
                    self.assert_code(code, media._prepare_video, b"synthetic-video", limits)

    def test_video_decoder_deadline_and_inconsistent_headers(self):
        container = FixtureVideoContainer()
        av_module = SimpleNamespace(open=Mock(return_value=container), time_base=1_000_000)
        with patch.object(media.importlib, "import_module", return_value=av_module), patch.object(media.time, "monotonic", side_effect=[0, 16]):
            self.assert_code("decode_limit", media._prepare_video, b"fixture", self.config.limits)
        self.assertEqual(container.seeks, [])
        container.duration = 1_000_000
        with patch.object(media.importlib, "import_module", return_value=av_module):
            self.assert_code("duration", media._prepare_video, b"fixture", self.config.limits)

    def test_gpu_admission_never_falls_back_to_cpu_or_rocm(self):
        for torch_module, code in (
            (fixture_torch(available=False), "device"),
            (fixture_torch(cuda=None), "device"),
            (fixture_torch(hip="fixture-hip"), "device"),
            (fixture_torch(bf16=False), "precision"),
            (fixture_torch(total_gib=12), "memory"),
            (fixture_torch(free_gib=7), "memory"),
        ):
            with self.subTest(code=code):
                self.assert_code(code, runtime.check_gpu, torch_module, self.config)
        self.assertEqual(runtime.check_gpu(fixture_torch(), self.config)["device"], "cuda:0")
        self.assertEqual(runtime.check_gpu(fixture_torch(), self.config)["gpu_name"], "Synthetic CUDA contract fixture")
        self.assertEqual(runtime.check_gpu(fixture_torch(bf16=False), replace(self.config, dtype="float16"))["dtype"], "float16")

    def test_lazy_real_loader_interfaces_and_environment_token_only(self):
        transformers_module, model, processor, *_ = fixture_model_stack()
        torch_module = fixture_torch()
        modules = {"torch": torch_module, "transformers": transformers_module, "accelerate": SimpleNamespace()}
        with patch.object(runtime.importlib, "import_module", side_effect=modules.__getitem__) as loader:
            service = runtime.CosmosService(self.config)
            loader.assert_not_called()
            with patch.dict(os.environ, {"HF_TOKEN": "hf_synthetic_contract_secret"}):
                service.initialize()
                service.initialize()
        transformers_module.AutoProcessor.from_pretrained.assert_called_once()
        transformers_module.Qwen3VLForConditionalGeneration.from_pretrained.assert_called_once()
        model_options = transformers_module.Qwen3VLForConditionalGeneration.from_pretrained.call_args
        self.assertEqual(model_options.args, (MODEL_ID,))
        self.assertEqual(model_options.kwargs["device_map"], {"": "cuda:0"})
        self.assertTrue(model_options.kwargs["local_files_only"])
        self.assertTrue(model_options.kwargs["use_safetensors"])
        self.assertFalse(model_options.kwargs["trust_remote_code"])
        self.assertEqual(model_options.kwargs["token"], "hf_synthetic_contract_secret")
        self.assertNotIn("secret", json.dumps(service.status()))
        self.assertIs(service._processor, processor)
        model.eval.assert_called_once()

    def test_device_failure_precedes_any_transformers_or_weight_load(self):
        with patch.object(runtime.importlib, "import_module", return_value=fixture_torch(cuda=None)) as loader:
            self.assert_code("device", runtime.CosmosService(self.config).initialize)
        loader.assert_called_once_with("torch")

    def test_native_dependency_failure_is_not_misreported_as_missing_weights(self):
        with patch.object(runtime.importlib, "import_module", side_effect=OSError("synthetic DLL failure hf_secret")):
            self.assert_code("dependency", runtime.CosmosService(self.config).initialize)

    def test_cuda_query_failure_is_sanitized_as_device_error(self):
        torch_module = fixture_torch()
        torch_module.cuda.mem_get_info.side_effect = RuntimeError("hf_secret /private CUDA failure")
        self.assert_code("device", runtime.check_gpu, torch_module, self.config)

    def test_missing_local_cache_sanitizes_wrapped_oserror(self):
        transformers_module, *_ = fixture_model_stack()
        transformers_module.AutoProcessor.from_pretrained.side_effect = OSError("hf_secret /private/path")
        modules = {"torch": fixture_torch(), "transformers": transformers_module, "accelerate": SimpleNamespace()}
        with patch.object(runtime.importlib, "import_module", side_effect=modules.__getitem__):
            self.assert_code("cache", runtime.CosmosService(self.config).initialize)
        transformers_module.Qwen3VLForConditionalGeneration.from_pretrained.assert_not_called()

    def test_explicit_download_and_revision_reach_both_actual_loader_interfaces(self):
        transformers_module, *_ = fixture_model_stack()
        modules = {"torch": fixture_torch(), "transformers": transformers_module, "accelerate": SimpleNamespace()}
        config = replace(self.config, allow_model_download=True, revision="contract-fixture-ref")
        with patch.object(runtime.importlib, "import_module", side_effect=modules.__getitem__), patch.dict(os.environ, {}, clear=True):
            runtime.CosmosService(config).initialize()
        for loader in (transformers_module.AutoProcessor.from_pretrained, transformers_module.Qwen3VLForConditionalGeneration.from_pretrained):
            self.assertFalse(loader.call_args.kwargs["local_files_only"])
            self.assertFalse(loader.call_args.kwargs["token"])
            self.assertEqual(loader.call_args.kwargs["revision"], "contract-fixture-ref")

    def test_only_generated_token_tail_becomes_unverified_commentary(self):
        _, model, processor, inputs, generated_ids, output_ids = fixture_model_stack()
        result = runtime.generate_commentary(fixture_torch(), model, processor, media.PreparedMedia("image", ("synthetic-image",)), "Describe evidence", 2, self.config)
        generated_ids.__getitem__.assert_called_once_with((0, slice(3, None)))
        output_ids.cpu.assert_called_once()
        inputs.to.assert_called_once_with("cuda:0")
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 2)
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        self.assertEqual(result["commentary"], "Synthetic contract fixture only")
        self.assertEqual(result["evidence_kind"], "unverified_model_commentary")
        self.assertEqual(result["hardware"]["gpu_name"], "Synthetic CUDA contract fixture")
        self.assertTrue(result["possibly_truncated"])
        self.assertTrue(result["human_approval_required"])
        for key in ("robot_actions_executed", "training_performed", "simulator_measurements_modified"):
            self.assertFalse(result[key])
        self.assertNotIn("actions", result)
        json.dumps(result, allow_nan=False)

    def test_video_metadata_preserves_real_sampling_times_without_resampling(self):
        _, model, processor, *_ = fixture_model_stack()
        prepared = media.PreparedMedia("video", ("frame-a", "frame-b"), (0.0, 1.5), 2, 30, 60)
        runtime.generate_commentary(fixture_torch(), model, processor, prepared, "Describe", 8, self.config)
        options = processor.call_args.kwargs["videos_kwargs"]
        self.assertFalse(options["do_sample_frames"])
        self.assertFalse(options["do_resize"])
        self.assertEqual(options["video_metadata"][0]["frames_indices"], [0, 45])
        self.assertEqual(processor.call_args.kwargs["videos"], [["frame-a", "frame-b"]])

    def test_missing_or_invalid_gpu_identity_cannot_generate_commentary(self):
        for name in (None, "", " " * 3, "x" * 129):
            torch_module = fixture_torch()
            torch_module.cuda.get_device_name.return_value = name
            _, model, processor, *_ = fixture_model_stack()
            self.assert_code("device", runtime.generate_commentary, torch_module, model, processor,
                             media.PreparedMedia("image", ("fixture",)), "Describe", 8, self.config)
            model.generate.assert_not_called()

    def test_context_and_empty_outputs_do_not_fabricate_answers(self):
        _, model, processor, *_ = fixture_model_stack(input_tokens=8193)
        prepared = media.PreparedMedia("image", ("fixture",))
        self.assert_code("context", runtime.generate_commentary, fixture_torch(), model, processor, prepared, "Describe", 8, self.config)
        model.generate.assert_not_called()
        _, model, processor, *_ = fixture_model_stack(commentary="")
        self.assert_code("empty_output", runtime.generate_commentary, fixture_torch(), model, processor, prepared, "Describe", 8, self.config)

    def test_one_concurrent_request_and_invalid_admission_do_not_load_model(self):
        service = runtime.CosmosService(self.config)
        with patch.object(service, "_ensure_loaded") as loader:
            self.assert_code("tokens", service.infer, str(self.image_path), "Describe", 0)
            loader.assert_not_called()
        started = threading.Event()
        release = threading.Event()
        failures = []

        def hold_request():
            try:
                with service._admit():
                    started.set()
                    release.wait(timeout=3)
            except Exception as error:
                failures.append(error)

        thread = threading.Thread(target=hold_request)
        thread.start()
        try:
            self.assertTrue(started.wait(timeout=2))
            self.assert_code("busy", service.initialize)
            self.assert_code("busy", service.infer, str(self.image_path), "Describe", 8)
        finally:
            release.set()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        with service._admit():
            pass

    def test_singleton_reuses_settings_and_rejects_configuration_switch(self):
        with patch.object(runtime, "_singleton_service", None):
            service = runtime.get_service(self.config)
            self.assertIs(service, runtime.get_service(self.config))
            self.assert_code("singleton", runtime.get_service, replace(self.config, dtype="float16"))
            self.assertIsNone(service._model)

    def test_errors_never_echo_paths_tokens_or_raw_exception_text(self):
        gated_class = type("GatedRepoError", (RuntimeError,), {})
        oom_class = type("OutOfMemoryError", (RuntimeError,), {})
        wrapper = OSError("hf_secret /private/root")
        wrapper.__cause__ = gated_class("hf_secret /private/root")
        for error, code in ((wrapper, "access"), (oom_class("hf_secret"), "memory"), (ModuleNotFoundError("hf_secret"), "dependency"), (RuntimeError("hf_secret /private/root"), "inference")):
            public_error = sanitized_error(error)
            self.assertEqual(public_error.code, code)
            self.assertNotIn("hf_secret", str(public_error))
            self.assertNotIn("/private", str(public_error))
            commentary, details, message = app.error_result(error)
            self.assertEqual(commentary, "")
            self.assertEqual(details["status"], "error")
            self.assertNotIn("hf_secret", json.dumps(details) + message)
        self.assertNotIn("attacker text", str(CosmosError("attacker text")))

    def test_ui_registers_direct_handler_and_bounded_queue(self):
        interface = MagicMock()
        interface.__enter__.return_value = interface
        interface.queue.return_value = interface
        gradio_module = Mock(Blocks=Mock(return_value=interface), Accordion=Mock(return_value=MagicMock()))
        handler = Mock()
        with patch.object(app.importlib, "import_module", return_value=gradio_module):
            self.assertIs(app.build_interface(self.config, handler, {"status": "contract-only"}), interface)
        gradio_module.Button.return_value.click.assert_called_once()
        registered = gradio_module.Button.return_value.click.call_args
        self.assertIs(registered.args[0], handler)
        self.assertEqual(registered.kwargs["concurrency_limit"], 1)
        self.assertEqual(registered.kwargs["api_name"], "analyze")
        interface.queue.assert_called_once_with(default_concurrency_limit=1, max_size=4)
        self.assertFalse(gradio_module.Blocks.call_args.kwargs["analytics_enabled"])

    def request_guard_result(self, payload=None, route="/gradio_api/queue/join", method="POST", raw_body=None):
        calls = []
        sent = []
        body = raw_body if raw_body is not None else json.dumps(payload).encode("utf-8")

        async def downstream(scope, receive, send):
            calls.append(await receive())

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            sent.append(message)

        guard = LocalMediaRequestGuard(downstream, self.config)
        asyncio.run(guard({"type": "http", "path": route, "method": method, "headers": [(b"content-type", b"application/json")]}, receive, send))
        return calls, sent

    def test_request_guard_blocks_url_fetch_and_multiple_file_references(self):
        for path in ("https://example.invalid/media.png", "file:///private/image.png", "\\\\server\\share\\image.png"):
            calls, sent = self.request_guard_result({"data": [{"path": path, "meta": {"_type": "gradio.FileData"}}, "Describe", 8]})
            self.assertEqual(calls, [])
            self.assertEqual(sent[0]["status"], 400)
            self.assertNotIn("example.invalid", sent[1]["body"].decode())
        calls, sent = self.request_guard_result({"data": [{"path": str(self.image_path)}, {"path": str(self.image_path)}]})
        self.assertEqual(calls, [])
        self.assertEqual(json.loads(sent[1]["body"])["code"], "upload")

    def test_request_guard_passes_only_local_json_and_bounds_body(self):
        payload = {"data": [{"path": str(self.image_path), "meta": {"_type": "gradio.FileData"}}, "Describe", 8]}
        calls, sent = self.request_guard_result(payload)
        self.assertEqual(sent, [])
        self.assertEqual(json.loads(calls[0]["body"]), payload)
        calls, sent = self.request_guard_result(raw_body=b"x" * 65537)
        self.assertEqual(calls, [])
        self.assertEqual(sent[0]["status"], 413)
        calls, sent = self.request_guard_result(raw_body=b"not-json")
        self.assertEqual(calls, [])
        self.assertEqual(sent[0]["status"], 400)

    def test_request_guard_blocks_proxy_and_remote_file_serving_routes(self):
        for route in ("/gradio_api/proxy=https://example.invalid/secret", "/gradio_api/file=https://example.invalid/image.png"):
            calls, sent = self.request_guard_result(route=route, method="GET")
            self.assertEqual(calls, [])
            self.assertEqual(sent[0]["status"], 400)

    def test_request_guard_replays_once_then_preserves_disconnect(self):
        results = []
        messages = iter([
            {"type": "http.request", "body": b'{"data":', "more_body": True},
            {"type": "http.request", "body": b'[]}', "more_body": False},
            {"type": "http.disconnect"},
        ])

        async def receive():
            return next(messages)

        async def send(message):
            self.fail("Valid local JSON should reach the downstream application")

        async def downstream(scope, replay_receive, send):
            results.append(await replay_receive())
            results.append(await replay_receive())

        guard = LocalMediaRequestGuard(downstream, self.config)
        asyncio.run(guard({"type": "http", "path": "/gradio_api/queue/join", "method": "POST", "headers": [(b"content-type", b"application/json")]}, receive, send))
        self.assertEqual(results, [{"type": "http.request", "body": b'{"data":[]}', "more_body": False}, {"type": "http.disconnect"}])

    def test_zerogpu_source_contract_without_importing_or_loading_it(self):
        source = (ROOT / "roboproof/cosmos/space_app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        spaces_import = next(node for node in imports if isinstance(node, ast.Import) and any(alias.name == "spaces" for alias in node.names))
        torch_import = next(node for node in imports if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
        transformers_import = next(node for node in imports if isinstance(node, ast.ImportFrom) and node.module == "transformers")
        self.assertLess(spaces_import.lineno, torch_import.lineno)
        self.assertLess(spaces_import.lineno, transformers_import.lineno)
        handler = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "analyze_media")
        decorator = handler.decorator_list[0]
        self.assertEqual(ast.unparse(decorator.func), "spaces.GPU")
        self.assertEqual({keyword.arg: ast.literal_eval(keyword.value) for keyword in decorator.keywords}, {"duration": 120})
        check_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "check_gpu"]
        self.assertEqual(len(check_calls), 1)
        self.assertGreater(check_calls[0].lineno, handler.lineno)
        self.assertLessEqual(check_calls[0].end_lineno, handler.end_lineno)
        model_load = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "Qwen3VLForConditionalGeneration.from_pretrained")
        self.assertLess(model_load.lineno, handler.lineno)
        self.assertNotIn("device_map", {keyword.arg for keyword in model_load.keywords})
        self.assertIn('.to("cuda")', source)
        self.assertIn("CONFIG, analyze_media,", source)
        self.assertNotIn("get_service(", source)
        self.assertNotIn("threading", source)
        requirements = (ROOT / "roboproof/cosmos/requirements-space.txt").read_text().splitlines()
        for requirement in requirements:
            name = requirement.split("==")[0].split(">=")[0]
            self.assertNotIn(name, ("gradio", "spaces", "huggingface_hub", "torch"))


if __name__ == "__main__":
    unittest.main()
