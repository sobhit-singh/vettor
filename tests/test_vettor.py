import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from config import QAConfig
from cache import RecipeCache
from agent import QAAgent
from engine import QAEngine
from assertions import QAAssertions
from mockinsurancesite import start_server_in_thread
from memory import QAMemory
from handoff import HandoffManager, stream_test_result_jsonl
import llm_manager
import reporting


class TestVettor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd, cls.server_thread = start_server_in_thread(port=0)
        cls.server_port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_config_serialization(self):
        cfg = QAConfig()
        cfg.target.url = "http://example.com/checkout"
        cfg.model.backend = "openai"
        cfg.model.model_name = "gpt-4o-mini"
        cfg.browser.engine = "playwright"

        dump_path = "test_config_dump.json"
        cfg.save(dump_path)
        self.assertTrue(Path(dump_path).exists())

        loaded = QAConfig.from_file(dump_path)
        self.assertEqual(loaded.start_url, "http://example.com/checkout")
        self.assertEqual(loaded.model.backend, "openai")
        self.assertEqual(loaded.model.model_name, "gpt-4o-mini")
        self.assertEqual(loaded.browser.engine, "playwright")
        Path(dump_path).unlink()

    def test_assertions_engine(self):
        # Numeric match
        res_num = QAAssertions.evaluate_case_assertions(
            row={"expected_premium": "450.00"},
            page_state={"bodyText": "Your quote is $450.00"},
            extracted={"premium": 450.00},
        )
        self.assertTrue(res_num[0].passed)

        # Numeric mismatch
        res_num_fail = QAAssertions.evaluate_case_assertions(
            row={"expected_premium": "500.00"},
            page_state={"bodyText": "Your quote is $450.00"},
            extracted={"premium": 450.00},
        )
        self.assertFalse(res_num_fail[0].passed)

        # Text containment
        res_text = QAAssertions.evaluate_case_assertions(
            row={"expected_text": "Confirmation Success"},
            page_state={"bodyText": "Order Confirmation Success! Thank you for your purchase."},
            extracted={},
        )
        self.assertTrue(res_text[0].passed)

        # Dynamic domain-agnostic expected_* assertion
        res_dynamic = QAAssertions.evaluate_case_assertions(
            row={"expected_order_status": "Shipped", "expected_balance": "129.99"},
            page_state={"bodyText": "Your order has been Shipped. Remaining balance is $129.99."},
            extracted={"order_status": "Shipped", "balance": 129.99},
        )
        self.assertEqual(len(res_dynamic), 2)
        self.assertTrue(res_dynamic[0].passed)
        self.assertTrue(res_dynamic[1].passed)

    def test_fixtures_match_mock_site_pricing(self):
        def calculate(age, coverage, smoker):
            return round(
                (coverage / 1000) * 0.5 + age * 4 + (300 if smoker == "yes" else 0) - 500 * 0.05,
                2,
            )

        with open("sample_cases.csv", newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        jsonl_rows = [
            json.loads(line)
            for line in Path("test_cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(len(csv_rows), 3)
        self.assertEqual(len(jsonl_rows), 3)
        for csv_row, jsonl_row in zip(csv_rows, jsonl_rows):
            actual = calculate(
                int(csv_row["age"]),
                int(csv_row["coverage_amount"]),
                csv_row["smoker"],
            )
            self.assertAlmostEqual(float(csv_row["expected_premium"]), actual, places=2)
            self.assertAlmostEqual(float(jsonl_row["expected_premium"]), actual, places=2)

    def test_html_report_generation(self):
        report_path = "test_report.html"
        sample_data = {
            "stats": {"total": 2, "passed": 2, "failed": 0, "errors": 0, "heals": 1, "duration_s": 4.5},
            "config": {"target": {"url": "http://localhost:8000/step1"}},
            "results": [
                {"case_id": "1", "status": "PASS", "actual": 345.0, "expected": 345.0, "duration_s": 2.1, "note": "OK"},
                {"case_id": "2", "status": "PASS", "actual": 955.0, "expected": 955.0, "duration_s": 2.4, "note": "OK"},
            ],
        }
        out = reporting.generate_html_report(sample_data, report_path)
        self.assertTrue(Path(out).exists())
        self.assertIn("Vettor: Web QA Test Execution Report", Path(out).read_text(encoding="utf-8"))
        Path(report_path).unlink()

    def test_recipe_cache(self):
        cache_path = "test_cache.json"
        cache = RecipeCache(cache_path)
        cache.clear()
        self.assertIsNone(cache.get("/step1"))

        cache.set("/step1", {"test": 123})
        self.assertEqual(cache.get("/step1"), {"test": 123})

        cache.clear()
        self.assertIsNone(cache.get("/step1"))
        if Path(cache_path).exists():
            Path(cache_path).unlink()

    @unittest.skipUnless(
        llm_manager.find_llama_executable() and llm_manager.find_model_gguf(),
        "requires a local llama.cpp binary and GGUF model",
    )
    def test_llm_manager_discovery(self):
        llama_bin = llm_manager.find_llama_executable()
        self.assertIsNotNone(llama_bin, "llama.exe or llama-server should be discoverable")
        self.assertTrue(llama_bin is not None and Path(llama_bin).exists())

        model_path = llm_manager.find_model_gguf()
        self.assertIsNotNone(model_path, "LFM3-1.7B GGUF model should be discoverable in HuggingFace cache")
        self.assertTrue(model_path is not None and Path(model_path).exists())

    def test_agent_with_lfm3_model_or_fallback(self):
        cfg = QAConfig(auto_start_llama=False)
        agent = QAAgent(cfg)

        payload = {
            "csv_columns": ["full_name", "age", "smoker", "coverage_amount"],
            "fields": [
                {"tag": "input", "id": "full_name", "selector": "#full_name", "type": "text", "label": "Full Name", "required": True},
                {"tag": "input", "id": "age", "selector": "#age", "type": "number", "label": "Age", "required": True},
                {"tag": "button", "id": "btn-next", "selector": "#btn-next", "label": "Next Step", "type": "submit"},
            ],
        }
        res = agent.call_tool("map_page_schema", payload)
        self.assertIn("field_map", res)
        self.assertIn("next_button", res)
        self.assertFalse(res["is_final_page"])

        # Test extract_final_result
        extract_payload = {
            "bodyText": "Your Insurance Quote Summary Applicant: Aarav Sharma Quote Reference: Q-7A8B9C Annual Premium: $450.00"
        }
        res_extract = agent.call_tool("extract_final_result", extract_payload)
        self.assertEqual(res_extract["premium"], 450.00)
        self.assertEqual(res_extract["quote_id"], "Q-7A8B9C")

    def test_agent_fallback_tools(self):
        cfg = QAConfig(auto_start_llama=False)
        agent = QAAgent(cfg)

        # map_page_schema fallback
        schema = agent._heuristic_fallback(
            "map_page_schema",
            {
                "csv_columns": ["full_name", "age"],
                "fields": [
                    {"tag": "input", "id": "full_name", "selector": "#full_name", "type": "text"},
                    {"tag": "input", "id": "age", "selector": "#age", "type": "number"},
                    {"tag": "button", "id": "btn-next", "selector": "#btn-next", "label": "Next Step"},
                ],
            },
        )
        self.assertFalse(schema["is_final_page"])
        self.assertEqual(schema["next_button"], "#btn-next")
        self.assertEqual(len(schema["field_map"]), 2)

        # resolve_delta_fields fallback
        delta = agent._heuristic_fallback(
            "resolve_delta_fields",
            {
                "row": {"full_name": "John Doe", "age": "30"},
                "fields": [
                    {"tag": "input", "id": "cigs_per_day", "selector": "#cigs_per_day", "type": "number"},
                ],
            },
        )
        self.assertEqual(len(delta["actions"]), 1)
        self.assertEqual(delta["actions"][0]["selector"], "#cigs_per_day")

        # triage_page_blocker fallback
        triage = agent._heuristic_fallback(
            "triage_page_blocker",
            {"errors": ["You must accept the terms to calculate a quote."], "fields": []},
        )
        self.assertEqual(triage["status"], "fix_format")
        self.assertEqual(triage["action"], "check")
        self.assertEqual(triage["fix_selector"], "#agree_terms")

        # extract_final_result fallback
        res = agent._heuristic_fallback(
            "extract_final_result",
            {"bodyText": "Your Quote Summary: Quote Reference: Q-789XYZ Annual Premium: $345.00"},
        )
        self.assertEqual(res["premium"], 345.00)
        self.assertEqual(res["quote_id"], "Q-789XYZ")

        # extract_final_result generic domain-agnostic fallback
        res_generic = agent._heuristic_fallback(
            "extract_final_result",
            {
                "bodyText": "Order Complete! Confirmation Number: ORD-99882 Total Paid: $1,450.00",
                "target_keys": ["total", "order_id"],
            },
        )
        self.assertEqual(res_generic["total"], 1450.00)
        self.assertEqual(res_generic["order_id"], "ORD-99882")

    @unittest.skipUnless(os.environ.get("QA_RUN_E2E") == "1", "set QA_RUN_E2E=1 to run browser end-to-end tests")
    def test_engine_e2e_playwright(self):
        recipe_path = "test_playwright_recipe.json"
        results_path = "test_playwright_results.csv"
        for p in (recipe_path, results_path):
            if Path(p).exists():
                Path(p).unlink()

        cfg = QAConfig(
            engine="playwright",
            start_url=f"http://127.0.0.1:{self.server_port}/step1",
            recipe_path=recipe_path,
            results_path=results_path,
            headless=True,
        )

        logs = []
        def handle_ev(ev):
            logs.append(ev)

        engine = QAEngine(cfg, on_event=handle_ev)
        stats = engine.run_all()

        self.assertGreater(stats.completed, 0)
        self.assertEqual(stats.errors, 0, f"Encountered unexpected errors: {logs}")
        self.assertTrue(Path(results_path).exists())

        for p in (recipe_path, results_path):
            if Path(p).exists():
                Path(p).unlink()

    @unittest.skipUnless(os.environ.get("QA_RUN_E2E") == "1", "set QA_RUN_E2E=1 to run browser end-to-end tests")
    def test_engine_e2e_rustwright(self):
        recipe_path = "test_rustwright_recipe.json"
        results_path = "test_rustwright_results.csv"
        for p in (recipe_path, results_path):
            if Path(p).exists():
                Path(p).unlink()

        cfg = QAConfig(
            engine="rustwright",
            start_url=f"http://127.0.0.1:{self.server_port}/step1",
            recipe_path=recipe_path,
            results_path=results_path,
            headless=True,
        )

        logs = []
        def handle_ev(ev):
            logs.append(ev)

        engine = QAEngine(cfg, on_event=handle_ev)
        stats = engine.run_all()

        self.assertGreater(stats.completed, 0)
        self.assertEqual(stats.errors, 0, f"Encountered unexpected errors: {logs}")
        self.assertTrue(Path(results_path).exists())

        for p in (recipe_path, results_path):
            if Path(p).exists():
                Path(p).unlink()

    def test_qa_memory_lifecycle(self):
        test_mem_file = Path("test_qa_memory.json")
        if test_mem_file.exists():
            test_mem_file.unlink()

        mem = QAMemory(test_mem_file)
        self.assertEqual(len(mem.records), 0)

        # 1. Record a fact
        f_rec = mem.record_fact(
            target_url="http://localhost:8000/step1",
            summary="Personal details step inputs",
            data={"field_map": [{"selector": "#full_name", "action": "fill"}], "next_button": "#btn-next"},
        )
        self.assertEqual(f_rec.kind, "fact")
        self.assertEqual(f_rec.trust, 1.0)

        # 2. Record a caution
        c_rec = mem.record_caution(
            target_url="http://localhost:8000/step7",
            summary="Requires terms checkbox",
            workaround={"fix_selector": "#agree_terms", "action": "check", "fix_value": "yes"},
        )
        self.assertEqual(c_rec.kind, "caution")

        # 3. Lookup
        found = mem.lookup("http://localhost:8000/step7", kind="caution")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].data["fix_selector"], "#agree_terms")

        # 4. Confirm & Contradict trust calibration
        mem.confirm(c_rec.id)
        self.assertEqual(c_rec.confirmed, 2)
        self.assertEqual(c_rec.trust, 1.0)

        mem.contradict(c_rec.id)
        self.assertEqual(c_rec.contradicted, 1)
        self.assertAlmostEqual(c_rec.trust, 0.67, delta=0.01)

        # 5. Persistence reload
        mem2 = QAMemory(test_mem_file)
        self.assertEqual(len(mem2.records), 2)
        self.assertIn(f_rec.id, mem2.records)

        # 6. Forget & clear
        self.assertTrue(mem2.forget(f_rec.id))
        self.assertEqual(len(mem2.records), 1)
        mem2.clear()
        self.assertEqual(len(mem2.records), 0)

        if test_mem_file.exists():
            test_mem_file.unlink()

    def test_handoff_and_jsonl_streaming(self):
        test_handoff_file = Path("test_DEVELOPMENT_HANDOFF.md")
        test_out_dir = Path("test_artifacts")
        if test_handoff_file.exists():
            test_handoff_file.unlink()
        if test_out_dir.exists():
            for f in test_out_dir.glob("*"):
                f.unlink()
            test_out_dir.rmdir()

        # 1. Record Issues in handoff
        mgr = HandoffManager(test_handoff_file)
        recorded = mgr.record_issue(
            title="Terms Checkbox Not Auto-Focused",
            issue_type="UX",
            severity="P2",
            observed="User clicked calculate premium but submission silently stalled.",
            expected="Checkbox should show validation error ring.",
            evidence="Case 1 on step 7",
            suggested_fix="Add required indicator or autofocus.",
            related_tests=["TC-001"],
        )
        self.assertTrue(recorded)

        # Deduplication check
        dup_recorded = mgr.record_issue(
            title="Terms Checkbox Not Auto-Focused",
            issue_type="UX",
            severity="P2",
        )
        self.assertFalse(dup_recorded)

        items = mgr.list_open_items()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["severity"], "P2")
        self.assertEqual(items[0]["type"], "UX")

        # 2. JSONL streaming
        res_entry = {
            "case_id": "1",
            "status": "PASS",
            "actual": 450.0,
            "expected": 450.0,
            "duration_s": 2.15,
            "note": "Quote calculated",
        }
        stream_path = stream_test_result_jsonl(test_out_dir, res_entry, date_str="2026-11-24")
        self.assertTrue(stream_path.exists())
        lines = stream_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["case_id"], "1")
        self.assertEqual(parsed["status"], "PASS")

        # Cleanup
        if test_handoff_file.exists():
            test_handoff_file.unlink()
        if stream_path.exists():
            stream_path.unlink()
        if test_out_dir.exists():
            for f in test_out_dir.glob("*"):
                f.unlink()
            test_out_dir.rmdir()

    def test_engine_jsonl_suite_loading(self):
        jsonl_path = Path("test_suite_loader.jsonl")
        jsonl_path.write_text(
            '{"id": "A1", "full_name": "Test User", "age": 25}\n'
            '{"id": "A2", "full_name": "Second User", "age": 35}\n',
            encoding="utf-8",
        )

        cfg = QAConfig(auto_start_llama=False)
        cfg.suite.cases_path = str(jsonl_path)
        engine = QAEngine(cfg)
        loaded = engine._load_test_cases()

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["case_id"], "A1")
        self.assertEqual(loaded[1]["case_id"], "A2")

        if jsonl_path.exists():
            jsonl_path.unlink()

    def test_testability_seams_config(self):
        cfg = QAConfig()
        cfg.target.url = "http://example.com"
        cfg.target.url_params = {"test_key": "banner.shown", "test_value": "false"}
        cfg.target.local_storage = {"auth_token": "mock-token-xyz"}
        cfg.target.session_storage = {"flow_step": "step1"}

        d = cfg.to_dict()
        self.assertEqual(d["target"]["url_params"]["test_key"], "banner.shown")
        self.assertEqual(d["target"]["local_storage"]["auth_token"], "mock-token-xyz")

        loaded = QAConfig.from_dict(d)
        self.assertEqual(loaded.target.url_params["test_value"], "false")
        self.assertEqual(loaded.target.session_storage["flow_step"], "step1")

    def test_flow_config_and_stage_guidance(self):
        cfg = QAConfig()
        cfg.flow.app_context = "E-commerce checkout funnel"
        cfg.flow.stages = [
            {"name": "Cart", "match": "/cart", "guidance": "Review items and click checkout"},
            {"name": "Payment", "match": "/checkout/payment", "guidance": "Enter credit card and billing zip"},
            {"name": "Confirmation", "match": "/checkout/success", "is_terminal": True},
            {"name": "Title Only", "match": "order done", "is_terminal": True},
        ]

        stage1 = cfg.get_stage_guidance("https://shop.com/cart")
        self.assertEqual(stage1["name"], "Cart")
        self.assertIn("Review items", stage1["guidance"])

        stage2 = cfg.get_stage_guidance("https://shop.com/checkout/payment?step=2")
        self.assertEqual(stage2["name"], "Payment")

        self.assertTrue(cfg.is_terminal_page("https://shop.com/checkout/success"))
        self.assertTrue(cfg.is_terminal_page("https://shop.com/other", body_text="Thank you for your order!"))
        self.assertFalse(cfg.is_terminal_page("https://shop.com/cart", body_text="Your cart is empty"))
        self.assertFalse(cfg.is_terminal_page("https://shop.com/step1", body_text="Please complete the form"))
        self.assertTrue(cfg.is_terminal_page("https://shop.com/page", page_title="Order Done"))

    def test_extraction_grounding(self):
        cfg = QAConfig(auto_start_llama=False)
        agent = QAAgent(cfg)
        body = "Annual Premium: $450.00 Quote Reference: Q-7A8B9C"
        self.assertTrue(agent._extraction_is_grounded({"premium": 450.0, "quote_id": "Q-7A8B9C"}, body))
        self.assertFalse(agent._extraction_is_grounded({"premium": 999.0, "quote_id": "Q-ABC123"}, body))


class TestModelProviders(unittest.TestCase):
    def test_multi_model_config_roundtrip(self):
        cfg = QAConfig(
            llm_backend="litellm",
            litellm_model="openai/gpt-4o-mini",
            fallback_model="ollama/llama3",
            provider_chain=["llamacpp", "litellm"],
            temperature=0.3,
            max_tokens=99,
        )
        self.assertEqual(cfg.llm_backend, "litellm")
        loaded = QAConfig.from_dict(cfg.to_dict())
        self.assertEqual(loaded.model.litellm_model, "openai/gpt-4o-mini")
        self.assertEqual(loaded.model.fallback_model, "ollama/llama3")
        self.assertEqual(loaded.model.provider_chain, ["llamacpp", "litellm"])
        self.assertEqual(loaded.temperature, 0.3)
        self.assertEqual(loaded.max_tokens, 99)

    def test_router_uses_available_provider(self):
        from model_provider import BaseProvider, Completion, ModelRouter

        class StubProvider(BaseProvider):
            name = "stub"

            def complete(self, messages, **kwargs):
                return Completion('{"ok": true}', "stub", "stub-model")

        cfg = QAConfig(auto_start_llama=False, llm_backend="litellm", litellm_model="openai/test")
        with patch.object(ModelRouter, "_build", return_value=[StubProvider(cfg, "stub-model")]):
            router = ModelRouter(cfg)
            result = router.complete(
                [], temperature=0.1, max_tokens=10, timeout=1,
                response_format={"type": "json_object"},
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.text, '{"ok": true}')
        self.assertEqual(result.provider, "stub")

    def test_llama_server_uses_jinja_template(self):
        with patch("llm_manager.is_server_alive", side_effect=[False, True]), \
             patch("llm_manager.find_llama_executable", return_value="llama.exe"), \
             patch("llm_manager.find_model_gguf", return_value="model.gguf"), \
             patch("llm_manager.subprocess.Popen") as popen:
            popen.return_value.poll.return_value = None
            started = llm_manager.ensure_llama_server("http://127.0.0.1:8099/v1", use_jinja=True)
            self.assertTrue(started)
            cmd = popen.call_args.args[0]
            self.assertIn("--jinja", cmd)
            self.assertNotIn("--chat-template", cmd)
            self.assertNotIn("--no-reasoning-preserve", cmd)


class TestTUIActions(unittest.IsolatedAsyncioTestCase):
    async def test_buttons_and_shortcuts_mutate_config(self):
        from tui import VettorTUI

        with tempfile.TemporaryDirectory() as tmp:
            cfg = QAConfig(auto_start_llama=False, llm_backend="heuristic")
            cfg.reporting.output_dir = tmp
            app = VettorTUI(cfg)
            async with app.run_test(size=(160, 50)) as pilot:
                await pilot.pause()
                await pilot.click("#btn-engine")
                await pilot.pause()
                self.assertEqual(cfg.engine, "playwright")
                await pilot.press("b")
                await pilot.pause()
                self.assertEqual(cfg.llm_backend, "llamacpp")
                await pilot.press("t")
                await pilot.pause()
                self.assertTrue(any(e["action"] == "show_trajectory" for e in app.trajectory.entries))
                self.assertTrue(Path(tmp, "tui_trajectory.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
