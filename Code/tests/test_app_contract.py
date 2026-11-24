"""Tests for the WildQuest UI MVP (issue #13): response parsing, mock providers, UI states."""

import io
import sys
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.contracts import REVIEW_THRESHOLD, ResponseError, parse_prediction  # noqa: E402
from app.mock_data import (SCENARIOS, MockInferenceProvider,  # noqa: E402
                           MockObservationProvider, choose_scenario)
from app.render import ImageLoadError, draw_boxes, load_image  # noqa: E402
from app.species import SPECIES  # noqa: E402

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # Streamlit is only needed for the UI tests
    AppTest = None

APP_PATH = str(REPO_ROOT / "app" / "streamlit_app.py")


def detection(predictions, box=(0.1, 0.1, 0.5, 0.5), detector_confidence=0.9):
    return {"box": list(box), "detector_confidence": detector_confidence,
            "predictions": [{"class_id": c, "confidence": p} for c, p in predictions]}


def photo_bytes(size=(64, 48), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size, 128).save(buffer, format="PNG")
    return buffer.getvalue()


class TestParsePrediction(unittest.TestCase):
    def test_single_animal_sorted_top3_with_names(self):
        response = {"status": "ok", "detections": [
            detection([(9, 0.03), (17, 0.94), (12, 0.01), (20, 0.005)])]}
        result = parse_prediction(response)
        self.assertEqual(result.status, "ok")
        animal = result.animals[0]
        self.assertEqual([g.class_id for g in animal.top], [17, 9, 12])  # top 3, sorted
        self.assertEqual(animal.best.common_name, "Raccoon")
        self.assertEqual(animal.best.scientific_name, "Procyon lotor")
        self.assertFalse(animal.needs_review)

    def test_multiple_detections_kept_in_order(self):
        response = {"status": "ok", "detections": [detection([(0, 0.9)]),
                                                   detection([(2, 0.8)])]}
        self.assertEqual([a.best.class_id for a in parse_prediction(response).animals], [0, 2])

    def test_low_confidence_needs_review(self):
        low = REVIEW_THRESHOLD - 0.01
        result = parse_prediction({"status": "ok", "detections": [detection([(18, low)])]})
        self.assertTrue(result.animals[0].needs_review)

    def test_empty_ok_response_becomes_no_animal(self):
        self.assertEqual(parse_prediction({"status": "ok", "detections": []}).status,
                         "no_animal")

    def test_error_response_keeps_message(self):
        result = parse_prediction({"status": "error", "message": "Service down"})
        self.assertEqual((result.status, result.message), ("error", "Service down"))

    def test_malformed_responses_rejected(self):
        bad = [
            None,
            {"status": "maybe"},
            {"status": "error"},                                         # no message
            {"status": "ok", "detections": "lots"},
            {"status": "ok", "detections": [detection([(25, 0.9)])]},    # class_id out of range
            {"status": "ok", "detections": [detection([(1, 1.5)])]},     # confidence > 1
            {"status": "ok", "detections": [detection([(True, 0.9)])]},  # bool is not an ID
            {"status": "ok", "detections": [detection([])]},             # no predictions
            {"status": "ok", "detections": [detection([(1, 0.9)], box=(0.5, 0.5, 0.2, 0.9))]},
            {"status": "no_animal", "detections": [detection([(1, 0.9)])]},
        ]
        for response in bad:
            with self.subTest(response=response), self.assertRaises(ResponseError):
                parse_prediction(response)


class TestMockProviders(unittest.TestCase):
    def test_every_mock_scenario_parses_and_is_labelled_mock(self):
        provider = MockInferenceProvider()
        expected = {"single": "ok", "multiple": "ok", "uncertain": "ok",
                    "no_animal": "no_animal", "error": "error"}
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                result = parse_prediction(provider.predict(b"x", scenario=scenario))
                self.assertEqual(result.status, expected[scenario])
                self.assertTrue(result.is_mock)
        self.assertEqual(len(parse_prediction(provider.predict(b"", scenario="multiple"))
                             .animals), 2)
        self.assertTrue(parse_prediction(provider.predict(b"", scenario="uncertain"))
                        .animals[0].needs_review)

    def test_scenario_choice_is_deterministic(self):
        data = photo_bytes()
        self.assertEqual(choose_scenario(data, "a.png"), choose_scenario(data, "a.png"))
        self.assertEqual(choose_scenario(data, "my_noanimal.jpg"), "no_animal")
        self.assertEqual(choose_scenario(data, "ERROR.png"), "error")

    def test_predict_returns_independent_copies(self):
        provider = MockInferenceProvider()
        first = provider.predict(b"", scenario="single")
        first["detections"].clear()
        self.assertEqual(len(provider.predict(b"", scenario="single")["detections"]), 1)

    def test_observation_mock_matches_service_shape_and_xp(self):
        obs = MockObservationProvider()
        summary, collection, history = obs.summary(), obs.collection(), obs.history()
        for key in ("total_xp", "discovered_species", "total_species", "observations",
                    "verified", "needs_review", "rejected"):
            self.assertIn(key, summary)
        self.assertEqual(len(collection["species"]), len(SPECIES))
        self.assertEqual(summary["discovered_species"], collection["discovered"])
        # The mock's total XP must equal the XP awarded across its history.
        self.assertEqual(summary["total_xp"], sum(h["xp_awarded"] for h in history))
        discovered = {e["class_id"] for e in collection["species"] if e["discovered"]}
        first = {h["class_id"] for h in history if h["outcome"] == "FIRST_DISCOVERY"}
        self.assertEqual(discovered, first)


class TestRendering(unittest.TestCase):
    def test_load_image_converts_grayscale_to_rgb(self):
        self.assertEqual(load_image(photo_bytes(mode="L")).mode, "RGB")

    def test_load_image_rejects_non_images(self):
        with self.assertRaises(ImageLoadError):
            load_image(b"definitely not a photo")

    def test_draw_boxes_keeps_size_and_marks_image(self):
        image = load_image(photo_bytes(size=(200, 100)))
        animals = parse_prediction({"status": "ok", "detections": [
            detection([(0, 0.9)]), detection([(2, 0.8)], box=(0.6, 0.2, 0.9, 0.8))]}).animals
        boxed = draw_boxes(image, animals)
        self.assertEqual(boxed.size, image.size)
        self.assertNotEqual(boxed.tobytes(), image.tobytes())


@unittest.skipIf(AppTest is None, "Streamlit not installed (pip install -r app/requirements.txt)")
class TestUiStates(unittest.TestCase):
    def run_app(self, page="Identify", scenario=None, upload=None):
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        if page != "Identify":
            at.sidebar.radio(key="page").set_value(page).run()
        if scenario:
            at.selectbox(key="scenario").set_value(scenario).run()
        if upload:
            at.file_uploader[0].upload(*upload).run()
        self.assertFalse(at.exception, at.exception)
        return at

    @staticmethod
    def text(elements):
        return " ".join(str(e.value) for e in elements)

    def test_start_screen_shows_demo_label_and_upload_prompt(self):
        at = self.run_app()
        self.assertIn("Demo mode", self.text(at.info))
        self.assertIn("Upload a photo", self.text(at.info))

    def test_confident_single_prediction(self):
        at = self.run_app(scenario="One animal", upload=("deer.png", photo_bytes(), "image/png"))
        self.assertIn("Found 1 animal", self.text(at.subheader))
        self.assertIn("Confident match", self.text(at.success))

    def test_multiple_detections(self):
        at = self.run_app(upload=("pond_multi.png", photo_bytes(), "image/png"))
        self.assertIn("Found 2 animals", self.text(at.subheader))
        self.assertEqual(len(at.table), 2)

    def test_uncertain_prediction_warns(self):
        at = self.run_app(scenario="Uncertain prediction",
                          upload=("x.png", photo_bytes(), "image/png"))
        self.assertIn("Needs review", self.text(at.warning))

    def test_no_animal_state(self):
        at = self.run_app(scenario="No animal found", upload=("x.png", photo_bytes(), "image/png"))
        self.assertIn("No animal found", self.text(at.warning))

    def test_service_error_state(self):
        at = self.run_app(scenario="Service error", upload=("x.png", photo_bytes(), "image/png"))
        self.assertIn("did not respond", self.text(at.error))

    def test_unreadable_upload_shows_error(self):
        at = self.run_app(upload=("broken.png", b"not an image", "image/png"))
        self.assertIn("could not be read", self.text(at.error))

    def test_collection_and_history_pages(self):
        at = self.run_app(page="Collection")
        self.assertIn("Your collection", self.text(at.header))
        self.assertIn("720", self.text(at.metric))
        at = self.run_app(page="History")
        self.assertIn("Observation history", self.text(at.header))
        self.assertEqual(len(at.dataframe), 1)


if __name__ == "__main__":
    unittest.main()