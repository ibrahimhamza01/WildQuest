"""WildQuest UI MVP (issue #13).

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts import REVIEW_THRESHOLD, ResponseError, parse_prediction  # noqa: E402
from app.mock_data import SCENARIOS, MockInferenceProvider, MockObservationProvider  # noqa: E402
from app.render import ImageLoadError, box_color, draw_boxes, load_image  # noqa: E402
from app.species import SPECIES  # noqa: E402

PAGES = ("Identify", "Collection", "History")
AUTO = "Automatic (based on photo)"
SCENARIO_LABELS = {"single": "One animal", "multiple": "Several animals",
                   "uncertain": "Uncertain prediction", "no_animal": "No animal found",
                   "error": "Service error"}
STATUS_LABELS = {"VERIFIED": "✅ Verified", "NEEDS_REVIEW": "🕒 Needs review",
                 "REJECTED": "❌ Rejected"}
OUTCOME_LABELS = {"FIRST_DISCOVERY": "First discovery", "NEW_ENCOUNTER": "New encounter",
                  "SAME_ENCOUNTER": "Same encounter (no XP)",
                  "DAILY_LIMIT_REACHED": "Daily limit reached",
                  "PENDING_REVIEW": "Pending review", "REJECTED": "Rejected",
                  "DUPLICATE_REQUEST": "Duplicate upload"}


@st.cache_resource
def get_providers():
    """Swap these for the real inference (#11) and observation (#12) services later."""
    return MockInferenceProvider(), MockObservationProvider(), True


def pct(value):
    return f"{value * 100:.0f}%"


def nice_time(text):
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%b %d, %Y %H:%M")
    except (AttributeError, ValueError):
        return text or "—"


def sidebar(observations):
    st.sidebar.title("🦝 WildQuest")
    summary = observations.summary()
    col1, col2 = st.sidebar.columns(2)
    col1.metric("XP", summary["total_xp"])
    col2.metric("Species", f"{summary['discovered_species']}/{summary['total_species']}")
    page = st.sidebar.radio("Go to", PAGES, key="page")
    with st.sidebar.expander("Demo settings"):
        choice = st.selectbox("Mock result", [AUTO] + [SCENARIO_LABELS[s] for s in SCENARIOS],
                              key="scenario",
                              help="Force a specific demo result to see each screen state.")
    scenario = None if choice == AUTO else next(
        s for s, label in SCENARIO_LABELS.items() if label == choice)
    return page, scenario


def show_animal(index, animal, is_mock):
    with st.container(border=True):
        best = animal.best
        color = box_color(index)
        st.markdown(f"<span style='color:{color};font-weight:700'>Box {index + 1}</span>"
                    f" &nbsp; **{best.common_name}** · *{best.scientific_name}*",
                    unsafe_allow_html=True)
        st.progress(best.confidence, text=f"Confidence {pct(best.confidence)}")
        if animal.needs_review:
            st.warning(f"Not sure about this one: confidence is below {pct(REVIEW_THRESHOLD)}. "
                       "It would be saved as **Needs review** and earn no XP until approved. "
                       "Check the other possibilities below.")
        else:
            st.success("Confident match")
        st.caption("Top 3 possibilities" + (" (demo prediction)" if is_mock else ""))
        st.table([{"Rank": rank, "Species": g.common_name, "Scientific name": g.scientific_name,
                   "Confidence": pct(g.confidence)} for rank, g in enumerate(animal.top, 1)])
        st.caption(f"Animal detector confidence: {pct(animal.detector_confidence)}")


def identify_page(inference, scenario, is_mock):
    st.header("Identify an animal")
    upload = st.file_uploader("Upload a wildlife photo", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        st.info("Upload a photo of a mammal, bird, reptile or amphibian from the DMV area "
                "to see what WildQuest finds.")
        return

    data = upload.getvalue()
    try:
        image = load_image(data)
    except ImageLoadError as exc:
        st.error(f"{exc} Please upload a JPG, PNG or WebP photo.")
        return

    with st.spinner("Looking for animals…"):
        try:
            result = parse_prediction(inference.predict(data, upload.name, scenario))
        except ResponseError as exc:
            result = None
            error = f"The identification service sent an unexpected response ({exc})."
        except Exception:  # the backend may be unreachable; the UI must not crash
            result = None
            error = "Something went wrong while identifying this photo. Please try again."

    left, right = st.columns([3, 2])
    if result is None or result.status == "error":
        left.image(image, caption=upload.name, width="stretch")
        right.error(error if result is None else result.message)
        return
    if result.status == "no_animal":
        left.image(image, caption=upload.name, width="stretch")
        right.warning("**No animal found in this photo.**\n\n"
                      "Try a photo where the animal is larger in the frame, in focus, "
                      "and not hidden behind branches or grass.")
        return

    left.image(draw_boxes(image, result.animals), caption=upload.name, width="stretch")
    count = len(result.animals)
    right.subheader(f"Found {count} animal{'s' if count != 1 else ''}")
    with right:
        for i, animal in enumerate(result.animals):
            show_animal(i, animal, result.is_mock)
    st.caption("Saving to your collection will use the observation service (#12), "
               "which is not connected in this demo.")


def collection_page(observations):
    st.header("Your collection")
    summary = observations.summary()
    collection = observations.collection()
    st.progress(collection["progress"],
                text=f"{collection['discovered']} of {collection['total_species']} species discovered")
    cols = st.columns(4)
    cols[0].metric("Total XP", summary["total_xp"])
    cols[1].metric("Observations", summary["observations"])
    cols[2].metric("Verified", summary["verified"])
    cols[3].metric("Needs review", summary["needs_review"])

    for group in ("Bird", "Mammal", "Reptile", "Amphibian"):
        st.subheader(group + "s")
        entries = [e for e in collection["species"] if SPECIES[e["class_id"]].group == group]
        grid = st.columns(4)
        for i, entry in enumerate(entries):
            sp = SPECIES[entry["class_id"]]
            with grid[i % 4].container(border=True):
                if entry["discovered"]:
                    st.markdown(f"✅ **{sp.common_name}**")
                    st.caption(f"*{sp.scientific_name}*  \n"
                               f"Seen {entry['verified_observations']}× · "
                               f"best {pct(entry['best_confidence'])}  \n"
                               f"First seen {nice_time(entry['first_seen'])}")
                else:
                    st.markdown("🔒 **Not discovered yet**")
                    hint = " · 1 photo awaiting review" if entry["pending_review"] else ""
                    st.caption(f"A {group.lower()}{hint}")


def history_page(observations):
    st.header("Observation history")
    rows = observations.history()
    if not rows:
        st.info("No observations yet. Identify an animal to start your collection.")
        return
    st.dataframe([{
        "When": nice_time(r["observed_at"]),
        "Species": SPECIES[r["class_id"]].common_name,
        "Confidence": pct(r["confidence"]),
        "Status": STATUS_LABELS.get(r["status"], r["status"]),
        "Result": OUTCOME_LABELS.get(r.get("outcome"), r.get("outcome") or "—"),
        "XP": r.get("xp_awarded", 0),
    } for r in rows], hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 3)


def main():
    st.set_page_config(page_title="WildQuest", page_icon="🦝", layout="wide")
    inference, observations, is_mock = get_providers()
    page, scenario = sidebar(observations)
    if is_mock:
        st.info("🧪 **Demo mode.** Predictions, collection and XP on these screens are mock "
                "data, not real model output.")
    if page == "Identify":
        identify_page(inference, scenario, is_mock)
    elif page == "Collection":
        collection_page(observations)
    else:
        history_page(observations)


main()