"""Streamlit demo - type a review, see the sentiment and what the model looked at.

Run with:
    streamlit run app/app.py

The app is a thin shell over ``src.predict.SentimentPredictor``: the same class
the CLI uses, which in turn reuses the training-time text cleaning. Nothing about
how text is processed is reimplemented here.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

# Make ``src`` importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import SentimentPredictor  # noqa: E402

# Project palette: sentiment is a polarity scale, so red <-> gray <-> blue.
COLORS = {"negative": "#e34948", "neutral": "#898781", "positive": "#2a78d6"}
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

EXAMPLES = [
    "The battery died after three weeks and support never replied. Complete waste of money.",
    "Works exactly as described, sound quality is excellent and setup took two minutes.",
    "The picture is sharp but the remote feels cheap and the menus are slow.",
    "It's okay for the price I guess. Nothing special either way.",
    "I really wanted to love this, but it keeps disconnecting which ruins the whole thing.",
]

st.set_page_config(page_title="Review Sentiment", page_icon="chart", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_key: str) -> SentimentPredictor:
    return SentimentPredictor(model_key)


def render_probabilities(probabilities: dict[str, float], predicted: str) -> None:
    """Horizontal probability bars, each in its class colour, all direct-labelled."""
    rows = []
    for name, p in probabilities.items():
        weight = "600" if name == predicted else "400"
        rows.append(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin:7px 0;">
              <div style="width:74px;color:{COLORS[name]};font-weight:{weight};
                          font-size:0.9rem;">{name}</div>
              <div style="flex:1;background:#eeeeec;border-radius:5px;height:13px;
                          overflow:hidden;">
                <div style="width:{p * 100:.1f}%;background:{COLORS[name]};
                            height:100%;border-radius:5px;"></div>
              </div>
              <div style="width:56px;text-align:right;color:{INK_SECONDARY};
                          font-variant-numeric:tabular-nums;font-size:0.9rem;
                          font-weight:{weight};">{p:.1%}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_attention(tokens: list[str], weights: list[float]) -> None:
    """Shade each token by its attention weight.

    Weights are normalised against the maximum in this review: attention is a
    distribution over one sequence, so raw values are not comparable between
    reviews of different lengths.
    """
    if not weights:
        st.info("This model has no attention layer, so there is nothing to highlight.")
        return

    peak = max(weights) or 1.0
    spans = []
    for token, weight in zip(tokens, weights):
        share = weight / peak
        # Opacity carries magnitude; the text darkens only where it must stay legible.
        spans.append(
            f'<span style="background:rgba(42,120,214,{share * 0.85:.3f});'
            f'padding:2px 5px;margin:2px 1px;border-radius:4px;'
            f'display:inline-block;'
            f'color:{"#ffffff" if share > 0.62 else "#0b0b0b"};">'
            f"{html.escape(token)}</span>"
        )
    st.markdown(
        f'<div style="line-height:2.25;font-size:0.97rem;">{"".join(spans)}</div>',
        unsafe_allow_html=True,
    )


st.title("Customer Review Sentiment")
st.caption(
    "Bi-LSTM with attention over GloVe embeddings, trained on 240,000 Amazon "
    "Electronics reviews. Type a review to see the prediction and the words the "
    "model weighted most."
)

with st.sidebar:
    st.subheader("Model")
    model_key = st.selectbox(
        "Architecture",
        ["bilstm_attention", "bigru", "lstm"],
        format_func=lambda k: {
            "bilstm_attention": "Bi-LSTM + Attention (main)",
            "bigru": "Bi-GRU",
            "lstm": "Plain LSTM",
        }[k],
    )
    st.caption("Only the Bi-LSTM + Attention model can show word highlighting.")

try:
    predictor = get_predictor(model_key)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.metric("Validation macro-F1", f"{predictor.val_macro_f1:.4f}")
    st.caption(f"Running on `{predictor.device}`")

if "review_text" not in st.session_state:
    st.session_state.review_text = EXAMPLES[2]

st.write("**Try an example**")
columns = st.columns(len(EXAMPLES))
for i, (column, example) in enumerate(zip(columns, EXAMPLES)):
    label = ["negative", "positive", "mixed", "lukewarm", "mixed"][i]
    if column.button(label, use_container_width=True, key=f"ex{i}"):
        st.session_state.review_text = example

review = st.text_area(
    "Your review", key="review_text", height=130,
    placeholder="Type or paste a product review...",
)

if not review.strip():
    st.info("Enter a review above to see a prediction.")
    st.stop()

try:
    prediction = predictor.predict(review)
except ValueError as exc:
    st.warning(f"Could not read that: {exc}")
    st.stop()

colour = COLORS[prediction.label_name]
st.markdown(
    f"""
    <div style="border-left:5px solid {colour};padding:12px 18px;margin:18px 0;
                background:rgba(0,0,0,0.02);border-radius:0 8px 8px 0;">
      <div style="font-size:1.55rem;font-weight:700;color:{colour};
                  text-transform:capitalize;">{prediction.label_name}</div>
      <div style="color:{INK_SECONDARY};font-size:0.9rem;">
        {prediction.confidence:.1%} confident
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_probabilities(prediction.probabilities, prediction.label_name)

st.divider()
st.subheader("What the model focused on")
st.caption("Darker background = higher attention weight.")
render_attention(prediction.tokens, prediction.attention)

top = prediction.top_tokens(6)
if top:
    st.write("**Strongest words**")
    st.markdown(
        "  ".join(
            f'<span style="background:#eeeeec;padding:3px 9px;border-radius:11px;'
            f'margin-right:6px;font-size:0.87rem;">{html.escape(t)} '
            f'<span style="color:{INK_MUTED};">{w:.3f}</span></span>'
            for t, w in top
        ),
        unsafe_allow_html=True,
    )

with st.expander("How this works"):
    st.markdown(
        """
        1. Your text is cleaned with **exactly the same functions used in training**
           (`normalise_text` then `clean_for_neural` from `src/data.py`) - otherwise
           the model would see a different kind of text than it learned on.
        2. Words become ids from the training vocabulary; unknown words map to `<unk>`.
        3. A bidirectional LSTM reads the sequence in both directions.
        4. An attention layer scores every token and produces a weighted summary,
           rather than relying only on the final hidden state. **Those scores are the
           highlighting you see above.**
        5. A linear layer turns that summary into three class probabilities.

        The model was trained with class-weighted loss because the data is 79%
        positive and only 7% neutral. Neutral remains the hardest class - a 3-star
        review is often a genuinely mixed opinion rather than a distinct sentiment.
        """
    )
