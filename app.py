"""
Streamlit Web App for AMP Discovery.
- Predict AMP probability for a given sequence
- Visualize attention maps
- Generate novel peptide candidates using VAE
- EDA dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────
#  Page config
# ─────────────────────────────────────
st.set_page_config(
    page_title="AMP Discovery · NLP",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}
.main-title {
    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 50%, #ff7b72 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #8b949e;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(145deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #a371f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.seq-input textarea {
    font-family: 'JetBrains Mono', monospace !important;
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}
.prediction-chip {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 1rem;
}
.amp {
    background: rgba(88, 166, 255, 0.15);
    border: 1px solid #58a6ff;
    color: #58a6ff;
}
.non-amp {
    background: rgba(255, 123, 114, 0.15);
    border: 1px solid #ff7b72;
    color: #ff7b72;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
#  Header
# ─────────────────────────────────────
st.markdown('<div class="main-title">🧬 AMP Discovery</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Antimicrobial Peptide Prediction · NLP-Powered</div>', unsafe_allow_html=True)

# ─────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    model_type = st.selectbox("Model", ["lightweight", "protbert"], index=0)
    threshold = st.slider("AMP Probability Threshold", 0.3, 0.95, 0.7, 0.05)
    max_len = st.number_input("Max Sequence Length", 32, 512, 128, 16)

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")

    csv_path = Path("data/processed/amp_dataset.csv")
    if csv_path.exists():
        df_info = pd.read_csv(csv_path)
        n_amp = (df_info["label"] == 1).sum()
        n_non = (df_info["label"] == 0).sum()
        st.metric("Total Sequences", f"{len(df_info):,}")
        st.metric("AMPs", f"{n_amp:,}")
        st.metric("non-AMPs", f"{n_non:,}")
    else:
        st.warning("Dataset not built yet. Run the pipeline first.")

    st.markdown("---")
    known_amps = {
        "Magainin 2": "GIGKFLHSAKKFGKAFVGEIMNS",
        "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "Melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
        "Defensin HNP-1": "ACYCRIPACIAGERRYGTCIYQGRLWAFCC",
    }
    st.markdown("### 🧪 Sample AMPs")
    selected_amp = st.selectbox("Load example", list(known_amps.keys()))

# ─────────────────────────────────────
#  Tabs
# ─────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Predict", "🎲 Generate", "📊 EDA Dashboard", "ℹ️ About"
])


# ─────────────────────────────────────
#  Tab 1: Prediction
# ─────────────────────────────────────
with tab1:
    st.markdown("### Sequence Prediction")
    col1, col2 = st.columns([3, 1])
    with col1:
        sequence_input = st.text_area(
            "Amino Acid Sequence",
            value=known_amps[selected_amp],
            height=100,
            help="Enter standard one-letter amino acid codes (A, C, D, E, …)",
            key="seq_input",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_attention = st.checkbox("Show Attention", value=True)
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)

    if predict_btn and sequence_input.strip():
        model_path = Path(f"models/best_{model_type}_model.pt")
        if not model_path.exists():
            st.error(f"Model not found at `{model_path}`. Please train the model first.")
        else:
            with st.spinner("Running inference..."):
                from src.predictor import AMPPredictor
                predictor = AMPPredictor(
                    model_path=str(model_path),
                    model_type=model_type,
                    max_len=int(max_len),
                )
                result = predictor.predict(
                    sequence_input.strip().upper(),
                    return_attention=(show_attention and model_type == "protbert")
                )

            # Results
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                f'<div class="metric-card"><div class="metric-value">'
                f'{result["amp_probability"]:.1%}</div>'
                f'<div class="metric-label">AMP Probability</div></div>',
                unsafe_allow_html=True,
            )
            chip_cls = "amp" if result["label"] == "AMP" else "non-amp"
            c2.markdown(
                f'<div class="metric-card"><div class="prediction-chip {chip_cls}">{result["label"]}</div>'
                f'<br><div class="metric-label">Prediction</div></div>',
                unsafe_allow_html=True,
            )
            c3.markdown(
                f'<div class="metric-card"><div class="metric-value">'
                f'{result["confidence"]:.1%}</div>'
                f'<div class="metric-label">Confidence</div></div>',
                unsafe_allow_html=True,
            )
            seq = result["sequence"]
            c4.markdown(
                f'<div class="metric-card"><div class="metric-value">'
                f'{len(seq)}</div>'
                f'<div class="metric-label">Sequence Length</div></div>',
                unsafe_allow_html=True,
            )

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["amp_probability"] * 100,
                title={"text": "AMP Probability (%)", "font": {"color": "#c9d1d9"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar": {"color": "#58a6ff"},
                    "steps": [
                        {"range": [0, float(threshold) * 100], "color": "#161b22"},
                        {"range": [float(threshold) * 100, 100], "color": "#1c3540"},
                    ],
                    "threshold": {
                        "line": {"color": "#ff7b72", "width": 3},
                        "thickness": 0.8,
                        "value": float(threshold) * 100,
                    },
                },
                number={"suffix": "%", "font": {"color": "#c9d1d9"}},
            ))
            fig_gauge.update_layout(
                height=250,
                paper_bgcolor="#0d1117",
                font={"color": "#c9d1d9"},
                margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Attention visualization (if available)
            if "position_importance" in result:
                st.markdown("#### 🔍 Per-Position Attention (CLS Token)")
                tokens = list(result["sequence"])
                scores = result["position_importance"][:len(tokens)]
                norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

                fig_att = go.Figure(go.Bar(
                    x=list(range(len(tokens))),
                    y=norm_scores,
                    text=tokens,
                    textposition="outside",
                    marker=dict(
                        color=norm_scores,
                        colorscale="Viridis",
                    ),
                ))
                fig_att.update_layout(
                    height=280,
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#161b22",
                    font={"color": "#c9d1d9"},
                    xaxis=dict(title="Position", showgrid=False, color="#8b949e"),
                    yaxis=dict(title="Attention Weight", showgrid=False, color="#8b949e"),
                    margin=dict(t=20, b=40),
                )
                st.plotly_chart(fig_att, use_container_width=True)

    elif predict_btn:
        st.warning("Please enter a sequence.")


# ─────────────────────────────────────
#  Tab 2: Generate
# ─────────────────────────────────────
with tab2:
    st.markdown("### Novel Peptide Generation (VAE)")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        n_gen = st.slider("Number of candidates to generate", 5, 100, 20, 5)
        temperature = st.slider("Sampling Temperature", 0.3, 2.0, 0.8, 0.1,
                                help="Lower = more conservative, Higher = more diverse")
    with g_col2:
        gen_threshold = st.slider("AMP threshold for filtering", 0.3, 0.95, 0.6, 0.05)
        st.markdown("<br>", unsafe_allow_html=True)
        gen_btn = st.button("🎲 Generate Peptides", type="primary", use_container_width=True)

    if gen_btn:
        vae_path = Path("models/peptide_vae.pt")
        classifier_path = Path(f"models/best_{model_type}_model.pt")

        if not vae_path.exists():
            st.error("VAE model not found. Please train the VAE first (see docs).")
        elif not classifier_path.exists():
            st.error("Classifier model not found. Please train the classifier first.")
        else:
            with st.spinner(f"Generating {n_gen} peptide candidates..."):
                import torch
                from src.generator import PeptideVAE
                device = "cuda" if torch.cuda.is_available() else "cpu"

                vae = PeptideVAE()
                vae.load_state_dict(torch.load(str(vae_path), map_location=device))
                vae.eval()

                raw_seqs = vae.generate(n_gen, temperature=temperature, device=device)

                from src.predictor import AMPPredictor
                predictor = AMPPredictor(str(classifier_path), model_type=model_type)
                candidates = predictor.screen_candidates(raw_seqs, threshold=gen_threshold)

            if candidates:
                st.success(f"✅ {len(candidates)} candidates passed the AMP threshold ({gen_threshold:.0%})")
                cand_df = pd.DataFrame(candidates)
                cand_df = cand_df[["sequence", "amp_probability", "confidence", "label"]].copy()
                cand_df.columns = ["Sequence", "AMP Probability", "Confidence", "Label"]
                cand_df["AMP Probability"] = cand_df["AMP Probability"].apply(lambda x: f"{x:.2%}")
                cand_df["Confidence"] = cand_df["Confidence"].apply(lambda x: f"{x:.2%}")
                st.dataframe(cand_df, use_container_width=True)

                csv_out = cand_df.to_csv(index=False)
                st.download_button("⬇️ Download Candidates (CSV)", csv_out,
                                   file_name="amp_candidates.csv", mime="text/csv")
            else:
                st.warning("No candidates passed the threshold. Try lowering the threshold or increasing temperature.")


# ─────────────────────────────────────
#  Tab 3: EDA Dashboard
# ─────────────────────────────────────
with tab3:
    st.markdown("### Dataset Analysis")

    if not csv_path.exists():
        st.warning("Run the data pipeline first: `python run_pipeline.py`")
    else:
        df = pd.read_csv(csv_path)
        df["label_name"] = df["label"].map({0: "non-AMP", 1: "AMP"})
        color_map = {"AMP": "#58a6ff", "non-AMP": "#ff7b72"}

        row1_c1, row1_c2 = st.columns(2)

        with row1_c1:
            fig_dist = px.pie(
                df["label_name"].value_counts().reset_index(),
                values="count", names="label_name",
                title="Class Distribution",
                color="label_name", color_discrete_map=color_map,
            )
            fig_dist.update_layout(
                paper_bgcolor="#0d1117", font={"color": "#c9d1d9"}, height=350
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with row1_c2:
            fig_len = px.histogram(
                df, x="length", color="label_name",
                barmode="overlay", opacity=0.7,
                title="Sequence Length Distribution",
                color_discrete_map=color_map,
            )
            fig_len.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font={"color": "#c9d1d9"}, height=350,
                legend_title_text="",
            )
            st.plotly_chart(fig_len, use_container_width=True)

        row2_c1, row2_c2 = st.columns(2)

        with row2_c1:
            if "charge_at_ph7" in df.columns and "gravy" in df.columns:
                fig_scatter = px.scatter(
                    df.sample(min(3000, len(df))),
                    x="charge_at_ph7", y="gravy",
                    color="label_name",
                    opacity=0.6,
                    title="Charge vs. Hydrophobicity (GRAVY)",
                    color_discrete_map=color_map,
                )
                fig_scatter.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font={"color": "#c9d1d9"}, height=350,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with row2_c2:
            if "isoelectric_point" in df.columns:
                fig_iep = px.histogram(
                    df, x="isoelectric_point", color="label_name",
                    barmode="overlay", opacity=0.7,
                    title="Isoelectric Point Distribution",
                    color_discrete_map=color_map,
                )
                fig_iep.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font={"color": "#c9d1d9"}, height=350,
                )
                st.plotly_chart(fig_iep, use_container_width=True)

        st.markdown("#### Source Distribution")
        src_fig = px.histogram(
            df, x="source", color="label_name",
            barmode="group", color_discrete_map=color_map,
            title="Sequences per Source",
        )
        src_fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font={"color": "#c9d1d9"}, height=300,
        )
        st.plotly_chart(src_fig, use_container_width=True)


# ─────────────────────────────────────
#  Tab 4: About
# ─────────────────────────────────────
with tab4:
    st.markdown("""
    ## 🧬 Antimicrobial Peptide Discovery with NLP

    This tool applies **Transformer-based NLP** to predict and generate Antimicrobial Peptides (AMPs).

    ### Models Available
    | Model | Architecture | Speed | Accuracy |
    |-------|-------------|-------|----------|
    | **Lightweight** | CNN-BiLSTM | Fast (CPU) | Good baseline |
    | **ProtBERT** | BERT fine-tuned on UniRef100 | Slower (GPU rec.) | State-of-the-art |

    ### Data Sources
    - **APD** (Antimicrobial Peptide Database) — [aps.unmc.edu](https://aps.unmc.edu/)
    - **DBAASP** — [dbaasp.org](https://dbaasp.org/)
    - **UniProt** — non-AMP negative controls

    ### Pipeline
    ```
    Data Collection → Preprocessing → Tokenization → Training → Prediction → Generation
    ```

    ### References
    - Wang et al. (2016) APD3
    - Pirtskhalava et al. (2021) DBAASP v3
    - Elnaggar et al. (2021) ProtTrans (ProtBERT)
    """)
