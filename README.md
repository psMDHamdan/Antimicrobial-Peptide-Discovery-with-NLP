# 🧬 Antimicrobial Peptide Discovery with NLP

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computational pipeline for the identification and de novo design of **Antimicrobial Peptides (AMPs)**. By treating peptide sequences as a biological language, this project leverages Transformer-based NLP and generative models to combat the global antimicrobial resistance crisis.

## 🚀 Key Features

*   **Deep Learning Classifier**: High-precision detection of AMPs using a CNN-BiLSTM architecture (98.9% test accuracy).
*   **Generative AI**: De novo peptide sequence design using a Variational Autoencoder (VAE).
*   **Physicochemical Profiling**: Automated calculation of charge, hydrophobicity (GRAVY), isoelectric point, and more via Biopython.
*   **Interactive Dashboard**: A powerful Streamlit-based web interface for sequence screening and generation.
*   **Interpretability**: Attention map extraction for identifying critical peptide regions (ProtBERT integration).

## 📂 Repository Structure

```text
├── src/
│   ├── data_utils.py     # Data curation & property calculation
│   ├── dataset.py        # Tokenization & PyTorch utilities
│   ├── model.py          # Classifier architectures
│   ├── train.py          # Modern training pipeline
│   ├── predictor.py      # Inference engine
│   ├── generator.py      # VAE sequence generator
│   └── visualization.py  # Advanced plotting (Seaborn/Plotly)
├── app.py                # Streamlit dashboard
├── run_pipeline.py       # Pipeline orchestration
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
```

## 🛠️ Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/psMDHamdan/Antimicrobial-Peptide-Discovery-with-NLP.git
    cd Antimicrobial-Peptide-Discovery-with-NLP
    ```

2.  **Set up a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Quick Start

### 1. Run the Full Discovery Pipeline
This command processes data, generates EDA plots, trains the classifier, and stabilizes the VAE generator:
```bash
python run_pipeline.py --model lightweight --epochs 15 --train-vae --vae-epochs 30
```

### 2. Launch the Web Interface
Screen sequences or generate new candidates in real-time:
```bash
streamlit run app.py
```

## 🧪 Model Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 98.92% |
| **F1-Score** | 0.9838 |
| **ROC-AUC** | 0.9995 |

## 🧬 Data Sources

*   **APD3**: [Antimicrobial Peptide Database](https://aps.unmc.edu/AP/)
*   **DBAASP**: [Database of Antimicrobial Activity/Structure](https://dbaasp.org/)
*   **UniProt**: [Protein Knowledgebase](https://www.uniprot.org/)

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Created by psMDHamdan as part of a computational biology initiative.*
