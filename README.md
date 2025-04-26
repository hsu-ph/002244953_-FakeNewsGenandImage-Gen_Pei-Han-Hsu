# 002244953_Spring25_Final-Project


# **Generative AI for Fake News Article Generation: Style-Controlled Synthesis with CNN and DailyMail & Generative Image**

## **1. Overview**

This project explores the use of **generative AI** models to synthesize fake and real news articles, with a particular focus on **style-controlled generation** across different media cultures—namely, U.S.-based **CNN** and U.K.-based **DailyMail**.  
By leveraging modern text-to-image and text-to-text generation architectures, the system is designed to produce stylistically faithful and semantically coherent news content from minimal textual prompts.

The project combines two key strands of investigation:
- **Fake News Generation**: Simulating the writing styles and rhetorical patterns commonly found in both authentic and fabricated news narratives.
- **Cross-Cultural Style Conditioning**: Controlling the generated outputs to emulate U.S. journalistic standards (CNN) versus U.K. tabloid-style reporting (DailyMail).

Through prompt engineering and modular generative pipelines, the system demonstrates the flexibility of latent diffusion models and transformer-based architectures in capturing complex stylistic and cultural nuances.



## **2. Technical Approach**

The generative workflow integrates multiple complementary models:

- **Prompt Encoding (CLIP/Transformer Models):**  
  Input prompts are encoded into dense semantic embeddings, enabling fine-grained conditioning during generation.

- **Latent-Space Text-to-Article Generation (Fine-Tuned LLMs):**  
  Based on the RealNews dataset (for headline expansion) and CNN/DailyMail summaries (for full article reconstruction), the models generate multi-paragraph news articles aligned with user-specified topics and stylistic controls.

- **Style Control Tokens:**  
  Special tokens such as `<fake>`, `<real>`, `<us>`, and `<uk>` are incorporated into prompts to steer the generative model toward specific writing tones, structures, and cultural framing.

- **Evaluation Metrics:**  
  Generated outputs are evaluated quantitatively (using BLEU, ROUGE scores) and qualitatively (based on sentence structure, emotional polarity, punctuation patterns, and lexical variation).



## **3. Experimental Results**
- **Structural Coherence:**  
  The models successfully expanded short prompts into coherent news narratives, demonstrating strong content organization.

- **Style Sensitivity:**  
  Using minimal prompt cues, the models reliably adapted their outputs to match stylistic expectations—emulating the factual tone of CNN or the sensationalized style of DailyMail.

- **Fake vs Real Differentiation:**  
  Fake news generations exhibited heightened emotional language, informal lexical choices, and exaggerated punctuation, while real news generations maintained a more formal, objective journalistic tone.

- **Semantic Alignment:**  
  The combination of latent-space diffusion techniques and transformer-based prompt understanding resulted in outputs that were both semantically consistent and stylistically appropriate.


## **4. Broader Impact and Future Work**

This study highlights the capabilities of modern generative AI systems not only to create linguistically rich content but also to emulate cultural and editorial styles with minimal supervision. Through responsible design and control, such systems can be powerful tools for education, media synthesis, and understanding linguistic bias and media framing across cultures. Future work may focus on:

- Expanding the system to multi-modal generation (e.g., headline + image + article bundles)
- Enhancing stylistic control to finer levels (e.g., political leaning, emotional framing)
- Integrating adversarial detectors to monitor and mitigate potential misuse in misinformation scenarios


## **Demo**
- **Demo:** [Watch Demo](https://youtu.be/OQLjNcveBJc)
- **Website:** [View Project Website](https://v0-python-text-to-image.vercel.app/)






了解！  
你的要求是要我幫你整理出：

- **完整格式的 README.md**，包含：
  - 項目目標（Project Goals）
  - 使用技術（Technologies）
  - 輸出內容（Outputs）
  - 安裝與執行指令（Setup, Execution, Usage）
  - 清楚結構和說明
- 可以直接配合你的 Final Project：**Generative AI + Fake News + CNN/DailyMail** 的主題

這裡是我為你**完整撰寫**的一版正式版 README.md：

---

# **Generative AI for Fake News and Style-Controlled News Article Generation**

---

## 📚 Project Overview

This project develops a modular generative AI system capable of producing full-length **fake** and **real** news articles from textual prompts.  
By fine-tuning transformer-based models and integrating **latent diffusion techniques**, the system supports **cross-cultural style control**, accurately emulating the journalistic writing styles of **CNN (U.S.)** and **DailyMail (U.K.)**.

The project addresses:
- Automated expansion of headlines or summaries into multi-paragraph news articles.
- Style-controlled text generation using lightweight prompt conditioning (e.g., `<fake>`, `<real>`, `<us>`, `<uk>` tokens).
- Comparative analysis between real vs. fake and U.S. vs. U.K. stylistic patterns.

---

## 🚀 Project Goals

- Build a text generation system conditioned on **fake/real** news labels and **U.S./U.K.** style indicators.
- Generate **full-length, coherent** articles from minimal prompts (headlines or summaries).
- Analyze **stylistic differences** in sentence structure, emotional tone, and lexical framing across generated outputs.
- Explore **lightweight prompt-controlled generation** strategies using latent diffusion and transformer-based models.

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **PyTorch** – Deep learning framework
- **Hugging Face Transformers** – For CLIP model, text encoders
- **Diffusers Library** – For Stable Diffusion latent generation
- **VAE (Variational Autoencoder)** – For latent representation compression
- **Real-ESRGAN** – For super-resolution post-processing
- **Datasets:**
  - **RealNews** – Title-to-article expansion dataset
  - **CNN/DailyMail** – Summary-to-article reconstruction dataset

Optional Components:
- **CLIP** (Prompt expansion module, extendable)
- **Scikit-learn**, **NLTK**, **spaCy** – For evaluation metrics and linguistic feature extraction

---

## 📄 Project Structure

```
/project-root/
│
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── generate_articles.py     # Main generation script
├── style_analysis.py        # Style evaluation and comparison
├── utils/
│   ├── dataset_loader.py    # Loading RealNews and CNN/DailyMail datasets
│   ├── text_cleaner.py      # Preprocessing utilities
│   └── evaluation.py        # BLEU, ROUGE, sentiment scoring
├── models/
│   ├── text_encoder.py      # CLIP prompt encoding (optional)
│   ├── diffusion_model.py   # Stable Diffusion latent generation
│   ├── vae_model.py         # Variational Autoencoder module
│   └── real_esrgan.py       # Super-resolution module
└── outputs/
    ├── generated_articles/  # Generated news outputs
    ├── evaluation_reports/  # Style evaluation results
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/fake-news-generation.git
   cd fake-news-generation
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**
   - **Stable Diffusion v1-5** (`runwayml/stable-diffusion-v1-5`)
   - **Real-ESRGAN** weights
   - (Optional) **OpenAI CLIP Model** for future prompt expansion

5. **Prepare datasets**
   - Load `RealNews` and `CNN/DailyMail` datasets using Hugging Face Datasets or custom loaders.
   - Preprocess text using `text_cleaner.py` utilities if needed.

---

## 🧩 Usage Instructions

### **Generating News Articles**

Run the main generation script:

```bash
python generate_articles.py --dataset realnews --style real
python generate_articles.py --dataset cnn_dailymail --style uk
```

**Options:**
- `--dataset` : Choose from `realnews` or `cnn_dailymail`
- `--style` : Choose from `real`, `fake`, `us`, `uk`

Generated articles will be saved in `/outputs/generated_articles/`.

---

### **Running Style Analysis**

Evaluate stylistic differences across generated outputs:

```bash
python style_analysis.py --input_folder outputs/generated_articles/
```

Generates statistical reports on:
- Average sentence length
- Punctuation density
- Sentiment polarity
- Lexical variation

Results saved under `/outputs/evaluation_reports/`.

---

## 📊 Outputs

- **Generated Articles**: Full-length articles based on short prompts (title or summary), conditioned by style.
- **Evaluation Reports**: Quantitative analysis of style, structure, and emotional tone across fake/real and U.S./U.K. generations.
- **Visualizations (Optional)**: Sentence length distribution, sentiment histograms, polarity comparison plots.

---

## 🌍 Broader Applications

- **Fake News Detection Research**: Studying rhetorical differences between fabricated and authentic articles.
- **Cross-Cultural Media Analysis**: Investigating stylistic framing differences between U.S. and U.K. news ecosystems.
- **Content Creation**: Building stylistically controllable AI writers for journalism, marketing, and creative industries.
