# **Generative AI for Fake News Article Generation: Style-Controlled Synthesis with CNN and DailyMail & Generative Image**


## 1. Project Overview

This project develops a modular generative AI system capable of producing full-length **fake** and **real** news articles from textual prompts.  
By fine-tuning transformer-based models and integrating **latent diffusion techniques**, the system supports **cross-cultural style control**, accurately emulating the journalistic writing styles of **CNN (U.S.)** and **DailyMail (U.K.)**. The project addresses:
- Automated expansion of headlines or summaries into multi-paragraph news articles.
- Style-controlled text generation using lightweight prompt conditioning (e.g., `<fake>`, `<real>`, `<us>`, `<uk>` tokens).
- Comparative analysis between real vs. fake and U.S. vs. U.K. stylistic patterns.



## 2. Project Goals

- Build a text generation system conditioned on **fake/real** news labels and **U.S./U.K.** style indicators.
- Generate **full-length, coherent** articles from minimal prompts (headlines or summaries).
- Analyze **stylistic differences** in sentence structure, emotional tone, and lexical framing across generated outputs.
- Explore **lightweight prompt-controlled generation** strategies using latent diffusion and transformer-based models.


## 3. Technologies Used

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



## 4. Project Structure

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



## 5. Setup Instructions

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



## 6. Usage Instructions

### **6.1 Generating News Articles**

Run the main generation script:

```bash
python generate_articles.py --dataset realnews --style real
python generate_articles.py --dataset cnn_dailymail --style uk
```

**Options:**
- `--dataset` : Choose from `realnews` or `cnn_dailymail`
- `--style` : Choose from `real`, `fake`, `us`, `uk`

Generated articles will be saved in `/outputs/generated_articles/`.



### **6.2 Running Style Analysis**

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



## 7. Outputs
- **Generated Articles**: Full-length articles based on short prompts (title or summary), conditioned by style.
- **Evaluation Reports**: Quantitative analysis of style, structure, and emotional tone across fake/real and U.S./U.K. generations.
- **Visualizations (Optional)**: Sentence length distribution, sentiment histograms, polarity comparison plots.


## 8. Broader Applications
- **Fake News Detection Research**: Studying rhetorical differences between fabricated and authentic articles.
- **Cross-Cultural Media Analysis**: Investigating stylistic framing differences between U.S. and U.K. news ecosystems.
- **Content Creation**: Building stylistically controllable AI writers for journalism, marketing, and creative industries.
