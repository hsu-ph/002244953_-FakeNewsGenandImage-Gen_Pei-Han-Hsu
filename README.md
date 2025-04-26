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
