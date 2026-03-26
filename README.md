# Multimodal Sentiment Analysis

This repository contains a progression of models designed to evaluate sentiment, humor, sarcasm, and offensiveness using both unimodal (text) and multimodal (text and image) inputs. The project is structured across several tasks, advancing from baseline text classification to complex multimodal multi-task learning architectures.

## Repository Structure

* **`imdb_data.py`**
    A baseline script for binary sentiment classification. It fine-tunes a `bert-base-uncased` model on a reduced subset of the IMDb dataset, utilizing Hugging Face's `Trainer` API for optimization and evaluation.
* **`task1.ipynb`**
    Implements a unimodal text classification pipeline using `distilbert-base-uncased`. It maps a custom dataset's overall sentiment into three discrete classes (positive, negative, neutral) and fine-tunes the transformer.
* **`task2.ipynb`**
    Introduces a multimodal early-fusion architecture. It extracts text embeddings using `bert-base-uncased` and image features using a pre-trained `ResNet50` (with the final classification head removed). The concatenated features are passed through a fully connected layer to output binary predictions for humor, sarcasm, and offensiveness, optimized via `BCEWithLogitsLoss`.
* **`task3.ipynb`**
    Expands on the multimodal approach by implementing a multi-task continuous learning model. It supports both `ResNet50` and Vision Transformers (`ViT`) for image feature extraction, fused with BERT embeddings. The architecture employs separate linear heads for humor, sarcasm, offense, and motivation. 
    *Methodology Note:* This task currently frames the classification problem as a regression task during training, utilizing Mean Squared Error (`MSELoss`) on mapped numerical labels. Continuous outputs are rounded to discrete integers during the evaluation phase to compute macro F1 and accuracy scores.

## Architectures Used

* **Text Encoders:** BERT (`bert-base-uncased`), DistilBERT (`distilbert-base-uncased`).
* **Image Encoders:** ResNet50, Vision Transformer (`vit_base_patch16_224`).
* **Fusion Strategy:** Feature concatenation followed by linear projections and ReLU activations.

## Dataset Requirements

To run the multimodal tasks (`task1`, `task2`, `task3`), the following data structure is expected:
1.  An `images/` directory containing the raw image files.
2.  A `labels.csv` file containing the following required columns: `image_name`, `text_corrected` (or `text_ocr`), `humour`, `sarcasm`, `offensive`, `motivational`, and `overall_sentiment`.

*Note: The script `imdb_data.py` expects a standard `IMDB Dataset.csv` file in the root directory.*

## Dependencies

The project relies on the following core libraries:
* `torch`
* `torchvision`
* `transformers`
* `pandas`
* `scikit-learn`
* `Pillow`
* `timm` (Required for ViT integration in `task3.ipynb`)

Install the required dependencies via pip:

```bash
pip install torch torchvision transformers pandas scikit-learn pillow timm
```

## Usage

1.  **Baseline Text Model:** Execute `python imdb_data.py` to train and evaluate the baseline IMDb model. The resulting model and tokenizer will be saved to the `bert-imdb-sentiment-model/` directory.
2.  **Jupyter Notebooks:** Execute the cells sequentially in `task1.ipynb`, `task2.ipynb`, and `task3.ipynb`. Ensure that file paths for `CSV_FILE_PATH` and `IMG_DIR_PATH` are correctly updated to point to your local dataset directories before execution.
