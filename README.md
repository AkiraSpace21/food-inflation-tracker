# AI-Powered Price Prediction & Early Warning System

## Project Overview
This project implements a multi-signal AI system designed to predict price trends and detect market anomalies. By combining historical price data with sentiment analysis and search trend signals, the model provides an early warning system for price spikes.

### Key Insights & Performance
* Price Trend: Analyzed price movements over the dataset period.
* Early Warning: Predicts price increases 2–4 weeks in advance.
* Accuracy: Achieved 70%+ accuracy on validation sets.
* Signals Used: Price History, Search Volume, Sentiment Analysis.

---

## Installation & Setup

### 1. Prerequisites
* Python 3.8+
* Git

### 2. Clone the Repository
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME

### 3. Create Virtual Environment & Install
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate

    # Then install dependencies
    pip install -r requirements.txt

---

## Usage Guide

### Run the Full Pipeline
    # Step 1: Preprocessing
    python src/preprocessing.py

    # Step 2: Train Models
    python src/train_models.py

    # Step 3: Final Analysis (Results saved to data/processed/final_results.csv)
    python main.py

---

## Project Structure
    data/
    ├── raw/                 # Input datasets
    └── processed/           # Cleaned data & results
    src/                     # Source code (cleaning, training, logic)
    main.py                  # Primary execution script
    requirements.txt         # Python library dependencies
    README.md                # This documentation file

---

## Development Cheat Sheet (Git)

    # 1. Check which files changed
    git status

    # 2. Stage all changes (new files + edits)
    git add .

    # 3. Save changes locally
    git commit -m "Description of changes"

    # 4. Upload to GitHub
    git push

    # 5. Download updates from GitHub
    git pull

---

## License
Distributed under the MIT License.
