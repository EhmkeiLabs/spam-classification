# Spam Classification Project

This project implements a machine learning model to classify text messages as spam or ham (not spam). The implementation is provided in a Jupyter notebook for easy exploration and analysis.

## Project Structure

```
spam-classification/
├── spam-classification.ipynb    # Jupyter notebook with the analysis
├── demo_dataset.csv             # Sample dataset for testing
└── .gitignore                  # Git ignore file
```

## Features

- **Text Preprocessing**: Cleans and prepares text data for classification
- **Machine Learning Model**: Implements a classification algorithm for spam detection
- **Performance Evaluation**: Includes accuracy, precision, recall, and F1-score metrics
- **Interactive Analysis**: Jupyter notebook for exploratory data analysis and visualization

## Prerequisites

- Python 3.7+
- Required Python packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - jupyter
  - matplotlib
  - seaborn

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/EhmkeiLabs/spam-classification.git
   cd spam-classification
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook spam-classification.ipynb
   ```

## Usage

1. **Using the Jupyter Notebook**:
   - Open `spam-classification.ipynb`
   - Run each cell sequentially to see the analysis
   - Modify parameters as needed for your specific use case

2. **Running Individual Classifiers**:
   ```bash
   # Run Logistic Regression
   python logistic-regression.py
   
   # Run Decision Tree Classifier
   python decisiontree-regression.py
   ```

## Dataset

The project uses a dataset of SMS messages labeled as spam or ham. The dataset is included in the repository as `demo_dataset.csv`.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| Decision Tree | - | - | - | - |

*Note: Replace the above placeholders with actual performance metrics from your models.*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset sourced from [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- Built with Python's amazing data science stack
