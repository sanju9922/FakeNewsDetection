# Fake News Detection

This project aims to develop a machine learning-based system for detecting fake news articles. It leverages natural language processing (NLP) techniques to classify news content as real or fake, enhancing the reliability of information dissemination.

## Key Features

- **Data Preprocessing**: Cleans and processes raw text data for analysis.
- **TF-IDF Feature Extraction**: Converts text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
- **Machine Learning Models**: Implements multiple models, including Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting, for accurate news classification.
- **Model Evaluation**: Provides detailed evaluation metrics, such as accuracy, precision, recall, and F1-score, to assess the performance of different models.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/fake-news-detection.git
    cd fake-news-detection
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Jupyter Notebook:

    ```bash
    jupyter notebook FakeNewsDetection.ipynb
    ```

2. Follow the steps in the notebook to preprocess the data, train the models, and evaluate their performance.

## Project Structure

- `FakeNewsDetection.ipynb` : Main Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `news.csv` : Dataset file containing news articles and their labels (real or fake).
- `requirements.txt` : A file containing all the required Python libraries and their versions.

## Models Implemented

- **Logistic Regression**: A statistical model for binary classification.
- **Decision Tree Classifier**: A tree-based model that makes decisions based on feature values.
- **Random Forest Classifier**: An ensemble method that combines multiple decision trees to enhance model robustness.
- **Gradient Boosting Classifier**: An ensemble technique that builds models sequentially to reduce errors.

## Data Preprocessing

- **Text Cleaning**: Converts text to lowercase, removes special characters and stop words, and applies lemmatization.
- **TF-IDF Vectorization**: Transforms cleaned text data into numerical vectors using TF-IDF, enabling machine learning algorithms to process it.

## Evaluation

- Evaluates the models using metrics such as accuracy, precision, recall, and F1-score.
- Displays a confusion matrix to visualize model performance.

## Contributions

Contributions to the project are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [your-email@example.com](mailto:your-email@example.com).
