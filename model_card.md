# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classification model trained using a Random Forest Classifier from scikit-learn. It is used to predict whether an individual's income exceeds $50K per year based on features from the U.S. Census Income dataset.

## Intended Use

The model is intended for educational and demonstration purposes in the context of learning how to build and deploy machine learning pipelines using FastAPI. It is not suitable for production use or real-world decision-making.

## Training Data

The model was trained on the U.S. Census Income dataset obtained from the UCI Machine Learning Repository. The dataset contains 32,561 rows and includes demographic and employment-related attributes. The data was split using an 80/20 train-test ratio.

The following features were treated as categorical:
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

## Evaluation Data

The test set consisted of 20% of the dataset (6,512 records). The data was processed using one-hot encoding for categorical features and label binarization for the target variable.

## Metrics

The model was evaluated using three metrics:
- **Precision**
- **Recall**
- **F1 Score**

Here are the model's overall performance metrics on the test set:

- **Precision:** 0.7372  
- **Recall:** 0.6366  
- **F1 Score:** 0.6832  

In addition, slice-based metrics were computed for each unique value of the categorical features, and results were logged in `slice_output.txt`.

## Ethical Considerations

This dataset reflects historical U.S. Census data and may encode biases related to gender, race, education, and employment. The model trained on this data may unintentionally perpetuate these biases.

These outputs should not be used in real-world applications, especially in areas such as hiring, lending, or housing, without thorough bias evaluation and fairness audits.

## Caveats and Recommendations

- Model performance varies across slices of the population. In particular, minority groups with fewer examples may result in skewed metrics or overfitting.
- One-hot encoding increases dimensionality and may result in sparse representations for infrequent values.
- Some values in the dataset (e.g., "?") may represent missing data and could negatively impact model generalization.

For better results:
- Consider imputing or removing unknown values
- Experiment with more powerful classifiers or feature engineering
- Analyze fairness and bias in predictions per demographic group
