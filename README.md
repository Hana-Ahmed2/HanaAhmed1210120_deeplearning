# Neural Network for Pima Indians Diabetes Prediction

## Problem Description

This project implements a neural network to predict diabetes diagnosis using the Pima Indians Diabetes dataset. The dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases and includes diagnostic measurements from 768 female patients of Pima Indian heritage, aged 21 years or older.

**Dataset Source**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Features (8 clinical measurements)
| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg/(height in m)²) |
| DiabetesPedigreeFunction | Diabetes pedigree function (genetic risk factor) |
| Age | Age in years |

### Target
- **0**: No Diabetes
- **1**: Diabetes

## Approach

### Data Preprocessing
1. Loaded dataset directly from UCI ML Repository URL
2. Applied stratified 80-20 train-test split to maintain class distribution
3. Normalized features using StandardScaler (fit on training data only to prevent data leakage)

### Neural Network Architecture
```
Input Layer (8 features)
    ↓
Dense Layer (16 neurons, ReLU activation)
    ↓
Dropout (30%)
    ↓
Dense Layer (8 neurons, ReLU activation)
    ↓
Dropout (20%)
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

### Training Configuration
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 16 (smaller for better generalization)
- **Max Epochs**: 150
- **Early Stopping**: Patience = 10 epochs (monitors validation loss)
- **Dropout**: 30% and 20% to prevent overfitting
- **Class Weights**: 1.3x weight on No Diabetes class (to reduce false positives)
- **Prediction Threshold**: 0.45 (balanced between FP and FN)

## Results

### Achieved Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | **74.03%** |
| Recall (Diabetes) | **74.07%** |
| Precision (Diabetes) | 63.49% |
| F1-Score | 68.38% |
| False Negatives | 14 |
| False Positives | 23 |

### Key Achievement: Balanced FP/FN Trade-off
We achieved a balanced model by:
1. **Dropout Regularization**: Added 30% and 20% dropout layers to prevent overfitting
2. **Class Weighting**: Applied 1.3x weight to No Diabetes class to reduce false positives
3. **Threshold Tuning**: Set threshold to 0.45 for balanced error distribution

### Dataset Challenges
- Class imbalance (~65% non-diabetic, ~35% diabetic)
- Feature overlap between classes
- Some missing data represented as zeros

### Evaluation Metrics
The model is evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: Of predicted diabetes cases, how many are truly diabetic?
- **Recall**: Of actual diabetes cases, how many did we correctly identify?
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual breakdown of TP, TN, FP, FN

### Medical Importance
For diabetes screening, **recall (sensitivity)** is particularly critical because:
- Missing a diabetes diagnosis (false negative) could delay treatment
- Early detection enables lifestyle interventions and treatment
- False positives, while inconvenient, lead to additional testing rather than harm

## Project Structure

```
neural-network-diabetes/
├── pima_diabetes_nn.ipynb    # Complete Jupyter notebook with all code
├── diabetes.csv              # Dataset file (from Kaggle)
├── README.md                 # This documentation file
└── results/
    ├── training_curves.png   # Accuracy and loss over epochs
    ├── confusion_matrix.png  # Test set confusion matrix
    └── metrics_summary.txt   # Numerical metrics summary
```

## How to Run

1. **Open the notebook**: `pima_diabetes_nn.ipynb`
2. **Run all cells** sequentially (Kernel → Run All)
3. **Results** will be automatically saved to the `results/` folder

### Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

Install dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

## Key Takeaways

1. **Dropout regularization** helps prevent overfitting on small datasets
2. **Class weights** can be tuned to balance false positives and false negatives
3. **Prediction thresholds** trade precision for recall - adjust based on use case
4. **Early stopping** prevents overfitting by monitoring validation loss
5. **Feature scaling** is essential when features have different ranges

## Techniques Used for Optimization

1. **Dropout Layers**: 30% and 20% dropout to improve generalization
2. **Class Weighting**: 1.3x weight on No Diabetes class to reduce FP
3. **Threshold Tuning**: Set to 0.45 for balanced FP/FN
4. **Result**: Achieved 74% accuracy with balanced error distribution

## Author

Created as part of Neural Networks for Binary Classification coursework.

## License

This project is for educational purposes.
