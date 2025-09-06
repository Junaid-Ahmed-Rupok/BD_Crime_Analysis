import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

df = joblib.load('df.pkl')  # loading the cleaned dataframe
encoder = joblib.load('encoder.pkl')  # loading the encoder

x = df.drop(columns=['crime'])  # features of the model
y = df['crime']  # label of the model

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=42)  # splitting the dataset into training and testing sections

# feature selection

# adaboost
ad_demo_model = AdaBoostClassifier(n_estimators=200, learning_rate=0.2, random_state=42)  #
ad_demo_model.fit(x_train, y_train)
ad_feature_importance = pd.Series(ad_demo_model.feature_importances_, index=x_train.columns).sort_values(
    ascending=False).head(10)
ad_features = ad_feature_importance.index.tolist()  # features for the adaboost model

# xgboost
xg_demo_model = XGBClassifier(n_estimators=200, learning_rate=0.2, random_state=42)  #
xg_demo_model.fit(x_train, y_train)
xg_feature_importance = pd.Series(xg_demo_model.feature_importances_, index=x_train.columns).sort_values(
    ascending=False).head(10)
xg_features = xg_feature_importance.index.tolist()  # features for the xgboost model

models = {
    'Adaboost': (AdaBoostClassifier(n_estimators=200, learning_rate=0.2, random_state=42), ad_features),
    'XGBoost': (XGBClassifier(n_estimators=200, learning_rate=0.2, random_state=42), xg_features),
}  # dictionary of the defined models

best_score, best_name, best_model, best_features = 0, None, None, None  # initials

for name, (model, features) in models.items():
    score = cross_val_score(model, x_train[features], y_train, cv=5).mean()  # cross-validation score
    print(f"cross validation score of {name}: {score}")

    if score > best_score:
        best_score, best_name, best_model, best_features = score, name, model, features

print(
    f"{best_name} is the best model with {best_score} cross-validation score.")  # printing the best model name and its cross-validation score

# training the best model
best_model.fit(x_train[best_features], y_train)  # trained
predictions = best_model.predict(x_test[best_features])  # predictions using the best model


def overfitting_check(model, name, features):
    train_predictions = model.predict(x_train[features])
    test_predictions = model.predict(x_test[features])

    train_accuracy = (y_train == train_predictions).mean()
    test_accuracy = (y_test == test_predictions).mean()

    if abs(train_accuracy - test_accuracy) > 0.1:
        print(f"--------> Warning: Overfitting")
    else:
        print(f"--------> No Significant Overfitting.")


print(f"Overfitting Checking of the best model: {best_name}")
overfitting_check(best_model, best_name, best_features)
print("\n" * 2)

print(f"Overfitting checking of all the models:\n\n")
for name, (model, features) in models.items():
    model.fit(x_train[features], y_train)
    print(f"{name} Overfitting Checking:")
    overfitting_check(model, name, features)

joblib.dump(best_model, 'model.pkl')  # dumping the best model

# classification report
print(classification_report(y_test, predictions))

# confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# learning curves
row_max, column_max = 1, 2  # adjust as needed
fig, ax = plt.subplots(row_max, column_max, figsize=(20, 10))
ax = ax.flatten()  # flatten to 1D array for easier indexing

# Loop through models
for i, (name, (model, features)) in enumerate(models.items()):
    # Generate learning curve data
    train_sizes, training_score, testing_score = learning_curve(
        model, x_train[features], y_train,
        train_sizes=np.linspace(0.1, 1, 10),
        cv=5, scoring='accuracy'
    )

    # Plot learning curves
    ax[i].set_xlabel('Training Sizes')
    ax[i].set_ylabel('Accuracy Scores')
    ax[i].set_title(f'{name} Learning Curves')
    ax[i].plot(train_sizes, training_score.mean(1), color='red', label='training curve')
    ax[i].plot(train_sizes, testing_score.mean(1), color='green', label='testing curve')
    ax[i].legend()
    ax[i].grid(True)

plt.tight_layout()
plt.show()

# feature importance graph
feature_importance = {
    'Adaboost': ad_feature_importance, 'XGBoost': xg_feature_importance
}

row, column = 0, 0
fig, bx = plt.subplots(row_max, column_max, figsize=(20, 10))
bx = bx.flatten()  # flatten to 1D array for easier indexing

# Loop through feature importances
for i, (name, value) in enumerate(feature_importance.items()):
    bx[i].set_xlabel('Features')
    bx[i].set_ylabel('Feature Importance')
    bx[i].set_title(f'{name} feature importance')
    bx[i].bar(value.index, value.values, color='green')
    bx[i].set_xticks(range(len(value.index)))  # set tick positions
    bx[i].set_xticklabels(value.index, rotation=45, ha='right')  # rotate labels
    bx[i].grid(True)

plt.tight_layout()
plt.show()