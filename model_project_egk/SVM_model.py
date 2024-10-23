import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

file_path = "/home/keg/workspace/openvino-AI-project/landmark_distances(ECK).csv"
data = pd.read_csv(file_path)

corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

data.hist(bins=20, figsize=(20, 15))
plt.show()
'''
X = data.drop(columns=['person_id'])
y = data['person_id']

def augment_data(X, y, num_augmented_samples=100, noise_factor=0.01):
    X_augmented = []
    y_augmented = []
    for i in range(num_augmented_samples):
        idx = np.random.randint(0, X.shape[0])
        sample = X.iloc[idx]

        noisy_sample = sample + noise_factor * np.random.randn(*sample.shape)

        X_augmented.append(noisy_sample)
        y_augmented.append(y.iloc[idx])

    X_aug = pd.DataFrame(X_augmented, columns=X.columns)
    y_aug = pd.Series(y_augmented)

    X_combined = pd.concat([X, X_aug], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y, y_aug], axis=0).reset_index(drop=True)

    return X_combined, y_combined

X_augmented, y_augmented = augment_data(X, y, num_augmented_samples=200)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_augmented)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_augmented, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("교차 검증 점수:", scores)
print("평균 점수:", scores.mean())

print(data['person_id'].value_counts())

train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

print(f"훈련 정확도: {train_accuracy}")
print(f"테스트 정확도: {test_accuracy}")

model_path = "/home/keg/workspace/openvino-AI-project/distance_model1.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"모델이 '{model_path}'에 저장되었습니다.")

with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

y_pred = loaded_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
'''