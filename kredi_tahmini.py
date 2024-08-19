import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment",
           "InstallmentRate", "PersonalStatus", "OtherDebtors", "ResidenceSince", "Property", "Age",
           "OtherInstallmentPlans", "Housing", "ExistingCredits", "Job", "LiablePeople", "Telephone",
           "ForeignWorker", "CreditRisk"]

df = pd.read_csv(url, delimiter=' ', header=None, names=columns)
df['CreditRisk'] = df['CreditRisk'].map({1: 1, 2: 0})

X = df.drop("CreditRisk", axis=1)
y = df["CreditRisk"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_res, y_train_res)


def user_input_features():
    print("Müşteri Bilgilerini Girin:")

    Status = input("Durum (A11, A12, A13, A14): ")
    Duration = int(input("Süre (ay): "))
    CreditHistory = input("Kredi Geçmişi (A30, A31, A32, A33, A34): ")
    Purpose = input("Kredi Amacı (A40, A41, A42, A43, A44, A45, A46, A47, A48, A49, A410): ")
    CreditAmount = int(input("Kredi Miktarı: "))
    Savings = input("Tasarruflar (A61, A62, A63, A64, A65): ")
    Employment = input("İstihdam Süresi (A71, A72, A73, A74, A75): ")
    InstallmentRate = int(input("Taksit Oranı (%): "))
    PersonalStatus = input("Kişisel Durum ve Cinsiyet (A91, A92, A93, A94): ")
    OtherDebtors = input("Diğer Borçlular (A101, A102, A103): ")
    ResidenceSince = int(input("İkamet Süresi: "))
    Property = input("Mülkiyet (A121, A122, A123, A124): ")
    Age = int(input("Yaş: "))
    OtherInstallmentPlans = input("Diğer Taksit Planları (A141, A142, A143): ")
    Housing = input("Konut (A151, A152, A153): ")
    ExistingCredits = int(input("Mevcut Krediler: "))
    Job = input("İş (A171, A172, A173, A174): ")
    LiablePeople = int(input("Bakmakla Yükümlü Kişiler: "))
    Telephone = input("Telefon (A191, A192): ")
    ForeignWorker = input("Yabancı İşçi (A201, A202): ")

    data = {
        'Status': Status,
        'Duration': Duration,
        'CreditHistory': CreditHistory,
        'Purpose': Purpose,
        'CreditAmount': CreditAmount,
        'Savings': Savings,
        'Employment': Employment,
        'InstallmentRate': InstallmentRate,
        'PersonalStatus': PersonalStatus,
        'OtherDebtors': OtherDebtors,
        'ResidenceSince': ResidenceSince,
        'Property': Property,
        'Age': Age,
        'OtherInstallmentPlans': OtherInstallmentPlans,
        'Housing': Housing,
        'ExistingCredits': ExistingCredits,
        'Job': Job,
        'LiablePeople': LiablePeople,
        'Telephone': Telephone,
        'ForeignWorker': ForeignWorker
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_df = scaler.transform(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

print("\nTahmin Sonucu")
print("İyi Kredi" if prediction[0] == 1 else "Kötü Kredi")

print("\nİyi Kredi Olasılığı")
print(prediction_proba[0][1])

print("\nPerformans Metrikleri")
y_pred = model.predict(X_test)
print("ROC AUC Skoru:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Precision Skoru:", precision_score(y_test, y_pred))
print("Recall Skoru (Sensitivity):", recall_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix Isı Haritası')
plt.show()
