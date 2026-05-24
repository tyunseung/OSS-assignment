from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#데이터 불러오기
wine = datasets.load_wine()
X = wine.data
y = wine.target

#분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

MLP = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', max_iter=500, random_state=42)

#학습
MLP.fit(X_train, y_train)

#예측
pred = MLP.predict(X_test)

#정답률
acc = accuracy_score(y_test, pred)

print(acc)