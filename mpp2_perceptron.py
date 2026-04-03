from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import random

iris = datasets.load_iris()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["class_name"] = df["target"].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

df.groupby("class_name").head(3)

X_all = iris.data
y_all = np.where(iris.target == 0, 1, 0)


X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.34,
    stratify=y_all,
    random_state=42
)

print(f"Number of features: {X_all.shape[1]}")
print(f"Train set: {X_train.shape}")
print(f"Setosa: {(y_train == 1).sum()}, Other: {(y_train == 0).sum()}")
print(f"Test set: {X_test.shape}")
print(f"Setosa: {(y_test == 1).sum()}, Other: {(y_test == 0).sum()}")

class Perceptron:
    def __init__(self, n_features, lr=0.1):
        self.weight = [random.random() for _ in range(n_features)]
        self.bias = random.random()
        self.lr = lr

    def predict(self, x):
        s = 0
        for i in range(len(self.weight)):
            s += self.weight[i] * x[i]
        s += self.bias

        return 1 if s >= 0 else 0

    def update(self, x, y):
        y_pred = self.predict(x)
        error = y - y_pred

        for i in range(len(self.weight)):
            self.weight[i] += self.lr * error * x[i]

        self.bias += self.lr * error


class Trainer:
    def train_arrays(self, model, X_train, y_train, X_test, y_test, epochs):
        acc_list = []

        for _ in range(epochs):
            for x, y in zip(X_train, y_train):
                model.update(x, y)

            acc = self.evaluate_arrays(model, X_test, y_test)
            acc_list.append(acc)

        return acc_list

    def evaluate_arrays(self, model, X, y):
        correct = 0
        for xi, yi in zip(X, y):
            if model.predict(xi) == yi:
                correct += 1
        return correct / len(X)


class UI:
    def run(self):
        trainer = Trainer()

        n_features = X_train.shape[1]
        model = Perceptron(n_features)

        acc_list = trainer.train_arrays(model, X_train, y_train, X_test, y_test, epochs=10)

        print("Final Accuracy:", acc_list[-1])

        plt.plot(acc_list)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per epoch")
        plt.show()

        print(f"Enter {n_features} features separated by space:")
        x = list(map(float, input().split()))

        if len(x) != n_features:
            print("Invalid")
        else:
            print("Prediction:", model.predict(x))

if __name__ == "__main__":
    UI().run()