import numpy as np
import re

class Dataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def text_to_vector(self, text):
        text = re.sub('[^a-z]', '', text.lower())
        vec = np.zeros(26)
        for ch in text:
            idx = ord(ch) - ord('a')
            if 0 <= idx < 26:
                vec[idx] += 1
        if np.sum(vec) != 0:
            vec = vec / np.sum(vec)
        return vec

    def get_vectors(self):
        X = np.array([self.text_to_vector(t) for t in self.texts])
        y = np.array(self.labels)
        return X, y

    def train_test_split(self, test_ratio=0.2):
        X, y = self.get_vectors()
        idx = np.random.permutation(len(X))
        split = int(len(X) * (1 - test_ratio))
        return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]

class Perceptron:
    def __init__(self, n_features, lr=0.01, epochs=3000):
        self.w = np.zeros(n_features)
        self.b = 0
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        return self.activation(np.dot(self.w, x) + self.b)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred

                self.w += self.lr * error * xi
                self.b += self.lr * error

class MultiClassPerceptron:
    def __init__(self, n_classes, n_features):
        self.perceptrons = [Perceptron(n_features) for _ in range(n_classes)]

    def fit(self, X, y):
        for i, perceptron in enumerate(self.perceptrons):
            y_binary = np.where(y == i, 1, 0)
            perceptron.fit(X, y_binary)  #each perceptron learn their language

    def predict(self, x):
        scores = [np.dot(p.w, x) + p.b for p in self.perceptrons]
        return np.argmax(scores)

class Metrics:
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def precision(y_true, y_pred, true_labels):
        true_positive = np.sum((y_pred == true_labels) & (y_true == true_labels))
        false_p = np.sum((y_pred == true_labels) & (y_true != true_labels))
        return true_positive / (true_positive + false_p) if (true_positive + false_p) != 0 else 0

class UI:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
    def classify_text(self, text):
        vec = self.dataset.text_to_vector(text)
        return self.model.predict(vec)

if __name__ == "__main__":
    texts = [
        "this is an english text",
        "another english sentence",
        "hello how are you",
        "the weather is nice today",
        "today is not friday",
        "that is for train",
        "im tired",
        "birds fly south during the cold winter months",
        "the recipe calls for flour sugar and butter",
        "this is a simple test",
        "i like learning new things",
        "the sun is shining today",
        "we are building a model",
        "data science is useful",
        "the cat is sleeping on the sofa",
        "i drink coffee every morning",
        "this program runs very fast",
        "we are solving a new problem",
        "the sky looks very blue today",
        "i am practicing my english skills",
        "this solution is quite effective",
        "we need to improve the model",
        "he is working on a project",
        "she likes to read novels",
        "the computer is very powerful",
        "this task is not difficult",
        "we are testing the algorithm",
        "i enjoy writing code",
        "the results are very accurate",
        "this method works better",
        "the car is parked outside",
        "i am watching a movie tonight",
        "this code needs optimization",
        "we are learning new concepts",
        "the book is on the table",
        "she is cooking dinner now",
        "the internet connection is slow",
        "i am fixing a small bug",
        "this example explains everything",
        "we are working as a team"

        "bu polyak dilində bir mətndir",
        "polyak dilində başqa bir cümlə",
        "necəsən",
        "bu gün hava gözəldir",
        "bu gün cümə deyil",
        "bu qatar üçündür",
        "mən yorğunam",
        "quşlar qışdan əvvəl cənuba uçur",
        "resept un şəkər və yağ tələb edir",
        "bu sadə bir testdir",
        "mən yeni şeylər öyrənməyi sevirəm",
        "bu gün günəş parlayır",
        "biz yaxi qururuq",
        "bu faydalıdır",
        "pişik divanda yatır",
        "hər səhər qəhvə içirəm",
        "bu proqram çox sürətli işləyir",
        "biz yeni bir problemi həll edirik",
        "bu gün səma çox mavidir",
        "mən ingilis dili bacarıqlarımı inkişaf etdirirəm",
        "bu həll çox effektivdir",
        "biz modeli yaxşılaşdırmalıyıq",
        "o bir layihə üzərində işləyir",
        "o roman oxumağı sevir",
        "bu kompüter çox güclüdür",
        "bu tapşırıq çətin deyil",
        "biz alqoritmi test edirik",
        "mən təmiz kod yazmağı sevirəm",
        "nəticələr çox dəqiqdir",
        "bu metod daha yaxşı işləyir",
        "maşın çöldə dayanıb",
        "bu axşam izləyirəm",
        "bu mətnə optimizasiya tələb edir",
        "biz yeni anlayışlar öyrənirik",
        "kitab masanın üstündədir",
        "o indi yemək bişirir",
        "çıxış yavaşdır",
        "mən kiçik bir səhvi düzəldirəm",
        "bu nümunə hər şeyi izah edir",
        "biz komanda olaraq işləyirik"

        "acesta este un text în poloneză",
        "o altă propoziție în poloneză",
        "ce mai faci",
        "astăzi vremea este frumoasă",
        "astăzi nu este vineri",
        "acesta este pentru tren",
        "sunt obosit",
        "păsările zboară spre sud înainte de iarnă",
        "rețeta necesită făină zahăr și unt",
        "acesta este un test simplu",
        "îmi place să învăț lucruri noi",
        "soarele strălucește astăzi",
        "construim un model",
        "este util",
        "pisica doarme pe canapea",
        "beau cafea în fiecare dimineață",
        "acest program rulează foarte rapid",
        "rezolvăm o problemă nouă",
        "cerul este foarte albastru astăzi",
        "îmi exersez abilitățile de engleză",
        "această soluție este foarte eficientă",
        "trebuie să îmbunătățim modelul",
        "el lucrează la un proiect",
        "ea îi place să citească romane",
        "acest computer este foarte performant",
        "această sarcină nu este dificilă",
        "testăm algoritmul",
        "îmi place să scriu cod curat",
        "rezultatele sunt foarte precise",
        "această metodă funcționează mai bine",
        "mașina este parcată afară",
        "mă uit la în această seară",
        "acest cod necesită optimizare",
        "învățăm concepte noi",
        "cartea este pe masă",
        "ea gătește acum",
        "internetul este lent",
        "repar o mică eroare",
        "acest exemplu explică totul",
        "lucrăm ca o echipă"
    ]
    labels = [
        *([0] * 40),
        *([1] * 40),
        *([2] * 40)
]
    dataset = Dataset(texts, labels)
    X_train, y_train, X_test, y_test = dataset.train_test_split()

    model = MultiClassPerceptron(3, 26)
    model.fit(X_train, y_train)
    y_pred = np.array([model.predict(x) for x in X_test])

    print("Accuracy:", Metrics.accuracy(y_test, y_pred))
    print("Precision (English):", Metrics.precision(y_test, y_pred, 0))
    print("Precision (Azerbaijani):", Metrics.precision(y_test, y_pred, 1))
    print("Precision (romanian):", Metrics.precision(y_test, y_pred, 2))
    ui = UI(model, dataset)
    text = input("Enter text: ")
    pred = ui.classify_text(text)
    languages = ["English", "Azerbaijani", "romanian"]
    print("Predicted:", languages[pred])
