import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from read_arff import read_arff
from sklearn.naive_bayes import GaussianNB

class NaiveBayesH:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.ancestrais = self.gera_dict_ancestrais(X, y) 

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.std = np.zeros((n_classes, n_features), dtype=np.float64)
        self.prior_prob = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            # Seleciona as instâncias da classe atual.
            X_c = X[np.isin(y, self.ancestrais[c])].astype(np.float64)
            print(f'hierarquico: {len(X_c)}')
            X_plano = X[y == c]
            print(f'plano: {len(X_plano)}')
            
            n_instancias_classe_c = X_c.shape[0]
            # Cálculo a média, variância e probabilidade a priori (probabilidade de cada classe na base de dados)
            self.mean[idx, :] = X_c.mean(axis=0)
            self.std[idx, :] = X_c.std(axis=0)
            self.prior_prob[idx] = n_instancias_classe_c / float(n_samples)
        

    def gaussian_pdf(self, class_idx, x):
        # Função que gera a curva normal
        mean = self.mean[class_idx, :]
        std = self.std[class_idx, :]
        return 1 / np.sqrt(2 * np.pi * std**2+1e-9) * np.exp(-(x - mean)**2/(2*std**2+1e-9))

    def predict(self, X_test, usefullness=False):
        if usefullness:
            usefullness_list = self.calculate_usefullness()
        X_test = X_test.astype(np.float64)
        predictions = []
        for x in X_test:
            posteriors = []
            for idx, c in enumerate(self.classes):
                
                # log(P(c)) + log(P(x1|c)) + log(P(x2|c)) + log(P(x3|c)) + log(P(x4|c))
                prior = np.log(self.prior_prob[idx])
                posterior = np.sum(np.log(self.gaussian_pdf(idx, x)))
                posterior += prior
                if usefullness:
                    posterior += np.log(usefullness_list[idx])
                posteriors.append(posterior)
            # np.argmax retorna o índice que maximiza o array "posteriors"
            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)

    def gera_dict_ancestrais(self, X, y):
        ancestrais = {}
        for c1 in self.classes:
            for c2 in self.classes:
                if c2 in c1 and c1 not in ancestrais:
                    ancestrais[c1] = [c2]
                elif c2 in c1 and c1 in ancestrais:
                    ancestrais[c1].append(c2)
        print(ancestrais)
        return ancestrais
    
    def calculate_usefullness(self):
        max_tree_size = max([len(value) for key, value in self.ancestrais.items()])
        usefullness = []
        for idx, c in enumerate(self.classes):
            tree_size_i = len(self.ancestrais[c])
            usefullness_i = 1 - (np.log2(tree_size_i) / max_tree_size)
            usefullness.append(usefullness_i)
        return usefullness

if __name__ == "__main__":
    data, hierarquia, columns = read_arff('Datasets/processados/cellcycle_sem_valores_ausentes.arff')

    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    # data = np.array([[0.12, 2, "R.1.1"],
    # [0.15, 3, "R.1.1"],
    # [0.24, 4, "R.2"],
    # [0.34, 5, "R.2.1"],
    # [2.26, 6, "R.1.1"],
    # [3.50, 7, "R.2.1"],
    # [3.50, 8, "R.2"],
    # [3.67, 9, "R.1.1.1"],
    # [4.10, 10, "R.2.1"],
    # [4.12, 11,"R.1.2"],
    # [5.17, 12,"R.1.1.1"],
    # [5.43, 13,"R.2.1"],
    # [6.50, 14,"R.1.2"],
    # [6.73, 15,"R.1.2"]])
    # X = data[:, :-1].astype(np.float64)
    # y = data[:, -1]

    model = NaiveBayesH()
    model.fit(X, y)
    predictions = model.predict(X, usefullness=True)
    print(sum(predictions == y) / len(y))

    model_sk = GaussianNB()
    model_sk.fit(X, y)
    predictions = model_sk.predict(X)
    print(model_sk.get_params())
    print(sum(predictions == y) / len(y))


    # Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # print(X)
    # print(y)


    # model = NaiveBayes()
    # model.fit(X, y)
    # predictions = model.predict(X)
    # print(predictions)
    # print(y)
