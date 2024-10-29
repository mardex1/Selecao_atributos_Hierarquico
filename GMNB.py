import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from read_arff import read_arff
from multirotulo_to_monorotulo import make_monorotulo
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB

def gera_ancestrais(classes):
        ancestrais = {}
        for c1 in classes:
            for c2 in classes:
                if c2 in c1 and c1 not in ancestrais:
                    ancestrais[c1] = [c2]
                elif c2 in c1 and c1 in ancestrais:
                    ancestrais[c1].append(c2)
        return ancestrais

def f_measure_hierarquica(predictions, y_true, classes):
    ancestrais = gera_ancestrais(classes)
    f_measure = 0
    numerador = 0
    denominador_precision = 0
    denominador_recall = 0
    for classe_predita, classe_verdadeira in zip(predictions, y_true):
        ancestrais_classe_predita = ancestrais[classe_predita]
        ancestrais_classe_verdadeira = ancestrais[classe_verdadeira]
        classes_comum = 0
        for c1 in ancestrais_classe_predita:
            for c2 in ancestrais_classe_verdadeira:
                if c1 == c2:
                    classes_comum += 1
        numerador += classes_comum
        denominador_precision += len(ancestrais_classe_predita)
        denominador_recall += len(ancestrais_classe_verdadeira)
    hierarchical_precision = numerador / denominador_precision
    hierarchical_recall = numerador / denominador_recall
    f_measure = (2 * hierarchical_precision * hierarchical_recall) / (hierarchical_precision + hierarchical_recall)
    return f_measure
        

    pass
class NaiveBayesH:
    def fit(self, X, y, hierarquia=None):
        n_samples, n_features = X.shape
        # self.classes = self.gera_hierarquia_completa(hierarquia)
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.descendentes = self.gera_descendentes(X, y) 
        self.ancestrais = gera_ancestrais(classes)
        
        # Lista com um dicionário para cada feature.
        # Nesse dicionário, teremos uma chave para cada classe, sendo os valores mais outros dicionários com os valores do atributo.
        self.feature_probs = [{} for _ in range(n_features)]
        self.prior_prob = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            # Seleciona as instâncias da classe atual.
            print(c)
            X_c = X[np.isin(y, self.ancestrais[c])].astype(np.float64)
            print(f'hierarquico: {len(X_c)}')
            X_plano = X[y == c]
            print(f'plano: {len(X_plano)}')

            n_instancias_classe_c = X_c.shape[0]
            # Para cada feature, seleciona as contagens de cada valor nessa feature.
            for feature_idx in range(n_features):
                feature_vals, counts = np.unique(X_c[:, feature_idx], return_counts=True)
                # Armazena em um dicionário, pares valor do atributo e sua probabilidade (contagem do atributo / instâncias da classe c)
                feature_prob = {val: count / n_instancias_classe_c for val, count in zip(feature_vals, counts)}
                self.feature_probs[feature_idx][c] = feature_prob
            self.prior_prob[idx] = n_instancias_classe_c / float(n_samples)

    def predict(self, X_test, usefullness=False):
        if usefullness:
            usefullness_list = self.calculate_usefullness()
        X_test = X_test.astype(np.float64)
        predictions = []
        # Para cada instância que deseja predizer
        for x in X_test:
            posteriors = []
            # Para cada classe
            for idx, c in enumerate(self.classes):
                # log(P(c)) + log(P(x1|c)) + log(P(x2|c)) + log(P(x3|c)) + log(P(x4|c))
                # Pega a probabilidade a priori da classe
                prior = np.log(self.prior_prob[idx])
                likelihood = 0
                # para cada feature
                for feature_idx, feature_val in enumerate(x):
                    print(feature_idx, feature_val)
                    # Get the feature probability given the class
                    if feature_val in self.feature_probs[feature_idx][c]:
                        likelihood += np.log(self.feature_probs[feature_idx][c][feature_val])
                    else:
                        # Handle unseen features (e.g., with Laplace smoothing, assume very small probability)
                        likelihood += np.log(1e-9)
                posterior = prior + likelihood
                if usefullness:
                    posterior += np.log(usefullness_list[idx])
                posteriors.append(posterior)
            # np.argmax retorna o índice que maximiza o array "posteriors"
            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)

    def gera_descendentes(self, X, y):
        descendentes = {}
        for c1 in self.classes:
            for c2 in self.classes:
                if c2 in c1 and c2 not in descendentes:
                    descendentes[c2] = [c1]
                elif c2 in c1 and c2 in descendentes:
                    descendentes[c2].append(c1)
        print(descendentes)
        return descendentes
    
    def calculate_usefullness(self):
        max_tree_size = max([len(value) for key, value in self.descendentes.items()])
        usefullness = []
        for idx, c in enumerate(self.classes):
            tree_size_i = len(self.descendentes[c])
            usefullness_i = 1 - (np.log2(tree_size_i) / max_tree_size)
            usefullness.append(usefullness_i)
        return usefullness

    def gera_hierarquia_completa(self, hierarquia):
        hierarquia_completa = []
        for classe in hierarquia:
            for _ in range(len(classe.split('/'))):
                if classe not in hierarquia_completa:
                    hierarquia_completa.append(classe)
                    classe = classe.split('/')[:-1]
                    print(classe)
                    classe = '/'.join(classe)
        return hierarquia_completa


if __name__ == "__main__":
    data, hier, cols = read_arff('Datasets/processados/cellcycle_discretizado.arff')
    print(data)
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
    classes = np.unique(y)


    model = NaiveBayesH()
    model.fit(X, y)
    predictions = model.predict(X, usefullness=True)
    print("Accuracy: ", sum(predictions == y) / len(y))
    f1_hier = f_measure_hierarquica(predictions, y, classes)
    print('F-measure-hierarquica: ', f1_hier)

    model_sk = CategoricalNB()
    model_sk.fit(X, y)
    predictions = model_sk.predict(X)
    print(model_sk.get_params())
    print("Accuracy: ", sum(predictions == y) / len(y))
    f1_hier = f_measure_hierarquica(predictions, y, classes)
    print('F-measure-hierarquica: ', f1_hier)


    # Xtr, Xts, y, yts = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # print(X)
    # print(y)


    # model = NaiveBayes()
    # model.fit(X, y)
    # predictions = model.predict(X)
    # print(predictions)
    # print(y)
