import numpy as np
from read_arff import read_arff
from multirotulo_to_monorotulo import make_monorotulo

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

class NaiveBayesH:
    def __init__(self, hierarquia, alpha=1):
        self.classes = self.gera_hierarquia_completa(hierarquia)
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_features -= 1
        self.descendentes = self.gera_descendentes(X, y) 
        self.ancestrais = gera_ancestrais(self.classes)
        # Lista com um dicionário para cada feature.
        # Nesse dicionário, teremos uma chave para cada classe, sendo os valores mais outros dicionários com os valores do atributo.
        self.feature_probs = [{} for _ in range(n_features)]
        self.prior_prob = {classe: 0 for classe in self.classes}
        # Número de valores por atributo, utilizado no LaPlace Smoothing
        self.n_values_per_att = {}
        for feature_idx in range(n_features):
            self.n_values_per_att[feature_idx] = len(np.unique(X[:, feature_idx]))
        # Número de ocorrências de classe, utilizado no LaPlace Smoothing
        self.n_class_occurances = {}
        classe, counts = np.unique(y, return_counts=True)
        for c, count in zip(classe, counts):
            self.n_class_occurances[c] = count
        for idx, c in enumerate(self.classes):
            # Seleciona as instâncias da classe atual
            X_c = X[y == c]
            X_c = X_c[:, :-1]
            n_instancias_classe_c = X_c.shape[0]
            # Para cada feature, seleciona as contagens de cada valor nessa feature.
            for feature_idx in range(n_features):
                feature_vals, counts = np.unique(X_c[:, feature_idx], return_counts=True)
                n_valores_unicos_feat = len(np.unique(X_c[:, feature_idx]))

                # Armazena em um dicionário, pares valor do atributo e sua probabilidade (contagem do atributo / instâncias da classe c)
                feature_prob = {val: (count + self.alpha) / (n_instancias_classe_c + self.alpha * n_valores_unicos_feat) for val, count in zip(feature_vals, counts)}
                # print(feature_prob)
                # return
                self.feature_probs[feature_idx][c] = feature_prob
            # Soma a contagem da classe em seus ancestrais e em si mesma
            for classe in self.ancestrais[c]:
                self.prior_prob[classe] += n_instancias_classe_c + self.alpha
        # Normaliza as probabilidades à priori, usando a formula de LaPlace
        for classe in self.prior_prob:
            self.prior_prob[classe] /= (n_samples + (self.alpha * len(self.classes)))
        # print(self.prior_prob)
        # print(self.feature_probs)

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
                prior = np.log(self.prior_prob[c])
                
                likelihood = 0
                # para cada feature
                for feature_idx, feature_val in enumerate(x):
                    feature_val = feature_val
                    # print(feature_val)
                    # Pega a probabilidade de feature, dado a classe e seu valor.
                    if feature_val in self.feature_probs[feature_idx][c]:
                        likelihood += np.log(self.feature_probs[feature_idx][c][feature_val])
                    else:
                        # Usa Laplace para estimar o valor
                        nvals = float(self.n_values_per_att[feature_idx])
                        likelihood += np.log(self.alpha / (self.n_class_occurances.get(c, 0) + nvals*self.alpha))
                # print(likelihood)

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
        return descendentes
    
    def calculate_usefullness(self):
        max_tree_size = max([len(value) for key, value in self.descendentes.items()])
        usefullness = []
        for c in self.classes:
            tree_size_i = len(self.descendentes[c])
            usefullness_i = 1 - (np.log2(tree_size_i) / max_tree_size)
            usefullness.append(usefullness_i)
        return usefullness

    def gera_hierarquia_completa(self, hierarquia):
        hierarquia_completa = []
        for classe in hierarquia:
            for _ in range(len(classe.split('.'))):
                if classe not in hierarquia_completa:
                    hierarquia_completa.append(classe)
                    classe = classe.split('.')[:-1]
                    if len(classe) == 1: # Sobrou so o R
                        break
                    classe = '.'.join(classe)
        return hierarquia_completa