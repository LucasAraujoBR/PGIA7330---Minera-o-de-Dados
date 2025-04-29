# 📑 Relatório Técnico — Classificação de Questões ENEM

## 1. Objetivo

Desenvolver um modelo de Machine Learning para classificar questões do ENEM Digital em duas categorias:
- `1` — Ciências Humanas e suas Tecnologias.
- `0` — Demais áreas do conhecimento.

O projeto envolveu criação de embeddings de texto, otimização de hiperparâmetros e construção de um super ensemble utilizando Stacking.

---

## 2. Código de Treinamento

### 2.1. Carregamento dos Dados

- Carregou-se a base `base_treino.csv` contendo as questões e alternativas.
- As colunas relevantes são: `ENUNCIADO_QUESTAO`, `ALTERNATIVA_A`, `ALTERNATIVA_B`, `ALTERNATIVA_C`, `ALTERNATIVA_D`, `ALTERNATIVA_E` e `LABEL_QUESTAO`.

### 2.2. Engenharia de Texto

- Para cada questão, criou-se uma nova coluna `TEXTO_FINAL` combinando:
  - O enunciado da questão (com peso maior, duplicado para dar mais importância).
  - Todas as alternativas concatenadas.

Objetivo: Criar um texto completo representando cada questão.

### 2.3. Geração de Embeddings

- Utilizou-se o modelo **Sentence-BERT** (`paraphrase-MiniLM-L6-v2`) para gerar representações vetoriais de cada `TEXTO_FINAL`.
- O Sentence-BERT foi escolhido por capturar o **significado semântico** dos textos, o que é superior a métodos clássicos como TF-IDF.

### 2.4. Separação dos Dados

- Divisão do conjunto de dados em **75% treino** e **25% teste**, estratificando a variável alvo para balanceamento.

### 2.5. Modelos Base

- Três modelos foram usados como base:
  - **Logistic Regression**.
  - **Random Forest Classifier**.
  - **XGBoost Classifier**.

### 2.6. Otimização com GridSearchCV

- Para cada modelo base, aplicou-se **GridSearchCV** para encontrar os melhores hiperparâmetros.
- As buscas testaram diferentes combinações de parâmetros como:
  - `C` para Logistic Regression.
  - `n_estimators` e `max_depth` para Random Forest e XGBoost.

### 2.7. Criação do StackingClassifier

- Em vez de um VotingClassifier, utilizou-se o **StackingClassifier**.
- Estrutura:
  - **Modelos base**: Logistic Regression, Random Forest e XGBoost (todos tunados).
  - **Meta-modelo final**: Logistic Regression.

O StackingClassifier usa as previsões dos modelos base como entrada para o meta-modelo, criando um ensemble mais inteligente.

### 2.8. Validação Cruzada

- Foi aplicada **validação cruzada estratificada** com 5 folds (`StratifiedKFold`) durante o treinamento do Stacking.
- A média e desvio padrão da acurácia foram impressos.

### 2.9. Treinamento Final e Avaliação

- O modelo Stacking foi treinado no conjunto de treino completo.
- Avaliação no conjunto de teste:
  - Impressão do **relatório de classificação** (precision, recall, f1-score).
  - Exibição da **matriz de confusão** com o Seaborn.

### 2.10. Salvamento do Modelo

- Foi salvo um arquivo `stacking_model.pkl`, contendo:
  - O modelo treinado.
  - O Sentence-BERT usado para geração dos embeddings.

---

## 3. Código de Teste

### 3.1. Carregamento do Modelo e Dados

- O arquivo `stacking_model.pkl` foi carregado utilizando o `joblib`.
- O novo arquivo `test.csv` foi lido, contendo as questões a serem classificadas.

### 3.2. Engenharia de Texto

- Aplicou-se a mesma função de combinação de enunciado + alternativas (`TEXTO_FINAL`) usada no treinamento.

### 3.3. Geração de Embeddings no Teste

- Gerou-se os embeddings do `TEXTO_FINAL` usando o mesmo Sentence-BERT carregado.

### 3.4. Predição

- O modelo Stacking realizou a predição das classes (`0` ou `1`) para o conjunto de teste.

### 3.5. Geração do Arquivo de Submissão

- Um arquivo `submissao.csv` foi gerado com duas colunas:
  - `id` — ID da questão.
  - `label` — Previsão do modelo.

O arquivo foi exportado sem índice, no formato solicitado.

---

# 📚 Resumo Técnico

| Etapa | Técnica Aplicada |
|:---|:---|
| Representação dos textos | Embeddings com Sentence-BERT |
| Algoritmos base | Logistic Regression, Random Forest, XGBoost |
| Ensemble | StackingClassifier |
| Otimização | GridSearchCV para cada modelo |
| Validação | Stratified 5-Fold Cross-validation |
| Salvar modelo | `stacking_model.pkl` (modelo + embeddings) |
| Teste | Geração de predições e submissão em CSV |

---

# ✅ Resultado

O projeto resultou em um pipeline robusto, usando técnicas modernas de NLP e Ensemble Learning para classificação de questões do ENEM Digital.

---