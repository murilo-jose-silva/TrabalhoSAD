# Gold Forecast — projeto pronto para teste

Este ZIP contém o código, o notebook Colab, o dashboard Streamlit e um CSV de demonstração para você conseguir executar tudo imediatamente.

> **ATENÇÃO:** `data/gold_price.csv` contém DADOS SINTÉTICOS (gerados artificialmente), apenas para testar o funcionamento do código. Não use os resultados desse CSV como resultados da atividade. Para a entrega acadêmica, substitua-o pelo dataset real escolhido no Kaggle e treine novamente.

## 1. Testar no computador (Windows)

1. Extraia o ZIP.
2. Abra a pasta `gold-forecast` no VS Code ou terminal.
3. Instale Python 3.10 ou superior.
4. Crie e ative um ambiente virtual:

```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
```

5. Instale as bibliotecas:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Treine com o CSV demonstrativo:

```powershell
python train_model.py --csv data/gold_price.csv
```

7. Abra o dashboard:

```powershell
streamlit run app.py
```

O Streamlit abrirá no navegador, normalmente em http://localhost:8501.

## 2. Testar no Google Colab

1. Abra `notebooks/gold_forecast.ipynb` no Colab.
2. Execute as células na ordem.
3. Quando solicitado, faça upload de um CSV com colunas de data e preço.
4. Confira os nomes das colunas e ajuste `COLUNA_DATA` e `COLUNA_PRECO`.
5. Execute até o final para treinar, avaliar e exportar o pacote do modelo.

O notebook foi preparado para o trabalho colaborativo no Colab. O treinamento local pelo `train_model.py` é a alternativa mais simples para testar o Streamlit.

## 3. Usar o dataset real

- Baixe o dataset escolhido no Kaggle.
- Confira as colunas com `pd.read_csv(...).head()` e `df.columns`.
- Se necessário, passe os nomes corretos:

```powershell
python train_model.py --csv data/gold_price.csv --date-col Date --price-col Price
```

Substitua `Date` e `Price` pelos nomes reais do seu arquivo. Salve o CSV real como `data/gold_price.csv` (sobrescrevendo o demonstrativo) ou informe outro caminho no comando.

Fonte sugerida para pesquisa: https://www.kaggle.com/datasets/lbronchal/gold-and-silver-prices-dataset

## 4. O que o projeto faz

- Limpa datas e preços, remove duplicatas e ordena cronologicamente.
- Cria variáveis defasadas (lags), médias móveis, volatilidade e variação percentual.
- Define como alvo o próximo preço observado.
- Separa os dados cronologicamente: 80% treino e 20% teste.
- Compara Decision Tree, Random Forest e HistGradientBoosting com `TimeSeriesSplit`.
- Seleciona o menor MAE médio na validação temporal.
- Avalia no teste com MAE, MSE, RMSE e R².
- Compara com baseline ingênuo (próximo preço igual ao último observado).
- Salva o modelo e os resultados na pasta `models/`.
- Exibe histórico, previsão, métricas e comparação no Streamlit.

## 5. Estrutura

```text
gold-forecast/
├── data/
│   └── gold_price.csv          # sintético para teste; substituir pelo real
├── models/                     # criado/preenchido após treinamento
├── notebooks/
│   └── gold_forecast.ipynb
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Interpretação e limitações

MAE, MSE e RMSE são medidas de erro (menor é melhor). R² mais próximo de 1 tende a indicar melhor ajuste, mas pode ser negativo. Compare sempre com o baseline: se o modelo não o superar, não afirme que trouxe ganho preditivo.

A avaliação no teste é de um passo à frente, usando observações anteriores já conhecidas. O dashboard faz previsão recursiva de vários passos, portanto a incerteza tende a crescer com o horizonte. As datas futuras são dias úteis de segunda a sexta, sem calendário específico de feriados.

O modelo usa apenas o histórico de preço. Não incorpora câmbio, juros, inflação, geopolítica ou outras variáveis externas. É um projeto acadêmico, não uma recomendação de investimento.

## 7. Publicar no GitHub

Antes de publicar, decidam se o dataset pode ser redistribuído. Se não puder, retirem `data/gold_price.csv` do repositório e documentem no README como obtê-lo. Não publiquem resultados obtidos com a série sintética como se fossem resultados reais.
