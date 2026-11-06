from pathlib import Path
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "preco_atual", "lag_1", "lag_2", "lag_3", "lag_5", "lag_7",
    "lag_14", "lag_30", "media_7", "media_30",
    "volatilidade_7", "variacao_1"
]

def detectar_coluna(colunas, candidatos):
    mapa = {str(c).strip().lower(): c for c in colunas}
    for nome in candidatos:
        if nome.lower() in mapa:
            return mapa[nome.lower()]
    return None

def carregar_dados(caminho, coluna_data=None, coluna_preco=None):
    raw = pd.read_csv(caminho)
    print("Colunas encontradas:", raw.columns.tolist())

    if coluna_data is None:
        coluna_data = detectar_coluna(
            raw.columns, ["Date", "date", "datetime", "timestamp", "observation_date"]
        )
    if coluna_data is None:
        raise ValueError("Não identifiquei a coluna de data. Informe coluna_data no código.")

    if coluna_preco is None:
        coluna_preco = detectar_coluna(
            raw.columns,
            ["Price", "Close", "USD (PM)", "USD PM", "USD", "Gold Price", "gold_price"]
        )

    if coluna_preco is None:
        numericas = raw.select_dtypes(include=np.number).columns.tolist()
        numericas = [c for c in numericas if c != coluna_data]
        print("Colunas numéricas disponíveis:", numericas)
        raise ValueError(
            "Não identifiquei automaticamente a coluna do preço. "
            "Informe coluna_preco com uma das colunas numéricas listadas."
        )

    dados = raw[[coluna_data, coluna_preco]].copy()
    dados.columns = ["Date", "Price"]
    dados["Date"] = pd.to_datetime(dados["Date"], errors="coerce")
    dados["Price"] = pd.to_numeric(dados["Price"], errors="coerce")
    dados = dados.dropna(subset=["Date", "Price"])
    dados = dados.drop_duplicates(subset=["Date"], keep="last")
    dados = dados.sort_values("Date").reset_index(drop=True)

    if (dados["Price"] <= 0).any():
        warnings.warn("Há preços menores ou iguais a zero; removendo essas observações.")
        dados = dados[dados["Price"] > 0].copy()

    if len(dados) < 100:
        raise ValueError("Poucas observações após limpeza. Verifique o arquivo e as colunas.")

    return dados, str(coluna_data), str(coluna_preco)

def criar_features(dados):
    d = dados.copy()
    p = d["Price"]
    d["preco_atual"] = p
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        d[f"lag_{lag}"] = p.shift(lag)
    d["media_7"] = p.rolling(7).mean()
    d["media_30"] = p.rolling(30).mean()
    d["volatilidade_7"] = p.rolling(7).std()
    d["variacao_1"] = p.pct_change()
    d["target"] = p.shift(-1)
    d["Target_Date"] = d["Date"].shift(-1)
    return d.dropna().reset_index(drop=True)

def calcular_metricas(y_real, y_previsto):
    return {
        "MAE": float(mean_absolute_error(y_real, y_previsto)),
        "MSE": float(mean_squared_error(y_real, y_previsto)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "R2": float(r2_score(y_real, y_previsto))
    }

def construir_modelos():
    return {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=5, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        )
    }

def construir_features_futuras(historico):
    v = list(map(float, historico))
    if len(v) < 31:
        raise ValueError("São necessárias pelo menos 31 observações para prever.")
    atual = v[-1]
    anterior = v[-2]
    linha = {
        "preco_atual": atual,
        "lag_1": v[-2],
        "lag_2": v[-3],
        "lag_3": v[-4],
        "lag_5": v[-6],
        "lag_7": v[-8],
        "lag_14": v[-15],
        "lag_30": v[-31],
        "media_7": float(np.mean(v[-7:])),
        "media_30": float(np.mean(v[-30:])),
        "volatilidade_7": float(np.std(v[-7:], ddof=1)),
        "variacao_1": float(atual / anterior - 1) if anterior else 0.0
    }
    return pd.DataFrame([linha], columns=FEATURES)

def prever_horizonte(modelo, historico, datas, horizonte):
    valores = list(map(float, historico))
    previsoes = []
    for data in datas[:horizonte]:
        x = construir_features_futuras(valores)
        pred = float(modelo.predict(x)[0])
        previsoes.append({"Date": data, "Predicted_Price": pred})
        valores.append(pred)
    return pd.DataFrame(previsoes)

def treinar(caminho_csv, coluna_data=None, coluna_preco=None, pasta_modelos="models"):
    pasta = Path(pasta_modelos)
    pasta.mkdir(parents=True, exist_ok=True)

    historico, nome_data, nome_preco = carregar_dados(
        caminho_csv, coluna_data, coluna_preco
    )
    dados = criar_features(historico)

    corte = int(len(dados) * 0.80)
    treino = dados.iloc[:corte].copy()
    teste = dados.iloc[corte:].copy()

    X_treino, y_treino = treino[FEATURES], treino["target"]
    X_teste, y_teste = teste[FEATURES], teste["target"]

    # Validação temporal somente dentro do conjunto de treino.
    n_splits = 5 if len(treino) >= 500 else 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    resultados_cv = []
    modelos = construir_modelos()

    for nome, modelo in modelos.items():
        scores = cross_val_score(
            modelo, X_treino, y_treino,
            cv=tscv, scoring="neg_mean_absolute_error", n_jobs=None
        )
        resultados_cv.append({
            "Modelo": nome,
            "MAE_CV": float(-scores.mean()),
            "Desvio_CV": float(scores.std())
        })

    cv_df = pd.DataFrame(resultados_cv).sort_values("MAE_CV")
    melhor_nome = cv_df.iloc[0]["Modelo"]
    melhor_modelo = modelos[melhor_nome]
    melhor_modelo.fit(X_treino, y_treino)

    previsoes = melhor_modelo.predict(X_teste)
    metricas = calcular_metricas(y_teste, previsoes)

    # Baseline ingênuo: amanhã será igual ao último preço conhecido.
    baseline = teste["preco_atual"].to_numpy()
    metricas_baseline = calcular_metricas(y_teste, baseline)

    comparacao = pd.DataFrame([
        {"Modelo": melhor_nome, **metricas},
        {"Modelo": "Baseline (último preço)", **metricas_baseline}
    ])

    pred_teste = pd.DataFrame({
        "Date": teste["Target_Date"].to_numpy(),
        "Real": y_teste.to_numpy(),
        "Predicted": previsoes
    })

    bundle = {
        "model": melhor_modelo,
        "model_name": melhor_nome,
        "features": FEATURES,
        "history": historico,
        "price_column_original": nome_preco,
        "date_column_original": nome_data,
        "metrics": metricas,
        "baseline_metrics": metricas_baseline,
        "cv_results": cv_df,
        "test_predictions": pred_teste,
        "training_cutoff": str(treino["Date"].max().date()),
        "test_start": str(teste["Date"].min().date()),
        "test_end": str(teste["Date"].max().date())
    }

    joblib.dump(bundle, pasta / "gold_forecast_bundle.joblib")
    cv_df.to_csv(pasta / "comparacao_validacao_temporal.csv", index=False)
    comparacao.to_csv(pasta / "metricas_teste.csv", index=False)
    pred_teste.to_csv(pasta / "previsoes_teste.csv", index=False)
    historico.to_csv(pasta / "historico_limpo.csv", index=False)

    print("\nValidação temporal (menor MAE é melhor):")
    print(cv_df.to_string(index=False))
    print("\nAvaliação no teste:")
    print(comparacao.to_string(index=False))
    print("\nModelo selecionado:", melhor_nome)
    print("Arquivos salvos em:", pasta.resolve())

    return bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/gold_price.csv")
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--price-col", default=None)
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    treinar(args.csv, args.date_col, args.price_col, args.models_dir)