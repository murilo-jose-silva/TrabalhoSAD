from pathlib import Path
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# IMPORTANTE (correcao de metrica):
# A versao anterior previa o NIVEL do preco e usava features em nivel (lags,
# medias). Modelos de arvore (Random Forest) NAO extrapolam: em uma serie com
# tendencia de alta, o periodo de teste fica acima da faixa vista no treino e a
# floresta "trava" num teto, gerando R2 negativo (~ -1,06).
#
# Agora o alvo e a VARIACAO (retorno de 1 dia) e as features sao retornos /
# volatilidade / distancia da media (grandezas estacionarias, que nao saem da
# faixa de treino). O preco previsto e reconstruido:
#     preco_previsto = preco_atual * (1 + retorno_previsto)
# As metricas (MAE, MSE, RMSE, R2) sao calculadas na escala de PRECO, para
# serem comparaveis com o baseline ingenuo.
# ---------------------------------------------------------------------------

FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_7", "ret_14", "ret_30",
    "ret_media_7", "ret_media_30",
    "volatilidade_7", "volatilidade_30",
    "dist_media_30",
]


def detectar_coluna(colunas, candidatos):
    mapa = {str(c).strip().lower(): c for c in colunas}
    for nome in candidatos:
        if nome.lower() in mapa:
            return mapa[nome.lower()]
    return None


def limpar_numero(serie):
    """Converte texto de preco para numero, aceitando varios formatos:
    '1823.50' (US), '1.823,50' (BR/EU) e '1,823.50' (US com milhar)."""
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")
    s = serie.astype(str).str.strip()
    s = s.str.replace(r"[^0-9.,\-]", "", regex=True)

    def conv(x):
        if x in ("", "-", "nan", "None"):
            return np.nan
        tem_ponto = "." in x
        tem_virgula = "," in x
        if tem_ponto and tem_virgula:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")  # formato BR/EU
            else:
                x = x.replace(",", "")                     # formato US c/ milhar
        elif tem_virgula:
            x = x.replace(",", ".")                        # so virgula -> decimal
        try:
            return float(x)
        except ValueError:
            return np.nan

    return s.map(conv)


def parsear_datas(serie):
    """Converte datas de forma robusta aos dois formatos comuns:
    - ISO 'YYYY-MM-DD' (ex.: dataset do Kaggle) -> parse direto, SEM dayfirst
      (usar dayfirst aqui corromperia, trocando mes por dia).
    - 'DD/MM/YYYY' (ex.: CSV sintetico, formato BR/EU) -> dayfirst=True.
    """
    s = serie.astype(str).str.strip()
    padrao_iso = s.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    if padrao_iso.mean() > 0.8:
        return pd.to_datetime(s, errors="coerce")  # ISO: nao usar dayfirst
    return pd.to_datetime(s, errors="coerce", dayfirst=True)  # assume dia primeiro


def carregar_dados(caminho, coluna_data=None, coluna_preco=None):
    raw = pd.read_csv(caminho, sep=None, engine="python")
    print("Colunas encontradas:", raw.columns.tolist())

    if coluna_data is None:
        coluna_data = detectar_coluna(
            raw.columns, ["Date", "date", "datetime", "timestamp", "observation_date"]
        )
    if coluna_data is None:
        raise ValueError("Nao identifiquei a coluna de data. Informe coluna_data no codigo.")

    if coluna_preco is None:
        coluna_preco = detectar_coluna(
            raw.columns,
            ["Price", "Close", "USD (PM)", "USD PM", "USD", "Gold Price", "gold_price"]
        )

    if coluna_preco is None:
        numericas = raw.select_dtypes(include=np.number).columns.tolist()
        numericas = [c for c in numericas if c != coluna_data]
        print("Colunas numericas disponiveis:", numericas)
        raise ValueError(
            "Nao identifiquei automaticamente a coluna do preco. "
            "Informe coluna_preco com uma das colunas numericas listadas."
        )

    dados = raw[[coluna_data, coluna_preco]].copy()
    dados.columns = ["Date", "Price"]
    dados["Date"] = parsear_datas(dados["Date"])
    dados["Price"] = limpar_numero(dados["Price"])
    dados = dados.dropna(subset=["Date", "Price"])
    dados = dados.drop_duplicates(subset=["Date"], keep="last")
    dados = dados.sort_values("Date").reset_index(drop=True)

    if (dados["Price"] <= 0).any():
        warnings.warn("Ha precos menores ou iguais a zero; removendo essas observacoes.")
        dados = dados[dados["Price"] > 0].copy()

    if len(dados) < 100:
        raise ValueError("Poucas observacoes apos limpeza. Verifique o arquivo e as colunas.")

    return dados, str(coluna_data), str(coluna_preco)


def criar_features(dados):
    """Cria features estacionarias (retornos/volatilidade) e o alvo = retorno
    do proximo dia. Mantem preco_atual e target_preco para reconstrucao."""
    d = dados.copy()
    p = d["Price"]
    r = p.pct_change()

    # retornos defasados
    d["ret_1"] = r
    d["ret_2"] = r.shift(1)
    d["ret_3"] = r.shift(2)
    d["ret_5"] = r.shift(4)
    d["ret_7"] = r.shift(6)
    d["ret_14"] = r.shift(13)
    d["ret_30"] = r.shift(29)

    # estatisticas de retorno
    d["ret_media_7"] = r.rolling(7).mean()
    d["ret_media_30"] = r.rolling(30).mean()
    d["volatilidade_7"] = r.rolling(7).std()
    d["volatilidade_30"] = r.rolling(30).std()

    # posicao relativa a media de 30 (estacionaria)
    d["dist_media_30"] = p / p.rolling(30).mean() - 1

    # alvo = retorno do proximo dia; guardamos tambem o preco para reconstruir
    d["preco_atual"] = p
    d["target_ret"] = p.shift(-1) / p - 1
    d["target_preco"] = p.shift(-1)
    d["Target_Date"] = d["Date"].shift(-1)

    return d.dropna().reset_index(drop=True)


def calcular_metricas(y_real, y_previsto):
    return {
        "MAE": float(mean_absolute_error(y_real, y_previsto)),
        "MSE": float(mean_squared_error(y_real, y_previsto)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "R2": float(r2_score(y_real, y_previsto)),
    }


N_ARVORES = 300        # arvores do modelo final
N_ARVORES_BUSCA = 150  # arvores durante a busca por CV (mais rapido, mesma escolha)

# Usamos SOMENTE o Random Forest. O que a validacao cruzada escolhe e o nivel de
# REGULARIZACAO (profundidade da arvore e tamanho minimo da folha). Arvores
# profundas (depth alto, folha pequena) decoram ruido dos retornos (overfitting);
# arvores rasas generalizam melhor. A grade cobre de muito regularizado a pouco.
GRADE_HIPERPARAMETROS = [
    {"max_depth": d, "min_samples_leaf": l}
    for d in [3, 4, 6, 10]
    for l in [20, 50, 100]
]


def _novo_rf(n_estimators=N_ARVORES, **kw):
    return RandomForestRegressor(
        n_estimators=n_estimators, random_state=42, n_jobs=-1, **kw
    )


def construir_features_futuras(historico):
    """Monta uma linha de features (retornos) a partir da lista de precos."""
    v = list(map(float, historico))
    if len(v) < 31:
        raise ValueError("Sao necessarias pelo menos 31 observacoes para prever.")
    ret = np.diff(v) / np.array(v[:-1])  # retornos; ret[-1] = retorno mais recente

    linha = {
        "ret_1": float(ret[-1]),
        "ret_2": float(ret[-2]),
        "ret_3": float(ret[-3]),
        "ret_5": float(ret[-5]),
        "ret_7": float(ret[-7]),
        "ret_14": float(ret[-14]),
        "ret_30": float(ret[-30]),
        "ret_media_7": float(np.mean(ret[-7:])),
        "ret_media_30": float(np.mean(ret[-30:])),
        "volatilidade_7": float(np.std(ret[-7:], ddof=1)),
        "volatilidade_30": float(np.std(ret[-30:], ddof=1)),
        "dist_media_30": float(v[-1] / np.mean(v[-30:]) - 1),
    }
    return pd.DataFrame([linha], columns=FEATURES)


def prever_horizonte(modelo, historico, datas, horizonte):
    """Previsao recursiva: o modelo preve o RETORNO e reconstruimos o preco."""
    valores = list(map(float, historico))
    previsoes = []
    for data in datas[:horizonte]:
        x = construir_features_futuras(valores)
        ret_prev = float(modelo.predict(x)[0])
        preco_prev = valores[-1] * (1.0 + ret_prev)
        previsoes.append({"Date": data, "Predicted_Price": preco_prev})
        valores.append(preco_prev)
    return pd.DataFrame(previsoes)


def mae_cv_preco(dados_treino, kw, n_splits):
    """MAE medio (e desvio) na escala de PRECO por TimeSeriesSplit, para uma
    configuracao de hiperparametros. Em cada fold: ajusta no retorno, preve o
    retorno na validacao e reconstroi o preco (comparavel com o teste)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = dados_treino[FEATURES]
    y_ret = dados_treino["target_ret"].to_numpy()
    preco_atual = dados_treino["preco_atual"].to_numpy()
    preco_real = dados_treino["target_preco"].to_numpy()

    maes = []
    for idx_tr, idx_val in tscv.split(X):
        m = _novo_rf(n_estimators=N_ARVORES_BUSCA, **kw).fit(X.iloc[idx_tr], y_ret[idx_tr])
        preco_pred = preco_atual[idx_val] * (1.0 + m.predict(X.iloc[idx_val]))
        maes.append(mean_absolute_error(preco_real[idx_val], preco_pred))
    return float(np.mean(maes)), float(np.std(maes))


def selecionar_hiperparametros_cv(dados_treino, n_splits):
    """Testa a grade de regularizacao (somente Random Forest) por validacao
    cruzada temporal, usando APENAS o conjunto de treino, e escolhe a config de
    menor MAE de validacao. Retorna a melhor config e a tabela comparativa."""
    linhas = []
    for kw in GRADE_HIPERPARAMETROS:
        mae, desvio = mae_cv_preco(dados_treino, kw, n_splits)
        linhas.append({
            "Modelo": f"Random Forest (depth={kw['max_depth']}, leaf={kw['min_samples_leaf']})",
            "max_depth": kw["max_depth"],
            "min_samples_leaf": kw["min_samples_leaf"],
            "MAE_CV": mae,
            "Desvio_CV": desvio,
        })
    cv_df = pd.DataFrame(linhas).sort_values("MAE_CV").reset_index(drop=True)
    melhor = {
        "max_depth": int(cv_df.iloc[0]["max_depth"]),
        "min_samples_leaf": int(cv_df.iloc[0]["min_samples_leaf"]),
    }
    return melhor, cv_df


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

    X_treino, y_treino = treino[FEATURES], treino["target_ret"]
    X_teste = teste[FEATURES]

    # Validacao temporal (escala de preco) somente dentro do treino.
    # A CV escolhe o nivel de regularizacao do Random Forest (anti-overfitting).
    n_splits = 5 if len(treino) >= 500 else 3
    melhor_kw, cv_df = selecionar_hiperparametros_cv(treino, n_splits)

    melhor_nome = "Random Forest"
    melhor_modelo = _novo_rf(**melhor_kw)
    melhor_modelo.fit(X_treino, y_treino)

    # Previsao no teste: retorno -> reconstrucao do preco.
    ret_pred_teste = melhor_modelo.predict(X_teste)
    preco_atual_teste = teste["preco_atual"].to_numpy()
    preco_real_teste = teste["target_preco"].to_numpy()
    preco_pred_teste = preco_atual_teste * (1.0 + ret_pred_teste)

    metricas = calcular_metricas(preco_real_teste, preco_pred_teste)

    # Baseline ingenuo: amanha = ultimo preco conhecido (retorno zero).
    metricas_baseline = calcular_metricas(preco_real_teste, preco_atual_teste)

    comparacao = pd.DataFrame([
        {"Modelo": melhor_nome, **metricas},
        {"Modelo": "Baseline (ultimo preco)", **metricas_baseline}
    ])

    pred_teste = pd.DataFrame({
        "Date": teste["Target_Date"].to_numpy(),
        "Real": preco_real_teste,
        "Predicted": preco_pred_teste
    })

    nome_completo = (
        f"{melhor_nome} (depth={melhor_kw['max_depth']}, "
        f"leaf={melhor_kw['min_samples_leaf']})"
    )

    bundle = {
        "model": melhor_modelo,
        "model_name": nome_completo,
        "best_params": melhor_kw,
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

    print("\nSelecao de hiperparametros por validacao temporal (menor MAE e melhor):")
    print(cv_df.to_string(index=False))
    print("\nConfig escolhida pela CV:", melhor_kw)
    print("\nAvaliacao no teste:")
    print(comparacao.to_string(index=False))
    print("\nModelo selecionado:", nome_completo)
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
