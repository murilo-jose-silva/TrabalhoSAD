from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib
from pandas.tseries.offsets import BDay

from train_model import prever_horizonte

st.set_page_config(
    page_title="Gold Forecast | Inteligência de Dados",
    page_icon="🪙",
    layout="wide"
)

st.title("🪙 Gold Forecast")
st.caption("Sistema acadêmico de previsão do preço do ouro com Machine Learning")

ARQUIVO_MODELO = Path("models/gold_forecast_bundle.joblib")

# Limites de horizonte: previsoes muito longas perdem confiabilidade porque o
# erro se acumula (previsao recursiva) e modelos de arvore nao extrapolam.
HORIZONTE_MAX = 30     # maximo de dias uteis permitido (~1,5 mes de pregoes)
HORIZONTE_ALERTA = 15  # a partir daqui, avisa que a confianca cai

@st.cache_resource
def carregar_bundle():
    return joblib.load(ARQUIVO_MODELO)

if not ARQUIVO_MODELO.exists():
    st.error("Modelo não encontrado. Execute primeiro o treinamento.")
    st.code("python train_model.py --csv data/gold_price.csv")
    st.stop()

bundle = carregar_bundle()
historico = bundle["history"].copy()
historico["Date"] = pd.to_datetime(historico["Date"])
modelo = bundle["model"]

with st.sidebar:
    st.header("Configurações")

    horizonte = st.number_input(
        f"Observações futuras (dias úteis, máximo {HORIZONTE_MAX})",
        min_value=1,
        max_value=HORIZONTE_MAX,
        value=7,
        step=1,
        format="%d"
    )

    horizonte = int(horizonte)

    # Aviso quando o prazo escolhido se aproxima do limite confiável.
    if horizonte > HORIZONTE_ALERTA:
        st.warning(
            f"Previsões acima de {HORIZONTE_ALERTA} dias úteis são menos "
            "confiáveis: o erro se acumula a cada passo e o modelo tende a "
            "estabilizar num patamar."
        )

    st.divider()

    st.write("**Modelo selecionado**")
    st.info(bundle["model_name"])
    st.caption(
        "Previsões recursivas: cada estimativa futura alimenta a próxima."
    )

ultima_data = historico["Date"].max()
datas_futuras = pd.bdate_range(
    start=ultima_data + BDay(1), periods=horizonte
)
forecast = prever_horizonte(
    modelo, historico["Price"].tolist(), datas_futuras, horizonte
)

preco_atual = float(historico["Price"].iloc[-1])
preco_final = float(forecast["Predicted_Price"].iloc[-1])
variacao = (preco_final / preco_atual - 1) * 100 if preco_atual else 0
metricas = bundle["metrics"]
baseline = bundle["baseline_metrics"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Último preço observado", f"US$ {preco_atual:,.2f}")
c2.metric(f"Estimativa em {horizonte} observações", f"US$ {preco_final:,.2f}", f"{variacao:+.2f}%")
c3.metric("MAE no teste", f"US$ {metricas['MAE']:,.2f}")
c4.metric("R² no teste", f"{metricas['R2']:.4f}")

st.info(
    "A previsão é uma estimativa estatística para observações futuras em dias úteis. "
    "Não representa cotação garantida nem recomendação de investimento. "
    "O calendário considera dias úteis de segunda a sexta, mas não incorpora feriados específicos do mercado."
)

st.subheader("Histórico e previsão")
dias_historico = st.slider("Período histórico exibido (Em dias)", 90, min(2000, len(historico)), min(365, len(historico)))
recorte = historico.tail(dias_historico)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recorte["Date"], y=recorte["Price"],
    mode="lines", name="Preço observado"
))

# Faixa de incerteza ilustrativa: parte do RMSE do teste e alarga com a raiz
# do número de passos à frente. Não é um intervalo de confiança rigoroso,
# apenas comunica que a incerteza cresce ao longo do horizonte.
rmse = float(bundle["metrics"]["RMSE"])
passos = np.arange(1, len(forecast) + 1)
margem = rmse * np.sqrt(passos)
fig.add_trace(go.Scatter(
    x=list(forecast["Date"]) + list(forecast["Date"][::-1]),
    y=list(forecast["Predicted_Price"] + margem)
      + list((forecast["Predicted_Price"] - margem)[::-1]),
    fill="toself", fillcolor="rgba(255,165,0,0.15)",
    line=dict(width=0), name="Faixa de incerteza", hoverinfo="skip"
))

fig.add_trace(go.Scatter(
    x=forecast["Date"], y=forecast["Predicted_Price"],
    mode="lines+markers", name="Previsão",
    line=dict(dash="dash")
))
fig.add_trace(go.Scatter(
    x=[ultima_data, forecast["Date"].iloc[0]],
    y=[preco_atual, forecast["Predicted_Price"].iloc[0]],
    mode="lines", showlegend=False,
    line=dict(dash="dot")
))
fig.update_layout(
    xaxis_title="Data",
    yaxis_title="Preço (USD por unidade conforme dataset)",
    hovermode="x unified",
    legend_title="Série",
    height=520
)
st.plotly_chart(fig, use_container_width=True)

col_esq, col_dir = st.columns(2)
with col_esq:
    st.subheader("Previsões futuras")
    tabela = forecast.copy()
    tabela["Date"] = tabela["Date"].dt.strftime("%d/%m/%Y")
    tabela["Predicted_Price"] = tabela["Predicted_Price"].round(2)
    st.dataframe(tabela, use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar previsões em CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name="previsoes_gold_forecast.csv",
        mime="text/csv"
    )

with col_dir:
    st.subheader("Avaliação do modelo")
    metricas_df = pd.DataFrame([
        {"Métrica": "MAE", "Modelo selecionado": metricas["MAE"]},
        {"Métrica": "MSE", "Modelo selecionado": metricas["MSE"]},
        {"Métrica": "RMSE", "Modelo selecionado": metricas["RMSE"]},
        {"Métrica": "R²", "Modelo selecionado": metricas["R2"]}
    ])
    st.dataframe(metricas_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Baseline MAE: US$ {baseline['MAE']:,.2f}. "
        "O baseline repete o último preço conhecido."
    )

st.subheader("Comparação dos modelos na validação temporal")
cv = bundle["cv_results"].copy()
st.dataframe(cv.round(4), use_container_width=True, hide_index=True)

st.subheader("Comparação: valores reais x previstos no conjunto de teste")
pred = bundle["test_predictions"].copy()
pred["Date"] = pd.to_datetime(pred["Date"])
pred = pred.tail(250)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=pred["Date"], y=pred["Real"], mode="lines", name="Real"))
fig2.add_trace(go.Scatter(x=pred["Date"], y=pred["Predicted"], mode="lines", name="Previsto"))
fig2.update_layout(
    xaxis_title="Data", yaxis_title="Preço (USD)",
    hovermode="x unified", height=430
)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Entenda as métricas"):
    st.markdown("""
    - **MAE:** erro absoluto médio, na mesma unidade do preço.
    - **MSE:** média dos erros quadráticos; penaliza mais os erros grandes.
    - **RMSE:** raiz do MSE, também na unidade do preço.
    - **R²:** proporção da variação explicada pelo modelo; pode ser negativo.
    - **Baseline:** referência simples que prevê que o próximo preço será igual ao último observado.
    """)

with st.expander("Detalhes técnicos e limitações"):
    st.write("Período de treinamento até:", bundle["training_cutoff"])
    st.write("Período de teste:", bundle["test_start"], "até", bundle["test_end"])
    st.write("Coluna original de preço:", bundle["price_column_original"])
    st.write(
        "A avaliação de teste é de um passo à frente: para cada data do teste, "
        "as observações anteriores já realizadas são consideradas conhecidas. "
        "Já a previsão de vários passos no dashboard é recursiva e acumula incerteza."
    )
    st.write(
        "O modelo usa apenas o histórico do preço. Não considera câmbio, juros, "
        "inflação, eventos geopolíticos ou outras variáveis externas."
    )

st.caption("Projeto acadêmico — Sistemas de Apoio à Decisão | Gold Forecast")