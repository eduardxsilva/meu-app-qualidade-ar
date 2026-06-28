# QUALAR — Consolidação e análise de qualidade do ar

App Streamlit para o trabalho ACH1026: consolida CSVs do QUALAR/CETESB, filtra 2021–2025, calcula estatística descritiva, completude, médias anuais, tendências exploratórias, sazonalidade mensal, correlação e gráficos meteorológicos.

## Instalação

```bash
pip install -r requirements_qualar.txt
streamlit run streamlit_app_qualar.py
```

## Uso recomendado

1. Baixe no QUALAR os CSVs da estação escolhida para 2021, 2022, 2023, 2024 e 2025.
2. Inclua os poluentes MP10, MP2.5, NOx, SO2 e O3 quando disponíveis.
3. Inclua temperatura, umidade, precipitação, direção e velocidade do vento se a estação medir esses parâmetros.
4. Rode o app e envie todos os CSVs.
5. Confira se pelo menos 3 poluentes foram detectados.
6. Exporte a planilha consolidada e os gráficos em PNG 300 DPI.

## Observação

A tendência linear é exploratória. Não trate correlação como causalidade sem discutir fontes locais, meteorologia, sazonalidade e limitações de completude.
