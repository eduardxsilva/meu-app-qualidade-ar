# QUALAR — Análise de Qualidade do Ar | ACH1026

Este projeto foi feito para a planilha consolidada `Planilha_Geral_Unica_Tudo.xlsx` e para o trabalho de levantamento e análise de dados de qualidade do ar.

## O que o código gera

- `QUALAR_tabelas_resultados.xlsx`: tabelas analíticas em Excel.
- `RELATORIO_TECNICO_QUALAR.md`: texto técnico automático para apoiar o relatório.
- `graficos_png/*.png`: gráficos em 300 DPI para inserir no relatório/apresentação.
- `QUALAR_outputs_tabelas_graficos_texto.zip`: pacote com tudo.

## Como rodar no computador

Instale as dependências:

```bash
pip install -r requirements.txt
```

Rode o gerador offline:

```bash
python gerar_outputs_qualar.py --entrada "Planilha_Geral_Unica_Tudo.xlsx" --saida "outputs_qualar" --estacao "Nome da estação"
```

Rode o app Streamlit:

```bash
streamlit run streamlit_app_qualar_final.py
```

## Observações importantes

1. A planilha precisa ter `Data` e `Hora`, ou uma coluna `Datetime`.
2. O código trata `24:00` como `00:00` do dia seguinte, mas preserva o ano-base pela coluna `Data` original.
3. As tendências são exploratórias. Não confunda correlação com causalidade.
4. A seção de fontes no entorno precisa ser completada com mapa, relatório CETESB e características reais da estação.
5. Variáveis com baixa completude não devem sustentar conclusão forte.

## Referências usadas no texto automático

- CETESB. Relatórios de Qualidade do Ar do Estado de São Paulo.
- AHRENS, C. Donald. *Meteorology Today: An Introduction to Weather, Climate, and the Environment*. 9. ed. Brooks/Cole, 2009.
- SANTOS, F. S. et al. Avaliação da influência das condições meteorológicas na qualidade do ar. *Engenharia Sanitária e Ambiental*, 2019.
