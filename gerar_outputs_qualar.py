from pathlib import Path
import argparse
from qualar_core import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Gera tabelas Excel, gráficos PNG e relatório textual para dados QUALAR consolidados.")
    parser.add_argument("--entrada", required=True, help="Caminho da planilha consolidada .xlsx/.csv")
    parser.add_argument("--saida", default="outputs_qualar", help="Pasta de saída")
    parser.add_argument("--estacao", default="Estação analisada", help="Nome da estação")
    parser.add_argument("--inicio", type=int, default=2021, help="Ano inicial")
    parser.add_argument("--fim", type=int, default=2025, help="Ano final")
    args = parser.parse_args()

    resultado = run_pipeline(args.entrada, Path(args.saida), station_name=args.estacao, start_year=args.inicio, end_year=args.fim)
    print("Processo concluído.")
    print(f"Excel: {resultado['excel_path']}")
    print(f"Texto: {resultado['text_path']}")
    print(f"ZIP: {resultado['zip_path']}")
    print(f"Gráficos: {len(resultado['chart_paths'])}")


if __name__ == "__main__":
    main()
