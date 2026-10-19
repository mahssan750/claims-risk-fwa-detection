import argparse

from claims_risk.download import download_dataset
from claims_risk.modeling import train_and_evaluate, write_model_outputs
from claims_risk.prepare import build_analytics_database
from claims_risk.reporting import create_reports


def run_modeling() -> None:
    outputs = train_and_evaluate()
    write_model_outputs(outputs)
    create_reports(outputs)


def run_all() -> None:
    print("Downloading and validating published source files...")
    download_dataset()
    print("Building SQLite claim rules and provider features...")
    build_analytics_database()
    print("Training, evaluating and explaining models...")
    run_modeling()
    print("Pipeline complete. Review artifacts/ for outputs.")


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description="Healthcare claims risk and FWA analysis")
    command_parser.add_argument("command", choices=["download", "prepare", "model", "all"])
    return command_parser


def main() -> None:
    command = parser().parse_args().command
    if command == "download":
        download_dataset()
    elif command == "prepare":
        build_analytics_database()
    elif command == "model":
        run_modeling()
    else:
        run_all()


if __name__ == "__main__":
    main()
