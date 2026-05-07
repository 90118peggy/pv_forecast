import argparse
import json

from app.services.prediction_service import predict_one


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="PV Forecast CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run a single PV forecast prediction"
    )
    predict_parser.add_argument("--datetime", required=True, help="Prediction datetime")
    predict_parser.add_argument("--temp-air", required=True, type=float, help="Air temperature")
    predict_parser.add_argument("--temp-dew", required=True, type=float, help="Dew temperature")
    predict_parser.add_argument("--ghi", required=True, type=float, help="Global horizontal irradiance")
    predict_parser.add_argument("--dni", required=True, type=float, help="Direct normal irradiance")
    predict_parser.add_argument("--dhi", required=True, type=float, help="Diffuse horizontal irradiance")
    predict_parser.add_argument("--wind-speed", required=True, type=float, help="Wind speed")
    predict_parser.add_argument("--wind-direction", required=True, type=float, help="Wind direction")
    predict_parser.add_argument("--albedo", required=True, type=float, help="Surface albedo")
    predict_parser.add_argument("--pressure", required=True, type=float, help="Air pressure")

    return parser


def run_predict(args: argparse.Namespace) -> None:
    payload = {
        "datetime": args.datetime,
        "temp_air": args.temp_air,
        "temp_dew": args.temp_dew,
        "ghi": args.ghi,
        "dni": args.dni,
        "dhi": args.dhi,
        "wind_speed": args.wind_speed,
        "wind_direction": args.wind_direction,
        "albedo": args.albedo,
        "pressure": args.pressure,
    }

    result = predict_one(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
