import os


def run_markowitz_strategy(
    risk_constant, return_estimate, vol_weighted, max_concentration
):
    command = f"nice -n -20 python3 Testback.py MarkowitzStrategy,risk_constant={risk_constant},return_estimate={return_estimate},vol_weighted={vol_weighted},max_concentration={max_concentration}"
    os.system(command)


if __name__ == "__main__":
    parameters = [
        (1, 0.000269, False, 0.05),
        (1, 0.000269, True, 1),
        (1, 0.000269, True, 0.05),
        (0.5, 0.000269, False, 0.05),
        (2, 0.000269, False, 0.05),
        (1, 0.00017, False, 0.05),
        (1, 0.00037, False, 0.05),
    ]

    for param in parameters:
        run_markowitz_strategy(*param)

    print("All tasks finished!")
