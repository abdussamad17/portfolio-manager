import datetime
import json
import os
import shutil


class ExtractDailyPrice:
    def __init__(self) -> None:
        self.date = datetime.datetime.now().date()

    def getprice(self):
        # delete directory if it exists
        path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        shutil.rmtree(f"{path}/DataExtracts", ignore_errors=True)
        os.mkdir(f"{path}/DataExtracts")
        stocks = os.listdir(f"{path}/RawDataStorage/prices")

        for stock in stocks:
            if not stock.endswith(".json"):
                continue

            with open(f"{path}/RawDataStorage/prices/{stock}") as f:
                print(stock)
                data = json.load(f)
                for item in data["historical"]:
                    with open(
                        f'{path}/DataExtracts/daily_{item["date"]}.json', "a"
                    ) as fo:
                        json.dump(item | {"ticker": data["symbol"]}, fo)
                        fo.write("\n")


def run():
    dailyprice = ExtractDailyPrice()
    print(dailyprice.__dict__)
    dailyprice.getprice()


run()
