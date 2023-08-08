import collections
import os
import json
import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser
import shutil

EARLY_EXIT = True

class UniverseConstructor:
    def __init__(self):
        self.storage_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "RawDataStorage"
        )
        # Define the subfolder path
        self.subfolder_path = os.path.join(self.storage_folder, "UniversebyDate")

        # If the subfolder already exists, delete it and all its contents
        if os.path.exists(self.subfolder_path):
            shutil.rmtree(self.subfolder_path)

    def _get_date(self, date_string):
        try:
            if "?" in date_string:
                return parser.parse(date_string.rstrip("?") + "-01-01").date()

            elif len(date_string) == 4:
                return parser.parse(date_string + "-01-01").date()

            else:
                return parser.parse(date_string).date()
        except Exception:
            return None

    def _load_json(self, filename):
        filepath = os.path.join(self.storage_folder, f"{filename}.json")
        with open(filepath, "r") as f:
            return json.load(f)

    def _save_json(self, filename, data):
        # Create the subfolder
        os.makedirs(self.subfolder_path, exist_ok=True)

        filepath = os.path.join(self.subfolder_path, f"{filename}.json")
        with open(filepath, "w") as f:
            # Modify the universe dictionary to convert datetime objects to string format
            data = list(data.keys())
            json.dump(data, f)

    def get_universe(self):
        constituents = self._load_json("current_constituents")
        # Initialize universe as a dictionary of lists
        universe = {
            c["symbol"]: [
                {"ticker": c["symbol"], "date": self._get_date(c["dateFirstAdded"])}
            ]
            for c in constituents
            if c["dateFirstAdded"]
        }
        # load thme constituent changes and ticker changes
        constituent_changes = self._load_json("historical_constituents")
        ticker_changes = self._load_json("symbol_changes")

        current_date = datetime.datetime.now().date()

        constituent_changes_by_date = collections.defaultdict(list)
        ticker_changes_by_date = collections.defaultdict(list)

        for change in constituent_changes:
            constituent_changes_by_date[self._get_date(change["date"])].append(change)

        for change in ticker_changes:
            ticker_changes_by_date[self._get_date(change["date"])].append(change)


        while current_date >= min(
            min(self._get_date(change["date"]) for change in constituent_changes),
            min(self._get_date(change["date"]) for change in ticker_changes),
        ):
            if os.path.isfile("universe" + str(current_date)) and EARLY_EXIT:
                return

            change_made = 0
            for change in constituent_changes_by_date[current_date]:
                if change["addedSecurity"]:
                    if change["symbol"] in universe:
                        # Delete the company's entry if it's removed from the S&P 500
                        print(
                            "deleting"
                            + change["symbol"]
                            + " "
                            + " on "
                            + str(current_date)
                            + " Size "
                            + str(len(universe))
                        )
                        del universe[change["symbol"]]
                        change_made = 1

                elif change["removedSecurity"]:
                    if change["symbol"] not in universe:
                        # Add a new entry for the company when it's added to the S&P 500
                        print(
                            "Adding"
                            + change["symbol"]
                            + " "
                            + " on "
                            + str(current_date)
                            + " Size "
                            + str(len(universe))
                        )
                        universe[change["symbol"]] = [
                            {"ticker": change["symbol"], "date": str(current_date)}
                        ]
                        change_made = 1

            for change in ticker_changes_by_date[current_date]:
                old_symbol = change["oldSymbol"]
                new_symbol = change["newSymbol"]

                if new_symbol in universe.keys():
                    # Add the new ticker to the company's list of tickers
                    print(
                        "symbol change from:"
                        + new_symbol
                        + " to "
                        + old_symbol
                        + " on  "
                        + str(current_date)
                    )
                    universe[new_symbol].append(
                        {"ticker": old_symbol, "date": current_date}
                    )
                    change_made = 1

            # check if the datefirstadded in the constituents is the same as the current date. if the symbol appears in the universe, delete it
            for change in constituents:
                if change["symbol"] in universe and (
                    self._get_date(change["dateFirstAdded"]) > current_date
                ):
                    # Delete the company's entry if it's removed from the S&P 500
                    print(
                        " deleting(Inception of Stock) "
                        + change["symbol"]
                        + " "
                        + " on "
                        + str(current_date)
                        + " Size "
                        + str(len(universe))
                    )
                    del universe[change["symbol"]]
                    change_made = 1

            if change_made == 1:
                self._save_json("universe" + str(current_date), universe)

            current_date -= relativedelta(days=1)

        self._save_json("earliest_universe", universe)


universeConstructor = UniverseConstructor()
universeConstructor.get_universe()
