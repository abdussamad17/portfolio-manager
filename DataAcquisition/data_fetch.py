import os
import json
import requests
from dotenv import load_dotenv
import datetime


class DataFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/"
        self.storage_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "RawDataStorage"
        )
        self.currdate = datetime.datetime.now().date()

    def _get_data(self, endpoint):
        """
        Helper function to get data from the API
        params:
            endpoint(str): endpoint to get data from
        return:
            data(dict): data from the API
        """
        url = f"{self.base_url}{endpoint}{self.api_key}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def _save_json(self, folder, filename, data):
        """
        Helper function to save data to a json file
        params:
            folder(str): folder to save the file in
            filename(str): name of the file to save
            data(dict): data to save to the file
        return:
            None
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, f"{filename}.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

    def _load_json(self, filename):
        """
        Helper function to load data from a json file
        params:
            filename(str): name of the file to load
        return:
            data(dict): data loaded from the file
        """
        filepath = os.path.join(self.storage_folder, f"{filename}.json")
        with open(filepath, "r") as f:
            data = json.load(f)
        return data

    def fetch_historical_constituents(self):
        """
        Fetch historical constituents data from the API
        Sort the data by timestamp
        """
        data = self._get_data("api/v3/historical/sp500_constituent?apikey=")
        data.sort(
            key=lambda x: datetime.datetime.strptime(x["date"], "%Y-%m-%d"),
            reverse=True,
        )
        self._save_json(self.storage_folder, "historical_constituents", data)
        print("Historical Constituents fetched successfully")

    def fetch_symbol_change(self):
        data = self._get_data("/api/v4/symbol_change?apikey=")
        self._save_json(self.storage_folder, "symbol_changes", data)
        print("Symbol Changes fetched successfully")

    def fetch_full_historical_price(self, symbol):
        """
        Fetch full historical price data for a given symbol
        params:
            symbol(str): symbol to fetch data for
        """
        data = self._get_data(
            f"/api/v3/historical-price-full/{symbol}?from=1950-01-01&to={self.currdate}&apikey="
        )

        prices_folder = os.path.join(self.storage_folder, "prices")
        self._save_json(prices_folder, f"historical_price_{symbol}", data)

    def fetch_current_constituents(self):
        data = self._get_data("/api/v3/sp500_constituent?apikey=")
        self._save_json(self.storage_folder, "current_constituents", data)
        print("Current Constituents fetched successfully")

    def get_symbols_set(self):
        # Ensure data is fetched before getting symbols
        self.fetch_current_constituents()
        data = self._load_json("current_constituents")
        symbols = {item["symbol"] for item in data}
        return symbols

    def fetch_all_historical_prices(self):
        symbols = self.get_symbols_set()
        for symbol in symbols:
            print(f"Fetching {symbol} pricing data")
            self.fetch_full_historical_price(symbol)
        print("Prcing data fetched successfully")


fetcher = DataFetcher()
fetcher.fetch_historical_constituents()
fetcher.fetch_current_constituents()
fetcher.fetch_symbol_change()
fetcher.fetch_all_historical_prices()
