import pandas as pd
import numpy as np

class MessageManager:
    def __init__(self,):
        pass

    def fetching(self, item, symbol, start_time):
        """fetching {item} of {symbol} from {start_time}."""
        print(f"fetching {item} of {symbol} from {start_time}.")


    def updating(self, item, nume, deno):
        """updating {item} of {nume}/{deno} symbol."""
        print(f"updating {item} of {nume}/{deno} symbol.")


    def necessary_creating(self, item, symbol):
        """{item} of {symbol} should be creaded."""
        print(f"{item} of {symbol} should be creaded.")


    def unnecessary_update(self, item, symbol):
        """{item} of {symbol} is no need to update."""
        print(f"{item} of {symbol} is no need to update.")


    def donload_completed(self, item, symbol, file_path):
        """{item} of {symbol} has been downloaded to {file_path}."""
        print(f"{item} of {symbol} has been downloaded to {file_path}.")


    def update_completed(self, item, symbol):
        """{item} of {symbol} has been updated."""
        print(f"{item} of {symbol} has been updated.")


    def empty_warning(self, item, symbol):
        """WARNING! {item} of {symbol} is empty."""
        print(f"WARNING! {item} of {symbol} is empty.") # not use ValueError(error_message) to stop the program.
