from pyVinted.vinted import Vinted
from datetime import timedelta, datetime
from typing import List
from prefect import task
import pandas as pd

vinted = Vinted()
items = vinted.items.search_colors()
print(items)