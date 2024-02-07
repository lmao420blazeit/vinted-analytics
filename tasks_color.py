from pyVinted.vinted import Vinted
from prefect import task
import pandas as pd

vinted = Vinted()
items = vinted.items.search_colors()
print(items)