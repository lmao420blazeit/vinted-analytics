
def test_fetch_data(data) -> None:
    assert data['id'].is_unique, "Column 'product_id' is not unique"
    assert data['id'].notnull().all(), "Column 'product_id' has null values"
    assert data['user_id'].notnull().all(), "Column 'user_id' has null values"
    assert data['catalog_id'].notnull().all(), "Column 'catalog_id' has null values"

def test_transforms() -> None:
    pass

def test_get_fct_orders() -> None:
    pass

def test_get_sales_mart() -> None:
    pass