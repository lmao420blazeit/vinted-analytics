# Pipeline Design

## Functional design

- **Atomicity**: a function should only do one task
- **Idempotency**: 
    - running the same code multiple times with the same input should return the same output
    - the output should not be duplicated
- **Data encapsulation**: function encapsulation should be limited to the scope it refers to, no additional external data.
- **Functional composition**: a higher order mechanism which allows to increase the level of abstraction and atomicity. Can be implemented in prefect via subflow calls.
    - factoring/atomicity
    - code reuse
    - + abstraction layers (f(g(x)))

#### Implementation in catalogs_flow: load block

``` python
@task(name="Load from api", 
      log_prints= True,
      retries=3, 
      retry_delay_seconds=exponential_backoff(backoff_factor=6),
      retry_jitter_factor=2)
def load_data_from_api(vinted, nbrRows: int, batch_size: int, item: str) -> pd.DataFrame:
    """
    Loads data from the Vinted API based on specified parameters.

    Args:
        nbrRows (int): Number of rows to fetch from the API.
        batch_size (int): Batch size for API requests.
        item (str): Item include in the API request.

    Returns:
        pd.DataFrame: DataFrame containing data fetched from the Vinted API.
    """
    df = vinted.items.search_catalog(url = f"https://www.vinted.pt/catalog/items?catalog_ids[]={item}&order=newest_first")
    # cant process latin characters
    df["catalog_id"] = item
    
    return (df)
```

- **Atomicity**: Loads data from API and assigns the catalog_id to dataframe (optimally should only load data from API, data transformations should be done in different blocks). However, the reason I did this is to have a cleaner code and prevent overloading other tasks with unnecessary variables.

- **Encapsulation**: the load function has no side effects. It starts by accepting an existing connection via dependency injection (vinted obj).

**Takeaways**: cursors should be passed outside of loading blocks; input variables should be limited to their use scope; transformation blocks should be apart from load/upload blocks.

#### Implementation in catalogs_flow: subflow block

``` python
@flow(name= "Fetch from vinted", 
      log_prints= True,
      description= """
      Main flow: 
      start node: fetch vinted/items endpoint 
      -> simple preprocessing 
      -> dumps into postgres staging table""")
def fetch_data_from_vinted(sample_frac = 0.01, 
                           item_ids = [], 
                           batch_size = 500, 
                           nbrRows = 1000):
    """
    Fetches data from the vinted/items endpoint, preprocesses it, and exports it to a PostgreSQL staging table.

    Parameters:
    - sample_frac (float): Fraction of data to sample.
    - item_ids (list): List of item IDs to fetch data for.
    - batch_size (int): Size of each batch to fetch.
    - nbrRows (int): Number of rows to fetch.

    Returns:
    None
    """
    vinted = Vinted()
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    for __item in item_ids:
        catalog_subflow(item = __item, 
                        nbrRows= nbrRows,
                        batch_size= batch_size,
                        vinted = vinted, 
                        engine= engine, 
                        sample_frac= sample_frac)
        time.sleep(60)

    return

@flow(name="Subflow for catalog.", 
      flow_run_name= "Subflow for catalog {item}",
      log_prints= True)
def catalog_subflow(item, nbrRows, batch_size, vinted, engine, sample_frac):
    df = load_data_from_api(vinted = vinted,
                            nbrRows = nbrRows,
                            batch_size = batch_size,
                            item = item)
    df = transform(data = df)
    df = parse_size_title(data = df)
    #create_artifacts(data = df)
    export_data_to_postgres(data = df, 
                            engine = engine)     # upload first to products due to FK referencing
    export_sample_to_postgres(df, 
                              sample_frac= sample_frac,
                              engine = engine)
    return

```

- **Functional decomposition**: increasing the levels of abstraction by decomposing a single flow into sub flows. 
    - Before: main_flow -> task1 -> task2 -> etc
    - After: main_flow -> subflow1(task1, task2, etc) -> subflow2(task1, task2, etc)

- **Atomicity**: Transformer, loading, export functions are in different blocks.

- **Encapsulation**: external connections are passed as argument to code blocks.

#### Implementation in catalogs_flow: export data block

``` python
@task(name="Export data to pg", log_prints=True)
def export_data_to_postgres(data: pd.DataFrame, engine) -> None:
    """
    Exports a pandas DataFrame to a specified PostgreSQL table using the provided database engine.

    Args:
        data (pd.DataFrame): The DataFrame to be exported to PostgreSQL.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'products_catalog'  # Specify the name of the table to export data to
    #engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False, 
                method = insert_on_conflict_nothing)

    return

```

- **Idempotency**: the insert_on_confict_nothing is a callable function which acts as an upsert preventing duplicate records. This block could execute multiple times with the same inputs but will only write to db once.

## Best practices

Although Python is a dynamic language, having type hints and type checkers can help with incompatibility issues and troubleshooting during runtime.

- **Typing**: matching inputs and outputs with expected types

``` python
from typing import List, Callable
```

- **Data class**: data representation classes which add some cool functionalities (__post__init__ method), inheritance and custom operators at the cost of a bigger overhead during runtime.

- **Testing**: SOON