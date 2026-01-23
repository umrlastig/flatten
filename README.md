# flatten

# installation

## from source

```shell
git clone git@github.com:umrlastig/flatten.git
cd flatten
```

[Install uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

Install dependencies:
```shell
uv sync
```

```shell
source .venv/bin/activate
```

# usage

You can now use the *flatten_split* script to split your data.
You can see the different parameters with:
```shell
flatten_split --help
```

For instance, with the test data in directory *hydro* and saving the results in *split.gpkg*, you can use the following:
```shell
flatten_split hydro/SURFACE_HYDROGRAPHIQUE.shp hydro/TRONCON_HYDROGRAPHIQUE.shp split.gpkg
```
