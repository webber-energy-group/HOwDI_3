"""
Creates plot from outputs of model
Author: Braden Pecora

In the current version, there are next to no features,
but the metadata should be fairly easy to access and utilize.
"""
import json
import warnings
from itertools import combinations

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.wkt import loads
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
import math

from HOwDI.arg_parse import parse_command_line

# ignore warning about plotting empty frame
warnings.simplefilter(action="ignore", category=UserWarning)

# Set to True to plot color markers for station delivered cost, False to not plot
plot_delivered_price=True

def roads_to_gdf(wd):
    """Converts roads.csv into a GeoDataFrame object

    This is necessary since .geojson files can not handle LineStrings with multiple points.
    Road geodata are stored as csv, where the geodata are stored as literal strings.
    The shapely.wkt function "loads" can interpret this literal string and convert into a LineString object
    """
    # wd is path where 'hubs.geojson' and 'roads.csv' are located

    # get hubs for crs
    hubs = gpd.read_file(wd / "hubs.geojson")

    # read csv and convert geometry column
    # roads = gpd.read_file(wd / "roads.csv")
    roads = pd.read_csv(wd / "roads.csv")
    roads["geometry"] = roads["road_geometry"].apply(
        loads
    )  # convert string into Linestring
    roads = gpd.GeoDataFrame(roads, geometry="geometry")
    roads = roads.set_crs(hubs.crs)
    del roads["road_geometry"]

    return roads


def _all_possible_combos(items: list, existing=False) -> list:
    """Returns a list of all possible combos as sets.

    If "existing" is True, extends items with a duplicate of items
    where each item in items is followed by "Existing"

    For example

    all_possible_combos([1,2,3])
    returns
    [{1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}]

    This function is used for filtering dataframes.
    The dataframe has a set corresponding to the production
    types at each hub. For example, a hub could have a corresponding
    production value of {"smr","smrExisting"} or just {"smr",}, and we
    want to see if all values of this set are in the list "items" of production
    types (defined by therm/elec_production.csv "type" column).
    To do this, we must turn "items" into a list of all possible combinations.

    """
    if existing == True:
        # append "Existing" versions
        items.extend([i + "Existing" for i in items])
    out = []
    for length in range(len(items)):
        out.extend([frozenset(combo) for combo in combinations(items, length + 1)])
    return out


def _find_max_length_set_from_list(a: list):
    """From a list of sublists (or similar sub-object),
    returns the longest sublist (or sub-object as a list)"""
    return list(max(a))


def _diff_of_list(a: list, b: list) -> list:
    """From two outputs of 'all_possible_combos',
    gets all possible difference sets

    For example, if the items in "a" are {A,B,C} and the
    items in "b" are {1,2,3}, this would return
    [{A,1},{A,2},{A,3},{A,1,2},{A,1,3},{A,2,3},{A,B,1},...]
    but wouldn't include sets where all items are contained by
    either a or b.
    """

    # get the unique items from a and b,
    # basically undo all_possible_combos
    # just an easy way of not writing the same thing twice
    a_unique, b_unique = map(_find_max_length_set_from_list, (a, b))

    # get all possible combos between a_unique and b_unique
    all_possible = _all_possible_combos(a_unique + b_unique)

    # find the difference
    difference = set(all_possible) - set(a) - set(b)
    return list(difference)


def create_plot(H, background="black", price_min=None, price_max=None):
    """
    Parameters:
    H is a HydrogenData object with the following:

    H.hubs_dir: directory where hubs geo files are stored (hubs.geojson, roads.csv)
    H.output_dict: output dictionary of model
    H.shpfile: location of shapefile to use as background
    H.prod_therm and H.prod_elec: DataFrame with column "Type",
     used for determining if a node has thermal, electric, or both types of production.

    Returns:
    fig: a matplotlib.plt object which is a figure of the results.
    """
    plt.rc("font", family="Franklin Gothic Medium")
    
    if plot_delivered_price:
        price_df = pd.read_csv(H.outputs_dir / "delivered_price.csv")
        price_lookup = dict(zip(price_df["site_id"], price_df["min_delivered_price"]))

    hub_data = json.load(open(H.hubs_dir / "hubs.geojson"))["features"]
    locations = {d["properties"]["hub"]: d["geometry"]["coordinates"] for d in hub_data}

    # clean data
    def get_relevant_dist_data(hub_data):
        # returns a list of dicts used in a dict comprehension with only the `relevant_keys`
        outgoing_dicts = hub_data["distribution"]["outgoing"]
        relevant_keys = ["source_class", "destination", "destination_class", "dist_h"]
        for _, outgoing_dict in outgoing_dicts.items():
            for key in list(outgoing_dict.keys()):
                if key not in relevant_keys:
                    del outgoing_dict[key]
        return [outgoing_dict for _, outgoing_dict in outgoing_dicts.items()]

    dist_data = {
        hub: get_relevant_dist_data(hub_data)
        for hub, hub_data in H.output_dict.items()
        if hub_data["distribution"] != {"local": {}, "outgoing": {}, "incoming": {}}
    }

    def get_relevant_p_or_c_data(hub_data_p_or_c):
        # p_or_c = production or consumption
        # turns keys of hub_data['production'] or hub_data['consumption'] into a set,
        # used in the dictionary comprehensions below
        if hub_data_p_or_c != {}:
            # return frozenset(hub_data_p_or_c.keys())
            # Strip off 'Existing' suffix to normalize tech type
            cleaned_keys = {k.replace("Existing", "") for k in hub_data_p_or_c.keys()}
            return frozenset(cleaned_keys)
        else:
            return None

    prod_data = {
        hub: get_relevant_p_or_c_data(hub_data["production"])
        for hub, hub_data in H.output_dict.items()
    }
    cons_data = {
        hub: get_relevant_p_or_c_data(hub_data["consumption"])
        for hub, hub_data in H.output_dict.items()
    }

    def get_production_capacity(hub_data_prod):
        if hub_data_prod != {}:
            return sum(
                [
                    prod_data_by_type["prod_capacity"]
                    for _, prod_data_by_type in hub_data_prod.items()
                ]
            )
        else:
            return 0

    prod_capacity = {
        hub: get_production_capacity(hub_data["production"])
        for hub, hub_data in H.output_dict.items()
    }

    # Station marker color by delivered cost -- only if plot_delivered_price is True
    if plot_delivered_price:
        # Extract min and max price for colormap normalization
        prices = list(price_lookup.values())

        # Use fixed if provided, else dynamic
        if price_min is None:
            price_min = min(prices)
        if price_max is None:
            price_max = max(prices)


        # price_min = min(prices)
        # price_max = max(prices)
        # price_min = 1000
        # price_max = 12000

        # Round min down, max up to nearest 1000
        price_min_rounded = math.floor(price_min / 100) * 100
        price_max_rounded = math.ceil(price_max / 100) * 100
        price_mid = (price_min_rounded + price_max_rounded) // 2

        # Define colormap and normalizer
        cmap = LinearSegmentedColormap.from_list("custom_price_cmap", ["#2ca02c", "#ffdd00", "#d62728"])
        norm = Normalize(vmin=price_min, vmax=price_max)

        # Function to get color from price
        def get_price_color(hub):
            price = price_lookup.get(hub, None)
            if price is None:
                return "gray"  # fallback
            return cmap(norm(price))

    marker_size_factor = 15 #10
    prod_marker_size_factor = 1.6 # scale production square markers to be larger than consumption circles

    def get_marker_size(prod_capacity):
        if prod_capacity != 0:
            size = marker_size_factor * prod_capacity * prod_marker_size_factor
        else:
            # prod capacity is zero for non-producers, which would correspond to a size of zero.
            # Thus, we use the default size for non-producers
            size = 75
        return size

    prod_capacity_marker_size = {
        hub: get_marker_size(prod_capacity)
        for hub, prod_capacity in prod_capacity.items()
    }

    def get_consumption_amount(hub_data_cons):
        if hub_data_cons != {}:
            return sum(
                cons_data_by_type["cons_h"]
                for _, cons_data_by_type in hub_data_cons.items()
            )
        else:
            return 0

    cons_amount = {
        hub: get_consumption_amount(hub_data["consumption"])
        for hub, hub_data in H.output_dict.items()
    }

    # Use same base as production to keep scales comparable
    def get_cons_marker_size(cons_value):
        return marker_size_factor * cons_value if cons_value > 0 else 75

    cons_marker_size = {
        hub: get_cons_marker_size(cons_value)
        for hub, cons_value in cons_amount.items()
    }


    features = []
    for hub, hub_connections in dist_data.items():
        hub_latlng = locations[hub]
        hub_geodata = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": hub_latlng},
            "properties": {
                "name": hub,
                "production": prod_data[hub],
                "consumption": cons_data[hub],
                "consumption_amount": cons_amount[hub],
                "consumption_marker_size": cons_marker_size[hub],
                "production_capacity": prod_capacity[hub],
                "production_marker_size": prod_capacity_marker_size[hub],
            },
        }
        features.append(hub_geodata)

        for hub_connection in hub_connections:
            dest = hub_connection["destination"]
            dist_type = hub_connection["source_class"]
            dest_latlng = locations[dest]

            line_geodata = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [hub_latlng, dest_latlng],
                },
                "properties": {
                    "name": hub + " to " + dest,
                    "start": hub,
                    "end": dest,
                    "dist_type": dist_type,
                    "dist_h": hub_connection.get("dist_h", 0),
                },
            }
            features.append(line_geodata)

    geo_data = {"type": "FeatureCollection", "features": features}
    distribution = gpd.GeoDataFrame.from_features(geo_data)

    ########
    # Plot

    # initialize figure
    fig, ax = plt.subplots(figsize=(20, 20), dpi=300)
    # ax.set_facecolor("black")
    if background == "black":
        fig.patch.set_facecolor("black") # set background
    if background == "white":
        fig.patch.set_alpha(0.0) 
    ax.set_facecolor("none" if background == "white" else "black")
    ax.axis("off")

    us_county = gpd.read_file(H.shpfile)
    states_names = ["Texas", "New Mexico", "Arizona", "California"] # names of states in region of interest
    states_outlines = []
    for state_name in states_names:
        state_county = us_county[us_county["STATE_NAME"] == state_name] # counties within state
        state_outline = state_county.dissolve() # dissolve county outlines within state
        states_outlines.append(state_outline) # keep outline of state

    combined_states = gpd.GeoDataFrame(pd.concat(states_outlines, ignore_index=True)) # combine outlines of states

    combined_states.plot(ax=ax, color="white", edgecolor="black")


    # Plot hubs
    hubs = distribution[distribution.type == "Point"]

    prod_types = H.get_prod_types()
    thermal_prod_combos = _all_possible_combos(prod_types["thermal"], existing=True)
    electric_prod_combos = _all_possible_combos(prod_types["electric"])
    both_prod_combos = _diff_of_list(thermal_prod_combos, electric_prod_combos)

    both_mask = (
        hubs["production"].notnull() & hubs["consumption"].notnull()
    )
    hubs_both = hubs[both_mask]

    # Options for hub by technology
    hub_plot_tech = {
        "default": {
            "name": "Only Consumption",
            "color": "white",
            "marker": ".",
            "set": None,
            "b": lambda df: df["production"].isnull(),
        },
        "thermal": {
            "name": "Thermal Production",
            "color": "#4E2261",
            "marker": "s",
            "b": lambda df: df["production"].isin(thermal_prod_combos),
        },
        "electric": {
            "name": "Electric Production",
            "color": "#75C3D6",
            "marker": "s",
            "b": lambda df: df["production"].isin(electric_prod_combos),
        },
        "both": {
            "name": "Therm. and Elec. Production",
            "color": "#E8588D",
            "marker": "s",
            "b": lambda df: df["production"].isin(both_prod_combos),
        },
    }
    # if hub_plot_tech["both"]["b"](hubs):

    # Options for hub by Production, Consumption, or both
    hub_plot_type = {
        "production": {
            "name": "Production",
            "b": lambda df: df["production"].notnull() & df["consumption"].isnull(),
            "edgecolors": None,
        },
        "consumption": {
            "name": "Consumption (Shape)",
            "b": lambda df: df["production"].isnull() & df["consumption"].notnull(),
            "edgecolors": "black",
        },
        "both": {
            "name": "Production and Consumption (Shape)",
            "b": lambda df: df["production"].notnull() & df["consumption"].notnull(),
            "edgecolors": "black",
        },
    }

    # Plot hubs based on production/consumption (marker) options and production tech (color) options
    # in short, iterates over both of the above option dictionaries
    # the 'b' (boolean) key is a lambda function that returns the locations of where the hubs dataframe
    #   matches the specifications. An iterable way of doing stuff like df[df['production'] == 'smr']

    for tech, tech_plot in hub_plot_tech.items():
        for type_name, type_plot in hub_plot_type.items():
            # mask = type_plot["b"](hubs) & tech_plot["b"](hubs)
            mask = type_plot["b"](hubs) & tech_plot["b"](hubs) & (~both_mask)
            df = hubs[mask]

            if df.empty:
                continue

            xs = df.geometry.x
            ys = df.geometry.y

            if tech_plot["marker"] == "s":
                sizes = df["production_marker_size"]
            elif tech_plot["marker"] == ".":
                sizes = df["consumption_marker_size"]
            else:
                sizes = 75

            # Apply delivered cost coloring here for consumption hubs
            if plot_delivered_price and tech_plot["marker"] == ".":
                colors = [get_price_color(hub_name) for hub_name in df["name"]]
            else:
                colors = tech_plot["color"]

            ax.scatter(
                xs,
                ys,
                s=sizes,
                color=colors,
                marker="s" if tech_plot["marker"] == "s" else "o",
                edgecolors=type_plot["edgecolors"],
                linewidth=0.5 if type_plot["edgecolors"] else 0,
                zorder=3 if tech_plot["marker"] == "s" else 5,
            )
            
    
    # Plot hubs with both production and consumption
    for _, row in hubs_both.iterrows():
        # Determine production color
        tech_set = row["production"]
        if any("smr" in t for t in tech_set) and any("elec" in t for t in tech_set):
            prod_color = "#E8588D"
        elif any("smr" in t for t in tech_set):
            prod_color = "#4E2261"
        elif any("elec" in t for t in tech_set):
            prod_color = "#75C3D6"

        # Plot production marker (square) first
        ax.scatter(
            row.geometry.x,
            row.geometry.y,
            s=row["production_marker_size"],
            color=prod_color,
            marker="s",
            zorder=3,
        )

        # Plot consumption marker (circle) on top
        if plot_delivered_price:
            fill_color = get_price_color(row["name"])
        else:
            fill_color = "white"
        ax.scatter(
            row.geometry.x,
            row.geometry.y,
            s=row["consumption_marker_size"],
            color=fill_color,
            edgecolors="black",
            marker="o",
            linewidth=0.5,
            zorder=5,
        )
    
    
    
    # Plot connections:
    dist_pipelineColor = "#6A6262"
    dist_truckColor = "#fb8500"

    connections = distribution[distribution.type == "LineString"]
    roads_connections = connections.copy()

    # Normalize dist_h to line width
    if not roads_connections.empty:
        roads_connections["dist_h"] = roads_connections["dist_h"].fillna(0)
        scaling_max_factor = 300
        min_linewidth = 1
        max_linewidth = 5

        roads_connections["linewidth"] = roads_connections["dist_h"].apply(
            lambda x: min_linewidth + (max_linewidth - min_linewidth) * x / scaling_max_factor
        )

    if not roads_connections.empty:
        # get data from roads csv, which draws out the road path along a connection
        roads = roads_to_gdf(H.hubs_dir)

        for row in roads.itertuples():
            # get road geodata for each connection in connections df
            hubA = row.startHub
            hubB = row.endHub
            roads_connections.loc[
                (roads_connections["start"] == hubA)
                & (roads_connections["end"] == hubB),
                "geometry",
            ] = row.geometry
            roads_connections.loc[
                (roads_connections["end"] == hubA)
                & (roads_connections["start"] == hubB),
                "geometry",
            ] = row.geometry

        for _, row in roads_connections.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            x, y = geom.xy

            if row["dist_type"] in ["dist_pipelineLowPurity", "dist_pipelineHighPurity"]:
                color = dist_pipelineColor
                linestyle = "-" # solid for pipelines
            else:
                color = dist_truckColor
                linestyle = "--" # dashed for trucks
                
            lw = row["linewidth"]

            ax.plot(x, y, color=color, linewidth=lw, linestyle=linestyle, zorder=2)

    legend_elements = []

    legend_elements.extend(
        [
            Line2D(
                [0],
                [0],
                color=tech_plot["color"],
                label=tech_plot["name"],
                marker=tech_plot["marker"],
                lw=0,
                markersize=12,
            )
            for tech, tech_plot in hub_plot_tech.items()
            if (tech_plot["name"] != "Only Consumption")
            and (tech_plot["name"] != "Therm. and Elec. Production")
        ]
    )
    legend_elements.extend(
        [
            Line2D(
                [0],
                [0],
                color="white",
                markeredgecolor="black",
                label="Consumption",
                marker=".",
                # marker=type_plot["marker"],
                lw=0,
                markersize=22,
                markeredgewidth=1,
            )
            # for type_name, type_plot in hub_plot_type.items()
        ]
    )
    legend_elements.extend(
        [
            Line2D(
                [0],
                [0],
                color=dist_pipelineColor,
                lw=2,
                linestyle="-",
                label="Pipeline",
            ),
            Line2D(
                [0],
                [0],
                color=dist_truckColor,
                lw=2,
                linestyle="--",
                label="Truck",
            )
        ]
    )

    ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.01),
        facecolor="white",
        edgecolor="#212121",
        framealpha=1,
        fontsize=18,
    )

    if plot_delivered_price:
        # Create scalar mappable for the colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([price_min_rounded, price_max_rounded])

        # Get position of ax to align vertical baseline
        ax_pos = ax.get_position()

        # Adjust vertical position slightly higher
        cbar_x0 = ax_pos.x0 + 0.24 #0.17
        cbar_y0 = ax_pos.y0 + 0.03
        cbar_width = 0.2
        cbar_height = 0.02
        cbar_box = [cbar_x0, cbar_y0, cbar_width, cbar_height]

        # Add colorbar axis
        cax = fig.add_axes(cbar_box)

        # Add colorbar
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([price_min_rounded, price_mid, price_max_rounded])
        cbar.set_ticklabels([
            f"${price_min_rounded / 1000:.2f}/kg",
            f"${price_mid / 1000:.2f}/kg",
            f"${price_max_rounded / 1000:.2f}/kg"
        ])
        # cbar.set_ticks([price_min, price_max])
        # cbar.set_ticklabels([f"${price_min / 1000:.0f}/kg", f"${price_max / 1000:.0f}/kg"])
    
        tick_color = "black" if background == "white" else "white"
        cbar.ax.tick_params(labelsize=18, length=0, colors=tick_color)
 
        # Add title
        fig.text(
            cbar_x0 + cbar_width / 2,
            cbar_y0 + cbar_height + 0.005,
            "Delivered Cost of Hydrogen",
            ha="center",
            va="bottom",
            fontsize=18,
            color=tick_color,
            zorder=5,
        )
        
    return fig


def main():
    from HOwDI.model.HydrogenData import HydrogenData

    args = parse_command_line()

    H = HydrogenData(
        scenario_dir=args.scenario_dir,
        inputs_dir=args.inputs_dir,
        outputs_dir=args.outputs_dir,
        raiseFileNotFoundError=False,
    )

    try:
        H.output_dict = json.load(open(H.outputs_dir / "outputs.json"))
    except FileNotFoundError:
        from HOwDI.postprocessing.generate_outputs import create_output_dict

        H.create_output_dfs()
        H.output_dict = create_output_dict(H)
        H.write_output_dict()



if __name__ == "__main__":
    main()
