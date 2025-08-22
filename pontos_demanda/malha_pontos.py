### MALHA DE PONTOS ###
# Bibliotecas
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import pandas as pd
import folium

# Carrega o GeoJSON com os municípios do Maranhão
geojson_url = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-21-mun.json"
gdf = gpd.read_file(geojson_url)

# Filtra para o município de Paço do Lumiar
gdf_paco_lumiar = gdf[gdf["name"] == "Paço do Lumiar"].reset_index(drop=True)
geometria_municipio = gdf_paco_lumiar.loc[0, "geometry"]
lon_min, lat_min, lon_max, lat_max = geometria_municipio.bounds

# Projeta para um sistema métrico (UTM zona 23S)
gdf_paco_lumiar = gdf_paco_lumiar.to_crs(epsg=31983)
geometria_proj = gdf_paco_lumiar.loc[0, "geometry"]
minx, miny, maxx, maxy = geometria_proj.bounds

# Espaçamento dos pontos da malha
espacamento = 700  # (d_pontos) metros
x_coords = np.arange(minx, maxx, espacamento)
y_coords = np.arange(miny, maxy, espacamento)

# Cria os pontos da malha
pontos = []
for x in x_coords:
    for y in y_coords:
        ponto = Point(x, y)
        if geometria_proj.contains(ponto):
            pontos.append(ponto)

# Converte os pontos em GeoDataFrame e reprojeta de volta para WGS84
gdf_pontos = gpd.GeoDataFrame(geometry=pontos, crs="EPSG:31983")
gdf_pontos_wgs84 = gdf_pontos.to_crs(epsg=4326)

# Extrai latitude e longitude
gdf_pontos_wgs84["latitude"] = gdf_pontos_wgs84.geometry.y
gdf_pontos_wgs84["longitude"] = gdf_pontos_wgs84.geometry.x
malha_pontos = gdf_pontos_wgs84[["latitude", "longitude"]].values.tolist()

# Gera o mapa com a malha de pontos
center_lat = (lat_min + lat_max) / 2
center_lon = (lon_min + lon_max) / 2
m = folium.Map(location=[center_lat, center_lon],
               zoom_start=13, control_scale=True)
folium.GeoJson(
    geometria_municipio,
    style_function=lambda x: {
        "fillColor": "blue",
        "color": "blue",
        "weight": 2,
        "fillOpacity": 0.1,
    }
).add_to(m)
for ponto in malha_pontos:
    folium.CircleMarker(
        location=[ponto[0], ponto[1]],
        radius=2,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=1,
        popup=f"Coordenadas: {ponto[0]}, {ponto[1]}"
    ).add_to(m)

# Salva o mapa em HTML
m.save("C:/Users/Romul/Documents/Visual_Studio/TCC/pontos_demanda/mapa_pontos.html")

# Salva os pontos em arquivo CSV
df_malha_pontos = pd.DataFrame(malha_pontos, columns=["latitude", "longitude"])
df_malha_pontos.to_csv(
    "C:/Users/Romul/Documents/Visual_Studio/TCC/pontos_demanda/malha_pontos.csv", index=False)