#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import time
import pydeck as pdk
from geopy.geocoders import OpenCage  # 🔄 trocado Nominatim -> OpenCage
import openrouteservice
import math

st.set_page_config(page_title="Localizador de Endereços", layout="wide")

# -------------------------------
# Chaves e clientes das APIs
# -------------------------------
API_KEY_ORS = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjQwMjU3NjM5ZmQ5MDQxMjI5YjQ3MWJkMzZlMzlkMzZkIiwiaCI6Im11cm11cjY0In0="
API_KEY_OPENCAGE = "480d28fce0a04bd4839c8cc832201807"

try:
    CLIENT_ORS = openrouteservice.Client(key=API_KEY_ORS)
except Exception as e:
    CLIENT_ORS = None
    st.warning(f"Falha ao inicializar OpenRouteService: {e}")

# -------------------------------
# Utilitários
# -------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_float_or_none(s):
    try:
        return float(str(s).strip())
    except Exception:
        return None

# -------------------------------
# Centro de Distribuição (com OpenCage + fallback manual)
# -------------------------------
st.sidebar.header("📍 Centro de distribuição")
cd_endereco = st.sidebar.text_input(
    "Endereço do Centro de Distribuição",
    "Travessa Francisco Marrocos Portela, Alto Alegre I, Maracanaú - CE, Brasil, 61922-120"
)

def geocode_opencage(endereco):
    geolocator = OpenCage(api_key=API_KEY_OPENCAGE, timeout=5)
    try:
        if not endereco or str(endereco).strip() == "":
            return None, None
        location = geolocator.geocode(endereco)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        st.warning(f"Erro na geocodificação do CD: {e}")
        return None, None

cd_lat, cd_lon = geocode_opencage(cd_endereco)

# Fallback manual caso geocodificação falhe
st.sidebar.markdown("### Coordenadas do CD (fallback manual)")
cd_lat_manual = st.sidebar.text_input("Latitude do CD", value=f"{cd_lat:.6f}" if cd_lat else "", placeholder="-3.831753")
cd_lon_manual = st.sidebar.text_input("Longitude do CD", value=f"{cd_lon:.6f}" if cd_lon else "", placeholder="-38.613147")
usar_manual = st.sidebar.checkbox("Usar coordenadas manuais do CD", value=(cd_lat is None or cd_lon is None))

if usar_manual:
    cd_lat = parse_float_or_none(cd_lat_manual)
    cd_lon = parse_float_or_none(cd_lon_manual)

if cd_lat is not None and cd_lon is not None and (-90 <= cd_lat <= 90) and (-180 <= cd_lon <= 180):
    st.sidebar.success(f"CD localizado: {cd_lat:.6f}, {cd_lon:.6f}")
else:
    st.sidebar.error("Não foi possível definir o CD. Informe endereço válido ou use as coordenadas manuais.")

# -------------------------------
# Upload de clientes
# -------------------------------
st.sidebar.header("📂 Importar clientes (.xlsx)")
arquivo = st.sidebar.file_uploader("Selecione um arquivo Excel", type=["xlsx"])

st.title("📍 Localizador de Endereços")
st.write("Geocodifique clientes, visualize no mapa e veja tempo/distância do CD até cada cliente, com rota real e cenários de otimização.")

@st.cache_data
def geocode_dataframe_opencage(df, endereco_col="Endereco"):
    results = []
    geolocator = OpenCage(api_key=API_KEY_OPENCAGE, timeout=5)
    for _, row in df.iterrows():
        addr = str(row.get(endereco_col, "")).strip()
        if not addr:
            results.append((np.nan, np.nan))
            continue
        try:
            location = geolocator.geocode(addr)
            if location:
                results.append((location.latitude, location.longitude))
            else:
                results.append((np.nan, np.nan))
        except Exception:
            results.append((np.nan, np.nan))
        time.sleep(1)  # respeitar taxa
    return results

# -------------------------------
# Rota real com ORS (CD -> clientes na ordem definida) + resumo
# -------------------------------
def gerar_rota_real(cd_lat, cd_lon, pontos):
    if CLIENT_ORS is None or not pontos:
        return [], None
    coords = [[cd_lon, cd_lat]] + [[p["lon"], p["lat"]] for p in pontos]
    try:
        rota = CLIENT_ORS.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson'
        )
        caminho = rota['features'][0]['geometry']['coordinates']
        resumo = rota['features'][0]['properties']['summary']  # distance (m), duration (s)
        return [{"lon": lon, "lat": lat} for lon, lat in caminho], resumo
    except Exception as e:
        st.warning(f"Erro ao gerar rota: {e}")
        return [], None

# -------------------------------
# Tempo e distância do CD até cada cliente (com proteção de cota)
# -------------------------------
def calcular_tempo_distancia(cd_lat, cd_lon, pontos, max_clientes=50):
    resultados = []
    if CLIENT_ORS is None:
        for p in pontos:
            resultados.append({"Cliente": p["name"], "Tempo (min)": None, "Distância (km)": None})
        return pd.DataFrame(resultados)

    if len(pontos) > max_clientes:
        st.warning(f"Calculando tempo/distância apenas para os primeiros {max_clientes} clientes para proteger sua chave ORS.")
    pontos_avaliar = pontos[:max_clientes]

    for p in pontos_avaliar:
        try:
            rota = CLIENT_ORS.directions(
                coordinates=[[cd_lon, cd_lat], [p["lon"], p["lat"]]],
                profile="driving-car",
                format="geojson"
            )
            summary = rota['features'][0]['properties']['summary']
            duracao_min = int(summary['duration'] / 60)
            distancia_km = round(summary['distance'] / 1000, 2)
            resultados.append({"Cliente": p["name"], "Tempo (min)": duracao_min, "Distância (km)": distancia_km})
        except Exception:
            resultados.append({"Cliente": p["name"], "Tempo (min)": None, "Distância (km)": None})
        time.sleep(0.3)  # suaviza taxa
    return pd.DataFrame(resultados)

# -------------------------------
# Processamento do Excel
# -------------------------------
if arquivo:
    try:
        df = pd.read_excel(arquivo)
    except Exception as e:
        st.error(f"Erro ao ler o Excel: {e}")
        st.stop()

    df.columns = [c.strip() for c in df.columns]
    col_obrig = ["Cliente_ID", "Endereco"]
    faltantes = [c for c in col_obrig if c not in df.columns]
    if faltantes:
        st.error(f"Arquivo inválido. Faltam colunas: {', '.join(faltantes)}.")
        st.stop()

    st.success(f"{len(df)} clientes carregados.")

    with st.spinner("Geocodificando endereços com OpenCage..."):
        coords = geocode_dataframe_opencage(df, endereco_col="Endereco")

    if len(coords) == 0:
        df["Latitude"] = np.nan
        df["Longitude"] = np.nan
    else:
        df["Latitude"], df["Longitude"] = zip(*coords)

    total = len(df)
    validos = df["Latitude"].notna().sum()
    st.info(f"Coordenadas obtidas para {validos}/{total} clientes.")

    # ✏️ Inserção manual de coordenadas (um campo só: lat, lon)
    df_faltantes = df[df["Latitude"].isna() | df["Longitude"].isna()]
    if not df_faltantes.empty:
        st.subheader("✏️ Inserir coordenadas manualmente")
        for i, row in df_faltantes.iterrows():
            st.markdown(f"**{row.get('Cliente_ID', i)} - {row.get('Cliente', 'Cliente')}**")
            coord_in = st.text_input(
                f"Coordenadas (lat, lon) para {row.get('Cliente_ID', i)}",
                key=f"coord_{i}",
                placeholder="-3.831753, -38.613147"
            )
            if st.button(f"Salvar coordenadas de {row.get('Cliente_ID', i)}", key=f"btn_{i}"):
                try:
                    lat_str, lon_str = coord_in.split(",")
                    lat_in, lon_in = float(lat_str.strip()), float(lon_str.strip())
                    if -90 <= lat_in <= 90 and -180 <= lon_in <= 180:
                        df.at[i, "Latitude"] = lat_in
                        df.at[i, "Longitude"] = lon_in
                        st.success("Coordenadas salvas.")
                    else:
                        st.warning("Coordenadas inválidas.")
                except Exception:
                    st.warning("Formato inválido. Use: lat, lon (ex: -3.831753, -38.613147)")

    # Pontos válidos
    pontos = [
        {
            "lat": r["Latitude"],
            "lon": r["Longitude"],
            "name": f"{r['Cliente_ID']} - {r['Cliente']}" if "Cliente" in df.columns else str(r["Cliente_ID"])
        }
        for _, r in df.iterrows()
        if not pd.isna(r["Latitude"]) and not pd.isna(r["Longitude"])
    ]

    # -------------------------------
    # Cenários e ordenação
    # -------------------------------
    st.sidebar.header("🧭 Simulação de Rotas")
    opcao = st.sidebar.radio(
        "Escolha uma simulação:",
        ["Rota 1 - Ordem original", "Rota 2 - Menor distância", "Rota 3 - Menor tempo"]
    )

    df_td = pd.DataFrame()
    if cd_lat is not None and cd_lon is not None and pontos:
        df_td = calcular_tempo_distancia(cd_lat, cd_lon, pontos, max_clientes=50)

    pontos_ordenados = pontos.copy()
    if cd_lat is not None and cd_lon is not None:
        if opcao == "Rota 2 - Menor distância":
            pontos_ordenados = sorted(pontos, key=lambda p: haversine_km(cd_lat, cd_lon, p["lat"], p["lon"]))
        elif opcao == "Rota 3 - Menor tempo" and not df_td.empty:
            tempo_por_cliente = {row["Cliente"]: row["Tempo (min)"] for _, row in df_td.iterrows()}
            pontos_ordenados = sorted(pontos, key=lambda p: tempo_por_cliente.get(p["name"], float("inf")))

    # Proteção de cota ORS na rota
    max_pontos_rota = 50
    if len(pontos_ordenados) > max_pontos_rota:
        st.warning(f"Gerando rota apenas para os primeiros {max_pontos_rota} clientes para proteger sua chave ORS.")
    pontos_ordenados_rota = pontos_ordenados[:max_pontos_rota]

    # -------------------------------
    # Mapa, rota e resumo
    # -------------------------------
    if cd_lat is not None and cd_lon is not None and pontos_ordenados_rota:
        st.subheader(f"🗺️ Mapa de clientes e rota simulada — {opcao}")

        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=pontos_ordenados + [{"lat": cd_lat, "lon": cd_lon, "name": "Centro de Distribuição"}],
            get_position='[lon, lat]',
            get_fill_color='[0, 122, 255]',
            get_radius=60,
            pickable=True
        )

        rota_caminho, resumo = gerar_rota_real(cd_lat, cd_lon, pontos_ordenados_rota)
        path_data = []
        if rota_caminho:
            path_data = [{"path": [[p["lon"], p["lat"]] for p in rota_caminho], "name": "Rota CD -> Clientes"}]

        path_layer = pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_width=4,
            get_color=[0, 128, 255],
            width_min_pixels=2
        )

        view_state = pdk.ViewState(latitude=cd_lat, longitude=cd_lon, zoom=11)
        st.pydeck_chart(pdk.Deck(layers=[scatter, path_layer], initial_view_state=view_state, tooltip={"text": "{name}"}))

        st.subheader("📊 Resumo da rota escolhida")
        col1, col2, col3 = st.columns(3)
        if resumo:
            total_min = int(resumo["duration"] / 60)
            total_km = round(resumo["distance"] / 1000, 2)
            col1.metric(label="Tempo total estimado", value=f"{total_min} min")
            col2.metric(label="Distância total", value=f"{total_km} km")
            col3.metric(label="Clientes atendidos", value=len(pontos_ordenados_rota))
        else:
            col1.metric(label="Tempo total estimado", value="—")
            col2.metric(label="Distância total", value="—")
            col3.metric(label="Clientes atendidos", value=len(pontos_ordenados_rota))

        st.subheader("⏱️ Tempo e distância do CD até cada cliente")
        if not df_td.empty:
            st.dataframe(df_td, use_container_width=True)
        else:
            st.info("Nenhum cliente com coordenadas válidas para calcular tempo e distância.")
    else:
        st.warning("Defina um endereço válido para o CD e importe clientes com coordenadas válidas.")
else:
    st.warning("Importe um arquivo Excel (.xlsx) com as colunas 'Cliente_ID' e 'Endereco'.")


# In[ ]:





# In[ ]:




