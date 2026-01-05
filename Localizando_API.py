#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import time
import pydeck as pdk
from geopy.geocoders import Nominatim
import openrouteservice
import math

st.set_page_config(page_title="Localizador de Endereços", layout="wide")

# -------------------------------
# Configurações e chave ORS (mantida)
# -------------------------------
API_KEY_ORS = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImZlZTI5OWZiMGU4MzQ0OTg4ZWU1YzdmMjc5OGMyNWQyIiwiaCI6Im11cm11cjY0In0="
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

# -------------------------------
# Centro de Distribuição (com fallback manual)
# -------------------------------
st.sidebar.header("📍 Centro de distribuição")
cd_endereco = st.sidebar.text_input(
    "Endereço do Centro de Distribuição",
    "Travessa Francisco Marrocos Portela, Alto Alegre I, Maracanaú - CE, Brasil, 61922-120"
)

def geocode_osm(endereco):
    geolocator = Nominatim(user_agent="localizador_enderecos", timeout=5)
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

cd_lat, cd_lon = geocode_osm(cd_endereco)

# Fallback manual caso geocodificação falhe
st.sidebar.markdown("### Coordenadas do CD (fallback manual)")
cd_lat_manual = st.sidebar.number_input("Latitude do CD", value=float(cd_lat) if cd_lat else 0.0, format="%.6f")
cd_lon_manual = st.sidebar.number_input("Longitude do CD", value=float(cd_lon) if cd_lon else 0.0, format="%.6f")
usar_manual = st.sidebar.checkbox("Usar coordenadas manuais do CD (se geocodificação falhar)", value=(cd_lat is None or cd_lon is None))

if usar_manual:
    cd_lat, cd_lon = cd_lat_manual, cd_lon_manual

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
def geocode_dataframe_osm(df, endereco_col="Endereco"):
    results = []
    geolocator = Nominatim(user_agent="localizador_enderecos", timeout=5)
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
        time.sleep(1)  # respeitar limite de 1 req/s
    return results

# -------------------------------
# Rota real com ORS (CD -> clientes na ordem fornecida)
# -------------------------------
def gerar_rota_real(cd_lat, cd_lon, pontos):
    if CLIENT_ORS is None:
        return [], None
    if not pontos:
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
# Tempo e distância do CD até cada cliente
# -------------------------------
def calcular_tempo_distancia(cd_lat, cd_lon, pontos):
    resultados = []
    if CLIENT_ORS is None:
        for p in pontos:
            resultados.append({"Cliente": p["name"], "Tempo (min)": None, "Distância (km)": None})
        return pd.DataFrame(resultados)

    for p in pontos:
        try:
            rota = CLIENT_ORS.directions(
                coordinates=[[cd_lon, cd_lat], [p["lon"], p["lat"]]],
                profile="driving-car",
                format="geojson"
            )
            summary = rota['features'][0]['properties']['summary']
            duracao_min = int(summary['duration'] / 60)                 # segundos → minutos
            distancia_km = round(summary['distance'] / 1000, 2)         # metros → km
            resultados.append({"Cliente": p["name"], "Tempo (min)": duracao_min, "Distância (km)": distancia_km})
        except Exception:
            resultados.append({"Cliente": p["name"], "Tempo (min)": None, "Distância (km)": None})
        time.sleep(0.25)  # suaviza taxa de chamadas
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

    with st.spinner("Geocodificando endereços com OpenStreetMap..."):
        coords = geocode_dataframe_osm(df, endereco_col="Endereco")

    if len(coords) == 0:
        df["Latitude"] = np.nan
        df["Longitude"] = np.nan
    else:
        df["Latitude"], df["Longitude"] = zip(*coords)

    total = len(df)
    validos = df["Latitude"].notna().sum()
    st.info(f"Coordenadas obtidas para {validos}/{total} clientes.")

    # ✏️ Inserção manual de coordenadas
    df_faltantes = df[df["Latitude"].isna() | df["Longitude"].isna()]
    if not df_faltantes.empty:
        st.subheader("✏️ Inserir coordenadas manualmente")
        for i, row in df_faltantes.iterrows():
            st.markdown(f"**{row.get('Cliente_ID', i)} - {row.get('Cliente', 'Cliente')}**")
            lat_in = st.number_input(f"Latitude para {row.get('Cliente_ID', i)}", key=f"lat_{i}", value=0.0, format="%.6f")
            lon_in = st.number_input(f"Longitude para {row.get('Cliente_ID', i)}", key=f"lon_{i}", value=0.0, format="%.6f")
            if st.button(f"Salvar coordenadas de {row.get('Cliente_ID', i)}", key=f"btn_{i}"):
                if -90 <= lat_in <= 90 and -180 <= lon_in <= 180:
                    df.at[i, "Latitude"] = lat_in
                    df.at[i, "Longitude"] = lon_in
                    st.success("Coordenadas salvas.")
                else:
                    st.warning("Coordenadas inválidas.")

    # Mapa e rota real + cenários
    if cd_lat is not None and cd_lon is not None and validos > 0:
        st.subheader("🗺️ Mapa de clientes e rota")

        pontos = [
            {
                "lat": r["Latitude"],
                "lon": r["Longitude"],
                "name": f"{r['Cliente_ID']} - {r['Cliente']}" if "Cliente" in df.columns else str(r["Cliente_ID"])
            }
            for _, r in df.iterrows()
            if not pd.isna(r["Latitude"]) and not pd.isna(r["Longitude"])
        ]

        # Cenários
        st.sidebar.header("🧭 Simulação de Rotas")
        opcao = st.sidebar.radio(
            "Escolha uma simulação:",
            ["Rota 1 - Ordem original", "Rota 2 - Menor distância", "Rota 3 - Menor tempo"]
        )

        # Tabela tempo/distância por cliente (mantida)
        df_td = calcular_tempo_distancia(cd_lat, cd_lon, pontos) if pontos else pd.DataFrame()

        # Ordenação conforme cenário
        pontos_ordenados = pontos.copy()
        if opcao == "Rota 2 - Menor distância":
            pontos_ordenados = sorted(pontos, key=lambda p: haversine_km(cd_lat, cd_lon, p["lat"], p["lon"]))
        elif opcao == "Rota 3 - Menor tempo" and not df_td.empty:
            tempo_por_cliente = {row["Cliente"]: row["Tempo (min)"] for _, row in df_td.iterrows()}
            pontos_ordenados = sorted(pontos, key=lambda p: tempo_por_cliente.get(p["name"], float("inf")))

        # Pontos sempre visíveis
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=pontos_ordenados + [{"lat": cd_lat, "lon": cd_lon, "name": "Centro de Distribuição"}],
            get_position='[lon, lat]',
            get_fill_color='[0, 122, 255]',  # azul
            get_radius=60,
            pickable=True
        )

        # Rota (se disponível) e resumo
        rota_caminho, resumo = gerar_rota_real(cd_lat, cd_lon, pontos_ordenados)
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

        # Resumo total da rota
        st.subheader("📊 Resumo da rota escolhida")
        col1, col2, col3 = st.columns(3)
        if resumo:
            total_min = int(resumo["duration"] / 60)
            total_km = round(resumo["distance"] / 1000, 2)
            col1.metric(label="Tempo total estimado", value=f"{total_min} min")
            col2.metric(label="Distância total", value=f"{total_km} km")
            col3.metric(label="Clientes atendidos", value=len(pontos_ordenados))
        else:
            col1.metric(label="Tempo total estimado", value="—")
            col2.metric(label="Distância total", value="—")
            col3.metric(label="Clientes atendidos", value=len(pontos_ordenados))

        # Tempo e distância por cliente
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




