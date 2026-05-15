# --- SIMPAN KODE INI SEBAGAI app.py ---
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Football Scouting System", layout="wide")

st.title("⚽ Football Scouting Recommendation (VFM)")
st.write("Cari pemain alternatif yang serupa dengan statistik bintang dunia namun dengan harga lebih terjangkau.")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv(
    r'D:\SEMESTER 6\PROYEK DATA MINING\penelitian\transfermarkt_fbref_201920.csv',
    sep=';'
)
    return data.dropna(subset=['value'])

df_app = load_data()

# Sidebar Input
st.sidebar.header("Konfigurasi Pencarian")
player_list = df_app['player'].unique()
selected_player = st.sidebar.selectbox("Pilih Pemain Target:", player_list)
k_val = st.sidebar.slider("Jumlah Rekomendasi:", 1, 20, 5)

if st.button("Cari Alternatif"):
    # Logika Pencarian
    target_data = df_app[df_app['player'] == selected_player].iloc[0]
    pos = target_data['position']
    pos_code = 'FW' if 'FW' in pos else 'DF' if 'DF' in pos else 'GK' if 'GK' in pos else 'MF'
    
    # Preprocessing Lokal
    from sklearn.preprocessing import MinMaxScaler
    features = ['goals', 'assists', 'xg', 'xa'] if pos_code == 'FW' else ['passes_completed', 'interceptions', 'tackles']
    
    df_pos = df_app[df_app['position'].str.contains(pos_code)].copy().reset_index(drop=True)
    X = df_pos[features].fillna(0)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model
    knn = NearestNeighbors(n_neighbors=k_val + 1)
    knn.fit(X_scaled)
    
    target_idx = df_pos[df_pos['player'] == selected_player].index[0]
    dist, ind = knn.kneighbors(X_scaled[target_idx].reshape(1, -1))
    
    # Hasil
    res = df_pos.iloc[ind[0]]
    res = res[res['player'] != selected_player] # Hapus diri sendiri
    res = res[res['value'] < target_data['value']] # Filter lebih murah
    
    st.subheader(f"Hasil Rekomendasi untuk {selected_player}")
    st.write(f"Harga Pasar Target: € {target_data['value']:,.0f}")
    st.table(res[['player', 'age', 'squad', 'value'] + features])