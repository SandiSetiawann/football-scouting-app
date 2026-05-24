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
    data = pd.read_csv('transfermarkt_fbref_201920.csv', sep=';')
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
    
    # PEMETAAN FITUR DINAMIS YANG SESUAI DENGAN 4 POSISI UTAMA
    if pos_code == 'FW':
        features = ['goals', 'assists', 'xg', 'xa', 'shots_on_target_pct', 'progressive_passes']
    elif pos_code == 'MF':
        features = ['assists', 'xa', 'passes_completed', 'passes_pct', 'progressive_passes']
    elif pos_code == 'DF':
        features = ['passes_completed', 'interceptions', 'clearances', 'blocks', 'tackles']
    elif pos_code == 'GK':
        features = ['gk_save_pct', 'gk_clean_sheets', 'gk_goals_against_per90']
    
    # Memfilter dataset berdasarkan posisi yang dicari
    df_pos = df_app[df_app['position'].str.contains(pos_code)].copy().reset_index(drop=True)
    X = df_pos[features].fillna(0)
    
    # Skalasi Fitur
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model Pencarian Tetangga Terdekat
    knn = NearestNeighbors(n_neighbors=k_val + 1)
    knn.fit(X_scaled)
    
    target_idx = df_pos[df_pos['player'] == selected_player].index[0]
    dist, ind = knn.kneighbors(X_scaled[target_idx].reshape(1, -1))
    
    # Memproses Hasil Rekomendasi
    res = df_pos.iloc[ind[0]].copy()
    res['similarity_distance'] = dist[0]
    
    # Filter Value-for-Money (VFM)
    res = res[res['player'] != selected_player] # Menghilangkan nama diri sendiri dari daftar rekomendasi
    res = res[res['value'] < target_data['value']] # Memastikan harga alternatif lebih murah
    res = res.sort_values(by='value', ascending=True) # Mengurutkan dari yang termurah
    
    # Tampilan Output Web
    st.success(f"Ditemukan pemain alternatif dengan spesifikasi profil {pos_code} yang serupa!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Pemain Target", value=selected_player)
    with col2:
        st.metric(label=f"Harga Pasar Target", value=f"€ {target_data['value']:,.0f}")
        
    st.subheader("Daftar Pemain Rekomendasi (Diurutkan dari Paling Terjangkau):")
    # Menampilkan tabel akhir lengkap dengan fitur statistik aslinya secara dinamis
    st.table(res[['player', 'age', 'squad', 'value', 'similarity_distance'] + features])
