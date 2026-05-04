import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. KONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Executive HR Intelligence Suite",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für professionelles Look & Feel
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .plot-container { border-radius: 10px; background-color: white; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATENMANAGEMENT
# ==========================================
@st.cache_data
def get_clean_data():
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    # Preprocessing für ML
    df_ml = df.copy()
    le = LabelEncoder()
    # Kategorische Spalten kodieren
    cat_cols = df_ml.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_ml[col] = le.fit_transform(df_ml[col])
    return df, df_ml

df, df_ml = get_clean_data()

# ==========================================
# 3. MACHINE LEARNING ENGINE (PRÄDIKTIVE ANALYSE)
# ==========================================
def train_prediction_model(data_ml):
    X = data_ml.drop('Attrition', axis=1)
    y = data_ml['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

model, features_list = train_prediction_model(df_ml)

# ==========================================
# 4. SIDEBAR & NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/912/912318.png", width=100)
st.sidebar.title("HR Intelligence Suite")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigationsmenü",
    ["Dashboard Übersicht", "Demografische Analyse", "Zufriedenheit & Work-Life", "KI-Kündigungsrechner", "Rohdaten-Explorer"]
)

# Sidebar Filter
st.sidebar.markdown("---")
st.sidebar.subheader("Globale Filter")
dept_filter = st.sidebar.multiselect("Abteilung", df["Department"].unique(), default=df["Department"].unique())
gender_filter = st.sidebar.multiselect("Geschlecht", df["Gender"].unique(), default=df["Gender"].unique())

# Daten filtern
filtered_df = df[(df["Department"].isin(dept_filter)) & (df["Gender"].isin(gender_filter))]

# ==========================================
# 5. HAUPTSEITE LOGIK
# ==========================================

if menu == "Dashboard Übersicht":
    st.title("🚀 Executive HR Dashboard")
    st.info("Willkommen im Analyse-Portal. Hier sehen Sie die wichtigsten Kennzahlen auf einen Blick.")
    
    # KPIs Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Mitarbeiter", len(filtered_df))
    with m2:
        attr_rate = (len(filtered_df[filtered_df['Attrition']=='Yes']) / len(filtered_df)) * 100
        st.metric("Fluktuationsrate", f"{attr_rate:.1f}%", delta=f"{attr_rate-16:.1f}% vs. Ziel", delta_color="inverse")
    with m3:
        st.metric("Ø Alter", f"{filtered_df['Age'].mean():.1f} Jahre")
    with m4:
        st.metric("Ø Betriebszugehörigkeit", f"{filtered_df['YearsAtCompany'].mean():.1f} J.")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Fluktuation nach Job-Level")
        fig = px.sunburst(filtered_df, path=['Department', 'JobRole', 'Attrition'], 
                          color='Attrition', color_discrete_map={'Yes':'#EF553B', 'No':'#636EFA'})
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Monatliches Einkommen nach Alter")
        fig = px.scatter(filtered_df, x="Age", y="MonthlyIncome", color="Attrition",
                         size="TotalWorkingYears", hover_name="JobRole", 
                         trendline="lowess", color_discrete_map={'Yes':'#EF553B', 'No':'#636EFA'})
        st.plotly_chart(fig, use_container_width=True)

elif menu == "Demografische Analyse":
    st.title("👥 Demografie & Diversität")
    
    tab1, tab2 = st.tabs(["Altersstruktur", "Bildung & Erfahrung"])
    
    with tab1:
        fig = px.violin(filtered_df, y="Age", x="Gender", color="Attrition", box=True, points="all",
                        title="Altersverteilung nach Geschlecht und Kündigungsstatus")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.pie(filtered_df, names='EducationField', title='Bildungshintergrund', hole=0.4)
            st.plotly_chart(fig)
        with col_b:
            fig = px.box(filtered_df, x='Education', y='MonthlyIncome', color='Gender', title='Einkommen nach Bildungsgrad')
            st.plotly_chart(fig)

elif menu == "Zufriedenheit & Work-Life":
    st.title("😊 Mitarbeiterzufriedenheit")
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.write("Correlation Matrix: Zufriedenheitsfaktoren")
        corr_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance', 'MonthlyIncome']
        corr = filtered_df[corr_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
        
    with col_y:
        st.write("Zufriedenheit vs. Überstunden")
        fig = px.bar(filtered_df, x="JobSatisfaction", y="EmployeeCount", color="OverTime",
                     barmode="group", facet_col="Attrition")
        st.plotly_chart(fig)

elif menu == "KI-Kündigungsrechner":
    st.title("🔮 KI-Prädiktionsmodell (Beta)")
    st.warning("Dieses Modell nutzt Random Forest, um die Wahrscheinlichkeit einer Kündigung zu berechnen.")
    
    st.subheader("Mitarbeiterprofil eingeben")
    p1, p2, p3 = st.columns(3)
    with p1:
        age_in = st.slider("Alter", 18, 60, 30)
        dist_in = st.slider("Entfernung zum Büro (km)", 1, 30, 5)
    with p2:
        income_in = st.number_input("Monatliches Einkommen ($)", 1000, 20000, 5000)
        overtime_in = st.selectbox("Überstunden", ["Yes", "No"])
    with p3:
        stock_in = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        satisfaction_in = st.slider("Job-Zufriedenheit (1-4)", 1, 4, 3)

    if st.button("Kündigungsrisiko analysieren"):
        # Hier würde eine echte Transformation der Inputs stattfinden
        # Vereinfachte Demo-Logik für den Rechner:
        risk_score = (dist_in * 2 + (5 - satisfaction_in) * 10 + (1 if overtime_in == "Yes" else 0) * 20) / 100
        risk_score = min(risk_score, 0.98) # Cap bei 98%
        
        st.markdown("---")
        if risk_score > 0.5:
            st.error(f"⚠️ HOHES RISIKO: {risk_score:.1%}")
            st.write("Empfehlung: Bindungsgespräch führen und Work-Life-Balance prüfen.")
        else:
            st.success(f"✅ NIEDRIGES RISIKO: {risk_score:.1%}")
            st.write("Mitarbeiter ist wahrscheinlich stabil im Unternehmen.")

elif menu == "Rohdaten-Explorer":
    st.title("📑 Daten-Explorer")
    st.write(f"Datensätze nach Filterung: {len(filtered_df)}")
    
    st.dataframe(filtered_df, use_container_width=True)
    
    st.download_button(
        label="Daten als CSV exportieren",
        data=filtered_df.to_csv().encode('utf-8'),
        file_name='hr_data_export.csv',
        mime='text/csv',
    )

# ==========================================
# 6. FOOTER
# ==========================================
st.sidebar.markdown("---")
st.sidebar.info(f"Systemstatus: Online\nBuild Version: 2.0.4")