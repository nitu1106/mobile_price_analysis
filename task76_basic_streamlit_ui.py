import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="wide")
st.markdown("""
<style>
.stApp {background:#f3f7fc}
[data-testid="stSidebar"] {background:#102a43}
[data-testid="stSidebar"] * {color:white !important}
h1,h2,h3 {color:#123b63}
div.stButton>button {background:#1769aa;color:white;border-radius:8px}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

@st.cache_resource
def train(data):
    X=data.drop(columns=["price_range","id"],errors="ignore")
    y=data["price_range"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model=RandomForestClassifier(n_estimators=150,random_state=42)
    model.fit(Xtr,ytr)
    return model,list(X.columns),accuracy_score(yte,model.predict(Xte))

try:
    data=load_data()
    model,features,accuracy=train(data)
except Exception as e:
    st.error(f"Could not load/train using train.csv: {e}")
    st.stop()

st.sidebar.title("📱 Mobile Price App")
page=st.sidebar.radio("Navigate",["Home","Prediction","Dataset"])
if page=="Home":
    st.title("Mobile Price Range Prediction")
    st.write("Enter mobile specifications to predict its price range.")
    a,b,c=st.columns(3)
    a.metric("Records",len(data)); b.metric("Features",len(features)); c.metric("Test accuracy",f"{accuracy:.1%}")
    st.info("Price range labels: 0 = low, 1 = medium, 2 = high, 3 = very high.")
elif page=="Dataset":
    st.title("Dataset Preview")
    st.dataframe(data.head(20),use_container_width=True)
    st.write("Dataset shape:",data.shape)
else:
    st.title("Predict Mobile Price")
    st.caption("Choose the mobile specifications.")
    values={}
    cols=st.columns(3)
    for i,f in enumerate(features):
        with cols[i%3]:
            lo=float(data[f].min()); hi=float(data[f].max()); default=float(data[f].median())
            if data[f].nunique()<=2:
                values[f]=st.selectbox(f,sorted(data[f].unique().tolist()))
            else:
                values[f]=st.number_input(f,min_value=lo,max_value=hi,value=default)
    if st.button("Predict Price Range",type="primary"):
        pred=int(model.predict(pd.DataFrame([values],columns=features))[0])
        st.success(f"Predicted price range: {pred}")
        st.write("0 = low, 1 = medium, 2 = high, 3 = very high")
