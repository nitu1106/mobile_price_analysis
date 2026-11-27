import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Live Mobile Price Prediction",page_icon="📲",layout="wide")
st.markdown("""
<style>
.stApp {background:#f3f7fc}
[data-testid="stSidebar"] {background:#102a43}
[data-testid="stSidebar"] * {color:white !important}
h1,h2,h3 {color:#123b63}
</style>
""",unsafe_allow_html=True)

@st.cache_data
def load_data(): return pd.read_csv("train.csv")

@st.cache_resource
def get_model():
    data=load_data()
    X=data.drop(columns=["price_range","id"],errors="ignore"); y=data["price_range"]
    model=RandomForestClassifier(n_estimators=150,random_state=42)
    model.fit(X,y)
    return model,list(X.columns),data

try: model,features,data=get_model()
except Exception as e:
    st.error(f"Could not load/train using train.csv: {e}"); st.stop()

st.title("Real-Time Mobile Price Prediction")
st.write("Adjust a specification. Prediction updates automatically without a Predict button.")
values={}
with st.sidebar:
    st.header("Mobile Specifications")
    for f in features:
        lo=float(data[f].min()); hi=float(data[f].max()); default=float(data[f].median())
        if data[f].nunique()<=2:
            values[f]=st.selectbox(f,sorted(data[f].unique().tolist()))
        else:
            values[f]=st.slider(f,min_value=lo,max_value=hi,value=default)

row=pd.DataFrame([values],columns=features)
pred=int(model.predict(row)[0]); probs=model.predict_proba(row)[0]
st.subheader("Current Prediction")
st.metric("Predicted Price Range",pred)
st.caption("0 = low, 1 = medium, 2 = high, 3 = very high")
st.subheader("Prediction Probabilities")
chart=pd.DataFrame({"Price Range":[str(x) for x in model.classes_],"Probability":probs})
st.bar_chart(chart.set_index("Price Range"))
st.dataframe(chart,hide_index=True,use_container_width=True)
