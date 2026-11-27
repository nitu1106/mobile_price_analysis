
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Explain Mobile Price Prediction",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background:#f3f7fc}
[data-testid="stSidebar"] {background:#102a43}
[data-testid="stSidebar"] * {color:white !important}
h1,h2,h3 {color:#123b63}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("train.csv")


@st.cache_resource
def get_model():
    data = load_data()

    X = data.drop(columns=["price_range", "id"], errors="ignore")
    y = data["price_range"]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    return model, list(X.columns), data


try:
    model, features, data = get_model()

except Exception as e:
    st.error(f"Could not load/train using train.csv: {e}")
    st.stop()


st.title("Explainable Mobile Price Prediction")
st.write("Predict a price range and inspect the features that influence the model.")


# Mobile specification inputs
values = {}

with st.sidebar:
    st.header("Mobile Specifications")

    for f in features:
        lo = float(data[f].min())
        hi = float(data[f].max())
        default = float(data[f].median())

        if data[f].nunique() <= 2:
            values[f] = st.selectbox(
                f,
                sorted(data[f].unique().tolist())
            )
        else:
            values[f] = st.slider(
                f,
                min_value=lo,
                max_value=hi,
                value=default
            )


# Prepare input
row = pd.DataFrame([values], columns=features)

# Prediction
pred = int(model.predict(row)[0])

st.success(f"Predicted price range: {pred}")
st.caption("0 = low, 1 = medium, 2 = high, 3 = very high")


# Feature Importance
st.subheader("Feature Importance")

imp = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

top = imp.head(10).sort_values("Importance")

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(top["Feature"], top["Importance"])
ax.set_xlabel("Importance")
ax.set_title("Top 10 Random Forest Features")

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    imp,
    hide_index=True,
    use_container_width=True
)


# SHAP Explanation
st.subheader("SHAP Explanation")

st.write(
    "SHAP explains how each feature influences the selected mobile's prediction."
)

if st.button("Try SHAP explanation"):

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row)
        class_index = list(model.classes_).index(pred)
        if isinstance(shap_values, list):
            class_values = shap_values[class_index][0]
        else:
            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:
                class_values = shap_values[0, :, class_index]

            elif shap_values.ndim == 2:
                class_values = shap_values[0]

            else:
                raise ValueError("Unexpected SHAP output shape")

     
        base_value = np.asarray(explainer.expected_value)

        if base_value.ndim > 0 and base_value.size > 1:
            base_value = base_value.flatten()[class_index]
        else:
            base_value = float(base_value.flatten()[0])

        
        explanation = shap.Explanation(
            values=class_values,
            base_values=base_value,
            data=row.iloc[0].values,
            feature_names=features
        )

        st.success(f"SHAP explanation for predicted class: {pred}")

        # Display Waterfall Plot
        st.write("### Individual Prediction Explanation")

        shap.plots.waterfall(
            explanation,
            show=False
        )

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

        # Display SHAP values in table
        st.write("### SHAP Values")

        shap_df = pd.DataFrame({
            "Feature": features,
            "Feature Value": row.iloc[0].values,
            "SHAP Value": class_values
        })

        shap_df["Impact"] = np.where(
            shap_df["SHAP Value"] > 0,
            "Positive",
            "Negative"
        )

        shap_df["Absolute Impact"] = shap_df["SHAP Value"].abs()

        shap_df = shap_df.sort_values(
            "Absolute Impact",
            ascending=False
        )

        st.dataframe(
            shap_df,
            hide_index=True,
            use_container_width=True
        )

    except ImportError:
        st.warning("SHAP is not installed. Run: pip install shap")

    except Exception as e:
        st.warning(f"Could not generate SHAP output: {e}")
