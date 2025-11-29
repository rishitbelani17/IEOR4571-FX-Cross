import streamlit as st

st.set_page_config(page_title="About Strategy", page_icon="📘", layout="wide")

st.title("📘 Strategy Overview & Notes")

st.write("""
### Exponential Smoothing FX Strategy

This trading model uses **dual exponential smoothing (α/β)** to detect momentum
shifts and trend reversals in FX markets.

#### Key Concepts
- **ES(α)** = slow smoothing → long-term direction  
- **ES(β)** = fast smoothing → short-term momentum  
- Signals generated when **ESβ − ESα crosses thresholds**  
- Optional **θ deceleration exit rule**

#### Why it works
- Captures transitions from range → trend  
- Identifies momentum loss using deceleration  
- Simple, robust, low-parameter system  

This dashboard includes:
- Live FX mode  
- Optimization (α, β, x, θ)  
- Trade log and return distribution  
- Preset manager  
""")

st.markdown("---")
st.subheader("📞 Contact")
st.write("Email: tp2899@columbia.edu")
