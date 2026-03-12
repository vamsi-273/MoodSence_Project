import streamlit as st

def show_emotion_card(emotion,confidence,uncertainty,stability,stress,model_used,recommend):

    st.markdown(f"""
<div class="card">

<h2>{recommend['emoji']} {emotion}</h2>

<p class="metric">Confidence: {confidence:.2f}</p>

<p class="metric">Uncertainty: {uncertainty:.2f}</p>

<p class="metric">Stability: {stability:.2f}</p>

<p class="metric">Stress Level: {stress:.2f}</p>

<p class="metric">Model Used: {model_used}</p>

<div class="recommend-box">

<h4>Recommended Activity</h4>

<p><b>{recommend['activity']}</b></p>

<p>{recommend['suggestion']}</p>

</div>

</div>
""", unsafe_allow_html=True)