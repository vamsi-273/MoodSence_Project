import streamlit as st
import pandas as pd
import plotly.express as px


# ---------------- EMOTION PIE CHART ----------------

def show_emotion_distribution(history, classes):

    if len(history) == 0:
        st.info("No emotion data yet")
        return

    emotions = [classes[i] for i in history]

    df = pd.DataFrame({"Emotion": emotions})

    counts = df["Emotion"].value_counts().reset_index()
    counts.columns = ["Emotion", "Count"]

    fig = px.pie(
        counts,
        names="Emotion",
        values="Count",
        title="Emotion Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------- MOOD SCORE ----------------

def compute_mood_score(history, classes):

    if len(history) == 0:
        return 50

    positive = ["Happiness", "Surprise"]
    negative = ["Sadness", "Anger", "Fear", "Disgust"]

    score = 50

    for i in history:

        emotion = classes[i]

        if emotion in positive:
            score += 2

        if emotion in negative:
            score -= 2

    score = max(0, min(100, score))

    return score


# ---------------- STRESS LEVEL ----------------

def compute_stress(history, classes):

    if len(history) == 0:
        return 0

    stress_emotions = ["Sadness", "Anger", "Fear"]

    stress_count = 0

    for i in history:

        emotion = classes[i]

        if emotion in stress_emotions:
            stress_count += 1

    stress_percentage = (stress_count / len(history)) * 100

    return round(stress_percentage)


# ---------------- DAILY REPORT ----------------

def show_daily_report(history, classes):

    if len(history) == 0:
        st.info("No daily report available yet")
        return

    emotions = [classes[i] for i in history]

    df = pd.DataFrame({"Emotion": emotions})

    report = df["Emotion"].value_counts().reset_index()
    report.columns = ["Emotion", "Count"]

    st.subheader("Daily Emotion Report")

    st.dataframe(report, use_container_width=True)