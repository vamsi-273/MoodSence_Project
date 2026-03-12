def load_css():

    return """
<style>

/* -------- MAIN APP BACKGROUND -------- */

.stApp {
background: linear-gradient(135deg,#1e3c72,#2a5298);
color:white;
font-family: 'Segoe UI', sans-serif;
}

/* -------- TITLE -------- */

.main-title{
font-size:42px;
font-weight:700;
text-align:center;
margin-bottom:30px;
color:white;
}

/* -------- CARD DESIGN -------- */

.card{

background: rgba(255,255,255,0.08);

backdrop-filter: blur(12px);

border-radius:18px;

padding:25px;

box-shadow: 0px 8px 25px rgba(0,0,0,0.3);

margin-bottom:25px;

border:1px solid rgba(255,255,255,0.15);
}

/* -------- EMOTION TITLE -------- */

.card h2{
font-size:32px;
margin-bottom:15px;
}

/* -------- METRIC TEXT -------- */

.metric{

font-size:18px;

margin-top:8px;

margin-bottom:8px;

color:#f0f0f0;

}

/* -------- RECOMMENDATION BOX -------- */

.recommend-box{

background: rgba(0,0,0,0.25);

padding:18px;

border-radius:12px;

margin-top:15px;

}

/* -------- BUTTON STYLE -------- */

.stButton>button {

background: linear-gradient(135deg,#ff7b00,#ff3c00);

color:white;

border:none;

padding:10px 25px;

border-radius:10px;

font-size:16px;

font-weight:600;

transition:0.3s;

}

.stButton>button:hover {

transform:scale(1.05);

box-shadow:0px 4px 15px rgba(0,0,0,0.3);

}

/* -------- CHART AREA -------- */

.block-container {

padding-top:2rem;

}

/* -------- PROGRESS BAR -------- */

.stProgress > div > div > div > div {

background-color:#00ffcc;

}

/* -------- SCROLL BAR -------- */

::-webkit-scrollbar {

width:8px;

}

::-webkit-scrollbar-thumb {

background:#3f6fd6;

border-radius:10px;

}

</style>
"""