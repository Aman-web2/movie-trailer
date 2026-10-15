"""
app.py
CineSense AI — Movie Trailer Sentiment Analyzer
Main Streamlit application: navigation, movie selection, trailer selection,
sentiment dashboard, and analytics.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from movie_api import search_movies, format_movie, has_credentials as tmdb_ok
from youtube import search_trailers, get_video_comments, embed_url, has_credentials as yt_ok
from sentiment import analyze_comments, build_summary

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CineSense AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — cinematic dark theme ─────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, [data-testid="stApp"] {
        background: #0a0a0f;
        color: #e8e8f0;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%);
        border-right: 1px solid #1e1e3f;
    }
    [data-testid="stSidebar"] * { color: #c8c8e8 !important; }

    /* ── Headers ── */
    h1 { color: #a855f7 !important; letter-spacing: -0.5px; }
    h2 { color: #7c3aed !important; }
    h3 { color: #c084fc !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #13132b 0%, #1a1a38 100%);
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 16px !important;
    }
    [data-testid="stMetricValue"] { color: #a855f7 !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #9999cc !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* ── Text inputs ── */
    .stTextInput > div > div > input {
        background: #13132b !important;
        border: 1px solid #2d2d5e !important;
        border-radius: 8px !important;
        color: #e8e8f0 !important;
    }

    /* ── Movie card ── */
    .movie-card {
        background: linear-gradient(135deg, #13132b, #1a1a38);
        border: 1px solid #2d2d5e;
        border-radius: 14px;
        padding: 14px;
        transition: border-color 0.2s;
        height: 100%;
    }
    .movie-card:hover { border-color: #7c3aed; }
    .movie-card img { border-radius: 8px; width: 100%; }
    .movie-title { font-weight: 700; font-size: 0.95rem; margin: 8px 0 4px; color: #e8e8f0; }
    .movie-meta  { font-size: 0.80rem; color: #9999cc; }
    .badge-rating {
        display: inline-block;
        background: #7c3aed22;
        border: 1px solid #7c3aed55;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        color: #c084fc;
    }

    /* ── Trailer card ── */
    .trailer-card {
        background: #13132b;
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 12px;
        cursor: pointer;
        transition: border-color 0.2s;
    }
    .trailer-card img { border-radius: 8px; width: 100%; }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #2d2d5e;
        margin: 2rem 0;
    }

    /* ── Info banner ── */
    .info-banner {
        background: #13132b;
        border-left: 4px solid #7c3aed;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #c8c8e8;
        margin-bottom: 1rem;
    }

    /* ── Insight box ── */
    .insight-box {
        background: linear-gradient(135deg, #13132b, #1a1a38);
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 1rem;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* ── Feature cards on home ── */
    .feature-card {
        background: linear-gradient(135deg, #13132b, #1a1a38);
        border: 1px solid #2d2d5e;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        height: 100%;
    }
    .feature-icon { font-size: 2.2rem; margin-bottom: 10px; }
    .feature-title { font-weight: 700; color: #c084fc; margin-bottom: 8px; }
    .feature-desc { font-size: 0.88rem; color: #9999cc; line-height: 1.5; }

    /* ── Hero section ── */
    .hero-title {
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 800;
        background: linear-gradient(90deg, #a855f7, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #9999cc;
        margin-top: 0.75rem;
        line-height: 1.6;
    }

    /* ── Step badge ── */
    .step-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        border-radius: 50%;
        width: 28px; height: 28px;
        line-height: 28px;
        text-align: center;
        font-weight: 700;
        font-size: 0.85rem;
        color: white;
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "page":             "home",
    "search_query":     "",
    "search_results":   [],
    "selected_movie":   None,
    "trailers":         [],
    "selected_trailer": None,
    "comments":         [],
    "scored":           [],
    "summary":          {},
    "using_demo":       False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown("## 🎬 CineSense AI")
        st.markdown("---")
        pages = {
            "🏠 Home":      "home",
            "🔍 Search":    "search",
            "📊 Dashboard": "dashboard",
            "ℹ️ About":      "about",
        }
        for label, key in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        # Credential status pills
        tmdb_status = "🟢 TMDB connected"   if tmdb_ok() else "🔴 TMDB key missing"
        yt_status   = "🟢 YouTube connected" if yt_ok()   else "🔴 YouTube key missing"
        st.markdown(f"`{tmdb_status}`")
        st.markdown(f"`{yt_status}`")

        if not tmdb_ok() or not yt_ok():
            st.markdown("---")
            st.markdown("**Setup:** Add keys to `.env` — see the About page.")

        st.markdown("---")
        st.caption("v1.0 · Powered by VADER, TMDB & YouTube")


# ── Page: Home ─────────────────────────────────────────────────────────────────

def _page_home():
    # Hero
    col_logo, col_hero = st.columns([1, 3])
    with col_logo:
        try:
            st.image("assets/logp.png", width=150)
        except Exception:
            st.markdown("🎬")
    with col_hero:
        st.markdown(
            '<div class="hero-title">Discover What the<br>Audience Really Feels</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-sub">CineSense AI analyses real YouTube trailer comments '
            'with VADER sentiment analysis to reveal how audiences react — '
            'before you even buy a ticket.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Quick search
    st.markdown("### 🔍 Search a Movie")
    q_col, btn_col = st.columns([5, 1])
    with q_col:
        query = st.text_input(
            "Movie title",
            placeholder="e.g. Inception, Dune, Spider-Man…",
            label_visibility="collapsed",
            key="home_search_input",
        )
    with btn_col:
        go = st.button("Search", type="primary", use_container_width=True)

    if go:
        if not query.strip():
            st.warning("Please enter a movie title.")
        else:
            st.session_state.search_query   = query
            st.session_state.search_results = []
            st.session_state.page           = "search"
            st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("### ✨ What CineSense AI Does")
    c1, c2, c3 = st.columns(3)
    cards = [
        ("🎯", "Sentiment Analysis",
         "VADER classifies each comment as Positive, Negative, or Neutral "
         "using linguistic rules — no training data required."),
        ("📊", "Interactive Analytics",
         "Visualise sentiment distribution with donut charts and bar graphs. "
         "Filter, search, and export results as CSV."),
        ("🎥", "Audience Insights",
         "Understand the dominant mood, see example reactions, and get a "
         "plain-English summary of what viewers think."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # How it works
    st.markdown("### 🚀 How It Works")
    steps = [
        ("Search", "Type any movie title and pick from TMDB results."),
        ("Pick a Trailer", "Choose an official trailer from YouTube."),
        ("Analyze", "CineSense fetches comments and scores each one."),
        ("Explore", "Interact with charts, filter comments, and download."),
    ]
    s_cols = st.columns(4)
    for i, (col, (step, desc)) in enumerate(zip(s_cols, steps), 1):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon"><span class="step-badge">{i}</span></div>'
                f'<div class="feature-title">{step}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Page: Search ──────────────────────────────────────────────────────────────

def _page_search():
    st.markdown("## 🔍 Movie Search")

    # Search bar
    q_col, btn_col = st.columns([5, 1])
    with q_col:
        query = st.text_input(
            "Search movies",
            value=st.session_state.search_query,
            placeholder="Enter a movie title…",
            label_visibility="collapsed",
            key="search_page_input",
        )
    with btn_col:
        go = st.button("Search", type="primary", use_container_width=True, key="search_go")

    if go or (query and not st.session_state.search_results):
        if not query.strip():
            st.warning("Please enter a movie title.")
        elif not tmdb_ok():
            _show_setup_warning("TMDB")
        else:
            st.session_state.search_query = query
            with st.spinner("Searching TMDB…"):
                raw = search_movies(query)
            if isinstance(raw, dict) and "error" in raw:
                st.error(f"❌ {raw['error']}")
                st.session_state.search_results = []
            else:
                st.session_state.search_results = [format_movie(m) for m in raw[:12]]
                st.session_state.selected_movie = None

    results = st.session_state.search_results
    if not results:
        if not tmdb_ok():
            _show_setup_warning("TMDB")
        return

    st.markdown(f"**{len(results)} results for \"{st.session_state.search_query}\"**")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Movie grid — 4 per row
    cols_per_row = 4
    for row_start in range(0, len(results), cols_per_row):
        row_movies = results[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, movie in zip(cols, row_movies):
            with col:
                _render_movie_card(movie)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Show selected movie details + trailer section
    if st.session_state.selected_movie:
        _render_movie_detail(st.session_state.selected_movie)


def _render_movie_card(movie: dict):
    """Render a clickable movie poster card."""
    poster = movie.get("poster") or "https://via.placeholder.com/500x750?text=No+Poster"
    rating = movie.get("rating", 0)
    year   = movie.get("release", "N/A")

    st.markdown(
        f'<div class="movie-card">'
        f'<img src="{poster}" alt="{movie["title"]}">'
        f'<div class="movie-title">{movie["title"]}</div>'
        f'<div class="movie-meta">{year} &nbsp; '
        f'<span class="badge-rating">⭐ {rating:.1f}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("Select", key=f"sel_{movie['id']}", use_container_width=True):
        st.session_state.selected_movie   = movie
        st.session_state.trailers         = []
        st.session_state.selected_trailer = None
        st.session_state.comments         = []
        st.session_state.scored           = []
        st.session_state.summary          = {}
        st.session_state.using_demo       = False
        st.rerun()


def _render_movie_detail(movie: dict):
    """Render full movie info + trailer picker."""
    st.markdown(f"## 🎬 {movie['title']}")

    d_col, info_col = st.columns([1, 3])
    with d_col:
        if movie.get("poster"):
            st.image(movie["poster"], width=220)
    with info_col:
        st.markdown(f"**Release year:** {movie['release']}")
        st.markdown(f"**TMDB Rating:** ⭐ {movie['rating']:.1f} / 10 ({movie['votes']:,} votes)")
        if movie.get("genres"):
            st.markdown(f"**Genres:** {', '.join(movie['genres'])}")
        st.markdown(f"**Overview:** {movie['overview']}")
        st.markdown(
            '<div class="info-banner">ℹ️ TMDB ratings reflect critic/user scores on TMDB and '
            'are separate from the YouTube comment sentiment below.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    _render_trailer_section(movie)


def _render_trailer_section(movie: dict):
    """Search YouTube for trailers and let the user pick one."""
    st.markdown("### 🎥 Select a Trailer")

    if not st.session_state.trailers:
        if not yt_ok():
            _show_setup_warning("YouTube")
            _offer_demo(movie["title"])
            return

        with st.spinner("Searching YouTube for trailers…"):
            result = search_trailers(movie["title"])

        if isinstance(result, dict) and "error" in result:
            st.error(f"❌ {result['error']}")
            _offer_demo(movie["title"])
            return

        st.session_state.trailers = result

    trailers = st.session_state.trailers
    if not trailers:
        st.info("No trailers found.")
        return

    # Show trailer cards
    t_cols_per_row = 3
    for row_start in range(0, len(trailers), t_cols_per_row):
        row_t = trailers[row_start: row_start + t_cols_per_row]
        t_cols = st.columns(t_cols_per_row)
        for col, t in zip(t_cols, row_t):
            with col:
                if t.get("thumbnail"):
                    st.image(t["thumbnail"], use_container_width=True)
                st.markdown(f"**{t['title'][:60]}{'…' if len(t['title'])>60 else ''}**")
                st.caption(t.get("channel", ""))
                if st.button("Analyze this trailer", key=f"tsel_{t['video_id']}", use_container_width=True):
                    st.session_state.selected_trailer = t
                    st.session_state.comments         = []
                    st.session_state.scored           = []
                    st.session_state.summary          = {}
                    st.session_state.using_demo       = False
                    st.rerun()

    # Once a trailer is selected, show embed + fetch comments
    if st.session_state.selected_trailer:
        _render_analysis_section(st.session_state.selected_trailer)


def _render_analysis_section(trailer: dict):
    """Embed trailer, fetch comments, run analysis, and navigate to dashboard."""
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(f"### ▶️ {trailer['title']}")
    st.markdown(
        f'<iframe width="100%" height="380" src="{embed_url(trailer["video_id"])}" '
        f'frameborder="0" allowfullscreen style="border-radius:12px;"></iframe>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.scored:
        if st.button("🔍 Analyze Comments", type="primary"):
            _run_analysis(trailer["video_id"])
    else:
        st.success("✅ Analysis complete! View results on the Dashboard.")
        if st.button("📊 Go to Dashboard", type="primary"):
            st.session_state.page = "dashboard"
            st.rerun()


def _run_analysis(video_id: str):
    """Fetch comments and run sentiment analysis, storing results in session state."""
    with st.spinner("Fetching YouTube comments…"):
        result = get_video_comments(video_id)

    if isinstance(result, dict) and "error" in result:
        st.error(f"❌ {result['error']}")
        _offer_demo(st.session_state.selected_movie["title"] if st.session_state.selected_movie else "")
        return

    if not result:
        st.warning("No comments were retrieved.")
        return

    with st.spinner("Running sentiment analysis…"):
        scored  = analyze_comments(result)
        summary = build_summary(scored)

    st.session_state.comments   = result
    st.session_state.scored     = scored
    st.session_state.summary    = summary
    st.session_state.using_demo = False
    st.session_state.page       = "dashboard"
    st.rerun()


def _offer_demo(movie_title: str):
    """Offer to run analysis on labeled demo comments when live API is unavailable."""
    st.markdown(
        '<div class="info-banner">💡 <strong>Demo mode:</strong> Your YouTube API key is '
        'unavailable or quota is exceeded. You can run analysis on sample comments instead. '
        'These are <em>not</em> real YouTube comments.</div>',
        unsafe_allow_html=True,
    )
    if st.button("▶ Run Demo Analysis (sample comments)", key="demo_btn"):
        demo = [
            "This trailer gave me chills, absolutely breathtaking visuals!",
            "I'm so hyped for this movie, can't wait for the release!",
            "The cinematography looks stunning, Oscar worthy.",
            "Best trailer I've seen this year by far.",
            "That plot twist at the end — wow!",
            "Looks generic and predictable honestly.",
            "Not a fan of the casting choices here.",
            "The CGI looks terrible and overdone.",
            "Disappointing compared to the original.",
            "Meh, just another sequel nobody asked for.",
            "Coming out next month, so excited!",
            "The soundtrack is available on Spotify now.",
            "New director for this installment apparently.",
            "Premiering at a film festival first.",
            "Tickets go on sale this Friday.",
        ]
        with st.spinner("Running demo analysis…"):
            scored  = analyze_comments(demo)
            summary = build_summary(scored)
        st.session_state.comments   = demo
        st.session_state.scored     = scored
        st.session_state.summary    = summary
        st.session_state.using_demo = True
        st.session_state.page       = "dashboard"
        st.rerun()


# ── Page: Dashboard ────────────────────────────────────────────────────────────

def _page_dashboard():
    scored  = st.session_state.scored
    summary = st.session_state.summary
    movie   = st.session_state.selected_movie
    trailer = st.session_state.selected_trailer

    st.markdown("## 📊 Sentiment Dashboard")

    if not scored:
        st.info("No analysis results yet. Go to Search, pick a movie and trailer, then run analysis.")
        if st.button("🔍 Go to Search"):
            st.session_state.page = "search"
            st.rerun()
        return

    # Demo banner
    if st.session_state.using_demo:
        st.warning(
            "⚠️ **Demo mode:** The results below are based on sample comments, "
            "not real YouTube data. Connect your YouTube API key to analyse live comments."
        )

    # Context header
    if movie:
        m_col, spacer = st.columns([4, 1])
        with m_col:
            st.markdown(f"**Movie:** {movie['title']} ({movie['release']})")
        if trailer:
            st.markdown(f"**Trailer:** {trailer['title']}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── KPI metrics ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💬 Comments",  summary["total"])
    k2.metric("😊 Positive",  f"{summary['positive_pct']}%")
    k3.metric("😡 Negative",  f"{summary['negative_pct']}%")
    k4.metric("😐 Neutral",   f"{summary['neutral_pct']}%")
    k5.metric("📈 Avg Score", f"{summary['avg_score']:.3f}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    chart_df = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count":     [summary["positive"], summary["negative"], summary["neutral"]],
        "Percent":   [summary["positive_pct"], summary["negative_pct"], summary["neutral_pct"]],
    })

    COLOR_MAP = {
        "Positive": "#a855f7",
        "Negative": "#ef4444",
        "Neutral":  "#3b82f6",
    }

    left_c, right_c = st.columns(2)

    with left_c:
        st.markdown("#### Sentiment Distribution")
        donut = px.pie(
            chart_df, names="Sentiment", values="Count",
            hole=0.52,
            color="Sentiment",
            color_discrete_map=COLOR_MAP,
        )
        donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8f0",
            legend=dict(font=dict(color="#e8e8f0")),
            margin=dict(t=20, b=20),
        )
        donut.update_traces(textfont_color="#ffffff")
        st.plotly_chart(donut, use_container_width=True)

    with right_c:
        st.markdown("#### Sentiment Comparison")
        bar = px.bar(
            chart_df, x="Sentiment", y="Count",
            text="Percent",
            color="Sentiment",
            color_discrete_map=COLOR_MAP,
        )
        bar.update_traces(texttemplate="%{text}%", textposition="outside")
        bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8f0",
            showlegend=False,
            yaxis=dict(gridcolor="#2d2d5e"),
            xaxis=dict(gridcolor="#2d2d5e"),
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(bar, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Audience insights ────────────────────────────────────────────────────
    _render_insights(scored, summary)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Comment table with filters ───────────────────────────────────────────
    st.markdown("#### 💬 Comment-Level Analysis")

    st.markdown(
        '<div class="info-banner">ℹ️ Sentiment analysis is an estimate. '
        'VADER may not perfectly interpret sarcasm, slang, or cultural context.</div>',
        unsafe_allow_html=True,
    )

    df = pd.DataFrame(scored).rename(columns={"text": "Comment", "label": "Sentiment", "score": "Score"})

    f_col1, f_col2 = st.columns([3, 2])
    with f_col1:
        search_term = st.text_input("🔎 Search within comments", placeholder="Type to filter…", key="comment_search")
    with f_col2:
        sentiment_filter = st.selectbox("Filter by sentiment", ["All", "Positive", "Negative", "Neutral"], key="sent_filter")

    view = df.copy()
    if sentiment_filter != "All":
        view = view[view["Sentiment"] == sentiment_filter]
    if search_term.strip():
        view = view[view["Comment"].str.contains(search_term.strip(), case=False, na=False)]

    st.caption(f"Showing {len(view)} of {len(df)} comments")
    st.dataframe(
        view.reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # ── CSV download ─────────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename  = f"cinesense_{movie['title'].replace(' ','_') if movie else 'analysis'}.csv"
    st.download_button(
        "⬇️ Download Full Results as CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def _render_insights(scored: list, summary: dict):
    """Display an audience insights summary section."""
    st.markdown("#### 🧠 Audience Insights")

    dominant = summary["dominant"]
    avg      = summary["avg_score"]

    # Overall reaction sentence
    if dominant == "Positive":
        reaction = "The overall audience reaction is **positive**. Most viewers seem excited about this trailer."
    elif dominant == "Negative":
        reaction = "The overall audience reaction is **negative**. Many viewers expressed disappointment or criticism."
    else:
        reaction = "The overall audience reaction is **mixed or neutral**. Opinions appear divided."

    st.markdown(
        f'<div class="insight-box">'
        f'<p style="font-size:1.05rem;">{reaction}</p>'
        f'<p style="color:#9999cc; font-size:0.9rem;">Average sentiment score: '
        f'<strong style="color:#a855f7">{avg:.4f}</strong> '
        f'(range: −1.0 = most negative → +1.0 = most positive)</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Example comments
    positives = [s for s in scored if s["label"] == "Positive"]
    negatives = [s for s in scored if s["label"] == "Negative"]

    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("**😊 Top Positive Comments**")
        top_pos = sorted(positives, key=lambda x: x["score"], reverse=True)[:3]
        for c in top_pos:
            st.markdown(f"> {c['text'][:160]}{'…' if len(c['text'])>160 else ''}")
        if not top_pos:
            st.caption("No positive comments found.")

    with ex_col2:
        st.markdown("**😡 Top Negative Comments**")
        top_neg = sorted(negatives, key=lambda x: x["score"])[:3]
        for c in top_neg:
            st.markdown(f"> {c['text'][:160]}{'…' if len(c['text'])>160 else ''}")
        if not top_neg:
            st.caption("No negative comments found.")

    st.markdown(
        '<div class="info-banner">⚠️ Audience sentiment from trailer comments does not '
        'predict box-office performance or a film\'s overall quality. '
        'TMDB ratings and comment sentiment are independent signals.</div>',
        unsafe_allow_html=True,
    )


# ── Page: About ───────────────────────────────────────────────────────────────

def _page_about():
    st.markdown("## ℹ️ About CineSense AI")

    st.markdown("""
**CineSense AI** is an open-source Movie Trailer Sentiment Analyzer built with:

| Component | Technology |
|---|---|
| Web interface | Streamlit |
| Movie data | TMDB API v3 |
| Trailer & comments | YouTube Data API v3 |
| Sentiment engine | VADER (vaderSentiment) |
| Charts | Plotly |

---

### 🔑 API Key Setup

**TMDB**
1. Create a free account at [themoviedb.org](https://www.themoviedb.org/)
2. Go to Settings → API → Request an API key (v3 auth)
3. Copy the key into `.env` as `TMDB_API_KEY`

**YouTube Data API v3**
1. Open [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable **YouTube Data API v3**
3. Credentials → Create API Key
4. Copy into `.env` as `YOUTUBE_API_KEY`

Your `.env` should look like:
```
TMDB_API_KEY=your_tmdb_v3_key_here
YOUTUBE_API_KEY=your_youtube_api_key_here
```

---

### ⚠️ Limitations
- VADER does not understand sarcasm, irony, or cultural context perfectly.
- YouTube API has a daily quota limit (10 000 units by default).
- Comment sentiment does not predict box-office results or film quality.
- Only top-level comments (up to 100) are fetched per video.
""")


# ── Helper utilities ───────────────────────────────────────────────────────────

def _show_setup_warning(service: str):
    key_name = "TMDB_API_KEY" if service == "TMDB" else "YOUTUBE_API_KEY"
    st.error(
        f"❌ {service} API key not found. "
        f"Add `{key_name}=your_key` to your `.env` file and restart the app. "
        f"See the About page for setup instructions."
    )


# ── Main router ────────────────────────────────────────────────────────────────

def main():
    _sidebar()

    page = st.session_state.page
    if page == "home":
        _page_home()
    elif page == "search":
        _page_search()
    elif page == "dashboard":
        _page_dashboard()
    elif page == "about":
        _page_about()
    else:
        _page_home()


if __name__ == "__main__":
    main()
