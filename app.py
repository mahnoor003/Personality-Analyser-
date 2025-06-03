import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocess import clean_text
from model.bert_model import get_bert_embedding, get_bert_embeddings_batch
from model.predictor import predict_personality,predict_personality_batch
from report.report_generator import generate_report

# ---------------------------
# Enhanced AI Typing Effect
# ---------------------------
def ai_typing_effect(text, delay=0.05):
    output = ""
    placeholder = st.empty()
    with st.container():
        for char in text:
            output += char
            placeholder.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; background-color: rgba(56, 189, 248, 0.1);
                border-radius: 0.5rem; padding: 1rem; border-left: 4px solid #38bdf8; margin: 1rem 0;">
                <div style="font-size: 1.5rem;">🤖</div>
                <div style="flex-grow: 1;">
                    <span style="font-family: 'Courier New', monospace;">{output}</span><span style="animation: blink 1s infinite;">_</span>
                </div>
            </div>
            <style>
                @keyframes blink {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0; }}
                    100% {{ opacity: 1; }}
                }}
            </style>
            """, unsafe_allow_html=True)
            time.sleep(delay)
        placeholder.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; background-color: rgba(56, 189, 248, 0.1);
            border-radius: 0.5rem; padding: 1rem; border-left: 4px solid #38bdf8; margin: 1rem 0;">
            <div style="font-size: 1.5rem;">🤖</div>
            <div style="flex-grow: 1;">
                <span style="font-family: 'Courier New', monospace;">{output}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Personality Predictor Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
    <style>
        :root {
            --primary: #4f46e5;
            --secondary: #10b981;
            --accent: #f59e0b;
            --dark-bg: #1e293b;
            --light-bg: #f8fafc;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--light-bg);
        }
        
        .main {
            background-color: var(--light-bg);
        }
        
        .block-container {
            padding: 2rem;
            border-radius: 12px;
            background-color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        h1 {
            color: var(--primary);
            font-weight: 800;
            border-bottom: 3px solid var(--accent);
            padding-bottom: 0.5rem;
            display: inline-block;
        }
        
        h2 {
            color: var(--primary);
            font-weight: 700;
            position: relative;
            padding-left: 1rem;
        }
        
        h2:before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 4px;
            background: var(--accent);
            border-radius: 2px;
        }
        
        h3 {
            color: var(--secondary);
            font-weight: 600;
        }
        
        .stButton>button, .stDownloadButton>button {
            background: linear-gradient(135deg, var(--primary), #7c3aed);
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button:hover, .stDownloadButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        
        .stRadio>div {
            display: flex;
            gap: 1rem;
        }
        
        .stRadio>div>label {
            flex: 1;
            text-align: center;
            padding: 0.75rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .stRadio>div>label:hover {
            border-color: var(--primary);
            background-color: rgba(79, 70, 229, 0.05);
        }
        
        .stRadio>div>label[data-baseweb="radio"]>div:first-child {
            display: none;
        }
        
        .stRadio>div>label[data-baseweb="radio"]>div:last-child {
            width: 100%;
        }
        
        section[data-testid="stSidebar"]>div {
            background: linear-gradient(180deg, #a18cd1, #fbc2eb);
            padding: 1.5rem;
            border-radius: 0 12px 12px 0;
            color: white;
        }
        
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid var(--accent);
            background-color: rgba(245, 158, 11, 0.1);
        }
        
        .stProgress>div>div>div {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }
        
        .footer {
            background-color: var(--dark-bg);
            color: white;
            font-size: 0.85rem;
            padding: 1.5rem;
            text-align: center;
            margin-top: 3rem;
            border-radius: 12px 12px 0 0;
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: var(--dark-bg);
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 0.5rem;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8rem;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        .trait-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border-left: 4px solid var(--accent);
        }
        
        .trait-card h4 {
            margin: 0;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .trait-card p {
            margin: 0.5rem 0 0;
            color: #64748b;
            font-size: 0.9rem;
        }
        
        @media (prefers-color-scheme: dark) {
            body {
                background-color: #0f172a;
                color: #f8fafc;
            }
            
            .main {
                background-color: #0f172a;
            }
            
            .block-container {
                background-color: #1e293b;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
            }
            
            h1, h2, h3 {
                color: #f8fafc;
            }
            
            .trait-card {
                background-color: #334155;
            }
            
            .trait-card p {
                color: #cbd5e1;
            }
        }
        
        /* Custom upload box styling */
        .upload-box {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
        }
        
        /* Text input styling */
        .stTextArea>div>div>textarea {
            min-height: 200px;
            border-radius: 8px;
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Theme Toggle (Light/Dark)
# ---------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.sidebar:
    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_toggle

# ---------------------------
# Start Page
# ---------------------------
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    # Hero section with Google image (not downloadable)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        <div style="display: flex; justify-content: center;">
            <img src="https://www.gstatic.com/images/branding/product/1x/docs_2020q4_48dp.png" 
                 alt="AI Brain" 
                 style="width: 150px; pointer-events: none; user-select: none;">
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.title("🧠 Personality Predictor Pro")
        st.markdown("""
            <div style="font-size: 1.1rem; color: #4b5563;">
                Unlock the power of AI to analyze personality traits from text content
            </div>
        """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 2rem 0;">
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                <h3 style="margin: 0 0 0.5rem 0;">Deep Analysis</h3>
                <p style="margin: 0; color: #64748b;">Advanced NLP models extract personality insights from text</p>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <h3 style="margin: 0 0 0.5rem 0;">Visual Reports</h3>
                <p style="margin: 0; color: #64748b;">Beautiful visualizations to understand personality traits</p>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔄</div>
                <h3 style="margin: 0 0 0.5rem 0;">Multi-Platform</h3>
                <p style="margin: 0; color: #64748b;">Analyze LinkedIn, GitHub, and compare results</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Motivational quote
    st.markdown("""
        <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 1.5rem; border-radius: 12px; margin: 2rem 0;">
            <div style="font-size: 1.2rem; font-style: italic; margin-bottom: 0.5rem;">"The best way to predict the future is to create it."</div>
            <div style="font-weight: 600;">— Abraham Lincoln</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Start Dashboard", type="primary"):
        st.session_state.started = True
        st.rerun()
    
    # Footer badges
    st.markdown("""
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 3rem;">
            <a href="https://streamlit.io" target="_blank" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/Made_with-Streamlit-FF4B4B?logo=streamlit" alt="Made with Streamlit">
            </a>
            <a href="https://python.org" target="_blank" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python" alt="Python 3.10+">
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    st.stop()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    # Avatar using Google image (not downloadable)
    st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 2rem;">
            <img src="https://www.gstatic.com/images/branding/product/1x/contacts_2022_48dp.png" 
                 width="100" 
                 style="border-radius: 50%; border: 4px solid white; margin-bottom: 1rem; pointer-events: none; user-select: none;">
            <h3 style="color: white; margin: 0;">AI Personality Analyst</h3>
            <div style="color: #e2e8f0; font-size: 0.9rem;">v2.1.0</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation with icons
    section = st.radio(
        "Navigate to Section",
        options=[
            "📘 LinkedIn Analysis", 
            "🐙 GitHub Analysis", 
            "📊 Compare Platforms"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Tooltips for traits - fixed duplicate key issue
    st.markdown("### 🎯 Personality Traits Guide")
    traits = {
        "Openness": "Creativity, curiosity, and preference for novelty",
        "Conscientiousness": "Self-discipline, organization, and dependability",
        "Extraversion": "Sociability, talkativeness, and assertiveness",
        "Agreeableness": "Compassion, cooperativeness, and trust",
        "Neuroticism": "Emotional instability and negative emotions"
    }
    
    for i, (trait, desc) in enumerate(traits.items()):
        st.markdown(f"""
            <div class="tooltip" style="background-color: rgba(255, 255, 255, 0.1);
                border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
                <strong>{trait}</strong>
                <span class="tooltiptext">{desc}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("💡 Upload or paste data to predict personality using NLP and AI.")
    st.caption("🚀 Built with ❤️ using Streamlit")

# ---------------------------
# Main Content
# ---------------------------
if st.button("🤖 Start AI Analysis"):
    st.session_state.started = True
    with st.spinner("AI is analyzing your content..."):
        ai_typing_effect("Hello! I'm analyzing your writing and generating insights just for you...")

# ----------------------------
# LINKEDIN SECTION
# ----------------------------
if section == "📘 LinkedIn Analysis":
    st.header("📘 LinkedIn Personality Analyzer")
    st.markdown("Analyze personality traits from LinkedIn profiles, posts, and experiences.")

    # Mode selection
    mode = st.radio(
        "Choose Input Mode",
        ["📄 Upload CSV", "✍️ Manual Input"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if mode == "📄 Upload CSV":
        st.markdown("""
        <div class="upload-box">
            Upload your LinkedIn CSV file
        </div>
        """, unsafe_allow_html=True)

        file = st.file_uploader(
            "Upload your LinkedIn CSV file",
            type=["csv"],
            label_visibility="collapsed"
        )

        required_columns = {'name', 'about', 'posts', 'experience', 'education'}

        if file:
            try:
                df = pd.read_csv(file)

                if not required_columns.issubset(df.columns):
                    st.error("❌ Required columns missing. Please ensure your CSV contains: name, about, posts, experience, education")
                else:
                    st.success("✅ File validated successfully!")

                    # Sub-mode for analysis
                    sub_mode = st.radio(
                        "Choose Analysis Mode",
                        ["👤 Individual", "👥 All Users"],
                        horizontal=True,
                        label_visibility="collapsed"
                    )

                    if sub_mode == "👤 Individual":
                        selected = st.selectbox("Select User", df['name'].unique())
                        person = df[df['name'] == selected].iloc[0]
                        raw_text = " ".join(str(person[col]) for col in required_columns if pd.notna(person[col]))

                        if raw_text.strip():
                            with st.spinner("🔍 Analyzing personality traits..."):
                                emb = get_bert_embedding(clean_text(raw_text))
                                result = predict_personality(np.array(emb).reshape(1, -1))

                            st.success("✅ Analysis complete!")

                            # Visualization
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                fig = px.bar(
                                    x=list(result.keys()),
                                    y=list(result.values()),
                                    labels={'x': 'Trait', 'y': 'Score'},
                                    color=list(result.keys()),
                                    color_discrete_sequence=px.colors.qualitative.Pastel,
                                    title=f"Personality Traits for {selected}"
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    xaxis_title=None,
                                    yaxis_title="Score",
                                    hovermode="x"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                st.markdown("### 📊 Trait Breakdown")
                                for trait, score in result.items():
                                    st.markdown(f"""
                                        <div style="background-color: #f8fafc; border-radius: 8px; padding: 1rem; 
                                            margin-bottom: 1rem; border-left: 4px solid #4f46e5;">
                                            <h4 style="margin: 0 0 0.25rem 0;">{trait}</h4>
                                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                                <div style="flex-grow: 1; height: 8px; background: #e2e8f0; border-radius: 4px;">
                                                    <div style="width: {score*100}%; height: 100%; background: linear-gradient(90deg, #4f46e5, #8b5cf6); border-radius: 4px;"></div>
                                                </div>
                                                <div style="font-weight: 600; color: #4f46e5;">{score:.2f}</div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)

                            st.markdown("""
                                <div style="background-color: #f8fafc; border-radius: 12px; padding: 1.5rem; margin-top: 2rem;">
                                    <h3>📄 Generate Report</h3>
                                </div>
                            """, unsafe_allow_html=True)

                            if st.button("✨ Generate PDF Report"):
                                with st.spinner("Generating beautiful PDF report..."):
                                    fname = generate_report(selected, result)
                                    with open(fname, "rb") as f:
                                        st.download_button(
                                            "📥 Download Report",
                                            f,
                                            file_name=fname,
                                            help="Download a detailed PDF report of the analysis"
                                        )

                            with st.expander("🔍 Show Raw Analysis Data"):
                                st.json(result)

                    else:  # 👥 All Users
                        df['combined'] = df[list(required_columns)].astype(str).agg(' '.join, axis=1)
                        df['clean'] = df['combined'].apply(clean_text)
                        texts = df['clean'].tolist()

                        st.info("Generating BERT embeddings...")
                        embs = get_bert_embeddings_batch(texts)

                        results = []
                        progress = st.progress(0)
                        for i, emb in enumerate(embs):
                            traits = predict_personality(np.array(emb).reshape(1, -1))
                            results.append(traits)
                            progress.progress(int((i + 1) / len(embs) * 100))
                        progress.empty()

                        traits_df = pd.DataFrame(results)
                        st.dataframe(pd.concat([df[['name']], traits_df], axis=1))

            except Exception as e:
                st.error(f"Error: {e}")

    else:  # ✍️ Manual Input mode
        text = st.text_area(
            "Paste LinkedIn profile content (about, posts, experience, education)",
            placeholder="Paste LinkedIn content here...",
            label_visibility="collapsed"
        )

        if st.button("🔍 Analyze", type="primary"):
            if text.strip():
                with st.spinner("🔍 Analyzing personality traits..."):
                    emb = get_bert_embedding(clean_text(text))
                    result = predict_personality(np.array(emb).reshape(1, -1))

                st.success("✅ Analysis complete!")

                # Visualization
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = px.bar(
                        x=list(result.keys()),
                        y=list(result.values()),
                        labels={'x': 'Trait', 'y': 'Score'},
                        color=list(result.keys()),
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        title="Personality Traits (Manual Input)"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title=None,
                        yaxis_title="Score",
                        hovermode="x"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### 📊 Trait Breakdown")
                    for trait, score in result.items():
                        st.markdown(f"""
                            <div style="background-color: #f8fafc; border-radius: 8px; padding: 1rem; 
                                margin-bottom: 1rem; border-left: 4px solid #4f46e5;">
                                <h4 style="margin: 0 0 0.25rem 0;">{trait}</h4>
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <div style="flex-grow: 1; height: 8px; background: #e2e8f0; border-radius: 4px;">
                                        <div style="width: {score*100}%; height: 100%; background: linear-gradient(90deg, #4f46e5, #8b5cf6); border-radius: 4px;"></div>
                                    </div>
                                    <div style="font-weight: 600; color: #4f46e5;">{score:.2f}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)


                with st.expander("🔎 Show JSON Result"):
                    st.json(result)
            else:
                st.warning("⚠️ Please enter some content to analyze!")


# ----------------------------
# GITHUB SECTION
# ----------------------------
elif section == "🐙 GitHub Analysis":
    st.header("🐙 GitHub Personality Analyzer")
    st.markdown("Analyze personality traits from GitHub profiles, READMEs, and commit messages.")
    
    mode = st.radio(
        "Choose Input Mode",
        ["📄 Upload CSV", "✍ Manual Input"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if mode == "📄 Upload CSV":
        st.markdown("""
        <div class="upload-box">
            Upload GitHub CSV
        </div>
        """, unsafe_allow_html=True)
        file = st.file_uploader(
            "Upload GitHub CSV",
            type=["csv"],
            label_visibility="collapsed"
        )
        
        required_columns = {'Username', 'Name', 'Description', 'Languages', 'Latest Commit', 'README'}
        
        if file:
            df = pd.read_csv(file)
            if not required_columns.issubset(df.columns):
                st.error("❌ Missing required columns. Please ensure your CSV contains: Username, Name, Description, Languages, Latest Commit, README")
            else:
                st.success("✅ File validated successfully!")
                
                sub_mode = st.radio(
                    "Choose Analysis Mode",
                    ["👤 Individual", "👥 All Users"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if sub_mode == "👤 Individual":
                    selected = st.selectbox("Select a User", df['Username'].unique())
                    user = df[df['Username'] == selected].iloc[0]
                    name = user['Name'] or selected
                    raw_text = " ".join(str(user[col]) for col in ['Description', 'Languages', 'Latest Commit', 'README'] if pd.notna(user[col]))
                    
                    if raw_text.strip():
                        with st.spinner("🔍 Analyzing GitHub personality..."):
                            emb = get_bert_embedding(clean_text(raw_text))
                            result = predict_personality(np.array(emb).reshape(1, -1))
                        
                        st.success("✅ Analysis complete!")
                        
                        # Enhanced visualization
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.bar(
                                x=list(result.keys()),
                                y=list(result.values()),
                                labels={'x': 'Trait', 'y': 'Score'},
                                color=list(result.keys()),
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                                title=f"GitHub Personality Traits for {name}"
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                xaxis_title=None,
                                yaxis_title="Score",
                                hovermode="x"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 📊 Trait Breakdown")
                            for trait, score in result.items():
                                st.markdown(f"""
                                    <div style="background-color: #f8fafc; border-radius: 8px; padding: 1rem; 
                                        margin-bottom: 1rem; border-left: 4px solid #10b981;">
                                        <h4 style="margin: 0 0 0.25rem 0;">{trait}</h4>
                                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                                            <div style="flex-grow: 1; height: 8px; background: #e2e8f0; border-radius: 4px;">
                                                <div style="width: {score*100}%; height: 100%; background: linear-gradient(90deg, #10b981, #34d399); border-radius: 4px;"></div>
                                            </div>
                                            <div style="font-weight: 600; color: #10b981;">{score:.2f}</div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        # Report generation
                        st.markdown("""
                            <div style="background-color: #f8fafc; border-radius: 12px; padding: 1.5rem; margin-top: 2rem;">
                                <h3>📄 Generate Report</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        if st.button("✨ Generate PDF Report"):
                            with st.spinner("Generating beautiful PDF report..."):
                                fname = generate_report(name, result, source="GitHub")
                                with open(fname, "rb") as f:
                                    st.download_button(
                                        "📥 Download Report",
                                        f,
                                        file_name=fname,
                                        help="Download a detailed PDF report of the analysis"
                                    )
                        
                        # Raw JSON data
                        with st.expander("🔍 Show Raw Analysis Data"):
                            st.json(result)
                
                else:  # All Users mode
                    with st.spinner("Analyzing all users in the dataset..."):
                        df['text'] = df[['Description', 'Languages', 'Latest Commit', 'README']].astype(str).agg(' '.join, axis=1)
                        df['clean'] = df['text'].apply(clean_text)
                        embs = get_bert_embeddings_batch(df['clean'].tolist())
                        results = [predict_personality(np.array(e).reshape(1, -1)) for e in embs]
                        
                        # Enhanced dataframe display
                        result_df = pd.concat([df[['Username', 'Name']], pd.DataFrame(results)], axis=1)
                        st.dataframe(
                            result_df.style
                            .background_gradient(cmap='Greens', subset=result_df.columns[2:])
                            .format("{:.2f}", subset=result_df.columns[2:]),
                            use_container_width=True
                        )
    
    else:  # Manual Input mode
        text = st.text_area(
            "Paste GitHub README or Commit Messages",
            placeholder="Paste GitHub content here...",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Analyze", type="primary"):
            if text.strip():
                with st.spinner("Analyzing personality traits..."):
                    emb = get_bert_embedding(clean_text(text))
                    result = predict_personality(np.array(emb).reshape(1, -1))
                
                st.success("✅ Analysis complete!")
                
                # Visualization
                fig = px.line_polar(
                    r=list(result.values()),
                    theta=list(result.keys()),
                    line_close=True,
                    color_discrete_sequence=['#10b981'],
                    template="plotly_white",
                    title="GitHub Personality Trait Radar Chart"
                )
                fig.update_traces(fill='toself')
                st.plotly_chart(fig, use_container_width=True)
                
                # Raw data
                with st.expander("🔎 Show JSON Result"):
                    st.json(result)
            else:
                st.warning("⚠️ Please enter some content to analyze!")

# ----------------------------
# COMPARISON SECTION
# ----------------------------
elif section == "📊 Compare Platforms":
    st.header("📊 Compare LinkedIn vs GitHub")
    st.markdown("Compare personality traits between LinkedIn and GitHub profiles.")
    
    # File uploaders in columns
    lnk_file, git_file = st.columns(2)
    
    with lnk_file:
        st.markdown("""
        <div class="upload-box">
            <h3>📘 LinkedIn Data</h3>
        </div>
        """, unsafe_allow_html=True)
        lnk_csv = st.file_uploader(
            "Upload LinkedIn CSV",
            type=["csv"],
            key="lnk_csv",
            label_visibility="collapsed"
        )
    
    with git_file:
        st.markdown("""
        <div class="upload-box">
            <h3>🐙 GitHub Data</h3>
        </div>
        """, unsafe_allow_html=True)
        git_csv = st.file_uploader(
            "Upload GitHub CSV",
            type=["csv"],
            key="git_csv",
            label_visibility="collapsed"
        )
    
    if lnk_csv and git_csv:
        try:
            with st.spinner("Analyzing and comparing data..."):
                lnk_df = pd.read_csv(lnk_csv)
                git_df = pd.read_csv(git_csv)

                if not {'name', 'about', 'posts', 'experience', 'education'}.issubset(lnk_df.columns):
                    st.error("❌ LinkedIn CSV missing required columns.")
                else:
                    lnk_df['combined'] = lnk_df[['name', 'about', 'posts', 'experience', 'education']].astype(str).agg(' '.join, axis=1)
                    lnk_df['clean'] = lnk_df['combined'].apply(clean_text)
                    lnk_traits = predict_personality_batch(lnk_df['clean'].tolist())
                    lnk_avg = pd.DataFrame(lnk_traits).mean()
                
                if 'Latest Commit' not in git_df.columns:
                    st.error("❌ GitHub CSV missing 'Latest Commit'.")
                else:
                    git_df['clean'] = git_df['Latest Commit'].astype(str).apply(clean_text)
                    git_traits = predict_personality_batch(git_df['clean'].tolist())
                    git_avg = pd.DataFrame(git_traits).mean()
                
                if 'lnk_avg' in locals() and 'git_avg' in locals():
                    comparison = pd.DataFrame({
                        "Trait": lnk_avg.index,
                        "LinkedIn": lnk_avg.values,
                        "GitHub": git_avg.values
                    })

                    # Display comparison table
                    st.markdown("### 📈 Trait Comparison Table")
                    st.dataframe(
                        comparison.style
                        .background_gradient(cmap='Purples', subset=['LinkedIn'])
                        .background_gradient(cmap='Greens', subset=['GitHub'])
                        .format({"LinkedIn": "{:.2f}", "GitHub": "{:.2f}"}),
                        use_container_width=True
                    )

                    # Visual comparison
                    st.markdown("### 📊 Platform Comparison")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=comparison["Trait"],
                        y=comparison["LinkedIn"],
                        name="LinkedIn",
                        marker_color="#4f46e5",
                        hovertemplate="<b>LinkedIn</b><br>%{x}: %{y:.2f}<extra></extra>"
                    ))
                    fig.add_trace(go.Bar(
                        x=comparison["Trait"],
                        y=comparison["GitHub"],
                        name="GitHub",
                        marker_color="#10b981",
                        hovertemplate="<b>GitHub</b><br>%{x}: %{y:.2f}<extra></extra>"
                    ))
                    fig.update_layout(
                        barmode='group',
                        title="Personality Trait Comparison: LinkedIn vs GitHub",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title=None,
                        yaxis_title="Score",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Key insights
                    st.markdown("### 🔍 Key Insights")
                    max_linkedin = comparison.loc[comparison['LinkedIn'].idxmax()]
                    max_github = comparison.loc[comparison['GitHub'].idxmax()]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                            <div style="background-color: #eef2ff; border-radius: 12px; padding: 1.5rem; 
                                border-left: 4px solid #4f46e5;">
                                <h4 style="margin: 0 0 1rem 0;">📘 LinkedIn Dominant Trait</h4>
                                <div style="font-size: 2rem; color: #4f46e5; margin-bottom: 0.5rem;">{max_linkedin['Trait']}</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #4f46e5;">{max_linkedin['LinkedIn']:.2f}</div>
                                <p style="margin: 0.5rem 0 0; color: #64748b;">This trait is most prominent in LinkedIn profiles</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                            <div style="background-color: #ecfdf5; border-radius: 12px; padding: 1.5rem; 
                                border-left: 4px solid #10b981;">
                                <h4 style="margin: 0 0 1rem 0;">🐙 GitHub Dominant Trait</h4>
                                <div style="font-size: 2rem; color: #10b981; margin-bottom: 0.5rem;">{max_github['Trait']}</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: #10b981;">{max_github['GitHub']:.2f}</div>
                                <p style="margin: 0.5rem 0 0; color: #64748b;">This trait is most prominent in GitHub profiles</p>
                            </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error during comparison: {str(e)}")
            st.exception(e)

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
    <div class='footer'>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
            <a href="#" style="color: white; text-decoration: none;">About</a>
            <a href="#" style="color: white; text-decoration: none;">Privacy</a>
            <a href="#" style="color: white; text-decoration: none;">Terms</a>
            <a href="#" style="color: white; text-decoration: none;">Contact</a>
        </div>
        <div>© 2023 Personality Predictor Pro. All rights reserved.</div>
    </div>
""", unsafe_allow_html=True)