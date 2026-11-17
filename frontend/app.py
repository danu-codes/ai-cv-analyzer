import streamlit as st
import requests
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="AI CV Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

# --- Custom CSS for Skill Badges & Suggestion Cards ---
st.markdown(
    """
    <style>
    .skill-badge-match {
        display: inline-block;
        background-color: #d1fae5;
        color: #065f46;
        padding: 5px 12px;
        margin: 4px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.88rem;
        border: 1px solid #a7f3d0;
    }
    .skill-badge-missing {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        padding: 5px 12px;
        margin: 4px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.88rem;
        border: 1px solid #fecaca;
    }
    .role-banner {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 18px;
        color: #1e3a8a;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_score_gauge(score: int):
    """Creates a clean radial gauge chart for the Overall CV Score."""
    if score >= 80:
        bar_color = "#10b981"  # Emerald green
    elif score >= 60:
        bar_color = "#f59e0b"  # Amber
    else:
        bar_color = "#ef4444"  # Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 36}},
            title={"text": "Overall ATS & Model Score", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 50], "color": "#fef2f2"},
                    {"range": [50, 75], "color": "#fffbeb"},
                    {"range": [75, 100], "color": "#ecfdf5"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def check_backend_health() -> bool:
    """Checks if the FastAPI backend is running and the model is loaded."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return res.status_code == 200 and res.json().get("model_loaded", False)
    except requests.RequestException:
        return False


# --- Sidebar: API Status & Instructions ---
with st.sidebar:
    st.header("System Status")
    if check_backend_health():
        st.success("FastAPI Backend: Online & Model Loaded")
    else:
        st.error("FastAPI Backend: Offline")
        st.caption("Make sure `uvicorn app.main:app --reload` is running in your backend terminal.")

    st.divider()
    st.subheader("How It Works")
    st.markdown(
        """
        1. **Upload** your resume (`.pdf` or `.docx`).
        2. *(Optional)* **Paste** a target Job Description to calculate semantic similarity.
        3. **Review** your custom model's predicted role, skill gaps, and actionable improvements.
        """
    )

# --- Main Header ---
st.title("📄 AI-Powered CV Analyzer")
st.caption("Evaluates resume structure, skill alignment, and measurable impact using your custom ML pipeline.")

# --- Input Section ---
col_upload, col_jd = st.columns([1, 1.2], gap="large")

with col_upload:
    st.subheader("1. Upload Your CV")
    uploaded_file = st.file_uploader(
        "Drag and drop your resume here",
        type=["pdf", "docx"],
        help="Maximum file size: 5 MB",
    )

with col_jd:
    st.subheader("2. Target Job Description (Optional)")
    job_description = st.text_area(
        "Paste the job posting to measure exact role fit & missing keywords",
        height=145,
        placeholder="e.g., Looking for a Python Backend Engineer with FastAPI, Docker, SQL, and REST API experience...",
    )

analyze_clicked = st.button("Analyze CV", type="primary", use_container_width=True)

# --- Analysis Execution & Results Dashboard ---
if analyze_clicked:
    if not uploaded_file:
        st.warning("Please upload a `.pdf` or `.docx` CV first.")
    else:
        with st.spinner("Extracting text and running custom ML inference..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                data = {"job_description": job_description} if job_description.strip() else {}

                response = requests.post(
                    f"{API_BASE_URL}/analyze",
                    files=files,
                    data=data,
                    timeout=30,
                )

                if response.status_code != 200:
                    error_detail = response.json().get("detail", "Unknown error occurred.")
                    st.error(f"Analysis failed: {error_detail}")
                else:
                    result = response.json()
                    st.divider()

                    # 1. Predicted Role Banner
                    detected_role = result["extracted_profile"]["detected_role"]
                    st.markdown(
                        f'<div class="role-banner"><strong>Custom Model Classification:</strong> {detected_role}</div>',
                        unsafe_allow_html=True,
                    )

                    # 2. Score Gauge & Sub-Metrics
                    gauge_col, metrics_col = st.columns([1, 1.3], gap="large")

                    with gauge_col:
                        st.plotly_chart(
                            render_score_gauge(result["overall_score"]),
                            use_container_width=True,
                        )

                    with metrics_col:
                        st.subheader("Score Breakdown")
                        breakdown = result["score_breakdown"]

                        st.write(f"**Skills & Role Match:** {breakdown['skills_match']}%")
                        st.progress(breakdown["skills_match"] / 100)

                        st.write(f"**Impact & Quantifiable Metrics:** {breakdown['impact_and_metrics']}%")
                        st.progress(breakdown["impact_and_metrics"] / 100)

                        st.write(f"**Formatting & ATS Sections:** {breakdown['formatting_and_sections']}%")
                        st.progress(breakdown["formatting_and_sections"] / 100)

                    st.divider()

                    # 3. Matched vs. Missing Skills Badges
                    st.subheader("Skill Gap Analysis")
                    skill_col1, skill_col2 = st.columns(2, gap="large")

                    matched = result["extracted_profile"]["matched_skills"]
                    missing = result["extracted_profile"]["missing_skills"]

                    with skill_col1:
                        st.markdown("**Matched Skills**")
                        if matched:
                            badges_html = "".join(
                                [f'<span class="skill-badge-match">✓ {s}</span>' for s in matched]
                            )
                            st.markdown(badges_html, unsafe_allow_html=True)
                        else:
                            st.info("No matching skills detected from the target taxonomy.")

                    with skill_col2:
                        st.markdown("**Missing / Recommended Keywords**")
                        if missing:
                            badges_html = "".join(
                                [f'<span class="skill-badge-missing">+ {s}</span>' for s in missing]
                            )
                            st.markdown(badges_html, unsafe_allow_html=True)
                        else:
                            st.success("Great job! No critical skill gaps detected.")

                    st.divider()

                    # 4. Actionable Improvement Suggestions
                    st.subheader("Recommended Changes & Improvements")
                    suggestions = result["suggested_changes"]

                    if not suggestions:
                        st.success("Your CV looks strong across all structural and impact checks!")
                    else:
                        for item in suggestions:
                            severity = item["severity"].lower()
                            label = f"[{item['severity'].upper()}] {item['section']} — {item['issue']}"

                            with st.expander(label, expanded=True):
                                if item.get("original_text"):
                                    st.markdown(f"**Detected Text:** `{item['original_text']}`")
                                st.markdown(f"**How to Fix:** {item['recommendation']}")

            except requests.ConnectionError:
                st.error(
                    "Could not connect to the FastAPI server at `http://127.0.0.1:8000`. "
                    "Verify that your backend server is running."
                )