import os
import re
import joblib
import numpy as np
from typing import Optional, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import HTTPException

from app.schemas.response import (
    CVAnalysisResponse,
    ScoreBreakdown,
    ExtractedProfile,
    SuggestedChange,
)

ROLE_SKILL_BANK = {
    "AI / Machine Learning Engineer": [
        "python", "pytorch", "tensorflow", "scikit-learn", "transformers",
        "nlp", "computer vision", "sql", "docker", "git", "pandas", "fastapi"
    ],
    "Backend Software Engineer": [
        "python", "java", "node.js", "fastapi", "django", "spring boot",
        "sql", "postgresql", "mongodb", "docker", "redis", "git", "rest api"
    ],
    "Frontend Developer": [
        "javascript", "typescript", "react", "next.js", "html", "css",
        "tailwind", "redux", "git", "figma", "rest api"
    ],
    "Data Analyst / Engineer": [
        "python", "sql", "pandas", "excel", "power bi", "tableau",
        "etl", "spark", "airflow", "statistics", "snowflake"
    ],
    "DevOps / Cloud Engineer": [
        "docker", "kubernetes", "aws", "azure", "linux", "ci/cd",
        "github actions", "terraform", "bash", "git", "prometheus"
    ],
}

STRONG_ACTION_VERBS = {
    "architected", "automated", "built", "configured", "decreased", "deployed",
    "designed", "developed", "engineered", "improved", "implemented", "increased",
    "launched", "led", "migrated", "optimized", "reduced", "refactored",
    "scaled", "spearheaded", "streamlined", "trained", "validated"
}

WEAK_PHRASE_REWRITES = {
    "responsible for": "Engineered and maintained",
    "worked on": "Developed and deployed",
    "helped with": "Collaborated to deliver",
    "duties included": "Executed core workflows for",
    "assisted in": "Contributed to building",
}


class CVCustomModel:
    def __init__(
        self,
        weights_path: str = "weights/cv_role_classifier.joblib",
        gatekeeper_path: str = "weights/cv_gatekeeper.joblib",
    ):
        self.weights_path = weights_path
        self.gatekeeper_path = gatekeeper_path
        self.gatekeeper = None
        self.role_classifier = None
        self.similarity_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

    def load(self):
        """Loads both the Binary Gatekeeper and Role Classifier at startup."""
        if not os.path.exists(self.gatekeeper_path) or not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                "Model weights missing. Run 'python training/train_model.py' first."
            )
        self.gatekeeper = joblib.load(self.gatekeeper_path)
        self.role_classifier = joblib.load(self.weights_path)
        print("Stage 1 Gatekeeper + Stage 2 Role Classifier loaded successfully.")

    def _verify_document_is_cv(self, cv_text: str):
        """Stage 1 ML Gatekeeper: Uses our trained binary classifier to reject non-CVs."""
        cv_prob = float(self.gatekeeper.predict_proba([cv_text])[0][1])
        has_email = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', cv_text))

        if cv_prob < 0.50 and not (cv_prob > 0.35 and has_email):
            raise HTTPException(
                status_code=422,
                detail=f"Rejected: Document classified as Non-Resume (CV probability: {int(cv_prob * 100)}%). Please upload a valid CV."
            )

    def _validate_is_cv(self, text: str, role_confidence: float):
        """Secondary rule-based safety net against exam papers and non-CV documents."""
        text_lower = text.lower()
        has_email = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text))

        cv_keywords = [
            "experience", "education", "skills", "projects", "summary",
            "objective", "certifications", "internship", "university", "bachelor", "degree"
        ]
        keyword_hits = sum(1 for kw in cv_keywords if kw in text_lower)

        quiz_keywords = ["question 1", "multiple choice", "marks:", "answer key", "exam paper", "quiz"]
        quiz_hits = sum(1 for qk in quiz_keywords if qk in text_lower)

        if quiz_hits >= 2 or (keyword_hits < 2 and not has_email and role_confidence < 0.50):
            raise HTTPException(
                status_code=422,
                detail="The uploaded file does not appear to be a valid CV or resume (missing standard resume sections or contact details)."
            )

    def _split_sections(self, text: str) -> Dict[str, str]:
        """Splits CV text into logical sections for targeted diagnostics."""
        headers = ["summary", "objective", "experience", "work experience", "projects", "education", "skills", "certifications"]
        sections = {"general": ""}
        current = "general"

        for line in text.split("\n"):
            clean = line.strip().lower().rstrip(":")
            if clean in headers and len(line.strip()) < 35:
                current = "experience" if "experience" in clean else clean
                sections.setdefault(current, "")
            else:
                sections[current] += line + "\n"
        return sections

    def _generate_bullet_rewrite(self, weak_line: str, matched_skills: List[str]) -> str:
        """Creates a tailored Before -> After suggestion using the candidate's own skills."""
        rewritten = weak_line.strip("•-* \t")
        for weak, strong in WEAK_PHRASE_REWRITES.items():
            pattern = re.compile(re.escape(weak), re.IGNORECASE)
            if pattern.search(rewritten):
                rewritten = pattern.sub(strong, rewritten, count=1)
                break

        sample_skill = matched_skills[0] if matched_skills else "Python"
        return (
            f"Suggested Rewrite: \"{rewritten.rstrip('.')}, leveraging {sample_skill} "
            f"to improve system efficiency by [X]%.\""
        )

    def predict(self, filename: str, cv_text: str, job_description: Optional[str] = None) -> CVAnalysisResponse:
        text_lower = cv_text.lower()
        sections = self._split_sections(cv_text)
        suggestions: list[SuggestedChange] = []

        # Stage 1: ML Gatekeeper check (Resume vs. Non-Resume)
        self._verify_document_is_cv(cv_text)

        # Stage 2: Role Classification
        predicted_role = self.role_classifier.predict([cv_text])[0]
        probabilities = self.role_classifier.predict_proba([cv_text])[0]
        role_confidence = float(np.max(probabilities))

        # Secondary rule-based verification
        self._validate_is_cv(cv_text, role_confidence)

        # 2. Vector Similarity & Evidence-Based Skill Matching
        role_skills = ROLE_SKILL_BANK.get(predicted_role, ["python", "git", "sql", "docker"])
        experience_and_projects = (sections.get("experience", "") + " " + sections.get("projects", "")).lower()

        if job_description and job_description.strip():
            tfidf_matrix = self.similarity_vectorizer.fit_transform([cv_text, job_description])
            cosine_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            semantic_score = int(np.clip(45 + (cosine_sim * 115), 40, 98))

            jd_lower = job_description.lower()
            all_known_skills = {s for skills in ROLE_SKILL_BANK.values() for s in skills}
            target_skills = [s for s in all_known_skills if s in jd_lower] or role_skills
            matched_skills = [s.title() for s in target_skills if s in text_lower]
            missing_skills = [s.title() for s in target_skills if s not in text_lower]

            keyword_ratio = int((len(matched_skills) / max(1, len(target_skills))) * 100)
            skills_score = int((semantic_score * 0.55) + (keyword_ratio * 0.45))
        else:
            matched_skills = [s.title() for s in role_skills if s in text_lower]
            missing_skills = [s.title() for s in role_skills if s not in text_lower]
            skills_score = int(np.clip(45 + (len(matched_skills) / len(role_skills)) * 55, 40, 96))

        # Check if skills are listed in Skills section but never proven in Experience/Projects
        unproven_skills = [
            s for s in matched_skills
            if experience_and_projects.strip() and s.lower() not in experience_and_projects
        ]
        if unproven_skills:
            suggestions.append(
                SuggestedChange(
                    section="Experience / Projects",
                    severity="medium",
                    issue=f"Skills listed without project proof: {', '.join(unproven_skills[:3])}.",
                    recommendation="Mention where and how you applied these tools inside your Experience or Project bullet points to boost recruiter trust."
                )
            )

        if missing_skills:
            suggestions.append(
                SuggestedChange(
                    section="Skills",
                    severity="medium",
                    issue=f"Missing high-value keywords for a {predicted_role}.",
                    recommendation=f"If you have hands-on familiarity with {', '.join(missing_skills[:4])}, integrate them into your Skills and Project descriptions."
                )
            )

        # 3. Formatting, Links & Word Count Diagnostics
        required_sections = ["experience", "education", "skills"]
        missing_sections = [sec for sec in required_sections if sec not in text_lower]
        formatting_score = max(40, 100 - (len(missing_sections) * 18))

        for sec in missing_sections:
            suggestions.append(
                SuggestedChange(
                    section="Structure",
                    severity="high",
                    issue=f"Missing standard '{sec.capitalize()}' section header.",
                    recommendation=f"Add a clear '{sec.capitalize()}' heading so ATS parsers index your background accurately."
                )
            )

        has_links = any(domain in text_lower for domain in ["github.com", "linkedin.com", "portfolio", "http"])
        if not has_links:
            formatting_score = max(40, formatting_score - 10)
            suggestions.append(
                SuggestedChange(
                    section="Header / Contact",
                    severity="low",
                    issue="No GitHub, LinkedIn, or portfolio URL detected.",
                    recommendation="Include clickable links to your GitHub profile and LinkedIn in the header so technical recruiters can inspect your code."
                )
            )

        word_count = len(cv_text.split())
        if word_count < 200:
            formatting_score = max(40, formatting_score - 10)
            suggestions.append(
                SuggestedChange(
                    section="Length & Depth",
                    severity="medium",
                    issue=f"CV is quite brief ({word_count} words).",
                    recommendation="Expand your project descriptions with architecture details, tools used, and measurable outcomes (aim for 350–650 words)."
                )
            )

        # 4. Impact, Action Verbs & Concrete Bullet Rewriting
        metrics_found = re.findall(r'(?:\$\d+|\b\d+(?:%|\+|k|m|x)\b)', text_lower)
        words = set(re.findall(r'\b[a-z]+\b', text_lower))
        action_verbs_used = words.intersection(STRONG_ACTION_VERBS)

        impact_score = min(100, 42 + (len(metrics_found) * 6) + (len(action_verbs_used) * 4))

        if len(metrics_found) < 3:
            suggestions.append(
                SuggestedChange(
                    section="Experience",
                    severity="high",
                    issue="Low density of quantifiable metrics (%, +, numbers).",
                    recommendation="Add concrete numbers to at least 3 bullet points (e.g., 'Reduced query time by 40%', 'Processed 15k+ records', or 'Achieved 94% model accuracy')."
                )
            )

        rewrites_added = 0
        for line in cv_text.split("\n"):
            if any(phrase in line.lower() for phrase in WEAK_PHRASE_REWRITES):
                suggestions.append(
                    SuggestedChange(
                        section="Experience Rewrite",
                        severity="medium",
                        issue="Passive phrasing weakens technical ownership.",
                        original_text=line.strip()[:130],
                        recommendation=self._generate_bullet_rewrite(line, matched_skills)
                    )
                )
                rewrites_added += 1
                if rewrites_added >= 2:
                    break

        # 5. Weighted Overall Score
        overall_score = int((skills_score * 0.45) + (impact_score * 0.35) + (formatting_score * 0.20))

        return CVAnalysisResponse(
            filename=filename,
            overall_score=overall_score,
            score_breakdown=ScoreBreakdown(
                skills_match=skills_score,
                impact_and_metrics=impact_score,
                formatting_and_sections=formatting_score,
            ),
            extracted_profile=ExtractedProfile(
                detected_role=f"{predicted_role} ({int(role_confidence * 100)}% confidence)",
                matched_skills=matched_skills,
                missing_skills=missing_skills,
            ),
            suggested_changes=suggestions,
        )