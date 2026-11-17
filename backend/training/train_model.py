import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Role Classification Dataset (Positive CV Examples) ---
ROLE_DATA = [
    # AI / Machine Learning
    ("Summary: ML Engineer with 3 years experience. Skills: Python, PyTorch, TensorFlow, NLP, BERT. Experience: Built deep learning models and deployed inference APIs. Education: BSc Computer Science.", "AI / Machine Learning Engineer"),
    ("Objective: Seeking AI roles. Skills: Scikit-learn, Pandas, NumPy, XGBoost, Feature Engineering. Projects: Customer churn prediction achieving 92% accuracy. Education: B.Tech Data Science.", "AI / Machine Learning Engineer"),
    ("Experience: Trained neural networks, Hugging Face transformers, RAG pipelines, and vector databases. Skills: Python, Docker, FastAPI, Git. Education: MSc Artificial Intelligence.", "AI / Machine Learning Engineer"),
    ("Projects: Computer vision object detection using YOLOv8 and OpenCV. Skills: Deep Learning, Python, SQL, AWS SageMaker. Education: Bachelor of Engineering.", "AI / Machine Learning Engineer"),
    # Backend Engineering
    ("Experience: Backend Developer. Designed scalable REST APIs using FastAPI, Django, and Python. Managed PostgreSQL databases and Redis caching. Skills: Docker, Git, SQL. Education: BSc Software Engineering.", "Backend Software Engineer"),
    ("Summary: Java Backend Engineer. Built microservices in Spring Boot and Node.js. Optimized SQL queries and JWT authentication. Skills: Java, Kafka, MySQL, AWS. Education: BSc IT.", "Backend Software Engineer"),
    ("Projects: E-commerce backend API with payment gateway integration and Celery async workers. Skills: Python, FastAPI, PostgreSQL, Unit Testing, Linux. Education: BSc Computer Science.", "Backend Software Engineer"),
    ("Experience: Developed GraphQL endpoints, database migrations, and high-throughput server logic. Skills: Node.js, Express, MongoDB, Docker, REST API. Education: BEng Software.", "Backend Software Engineer"),
    # Frontend Engineering
    ("Summary: Frontend Developer skilled in React, TypeScript, Next.js, and Tailwind CSS. Experience: Built responsive web dashboards and improved Core Web Vitals by 40%. Education: BSc Web Technologies.", "Frontend Developer"),
    ("Skills: HTML5, CSS3, JavaScript ES6, Vue.js, Redux, Figma. Projects: Interactive portfolio and SaaS admin panel with REST API integration. Education: BSc Computer Science.", "Frontend Developer"),
    ("Experience: UI Engineer creating accessible React components, client-side routing, and state management. Skills: React, TypeScript, CSS, Git. Education: Bachelor of Design & Computing.", "Frontend Developer"),
    ("Projects: Real-time chat web app built with Next.js and Tailwind. Skills: JavaScript, React, Responsive Design, Webpack. Education: BSc Information Systems.", "Frontend Developer"),
    # Data Analyst / Data Engineering
    ("Summary: Data Analyst proficient in SQL, Python Pandas, Power BI, and Tableau. Experience: Built automated ETL pipelines and executive KPI dashboards. Education: BSc Statistics.", "Data Analyst / Engineer"),
    ("Skills: Excel, SQL, Snowflake, A/B Testing, Matplotlib, Python. Experience: Analyzed customer retention data and delivered revenue forecasts. Education: BSc Business Analytics.", "Data Analyst / Engineer"),
    ("Experience: Data Engineer building batch and streaming pipelines using Apache Spark, Airflow, and SQL. Skills: Python, ETL, BigQuery, Git. Education: BSc Data Science.", "Data Analyst / Engineer"),
    ("Projects: Sales forecasting dashboard in Power BI and SQL data warehouse modeling. Skills: SQL, Python, Data Visualization, Statistics. Education: BCom & Data Analytics.", "Data Analyst / Engineer"),
    # DevOps / Cloud
    ("Summary: DevOps Engineer. Experience: Managed Kubernetes clusters, Docker containers, Terraform IaC, and GitHub Actions CI/CD pipelines. Education: BSc Network Engineering.", "DevOps / Cloud Engineer"),
    ("Skills: AWS EC2, S3, IAM, Linux, Bash, Jenkins, Prometheus, Grafana. Experience: Automated zero-downtime cloud deployments. Education: BSc Computer Systems.", "DevOps / Cloud Engineer"),
    ("Projects: Multi-container microservice deployment on AWS EKS using Helm charts and Terraform. Skills: Docker, Kubernetes, CI/CD, Git. Education: BEng Computing.", "DevOps / Cloud Engineer"),
    ("Experience: Site Reliability Engineer maintaining Linux production servers, ELK logging, and automated cloud infrastructure. Skills: Azure, Docker, Bash, Python. Education: BSc IT.", "DevOps / Cloud Engineer"),
]

# --- 2. Non-Resume (Negative) Examples for Binary Gatekeeper ---
NON_RESUME_DATA = [
    "Question 1: What is the time complexity of binary search? A) O(n) B) O(log n) C) O(n^2) D) O(1). Answer Key: B. Total Marks: 20.",
    "Midterm Examination 2026. Instructions: Answer all 5 questions clearly. Show your step-by-step calculations in the space provided. Maximum Marks: 100.",
    "Quiz 4 Solutions: 1. True 2. False 3. The derivative of x^2 is 2x. 4. Photosynthesis occurs in the chloroplasts of plant cells.",
    "Multiple Choice Questions: Choose the correct option for each statement below. Each question carries 2 marks. Do not use a calculator.",
    "Invoice #INV-2026-089. Bill To: Acme Corp. Description: Cloud Hosting Services. Quantity: 2. Unit Price: $150.00. Total Due: $300.00. Thank you for your business.",
    "Tax Receipt and Payment Confirmation. Transaction ID: 98472910. Date: 12/10/2026. Subtotal: $45.00. VAT (15%): $6.75. Amount Paid: $51.75.",
    "Chapter 3: Introduction to Operating Systems. An operating system acts as an intermediary between the user of a computer and the computer hardware.",
    "Lecture Notes - Week 6: Supervised vs Unsupervised Learning. In supervised learning, the algorithm learns from labeled training data to predict outcomes.",
    "Abstract: This research paper investigates the effects of climate change on coastal ecosystems using satellite imagery collected between 2010 and 2025. References: [1] Smith et al.",
    "Terms and Conditions of Service. By accessing this website, you agree to be bound by these terms, all applicable laws, and regulations.",
    "Meeting Minutes - Product Sync. Attendees: Sarah, John, David. Action Items: Finalize Q3 roadmap by Friday and review customer feedback tickets.",
    "Recipe for Classic Pancakes: 1 cup all-purpose flour, 2 tablespoons sugar, 1 teaspoon baking powder, 1 cup milk, 1 egg. Whisk ingredients until smooth.",
    "User Manual and Safety Guide. Warning: Do not expose the device to water or extreme temperatures. Press and hold the power button for 3 seconds to turn on.",
    "Lab Report: Determination of Acid Concentration by Titration. Objective: To find the molarity of HCl using standard NaOH solution. Apparatus: Burette, pipette, conical flask.",
    "Purchase Order #PO-4412. Supplier: Global Tech Supplies. Item: Wireless Mechanical Keyboard. Delivery Address: Colombo, Sri Lanka. Net 30 Days.",
    "Assignment 2: Write a 1500-word essay discussing the ethical implications of generative artificial intelligence in modern journalism. Submit via LMS before midnight.",
    "Bank Statement for Account ending in 4821. Opening Balance: $2,400.00. Debit: Grocery Store -$85.40. Credit: Salary Deposit +$3,200.00. Closing Balance: $5,514.60.",
    "Syllabus: CS304 Database Management Systems. Grading Policy: Midterm 30%, Final Exam 40%, Lab Assignments 30%. Textbook: Database System Concepts.",
    "Travel Itinerary: Flight UL-504 departing at 08:30 AM. Hotel Check-in: 2:00 PM. Booking Reference: XYZ789.",
    "Warranty Certificate: This product is warranted against defects in materials and workmanship for a period of 12 months from the original date of purchase.",
]


def train_and_save():
    os.makedirs("weights", exist_ok=True)

    # ==========================================
    # PART A: Train Binary Gatekeeper (CV vs Non-CV)
    # ==========================================
    cv_texts = [text for text, _ in ROLE_DATA]
    gatekeeper_texts = cv_texts + NON_RESUME_DATA
    gatekeeper_labels = [1] * len(cv_texts) + [0] * len(NON_RESUME_DATA)  # 1 = CV, 0 = Non-CV

    # Duplicate slightly for clean stratified train/test split
    gk_df = pd.DataFrame({"text": gatekeeper_texts, "is_cv": gatekeeper_labels})
    gk_df = pd.concat([gk_df] * 3, ignore_index=True)

    X_gk_train, X_gk_test, y_gk_train, y_gk_test = train_test_split(
        gk_df["text"], gk_df["is_cv"], test_size=0.2, random_state=42, stratify=gk_df["is_cv"]
    )

    gatekeeper_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=4000)),
        ("clf", LogisticRegression(C=2.0, random_state=42))
    ])

    print("--- Training Stage 1: Binary CV Gatekeeper (Resume vs. Non-Resume) ---")
    gatekeeper_pipeline.fit(X_gk_train, y_gk_train)
    gk_preds = gatekeeper_pipeline.predict(X_gk_test)
    print(f"Gatekeeper Validation Accuracy: {accuracy_score(y_gk_test, gk_preds) * 100:.2f}%")
    print(classification_report(y_gk_test, gk_preds, target_names=["Non-Resume", "Resume"]))

    gk_path = os.path.join("weights", "cv_gatekeeper.joblib")
    joblib.dump(gatekeeper_pipeline, gk_path)
    print(f"Saved Gatekeeper model to: {gk_path}\n")

    # ==========================================
    # PART B: Train Stage 2 Role Classifier
    # ==========================================
    role_df = pd.DataFrame(ROLE_DATA, columns=["resume_text", "category"])
    role_df = pd.concat([role_df] * 4, ignore_index=True)

    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        role_df["resume_text"], role_df["category"], test_size=0.2, random_state=42, stratify=role_df["category"]
    )

    role_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)),
        ("clf", CalibratedClassifierCV(LinearSVC(random_state=42), cv=3))
    ])

    print("--- Training Stage 2: Multi-Class CV Role Classifier ---")
    role_pipeline.fit(X_r_train, y_r_train)
    role_preds = role_pipeline.predict(X_r_test)
    print(f"Role Classifier Validation Accuracy: {accuracy_score(y_r_test, role_preds) * 100:.2f}%")

    role_path = os.path.join("weights", "cv_role_classifier.joblib")
    joblib.dump(role_pipeline, role_path)
    print(f"Saved Role Classifier model to: {role_path}")


if __name__ == "__main__":
    train_and_save()