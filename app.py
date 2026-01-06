import streamlit as st
from pathlib import Path
from PIL import Image
import os

from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils import load_keras, load_joblib, load_faiss, load_npy
from src.config.configuration import ConfigurationManager

# ------------------ Paths ------------------
PROJECT_ROOT = Path(__file__).resolve().parent / "notebooks"
CSV_PATH = PROJECT_ROOT / "data" / "recipe_meta_topics.csv"

# ------------------ Helper functions ------------------
def extract_lemmatized_name(image_path):
    """Extract lemmatized name from image filename"""
    filename = Path(image_path).stem
    parts = filename.split('_')
    name_parts = []
    for part in parts:
        if part.isdigit():
            break
        name_parts.append(part)
    return ' '.join(name_parts)

def get_recipe_data(lemmatized_name):
    """Load CSV and get recipe data - import pandas only when needed"""
    # Cache the dataframe in session state to avoid reloading
    if 'recipe_df' not in st.session_state:
        import pandas as pd
        st.session_state.recipe_df = pd.read_csv(CSV_PATH)
    
    df = st.session_state.recipe_df
    match = df[df['lemmatized_name'] == lemmatized_name]
    if not match.empty:
        return {
            'original_name': match.iloc[0]['original_name'],
            'recipe': match.iloc[0]['recipe']
        }
    return None

@st.dialog("Recipe Details", width="large")
def show_recipe_modal(image_path):
    """Display recipe in a modal dialog"""
    lemmatized_name = extract_lemmatized_name(image_path)
    recipe_data = get_recipe_data(lemmatized_name)
    
    if recipe_data:
        col_img, col_recipe = st.columns([1, 2])
        
        with col_img:
            img_full_path = (PROJECT_ROOT / image_path).resolve()
            st.image(str(img_full_path), use_container_width=True)
        
        with col_recipe:
            st.markdown(f"### {recipe_data['original_name']}")
            st.markdown("---")
            st.markdown("#### 📝 Instructions")
            
            steps = recipe_data['recipe'].split('|')
            for step in steps:
                step = step.strip()
                if step:
                    st.markdown(f"{step}")
                    st.markdown("")
    else:
        st.error(f"Recipe not found for: {lemmatized_name}")

# ------------------ Page config ------------------
st.set_page_config(
    page_title="Recipe Image Similarity Search",
    layout="wide"
)

# ------------------ Styling ------------------
st.markdown(
    """
    <style>
    .image-grid {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 14px;
        margin-top: 20px;
    }
    /* Hide button text and make it invisible */
    div[data-testid="column"] button {
        background-color: transparent;
        border: none;
        padding: 0;
        margin: 0;
        height: 0;
        min-height: 0;
    }
    div[data-testid="column"] button p {
        display: none;
    }
    /* Make images clickable-looking */
    .stImage {
        cursor: pointer;
        transition: transform 0.2s;
    }
    .stImage:hover {
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Load heavy assets ONCE ------------------
@st.cache_resource
def load_pipeline():
    config = ConfigurationManager()
    model_dir = config.get_model_training_config().save_path

    embedding_model = load_keras(os.path.join(model_dir, "embedding_model.keras"))
    pca = load_joblib(os.path.join(model_dir, "pca.joblib"))
    index = load_faiss(os.path.join(model_dir, "recipes.faiss"))
    image_paths = load_npy(os.path.join(model_dir, "image_paths.npy"))

    return PredictionPipeline(
        embedding_model=embedding_model,
        pca=pca,
        index=index,
        image_paths=image_paths
    )

pipeline = load_pipeline()

# ------------------ App title ------------------
st.title("🍲 Recipe Image Similarity Search")

# ------------------ Upload image ------------------
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"],
    help="Drag and drop a file or click to browse"
)

if uploaded_file is not None:
    # ---- Show query image ----
    input_image = Image.open(uploaded_file)
    st.subheader("📷 Query Image")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(input_image, width=350)

    # ---- Similarity search ----
    with st.spinner("Finding similar recipes..."):
        image_paths = pipeline.initiate_pipeline(uploaded_file, k=25)

    # ---- Display results ----
    st.subheader("🔍 Most Similar Recipes")
    st.write("Click on any image to view the recipe")
    st.markdown('<div class="image-grid">', unsafe_allow_html=True)

    cols = st.columns(5, gap="small")
    for i, rel_path in enumerate(image_paths):
        img_path = (PROJECT_ROOT / rel_path).resolve()
        with cols[i % 5]:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
                if st.button("View Recipe", key=f"btn_{i}", use_container_width=True):
                    show_recipe_modal(rel_path)
            else:
                st.warning("Image not found")

    st.markdown('</div>', unsafe_allow_html=True)