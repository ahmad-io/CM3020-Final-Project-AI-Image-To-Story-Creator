# ==============================================================================
# AI IMAGE-TO-STORY CREATOR
# Final Project Script
#
# This Streamlit application orchestrates multiple AI models to generate
# creative stories based on user-uploaded images, with advanced controls
# for genre, style, and narrative continuation.
# ==============================================================================

# === 1. IMPORTS & GLOBAL SETUP ===
import os
from datetime import datetime
from dotenv import load_dotenv
from io import BytesIO
from gtts import gTTS

import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import google.generativeai as genai

# Page configuration must be the first Streamlit command.
st.set_page_config(page_title="AI Image-to-Story Creator", page_icon="📖", layout="centered")

# Load environment variables from a .env file for secure API key management.
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Check for API key availability and configure the Gemini client globally.
API_AVAILABLE = False
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        API_AVAILABLE = True
    except Exception:
        API_AVAILABLE = False


# === 2. AI MODEL LOADING ===

# The @st.cache_resource decorator is crucial for performance. It ensures these
# large, slow-to-load models are loaded into memory only once when the app starts,
# preventing reloads on every user interaction.

@st.cache_resource
def load_blip():
    # Loads the Salesforce BLIP model and processor for image captioning.
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_emotion_pipeline():
    # Loads a lightweight DistilRoBERTa model fine-tuned for emotion classification.
    try:
        emo_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
        return emo_pipe
    except Exception:
        return None

# Load models into global variables for use throughout the app.
caption_processor, caption_model = load_blip()
emotion_pipe = load_emotion_pipeline()


# === 3. AI HELPER FUNCTIONS ===

def generate_caption(image: Image.Image):
    # Generates a descriptive text caption for a given PIL Image object.
    # 1. Prepares the image for the model using the processor.
    inputs = caption_processor(images=image, return_tensors="pt")
    # 2. Disables gradient calculation for faster, more efficient inference.
    with torch.no_grad():
        # 3. Generates token IDs using beam search for higher quality captions.
        ids = caption_model.generate(**inputs, max_length=40, num_beams=3, repetition_penalty=1.1)
    # 4. Decodes the token IDs back into a human-readable string.
    return caption_processor.decode(ids[0], skip_special_tokens=True)

def detect_emotion(text: str):
    # Analyzes a text string to determine the dominant emotion.
    # Includes robust error handling to default to 'neutral' if anything fails.
    if not text or not emotion_pipe:
        return "neutral"
    try:
        out = emotion_pipe(text)
        # The model's output format can vary, so we handle multiple possibilities.
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict) and "label" in first:
                return first["label"].lower()
            if isinstance(first, list) and first and "label" in first[0]:
                return first[0]["label"].lower()
        return "neutral"
    except Exception:
        return "neutral"

def call_genai_safe(prompt, fallback_text=""):
    # Safely calls the Gemini API to generate content.
    # Provides a fallback text if the API is unavailable or fails, preventing crashes.
    if not API_AVAILABLE:
        return fallback_text
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        resp = model.generate_content(prompt)
        # Ensure the response is a clean string, falling back if empty.
        return (resp.text or "").strip() or fallback_text
    except Exception:
        return fallback_text

def fallback_new_story(caps):
    # A simple, non-AI function to generate a story if the Gemini API fails.
    out = ["A small scene unfolded:"]
    for i, c in enumerate(caps, 1):
        out.append(f"{['First','Next','Then'][min(i-1,2)]}, {c}.")
    out.append("The moments threaded together into a quiet, memorable scene.")
    return " ".join(out)

def fallback_continuation(old_story, caps):
    # A simple, non-AI function to generate a story continuation if the API fails.
    out = ["Later, the story moved forward:"]
    for c in caps:
        out.append(f"{c}.")
    out.append("It felt like the next small chapter in the same world.")
    return " ".join(out)


# === 4. PROMPT ENGINEERING FRAMEWORK ===

# The "Master Storyteller" framework uses detailed, structured instructions
# to guide the LLM's creative process, ensuring higher quality and more
# controlled narrative outputs.

GENRE_INSTRUCTIONS = {
    "Fantasy": "Focus on a sense of wonder, ancient magic, and a character with a clear personal quest. The world should feel historic and lived-in.",
    "Mystery": "Introduce a central puzzle, crime, or paradox early. Use foreshadowing and clues. The story should build towards a logical yet surprising revelation.",
    "Sci-Fi": "Explore a 'what-if' technological or societal concept. Ground the futuristic elements in relatable human emotions and consequences. Focus on the implications of the technology.",
    "Adventure": "Emphasize action, a journey to a compelling destination, and overcoming external obstacles. The pacing should be brisk and exciting.",
    "Poetic": "Focus on imagery, metaphor, and emotional resonance over a clear plot. The language should be evocative and lyrical.",
    "Horror": "Build suspense and a sense of dread. Focus on psychological fear or an unsettling atmosphere. What is unseen is often more frightening than what is shown.",
    "Romance": "Center the story on the development of a relationship between two characters. Focus on their emotional journey, internal thoughts, and the obstacles to their connection."
}
AUTHOR_INSTRUCTIONS = {
    "Edgar Allan Poe": "Emulate the distinct literary voice of Edgar Allan Poe. Use a first-person narrator who may be unreliable. Focus on themes of psychological dread, madness, and the supernatural. Employ long, complex sentences and a rich, gothic vocabulary.",
    "Jane Austen": "Emulate the distinct literary voice of Jane Austen. Focus on social commentary, irony, and the complexities of relationships and class. The tone should be witty and observational, with a focus on character dialogue and internal thoughts.",
    "Ernest Hemingway": "Emulate the distinct literary voice of Ernest Hemingway. Use short, declarative sentences and sparse, direct prose. Focus on action and dialogue, avoiding excessive adverbs and adjectives. Imply emotion rather than stating it outright (the 'iceberg theory').",
    "H.P. Lovecraft": "Emulate the distinct literary voice of H.P. Lovecraft. Focus on cosmic horror and the fear of the unknown. Use a descriptive, often academic, tone to describe indescribable, ancient entities. Build a sense of insignificance and creeping madness.",
    "Classic Fairytale": "Emulate the style of a classic fairytale, like those from the Brothers Grimm or Hans Christian Andersen. Begin with 'Once upon a time...' and use archetypal characters and a clear moral lesson. The tone should be simple, magical, and timeless."
}

def build_new_story_prompt(caps, genre="General", author_style="Default", emotion=None):
    # Constructs the prompt for generating a brand new story.
    joined = " || ".join(caps)
    p = "You are a master storyteller, an acclaimed author. Your task is to write a compelling, nuanced, and original short story inspired by the following elements.\n\n"
    p += f"INSPIRATIONAL ELEMENTS: {joined}\n\n"
    p += "CORE WRITING INSTRUCTIONS:\n"
    p += "- **Narrative First:** Do not just describe a scene. Create a story with a character, a setting, a clear goal or desire, and a conflict or obstacle.\n"
    p += "- **Show, Don't Tell:** Instead of saying a character is brave, show them performing a brave act. Imply emotions through actions and dialogue.\n"
    p += "- **Avoid Clichés:** Do not use overused phrases or plot devices. Critically, avoid common generic character names (e.g., Elara, Kael, Lyra, Aiden). Create a unique name that fits the story's tone.\n"
    p += "- **Pacing:** The story must have a clear beginning, a developing middle, and a conclusive end. It should not feel like a static snapshot.\n\n"
    p += "STYLE AND GENRE INSTRUCTIONS:\n"
    if author_style and author_style != "Default":
        author_rules = AUTHOR_INSTRUCTIONS.get(author_style, "")
        p += f"- **Authorial Style:** {author_rules}\n"
    elif genre and genre != "General":
        genre_rules = GENRE_INSTRUCTIONS.get(genre, "")
        p += f"- **Genre:** Write in a {genre.lower()} style. {genre_rules}\n"
    if emotion:
        p += f"- **Emotional Tone:** The story should evoke a feeling of {emotion}.\n\n"
    p += "FINAL OUTPUT REQUIREMENTS:\n"
    p += "- **Length:** 200–350 words.\n"
    p += "- **Format:** A single, cohesive block of prose.\n"
    p += "- **Crucially:** Do not mention the inspirational elements, AI, captions, or these instructions in your response. Simply tell the story as the author."
    return p

def build_continuation_prompt(old, caps, genre="General", author_style="Default", emotion=None):
    # Constructs the prompt for continuing an existing story, ensuring consistency.
    joined = " || ".join(caps)
    p = "You are a master storyteller continuing a narrative. Your task is to write the next logical and compelling paragraph(s) for the story so far.\n\n"
    p += f"STORY SO FAR:\n{old}\n\n"
    p += f"INSPIRATIONAL ELEMENTS FOR CONTINUATION: {joined}\n\n"
    p += "CORE WRITING INSTRUCTIONS:\n"
    p += "- **Seamless Transition:** The new part must flow naturally from the 'STORY SO FAR'. Do not repeat or summarize the previous text.\n"
    p += "- **Advance the Plot:** Use the new elements to move the story forward, introduce a new challenge, or reveal something new about the character or world.\n"
    p += "- **Maintain Consistency:** Keep the tone, style, and characters consistent with the established narrative.\n\n"
    p += "STYLE AND GENRE INSTRUCTIONS:\n"
    if author_style and author_style != "Default":
        author_rules = AUTHOR_INSTRUCTIONS.get(author_style, "")
        p += f"- **Authorial Style:** Maintain this voice: {author_rules}\n"
    elif genre and genre != "General":
        genre_rules = GENRE_INSTRUCTIONS.get(genre, "")
        p += f"- **Genre:** Continue in a {genre.lower()} style. {genre_rules}\n"
    if emotion:
        p += f"- **Emotional Tone:** The continuation should evoke a feeling of {emotion}.\n\n"
    p += "FINAL OUTPUT REQUIREMENTS:\n- **Format:** Append the next paragraph(s) only. Do not re-write the whole story."
    return p


# === 5. SESSION STATE MANAGEMENT ===

# st.session_state preserves variables across reruns, acting as the app's memory.
if "history" not in st.session_state:
    st.session_state.history = []
if "mode" not in st.session_state:
    st.session_state.mode = None
if "full_story_audio_bytes" not in st.session_state:
    st.session_state.full_story_audio_bytes = None
if "thread_id_counter" not in st.session_state:
    st.session_state.thread_id_counter = 0
if "genre" not in st.session_state:
    st.session_state.genre = "General"
if "emotion_override" not in st.session_state:
    st.session_state.emotion_override = "Auto-detect"
if "author_style" not in st.session_state:
    st.session_state.author_style = "Default"


# === 6. UI RENDERING LOGIC ===

st.title("📖 AI Image-to-Story Creator")

# --- 6.1. Sidebar Controls ---
with st.sidebar:
    st.header("🎨 Creative Controls")
    genres = ["General", "Fantasy", "Mystery", "Sci-Fi", "Adventure", "Poetic", "Horror", "Romance"]
    chosen_genre = st.selectbox("Choose a story genre:", genres, index=genres.index(st.session_state.get("genre", "General")))
    st.session_state.genre = chosen_genre

    author_styles = ["Default", "Edgar Allan Poe", "Jane Austen", "Ernest Hemingway", "H.P. Lovecraft", "Classic Fairytale"]
    chosen_author = st.selectbox("Emulate an author's style (optional):", author_styles, index=author_styles.index(st.session_state.get("author_style", "Default")))
    st.session_state.author_style = chosen_author
    
    emotions = ["Auto-detect", "joy", "sadness", "fear", "anger", "surprise", "love", "hope", "neutral"]
    chosen_emotion = st.selectbox("Override story emotion (optional):", emotions, index=emotions.index(st.session_state.get("emotion_override", "Auto-detect")))
    st.session_state.emotion_override = chosen_emotion
    
    st.markdown("---")
    st.caption("Genre, Style, and Emotion influence the story. Author Style overrides Genre.")

# --- 6.2. Main Page & History Rendering ---
if API_AVAILABLE:
    st.info("✨ Generative API available. High-quality output enabled.")
else:
    st.warning("⚠️ No generative API key detected. Fallback generator will be used for stories.")

last_thread_id = None
for i, entry in enumerate(st.session_state.history):
    current_thread_id = entry.get("thread_id", 0)
    if current_thread_id != last_thread_id:
        st.markdown("---")
        st.header(f"Story Thread #{current_thread_id + 1}")
        last_thread_id = current_thread_id
    
    with st.container(border=True):
        label = "✨ New Story" if entry["type"] == "new" else "📖 Continuation"
        st.subheader(label)
        
        meta = f"**Genre:** {entry.get('genre','General')} — **Style:** {entry.get('author_style','Default')} — **Emotion:** {entry.get('applied_emotion','neutral')}"
        st.info(meta)
        
        cols = st.columns(len(entry["images"])) if entry["images"] else [st]
        for j, img in enumerate(entry["images"]):
            with cols[j]:
                st.image(img, use_container_width=True)
                st.caption(f"AI saw: *{entry['captions'][j]}*")
        
        st.write(entry["text"])

        if entry.get("audio_bytes"):
            st.audio(entry["audio_bytes"])
        else:
            if st.button("🎧 Generate & Play Audio", key=f"tts_{entry['timestamp']}"):
                with st.spinner("Generating audio..."):
                    try:
                        sound_file = BytesIO()
                        tts = gTTS(text=entry["text"], lang='en', slow=False)
                        tts.write_to_fp(sound_file)
                        st.session_state.history[i]["audio_bytes"] = sound_file
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not generate audio: {e}", icon="⚠️")
        
        st.caption(f"Generated at {entry['timestamp']}")

# --- 6.3. Initial Uploader (First Story) ---
if not st.session_state.history and st.session_state.mode is None:
    st.header("✨ Start Your First Story")
    start_files = st.file_uploader("Upload 1–3 images", type=["png","jpg","jpeg"], accept_multiple_files=True, key="start")
    if start_files:
        if len(start_files) > 3:
            st.error("Max 3 images please.")
        else:
            imgs, caps = [], []
            cols = st.columns(len(start_files))
            for i,f in enumerate(start_files):
                img = Image.open(f).convert("RGB")
                imgs.append(img)
                cap = generate_caption(img)
                caps.append(cap)
                with cols[i]:
                    st.image(img, use_container_width=True)
                    st.caption(f"AI sees: *{cap}*")
            
            detected = detect_emotion(" ".join(caps))
            applied_emotion = st.session_state.emotion_override if st.session_state.emotion_override != "Auto-detect" else detected

            if st.button("Generate story from these images"):
                with st.spinner("Crafting your story..."):
                    prompt = build_new_story_prompt(caps, genre=st.session_state.genre, author_style=st.session_state.author_style, emotion=applied_emotion)
                    fallback = fallback_new_story(caps)
                    story = call_genai_safe(prompt, fallback_text=fallback)
                    
                    st.session_state.history.append({
                        "type":"new", "thread_id": st.session_state.thread_id_counter,
                        "timestamp":datetime.now().isoformat(timespec="seconds"),
                        "images":imgs, "captions":caps, "text":story, "genre":st.session_state.genre,
                        "author_style": st.session_state.author_style, "applied_emotion":applied_emotion, "audio_bytes": None
                    })
                    st.toast(f"New story created!", icon="🎉")
                    st.session_state.mode = "generated"
                    st.rerun()

# --- 6.4. Next Action Buttons ---
if st.session_state.history and st.session_state.mode == "generated":
    st.markdown("---")
    st.subheader("Next actions")
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("➕ Continue Story", use_container_width=True):
            st.session_state.mode="continue"
            st.rerun()
    with c2:
        if st.button("🆕 New Story", use_container_width=True):
            st.session_state.thread_id_counter += 1
            st.session_state.mode = "new"
            st.rerun()
    with c3:
        if st.button("♻️ Reset All", use_container_width=True):
            st.session_state.history = []
            st.session_state.mode = None
            st.session_state.full_story_audio_bytes = None
            st.session_state.thread_id_counter = 0
            st.rerun()

# --- 6.5. Uploader for Continue/New Modes ---
if st.session_state.mode in ("continue", "new"):
    st.markdown("---")
    if st.session_state.mode == "continue":
        st.subheader("📖 Continue Current Story")
        st.info("Upload images to add to the current story thread.")
        if st.session_state.history:
            current_thread = st.session_state.history[-1].get("thread_id", 0)
            current_story_segments = [e for e in st.session_state.history if e.get("thread_id") == current_thread]
            combined_text = "\n\n".join(e["text"] for e in current_story_segments)
            st.write("**Story so far (current thread):**")
            st.write(combined_text)
            st.markdown("---")
            if st.session_state.full_story_audio_bytes:
                st.audio(st.session_state.full_story_audio_bytes)
            else:
                if st.button("🎧 Generate Audio for Full Story"):
                    with st.spinner("Generating full story audio..."):
                        try:
                            full_audio_file = BytesIO()
                            tts = gTTS(text=combined_text, lang='en', slow=False)
                            tts.write_to_fp(full_audio_file)
                            st.session_state.full_story_audio_bytes = full_audio_file
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not generate full audio: {e}", icon="⚠️")
            with st.expander("📖 Export Full Story"):
                st.text_area("Full story text for copying:", value=combined_text, height=250, key=f"export_text_{current_thread}")
                st.download_button(label="📥 Download Story as .txt", data=combined_text, file_name=f"story_thread_{current_thread + 1}.txt", mime="text/plain", key=f"export_btn_{current_thread}")
    else: # mode is "new"
        st.subheader("✨ Start a New Story Thread")
        st.info("Upload images to begin a new story.")

    more_files = st.file_uploader("Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True, key="more")
    if more_files:
        if len(more_files) > 3:
            st.error("Max 3 images please.")
        else:
            imgs2, caps2 = [], []
            cols = st.columns(len(more_files))
            for i,f in enumerate(more_files):
                img = Image.open(f).convert("RGB")
                imgs2.append(img)
                cap = generate_caption(img)
                caps2.append(cap)
                with cols[i]:
                    st.image(img, use_container_width=True)
                    st.caption(f"AI sees: *{cap}*")
            
            detected2 = detect_emotion(" ".join(caps2))
            applied_emotion2 = st.session_state.emotion_override if st.session_state.emotion_override != "Auto-detect" else detected2

            if st.button("Generate from these images"):
                with st.spinner("Generating..."):
                    if st.session_state.mode == "continue":
                        current_thread = st.session_state.history[-1].get("thread_id", 0)
                        current_story_segments = [e for e in st.session_state.history if e.get("thread_id") == current_thread]
                        old_text = "\n\n".join(e["text"] for e in current_story_segments)
                        
                        prompt = build_continuation_prompt(old_text, caps2, genre=st.session_state.genre, author_style=st.session_state.author_style, emotion=applied_emotion2)
                        fallback = fallback_continuation(old_text, caps2)
                        new_text = call_genai_safe(prompt, fallback_text=fallback)
                        
                        st.session_state.history.append({
                            "type":"continuation", "thread_id": current_thread, "timestamp":datetime.now().isoformat(timespec="seconds"),
                            "images":imgs2, "captions":caps2, "text":new_text, "genre":st.session_state.genre,
                            "author_style": st.session_state.author_style, "applied_emotion":applied_emotion2, "audio_bytes": None
                        })
                        st.session_state.full_story_audio_bytes = None
                    else: # mode is "new"
                        prompt = build_new_story_prompt(caps2, genre=st.session_state.genre, author_style=st.session_state.author_style, emotion=applied_emotion2)
                        fallback = fallback_new_story(caps2)
                        story = call_genai_safe(prompt, fallback_text=fallback)
                        
                        st.session_state.history.append({
                            "type":"new", "thread_id": st.session_state.thread_id_counter, "timestamp":datetime.now().isoformat(timespec="seconds"),
                            "images":imgs2, "captions":caps2, "text":story, "genre":st.session_state.genre,
                            "author_style": st.session_state.author_style, "applied_emotion":applied_emotion2, "audio_bytes": None
                        })
                st.toast("New content generated!", icon="🖋️")
                st.session_state.mode="generated"
                st.rerun()

# === 7. FOOTER ===
st.markdown("---")
st.caption("All stories & images persist in this chat until you reset. Use the sidebar to control Genre, Style, and Emotion override.")