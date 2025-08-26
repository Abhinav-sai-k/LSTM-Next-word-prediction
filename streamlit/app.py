import streamlit as st
import torch
import torch.nn.functional as F
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import NextWordLSTM  # Import your LSTM model class


# Load tokenizer
tokenizer = joblib.load("tokenizer.pkl")
# Vocabulary size (needed for model definition)
total_words = len(tokenizer.word_index) + 1

# Load model
model = NextWordLSTM(vocab_size=total_words, embed_dim=100, hidden_dim=150)
model.load_state_dict(torch.load("best_lstm.pth", map_location=torch.device("cpu")))
model.eval()

# -------------------------------
# Prediction Function (Top-k)
# -------------------------------
def predict_top_k_words(model, tokenizer, text, max_sequence_len, k=5):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    token_tensor = torch.tensor(token_list, dtype=torch.long)

    with torch.no_grad():
        logits = model(token_tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k)

    top_words = []
    for i in range(k):
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == top_indices[0][i].item():
                word = w
                break
        top_words.append((word, top_probs[0][i].item()))

    return top_words


# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("📖 Next Word Prediction (Auto-complete LSTM)")
st.write("Type a sentence and choose from suggested next words to build text interactively.")

# Load corpus samples

with open("paradise.txt", "r") as f:
    corpus_text = f.split("\n")

sample_sentences = [line for line in corpus_text if len(line.split()) > 5][:5]

st.subheader("Sample Sentences from Training Corpus:")
for sent in sample_sentences:
    st.text(f"- {sent.strip()}")

st.write("---")

# Maintain session state for interactive text
if "user_text" not in st.session_state:
    st.session_state.user_text = "Of man’s first disobedience"

st.subheader("✍️ Current Text:")
st.write(st.session_state.user_text)

# Predict top 5
max_sequence_len = model.embedding.num_embeddings
predictions = predict_top_k_words(model, tokenizer, st.session_state.user_text, max_sequence_len, k=5)

st.subheader("🔮 Top 5 Predictions:")
cols = st.columns(5)
for i, (word, prob) in enumerate(predictions):
    if cols[i].button(f"{word}\n({prob:.2f})"):
        st.session_state.user_text += " " + word
        st.experimental_rerun()

# Option to reset
if st.button("🔄 Reset"):
    st.session_state.user_text = "Of man’s first disobedience"
    st.experimental_rerun()
