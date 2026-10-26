import streamlit as st
import json
import os
import asyncio
import tempfile
from datetime import datetime
import edge_tts
from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile

st.set_page_config(page_title="HipnoInglés", page_icon="🎧", layout="centered")

DB_FILE = "phrases_db.json"

def load_phrases():
    if "phrases" not in st.session_state:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "r", encoding="utf-8") as f:
                st.session_state.phrases = json.load(f)
        else:
            st.session_state.phrases = []
    return st.session_state.phrases

def save_phrases(phrases):
    st.session_state.phrases = phrases
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)

def select_daily_phrases(phrases, num_new=3, num_review=2):
    new_p = [p for p in phrases if p.get("times_reviewed", 0) == 0]
    rev_p = [p for p in phrases if p.get("times_reviewed", 0) > 0]
    rev_p.sort(key=lambda x: x.get("last_review") or "1970-01-01")
    
    selected_new = new_p[:num_new]
    selected_rev = rev_p[:num_review]
    
    # Si ya no quedan nuevas suficientes, completar con repaso
    if len(selected_new) < num_new:
        needed = num_new - len(selected_new)
        selected_rev = rev_p[:num_review + needed]
        
    return selected_new, selected_rev

async def tts_to_segment(text, voice, rate, pitch="+0Hz"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_name = fp.name
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(temp_name)
    seg = AudioSegment.from_mp3(temp_name)
    os.remove(temp_name)
    return seg

def create_binaural_theta(duration_ms, left_hz=200, right_hz=206, sample_rate=44100):
    t = np.linspace(0, duration_ms / 1000.0, int(sample_rate * (duration_ms / 1000.0)), False)
    left = (np.sin(2 * np.pi * left_hz * t) * 32767).astype(np.int16)
    right = (np.sin(2 * np.pi * right_hz * t) * 32767).astype(np.int16)
    stereo = np.column_stack((left, right))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        temp_wav = fp.name
    wavfile.write(temp_wav, sample_rate, stereo)
    seg = AudioSegment.from_wav(temp_wav)
    os.remove(temp_wav)
    return seg - 28  # Atenuar volumen para fondo suave

async def build_session_audio(induction_text, new_phrases, rev_phrases, exit_mode, use_theta):
    es_voice = "es-MX-DaliaNeural"
    en_voice = "en-US-ChristopherNeural"
    master = AudioSegment.silent(duration=1000)

    # 1. Inducción
    parts = [p.strip() for p in induction_text.split("[PAUSA]") if p.strip()]
    for part in parts:
        seg = await tts_to_segment(part, es_voice, rate="-15%", pitch="-5Hz")
        master += seg + AudioSegment.silent(duration=4000)

    # 2. Frases Nuevas (3 repeticiones)
    for p in new_phrases:
        intro = f"{p['mental_image']} ... En inglés dices: ... {p['meaning']}"
        master += await tts_to_segment(intro, es_voice, rate="-15%", pitch="-5Hz")
        master += AudioSegment.silent(duration=2000)
        en_seg = await tts_to_segment(p["english"], en_voice, rate="-20%")
        for _ in range(3):
            master += en_seg + AudioSegment.silent(duration=4000)
        master += AudioSegment.silent(duration=4000)

    # 3. Frases de Repaso (2 repeticiones)
    if rev_phrases:
        rep_intro = await tts_to_segment("Ahora reafirmamos las frases que ya forman parte de tu memoria.", es_voice, rate="-15%", pitch="-5Hz")
        master += rep_intro + AudioSegment.silent(duration=3000)
        for p in rev_phrases:
            master += await tts_to_segment(p["mental_image"], es_voice, rate="-15%", pitch="-5Hz")
            master += AudioSegment.silent(duration=2000)
            en_seg = await tts_to_segment(p["english"], en_voice, rate="-15%")
            for _ in range(2):
                master += en_seg + AudioSegment.silent(duration=3500)
            master += AudioSegment.silent(duration=3000)

    # 4. Salida
    if exit_mode == "Despertar activo (Conteo 1 a 3)":
        outro = "Uno, respiras profundo. Dos, mueves tus manos y pies. Tres, abres los ojos tranquilo y renovado."
    else:
        outro = "Ahora deja que estas frases se integren mientras pasas a un sueño profundo y reparador."
    master += await tts_to_segment(outro, es_voice, rate="-15%", pitch="-5Hz")
    master += AudioSegment.silent(duration=3000)

    # 5. Mezclar onda Theta de fondo
    if use_theta:
        theta_bg = create_binaural_theta(len(master))
        master = master.overlay(theta_bg)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out_fp:
        out_path = out_fp.name
    master.export(out_path, format="mp3", bitrate="128k")
    return out_path

st.title("🎧 HipnoInglés")
phrases = load_phrases()

tab1, tab2 = st.tabs(["🌙 Sesión de Hoy", "➕ Mis Frases y Progreso"])

with tab1:
    new_p, rev_p = select_daily_phrases(phrases)
    st.subheader("Tu bloque de hoy")
    st.write(f"**Nuevas ({len(new_p)}):** " + (", ".join([f"`{p['english']}`" for p in new_p]) if new_p else "Ninguna"))
    st.write(f"**Repaso ({len(rev_p)}):** " + (", ".join([f"`{p['english']}`" for p in rev_p]) if rev_p else "Ninguna aún"))

    with st.expander("Ajustes de Inducción y Fondo"):
        default_ind = (
            "Cierra los ojos, toma una respiración profunda y al exhalar suelta toda la tensión del día. [PAUSA] "
            "Siente el peso de tu cuerpo y relaja los hombros y la mandíbula. [PAUSA] "
            "Cinco, cuatro, tres, dos, uno. Tu mente está tranquila, receptiva y lista para absorber bloques de inglés sin traducir. [PAUSA]"
        )
        induction_txt = st.text_area("Guion de inducción:", value=default_ind, height=130)
        exit_choice = st.radio("Al finalizar el audio:", ["Continuar durmiendo", "Despertar activo (Conteo 1 a 3)"])
        use_theta = st.checkbox("Incluir ondas binaurales Theta (6 Hz) de fondo", value=True)

    if st.button("✨ Generar Audio Hipnótico MP3", type="primary", use_container_width=True):
        with st.spinner("Generando voces neuronales y ondas Theta (toma unos 20 segundos)..."):
            mp3_path = asyncio.run(build_session_audio(induction_txt, new_p, rev_p, exit_choice, use_theta))
            
            # Actualizar contadores de repaso
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            selected_ids = {p["id"] for p in (new_p + rev_p)}
            for p in phrases:
                if p["id"] in selected_ids:
                    p["times_reviewed"] = p.get("times_reviewed", 0) + 1
                    p["last_review"] = now_str
            save_phrases(phrases)

            with open(mp3_path, "rb") as f:
                audio_bytes = f.read()
            os.remove(mp3_path)

            st.success("¡Sesión generada y progreso actualizado!")
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                label="📲 Descargar MP3 al iPhone (Escuchar con pantalla bloqueada)",
                data=audio_bytes,
                file_name=f"HipnoIngles_{datetime.now().strftime('%Y-%m-%d')}.mp3",
                mime="audio/mp3",
                use_container_width=True
            )

with tab2:
    st.subheader("Agregar nueva frase")
    with st.form("add_phrase"):
        en_in = st.text_input("Frase en inglés (Bloque):", placeholder="Ej: Let's wrap it up.")
        ph_in = st.text_input("Pronunciación figurada:", placeholder="Ej: Lets rap-it-ap")
        me_in = st.text_input("Significado práctico:", placeholder="Ej: Vamos terminando / Cerremos esto.")
        im_in = st.text_area("Imagen mental (Empieza con 'Imagina que...'):", placeholder="Imagina que guardas las herramientas al final del turno...")
        if st.form_submit_button("Guardar frase", use_container_width=True):
            if en_in and me_in and im_in:
                new_id = max([p["id"] for p in phrases], default=0) + 1
                phrases.append({
                    "id": new_id, "category": "Personalizada",
                    "english": en_in, "phonetic": ph_in, "meaning": me_in,
                    "mental_image": im_in, "times_reviewed": 0, "last_review": None
                })
                save_phrases(phrases)
                st.success("Frase agregada correctamente.")
                st.rerun()

    st.divider()
    st.subheader("Estado de tus frases")
    for p in phrases:
        st.markdown(f"**{p['english']}** (*{p.get('phonetic','')}*) — {p['meaning']} | 🔄 Repasos: `{p['times_reviewed']}`")

    st.divider()
    st.caption("Respaldo de progreso (Descarga o sube tu JSON para no perder tus repasos):")
    st.download_button("⬇️ Descargar Respaldo JSON", data=json.dumps(phrases, ensure_ascii=False, indent=2), file_name="phrases_db.json", mime="application/json")
    up_file = st.file_uploader("⬆️ Restaurar Respaldo JSON", type=["json"])
    if up_file is not None:
        save_phrases(json.load(up_file))
        st.success("Progreso restaurado.")
        st.rerun()
