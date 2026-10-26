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

def get_categories(phrases):
    cats = []
    for p in phrases:
        c = p.get("category", "General")
        if c not in cats:
            cats.append(c)
    return cats

def select_daily_phrases(phrases, category_filter="🌐 Todas las situaciones", num_new=4, num_review=2):
    if category_filter != "🌐 Todas las situaciones":
        pool = [p for p in phrases if p.get("category") == category_filter]
    else:
        pool = phrases

    new_p = [p for p in pool if p.get("times_reviewed", 0) == 0]
    rev_p = [p for p in pool if p.get("times_reviewed", 0) > 0]
    rev_p.sort(key=lambda x: x.get("last_review") or "1970-01-01")
    
    selected_new = new_p[:num_new]
    selected_rev = rev_p[:num_review]
    
    # Si ya no quedan nuevas suficientes en esta categoría, completar con repaso
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

async def build_session_audio(induction_text, new_phrases, rev_phrases, exit_mode, use_theta, target_minutes=5):
    es_voice = "es-MX-DaliaNeural"
    en_voice = "en-US-ChristopherNeural"
    target_ms = target_minutes * 60 * 1000
    master = AudioSegment.silent(duration=2000)

    # 1. Inducción relajante (~50-60 seg)
    parts = [p.strip() for p in induction_text.split("[PAUSA]") if p.strip()]
    for part in parts:
        seg = await tts_to_segment(part, es_voice, rate="-18%", pitch="-5Hz")
        master += seg + AudioSegment.silent(duration=5000)

    # Cachear audios en inglés para reutilizar en la integración final
    en_slow_cache = {}
    en_nat_cache = {}
    es_meaning_cache = {}

    # 2. Frases Nuevas (4 repeticiones: 2 lentas + anclaje + 2 naturales)
    for p in new_phrases:
        intro = f"{p['mental_image']} ... Cuando quieras decir: {p['meaning']} ... simplemente di:"
        master += await tts_to_segment(intro, es_voice, rate="-15%", pitch="-5Hz")
        master += AudioSegment.silent(duration=2500)

        en_slow = await tts_to_segment(p["english"], en_voice, rate="-22%")
        en_nat = await tts_to_segment(p["english"], en_voice, rate="-8%")
        en_slow_cache[p["id"]] = en_slow
        en_nat_cache[p["id"]] = en_nat
        es_meaning_cache[p["id"]] = await tts_to_segment(p["meaning"], es_voice, rate="-15%", pitch="-5Hz")

        # 2 repeticiones lentas para absorber pronunciación
        master += en_slow + AudioSegment.silent(duration=4500)
        master += en_slow + AudioSegment.silent(duration=4500)

        # 2 repeticiones a ritmo natural
        master += en_nat + AudioSegment.silent(duration=4500)
        master += en_nat + AudioSegment.silent(duration=5000)

    # 3. Frases de Repaso
    if rev_phrases:
        rep_intro = await tts_to_segment("Ahora reafirmamos con naturalidad las frases que ya están en tu memoria.", es_voice, rate="-15%", pitch="-5Hz")
        master += rep_intro + AudioSegment.silent(duration=3500)
        for p in rev_phrases:
            master += await tts_to_segment(p["mental_image"], es_voice, rate="-15%", pitch="-5Hz")
            master += AudioSegment.silent(duration=2000)
            en_nat = await tts_to_segment(p["english"], en_voice, rate="-10%")
            en_nat_cache[p["id"]] = en_nat
            es_meaning_cache[p["id"]] = await tts_to_segment(p["meaning"], es_voice, rate="-15%", pitch="-5Hz")
            for _ in range(2):
                master += en_nat + AudioSegment.silent(duration=4000)
            master += AudioSegment.silent(duration=3500)

    # 4. Pre-generar el cierre para calcular el tiempo exacto restante hasta los 5 minutos
    if exit_mode == "Despertar activo (Conteo 1 a 3)":
        outro_txt = "Uno, inhala profundo y siente energía nueva. Dos, mueve suavemente tus manos y tus pies. Tres, abre los ojos sintiéndote despejado, seguro y renovado."
    else:
        outro_txt = "Ahora deja que estas frases se integren profundamente en tu mente mientras descansas en un sueño tranquilo y reparador."
    outro_seg = await tts_to_segment(outro_txt, es_voice, rate="-18%", pitch="-5Hz") + AudioSegment.silent(duration=4000)

    # 5. Ciclo de Integración Subconsciente para alcanzar la duración objetivo (ej. 5 minutos)
    all_session_phrases = new_phrases + rev_phrases
    if all_session_phrases and (len(master) + len(outro_seg) < target_ms - 15000):
        bridge = await tts_to_segment(
            "Deja que tu respiración fluya mientras tu mente repite e integra cada bloque de forma automática.",
            es_voice, rate="-18%", pitch="-5Hz"
        )
        master += bridge + AudioSegment.silent(duration=4500)

        idx = 0
        while len(master) + len(outro_seg) < target_ms - 8000:
            p = all_session_phrases[idx % len(all_session_phrases)]
            master += es_meaning_cache[p["id"]] + AudioSegment.silent(duration=2000)
            master += en_nat_cache[p["id"]] + AudioSegment.silent(duration=5000)
            if len(master) + len(outro_seg) < target_ms - 6000:
                master += en_nat_cache[p["id"]] + AudioSegment.silent(duration=5500)
            idx += 1

    # Si falta un pequeño margen de segundos para llegar a los 5 minutos exactos, rellenar con baño de ondas Theta
    remaining_ms = target_ms - (len(master) + len(outro_seg))
    if 0 < remaining_ms < 20000:
        master += AudioSegment.silent(duration=remaining_ms)

    # Añadir salida
    master += outro_seg

    # 6. Mezclar onda Theta de fondo durante toda la sesión
    if use_theta:
        theta_bg = create_binaural_theta(len(master))
        master = master.overlay(theta_bg)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out_fp:
        out_path = out_fp.name
    master.export(out_path, format="mp3", bitrate="128k")
    return out_path

st.title("🎧 HipnoInglés")
phrases = load_phrases()
categories = get_categories(phrases)

tab1, tab2 = st.tabs(["🌙 Sesión de Hoy", "➕ Mis Frases y Progreso"])

with tab1:
    st.subheader("1. Elige la situación que quieres practicar")
    selected_cat = st.selectbox(
        "Situación / Categoría:",
        options=["🌐 Todas las situaciones"] + categories,
        index=0
    )

    col_d, col_n, col_r = st.columns(3)
    with col_d:
        target_min = st.selectbox("⏱️ Duración del audio:", options=[5, 3, 8, 10], format_func=lambda x: f"{x} minutos", index=0)
    with col_n:
        num_new = st.slider("Frases nuevas:", min_value=1, max_value=6, value=4)
    with col_r:
        num_rev = st.slider("Frases repaso:", min_value=0, max_value=5, value=2)

    new_p, rev_p = select_daily_phrases(phrases, selected_cat, num_new=num_new, num_review=num_rev)

    st.subheader("2. Tu bloque para esta sesión")
    if new_p:
        st.markdown("**🌟 Nuevas:**")
        for p in new_p:
            st.markdown(f"- `{p['english']}` *(🔊 {p.get('phonetic', '')})* — {p['meaning']}")
    else:
        st.write("**🌟 Nuevas:** Ninguna pendiente en esta categoría")

    if rev_p:
        st.markdown("**🔄 Repaso:**")
        for p in rev_p:
            st.markdown(f"- `{p['english']}` *(🔊 {p.get('phonetic', '')})* — {p['meaning']}")
    else:
        st.write("**🔄 Repaso:** Ninguna aún")

    with st.expander("Ajustes de Inducción y Fondo"):
        default_ind = (
            "Cierra los ojos, toma una respiración profunda y al exhalar suelta toda la tensión del día. [PAUSA] "
            "Siente el peso de tu cuerpo y relaja los hombros, el cuello y la mandíbula. [PAUSA] "
            "Inhala calma... y exhala cualquier pensamiento. [PAUSA] "
            "Cinco, cuatro, tres, dos, uno. Tu mente está tranquila, receptiva y lista para absorber bloques de inglés de forma natural y sin traducir. [PAUSA]"
        )
        induction_txt = st.text_area("Guion de inducción:", value=default_ind, height=130)
        exit_choice = st.radio("Al finalizar el audio:", ["Continuar durmiendo", "Despertar activo (Conteo 1 a 3)"])
        use_theta = st.checkbox("Incluir ondas binaurales Theta (6 Hz) de fondo", value=True)

    if st.button(f"✨ Generar Audio Hipnótico ({target_min} min) MP3", type="primary", use_container_width=True):
        with st.spinner(f"Generando sesión hipnótica de {target_min} minutos con voces neuronales y ondas Theta..."):
            mp3_path = asyncio.run(build_session_audio(induction_txt, new_p, rev_p, exit_choice, use_theta, target_minutes=target_min))
            
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

            st.success(f"¡Sesión de {target_min} minutos generada y progreso actualizado!")
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                label=f"📲 Descargar MP3 ({target_min} min) al iPhone",
                data=audio_bytes,
                file_name=f"HipnoIngles_{target_min}min_{datetime.now().strftime('%Y-%m-%d')}.mp3",
                mime="audio/mp3",
                use_container_width=True
            )

with tab2:
    st.subheader("Agregar nueva frase")
    with st.form("add_phrase"):
        cat_choice = st.selectbox(
            "Categoría / Situación:",
            options=categories + ["➕ Crear nueva categoría..."]
        )
        custom_cat = st.text_input("Si elegiste crear nueva categoría, escríbela aquí:", placeholder="Ej: 🏥 Médico o Emergencias")
        en_in = st.text_input("Frase en inglés (Bloque):", placeholder="Ej: Let's wrap it up.")
        ph_in = st.text_input("Pronunciación figurada:", placeholder="Ej: Lets rap-it-ap")
        me_in = st.text_input("Significado práctico:", placeholder="Ej: Vamos terminando / Cerremos esto.")
        im_in = st.text_area("Imagen mental (Empieza con 'Imagina que...'):", placeholder="Imagina que guardas las herramientas al final del turno...")
        if st.form_submit_button("Guardar frase", use_container_width=True):
            if en_in and me_in and im_in:
                final_cat = custom_cat.strip() if (cat_choice == "➕ Crear nueva categoría..." and custom_cat.strip()) else cat_choice
                new_id = max([p["id"] for p in phrases], default=0) + 1
                phrases.append({
                    "id": new_id, "category": final_cat,
                    "english": en_in, "phonetic": ph_in, "meaning": me_in,
                    "mental_image": im_in, "times_reviewed": 0, "last_review": None
                })
                save_phrases(phrases)
                st.success(f"Frase agregada en '{final_cat}'.")
                st.rerun()

    st.divider()
    st.subheader(f"Estado de tus frases ({len(phrases)} en total)")
    filter_view = st.selectbox("Filtrar lista por categoría:", options=["Todas"] + categories)
    for p in phrases:
        if filter_view == "Todas" or p.get("category") == filter_view:
            st.markdown(f"**[{p.get('category', 'General')}]** **{p['english']}** (*{p.get('phonetic','')}*) — {p['meaning']} | 🔄 Repasos: `{p['times_reviewed']}`")

    st.divider()
    st.caption("Respaldo de progreso (Descarga o sube tu JSON para no perder tus repasos):")
    st.download_button("⬇️ Descargar Respaldo JSON", data=json.dumps(phrases, ensure_ascii=False, indent=2), file_name="phrases_db.json", mime="application/json")
    up_file = st.file_uploader("⬆️ Restaurar Respaldo JSON", type=["json"])
    if up_file is not None:
        save_phrases(json.load(up_file))
        st.success("Progreso restaurado.")
        st.rerun()
