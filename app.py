import streamlit as st
import json
import os
import re
import random
import asyncio
import tempfile
import difflib
from datetime import datetime
import edge_tts
from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import speech_recognition as sr

st.set_page_config(page_title="HipnoInglés", page_icon="🎧", layout="centered")

DB_FILE = "phrases_db.json"

def load_phrases():
    if "phrases" not in st.session_state or (
        st.session_state.phrases and "roleplay_en" not in st.session_state.phrases[0]
    ):
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

async def tts_to_bytes(text, voice="en-US-ChristopherNeural", rate="+0%", pitch="+0Hz"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_name = fp.name
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(temp_name)
    with open(temp_name, "rb") as f:
        data = f.read()
    os.remove(temp_name)
    return data

def normalize_english(text):
    text = text.lower().strip()
    replacements = {
        "i'm": "i am",
        "i'll": "i will",
        "i'd": "i would",
        "i've": "i have",
        "it's": "it is",
        "don't": "do not",
        "we're": "we are",
        "you're": "you are",
        "can't": "cannot",
        "let's": "let us",
        "what's": "what is",
        "that's": "that is",
        "carry-on": "carry on"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def evaluate_pronunciation(audio_file_obj, target_english):
    try:
        raw_seg = AudioSegment.from_file(audio_file_obj)
        raw_seg = raw_seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_fp:
            wav_path = wav_fp.name
        raw_seg.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        os.remove(wav_path)

        spoken_text = recognizer.recognize_google(audio_data, language="en-US")
        norm_spoken = normalize_english(spoken_text)
        norm_target = normalize_english(target_english)
        raw_ratio = difflib.SequenceMatcher(None, norm_spoken, norm_target).ratio()
        exact_ratio = difflib.SequenceMatcher(
            None,
            re.sub(r"[^a-z0-9\s]", "", spoken_text.lower()).strip(),
            re.sub(r"[^a-z0-9\s]", "", target_english.lower()).strip()
        ).ratio()
        score = int(round(max(raw_ratio, exact_ratio) * 100))
        return True, spoken_text, score
    except sr.UnknownValueError:
        return False, "No se pudo distinguir la voz con claridad. Intenta hablar un poco más cerca del micrófono.", 0
    except Exception as e:
        return False, f"Error al procesar el audio: {e}", 0

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

    # 1. Inducción relajante
    parts = [p.strip() for p in induction_text.split("[PAUSA]") if p.strip()]
    for part in parts:
        seg = await tts_to_segment(part, es_voice, rate="-18%", pitch="-5Hz")
        master += seg + AudioSegment.silent(duration=5000)

    en_slow_cache = {}
    en_nat_cache = {}
    es_meaning_cache = {}

    # 2. Frases Nuevas (2 lentas + 2 naturales)
    for p in new_phrases:
        intro = f"{p['mental_image']} ... Cuando quieras decir: {p['meaning']} ... simplemente di:"
        master += await tts_to_segment(intro, es_voice, rate="-15%", pitch="-5Hz")
        master += AudioSegment.silent(duration=2500)

        en_slow = await tts_to_segment(p["english"], en_voice, rate="-22%")
        en_nat = await tts_to_segment(p["english"], en_voice, rate="-8%")
        en_slow_cache[p["id"]] = en_slow
        en_nat_cache[p["id"]] = en_nat
        es_meaning_cache[p["id"]] = await tts_to_segment(p["meaning"], es_voice, rate="-15%", pitch="-5Hz")

        master += en_slow + AudioSegment.silent(duration=4500)
        master += en_slow + AudioSegment.silent(duration=4500)
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

    # 4. Pre-generar salida
    if exit_mode == "Despertar activo (Conteo 1 a 3)":
        outro_txt = "Uno, inhala profundo y siente energía nueva. Dos, mueve suavemente tus manos y tus pies. Tres, abre los ojos sintiéndote despejado, seguro y renovado."
    else:
        outro_txt = "Ahora deja que estas frases se integren profundamente en tu mente mientras descansas en un sueño tranquilo y reparador."
    outro_seg = await tts_to_segment(outro_txt, es_voice, rate="-18%", pitch="-5Hz") + AudioSegment.silent(duration=4000)

    # 5. Ciclo de Integración Subconsciente hasta completar target_minutes
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

    remaining_ms = target_ms - (len(master) + len(outro_seg))
    if 0 < remaining_ms < 20000:
        master += AudioSegment.silent(duration=remaining_ms)

    master += outro_seg

    if use_theta:
        theta_bg = create_binaural_theta(len(master))
        master = master.overlay(theta_bg)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out_fp:
        out_path = out_fp.name
    master.export(out_path, format="mp3", bitrate="128k")
    return out_path

st.title("🎧 HipnoInglés Interactivo")
phrases = load_phrases()
categories = get_categories(phrases)

tab1, tab2, tab3, tab4 = st.tabs([
    "🌙 Audio 5 Min",
    "🔊 Escucha y Habla",
    "🎭 Simulador (Roleplay)",
    "➕ Frases y Progreso"
])

# =========================================================
# PESTAÑA 1: SESIÓN HIPNÓTICA DE 5 MINUTOS
# =========================================================
with tab1:
    st.subheader("1. Elige la situación que quieres practicar")
    selected_cat = st.selectbox(
        "Situación / Categoría:",
        options=["🌐 Todas las situaciones"] + categories,
        index=0,
        key="tab1_cat"
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

# =========================================================
# PESTAÑA 2: ESCUCHA AL TOQUE + PRÁCTICA CON MICRÓFONO
# =========================================================
with tab2:
    st.subheader("🔊 Escucha frase por frase y entrena tu pronunciación")
    prac_cat = st.selectbox("Elige una situación:", options=categories, key="prac_cat")
    cat_phrases = [p for p in phrases if p.get("category") == prac_cat]

    selected_phrase = st.selectbox(
        "Selecciona la frase a practicar:",
        options=cat_phrases,
        format_func=lambda p: f"{p['english']} — ({p['meaning']})"
    )

    if selected_phrase:
        st.markdown("---")
        st.markdown(f"### 🎯 `{selected_phrase['english']}`")
        st.markdown(f"- **Pronunciación figurada:** `{selected_phrase.get('phonetic', '')}`")
        st.markdown(f"- **Significado:** {selected_phrase['meaning']}")
        st.info(f"🧠 **Anclaje mental:** {selected_phrase['mental_image']}")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🐢 Escuchar Lento", use_container_width=True):
                slow_bytes = asyncio.run(tts_to_bytes(selected_phrase["english"], rate="-28%"))
                st.audio(slow_bytes, format="audio/mp3", autoplay=True)
        with col_b2:
            if st.button("🐇 Escuchar Velocidad Natural", use_container_width=True):
                nat_bytes = asyncio.run(tts_to_bytes(selected_phrase["english"], rate="+0%"))
                st.audio(nat_bytes, format="audio/mp3", autoplay=True)

        st.markdown("#### 🎤 Ahora dilo tú con el micrófono:")
        st.caption("Toca el micrófono para grabar, di la frase en inglés y vuelve a tocar para detener.")
        mic_audio = st.audio_input(
            f"Di en voz alta: \"{selected_phrase['english']}\"",
            key=f"mic_prac_{selected_phrase['id']}"
        )

        if mic_audio is not None:
            with st.spinner("Escuchando y evaluando tu pronunciación..."):
                ok, spoken, score = evaluate_pronunciation(mic_audio, selected_phrase["english"])
                if ok:
                    st.write(f"🗣️ **Se escuchó:** `{spoken}`")
                    st.progress(score / 100.0)
                    if score >= 80:
                        st.success(f"🎉 ¡Excelente pronunciación! Precisión: **{score}%**")
                        selected_phrase["times_reviewed"] = selected_phrase.get("times_reviewed", 0) + 1
                        selected_phrase["last_review"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                        save_phrases(phrases)
                    elif score >= 55:
                        st.warning(f"👍 ¡Muy cerca! Precisión: **{score}%**. Escúchala en modo lento e inténtalo una vez más.")
                    else:
                        st.error(f"🔄 Precisión: **{score}%**. Fíjate en la guía fonética (`{selected_phrase.get('phonetic', '')}`) y vuelve a intentarlo.")
                else:
                    st.warning(spoken)

# =========================================================
# PESTAÑA 3: SIMULADOR INTERACTIVO DE SITUACIONES (ROLEPLAY)
# =========================================================
with tab3:
    st.subheader("🎭 Simulador de Situaciones en Vivo")
    st.caption("La otra persona te hablará en inglés en una situación real y tú debes responderle.")

    sim_cat = st.selectbox("¿Qué situación quieres simular?", options=categories, key="sim_cat")
    sim_pool = [p for p in phrases if p.get("category") == sim_cat]

    if "sim_idx" not in st.session_state:
        st.session_state.sim_idx = 0

    col_prev, col_rand, col_next = st.columns(3)
    with col_prev:
        if st.button("⬅️ Escenario anterior", use_container_width=True):
            st.session_state.sim_idx = (st.session_state.sim_idx - 1) % len(sim_pool)
    with col_rand:
        if st.button("🎲 Al azar", use_container_width=True):
            st.session_state.sim_idx = random.randint(0, len(sim_pool) - 1)
    with col_next:
        if st.button("Siguiente escenario ➡️", use_container_width=True):
            st.session_state.sim_idx = (st.session_state.sim_idx + 1) % len(sim_pool)

    current_sim = sim_pool[st.session_state.sim_idx % len(sim_pool)]
    prompt_en = current_sim.get("roleplay_en", "Hello! How can I help you today?")
    prompt_es = current_sim.get("roleplay_es", f"Situación: Quieres decir '{current_sim['meaning']}'")

    st.markdown("---")
    st.markdown(f"**📍 Contexto:** {prompt_es}")
    st.markdown(f"💬 **La otra persona te dice:** *\"{prompt_en}\"*")

    if st.button("🔊 Escuchar a la otra persona", type="primary", use_container_width=True):
        interlocutor_bytes = asyncio.run(tts_to_bytes(prompt_en, voice="en-US-GuyNeural", rate="-8%"))
        st.audio(interlocutor_bytes, format="audio/mp3", autoplay=True)

    st.markdown("#### ¿Cómo le respondes tú?")
    st.caption(f"💡 Pista en español: *{current_sim['meaning']}*")

    # Construir 4 opciones deterministas para este escenario
    other_options = [p["english"] for p in sim_pool if p["id"] != current_sim["id"]]
    rng = random.Random(current_sim["id"])
    distractors = rng.sample(other_options, min(3, len(other_options)))
    options_list = distractors + [current_sim["english"]]
    rng.shuffle(options_list)

    user_choice = st.radio(
        "Opción A — Elige la respuesta correcta:",
        options=options_list,
        index=None,
        key=f"quiz_{current_sim['id']}"
    )

    if user_choice:
        if user_choice == current_sim["english"]:
            st.success(f"✅ ¡Correcto! **{current_sim['english']}** (*{current_sim.get('phonetic', '')}*)")
            ans_bytes = asyncio.run(tts_to_bytes(current_sim["english"], rate="-10%"))
            st.audio(ans_bytes, format="audio/mp3", autoplay=True)
        else:
            st.error("❌ Esa frase significa otra cosa. ¡Intenta con otra opción!")

    st.markdown("---")
    st.markdown("**Opción B — Respóndele hablando con tu micrófono:**")
    sim_mic = st.audio_input(
        "Graba tu respuesta en inglés:",
        key=f"sim_mic_{current_sim['id']}"
    )
    if sim_mic is not None:
        with st.spinner("Evaluando tu respuesta en la conversación..."):
            ok, spoken, score = evaluate_pronunciation(sim_mic, current_sim["english"])
            if ok:
                st.write(f"🗣️ **Respondiste:** `{spoken}`")
                if score >= 75:
                    st.success(f"🎉 ¡Te entendieron perfectamente! ({score}% de coincidencia con `{current_sim['english']}`)")
                    current_sim["times_reviewed"] = current_sim.get("times_reviewed", 0) + 1
                    current_sim["last_review"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    save_phrases(phrases)
                else:
                    st.warning(f"Dijiste `{spoken}` ({score}%). La respuesta ideal es: **`{current_sim['english']}`** (*{current_sim.get('phonetic', '')}*). ¡Inténtalo de nuevo!")
            else:
                st.warning(spoken)

# =========================================================
# PESTAÑA 4: MIS FRASES Y PROGRESO
# =========================================================
with tab4:
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
        rp_en = st.text_input("Frase de la otra persona en el Simulador (Opcional, en inglés):", placeholder="Ej: We finished all the tasks for today.")
        if st.form_submit_button("Guardar frase", use_container_width=True):
            if en_in and me_in and im_in:
                final_cat = custom_cat.strip() if (cat_choice == "➕ Crear nueva categoría..." and custom_cat.strip()) else cat_choice
                new_id = max([p["id"] for p in phrases], default=0) + 1
                phrases.append({
                    "id": new_id,
                    "category": final_cat,
                    "english": en_in,
                    "phonetic": ph_in,
                    "meaning": me_in,
                    "mental_image": im_in,
                    "roleplay_en": rp_en.strip() if rp_en.strip() else "Hello! What would you like to say?",
                    "roleplay_es": f"Situación: Quieres decir '{me_in}'",
                    "times_reviewed": 0,
                    "last_review": None
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
