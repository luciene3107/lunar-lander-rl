import glob
import io
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pouso Lunar com RL", page_icon="🚀", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
GIFS = os.path.join(BASE, "assets", "gifs")
IMGS = os.path.join(BASE, "assets", "img")
MODELS = os.path.join(BASE, "assets", "models")


def mostrar_gif(nome, legenda):
    caminho = os.path.join(GIFS, nome)
    if os.path.exists(caminho):
        st.image(caminho, caption=legenda, width="stretch")
    else:
        st.info(f"Arquivo não encontrado: assets/gifs/{nome}")


st.title("🚀 Pouso lunar com aprendizado por reforço")
st.caption("PPO e DQN no LunarLander, com teste de robustez ao vento")

aba_projeto, aba_resultados, aba_demos, aba_ao_vivo = st.tabs(
    ["Projeto", "Resultados", "Demos (GIFs)", "Teste ao vivo"]
)

# ---------------------------------------------------------------- Projeto
with aba_projeto:
    st.subheader("Problema")
    st.write(
        "Uma nave precisa pousar entre as bandeiras controlando os propulsores. "
        "O agente aprende por tentativa e erro, sem ninguém dizer qual ação é a certa."
    )
    st.subheader("Modelagem")
    st.table(
        pd.DataFrame(
            {
                "Elemento": ["Estado", "Ações", "Recompensa"],
                "Descrição": [
                    "8 valores: posição, velocidade, ângulo, velocidade angular e contato das pernas",
                    "4 ações: nada, propulsor esquerdo, principal ou direito",
                    "Positiva ao pousar; negativa ao bater e ao gastar combustível",
                ],
            }
        ).set_index("Elemento")
    )
    st.subheader("Método")
    st.write(
        "DQN e PPO (Stable-Baselines3) treinados no Google Colab. Treino base de 300 mil passos "
        "e PPO até 1 milhão. Cada modelo foi avaliado em 100 episódios. "
        "Recompensa média acima de 200 indica pouso consistente."
    )

# -------------------------------------------------------------- Resultados
with aba_resultados:
    st.subheader("Mais treino, pouso consistente")
    c1, c2 = st.columns(2)
    c1.metric("PPO com 300 mil passos", "79,8", "± 106,7 (irregular)", delta_color="off")
    c2.metric("PPO com 1 milhão de passos", "245,4", "± 52,0 (acima da meta de 200)", delta_color="off")
    st.bar_chart(
        pd.DataFrame(
            {"Recompensa média": [79.8, 245.4]},
            index=["PPO 300k", "PPO 1M"],
        )
    )
    st.caption("Média de 100 episódios. O modelo selecionado pelo callback chegou a 256,1 ± 31,0, mas essa escolha pode inflar o valor, então o resultado principal é o modelo final.")

    grafico = os.path.join(IMGS, "grafico_recompensa_300k.png")
    if os.path.exists(grafico):
        st.image(grafico, caption="DQN vs. PPO nos primeiros 300 mil passos", width="stretch")

    st.subheader("O vento derruba o agente")
    st.table(
        pd.DataFrame(
            {
                "Treinou ↓ / Testou →": ["Sem vento", "Com vento (1M)", "Com vento (2M, final)"],
                "Sem vento": ["253,4 ± 39,2", "118,5 ± 74,2", "não testado"],
                "Com vento": ["103,4 ± 131,9", "93,3 ± 100,7", "27,3 ± 156,9"],
            }
        ).set_index("Treinou ↓ / Testou →")
    )
    st.write(
        "- Treinado sem vento, o agente cai de 253 para 103 quando o vento é ligado, e a variância explode.\n"
        "- Treinar com vento não ajudou: 1M deu 93 e 2M terminou instável em 27, sem superar o agente treinado sem vento.\n"
        "- Limitação: uma única semente de treino (42)."
    )

# -------------------------------------------------------------------- Demos
with aba_demos:
    st.subheader("Antes e depois")
    a, b = st.columns(2)
    with a:
        mostrar_gif("ppo_300k.gif", "PPO com 300 mil passos")
    with b:
        mostrar_gif("ppo_1M.gif", "PPO com 1 milhão de passos")

    if os.path.exists(os.path.join(GIFS, "treino_progresso.gif")):
        st.subheader("Evolução durante o treino")
        mostrar_gif("treino_progresso.gif", "Mesmo cenário em vários momentos do treino")

    st.subheader("Onde o agente erra")
    c, d = st.columns(2)
    with c:
        mostrar_gif("dqn_300k_falha_seed23.gif", "DQN com 300 mil passos: episódio de falha")
    with d:
        mostrar_gif("ppo_sem_vento_testado_com_vento.gif", "PPO treinado sem vento, testado com vento")

    st.subheader("Com vento: melhor e pior episódio")
    e, f = st.columns(2)
    with e:
        mostrar_gif("ppo_vento_treinado_melhor.gif", "Melhor episódio de 30")
    with f:
        mostrar_gif("ppo_vento_treinado_pior_seed15.gif", "Pior episódio de 30 (semente 15)")

# ---------------------------------------------------------------- Ao vivo
with aba_ao_vivo:
    st.subheader("Teste ao vivo")

    def importar_rl():
        try:
            import gymnasium as gym
            from stable_baselines3 import DQN, PPO

            return gym, PPO, DQN, None
        except Exception as erro:  # biblioteca ausente ou Box2D não instalado
            return None, None, None, erro

    gym, PPO, DQN, erro = importar_rl()
    modelos = sorted(glob.glob(os.path.join(MODELS, "*.zip")))

    if erro is not None:
        st.warning(
            "A parte ao vivo precisa de gymnasium[box2d] e stable-baselines3, que não "
            "puderam ser carregados neste ambiente. As outras abas continuam funcionando."
        )
        st.code(str(erro))
    elif not modelos:
        st.info(
            "Coloque os modelos (arquivos .zip do Stable-Baselines3) na pasta "
            "assets/models, por exemplo ppo_1M.zip e ppo_base.zip."
        )
    else:
        nomes = [os.path.basename(m) for m in modelos]

        @st.cache_resource
        def carregar(caminho):
            classe = DQN if os.path.basename(caminho).lower().startswith("dqn") else PPO
            return classe.load(caminho, device="cpu")

        def criar_env(vento, render):
            kwargs = {}
            if render:
                kwargs["render_mode"] = "rgb_array"
            if vento:
                kwargs.update(enable_wind=True, wind_power=15.0, turbulence_power=1.5)
            return gym.make("LunarLander-v3", **kwargs)

        def rodar_episodio(modelo, vento, seed, gravar, max_passos=700):
            env = criar_env(vento, gravar)
            obs, _ = env.reset(seed=seed)
            frames, total, passos = [], 0.0, 0
            for _ in range(max_passos):
                if gravar:
                    frames.append(env.render())
                acao, _ = modelo.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(int(acao))
                total += r
                passos += 1
                if term or trunc:
                    if gravar:
                        frames.append(env.render())
                    break
            env.close()
            return frames, total, passos

        col1, col2, col3 = st.columns(3)
        escolhido = col1.selectbox("Modelo", nomes)
        vento = col2.checkbox("Ligar o vento", value=False)
        seed = col3.number_input("Semente do cenário", min_value=0, max_value=9999, value=7, step=1)
        modelo = carregar(os.path.join(MODELS, escolhido))

        b1, b2 = st.columns(2)
        if b1.button("▶️ Rodar um pouso"):
            import imageio

            with st.spinner("Simulando..."):
                frames, total, passos = rodar_episodio(modelo, vento, int(seed), gravar=True)
                buffer = io.BytesIO()
                imageio.mimsave(buffer, frames[::2], format="GIF", fps=30)
            st.image(buffer.getvalue(), caption=f"{escolhido} | vento: {'sim' if vento else 'não'} | semente {int(seed)}")
            st.metric("Recompensa do episódio", f"{total:.1f}", f"{passos} passos", delta_color="off")
            st.caption("Cada pouso é um único episódio. A média de vários episódios está no botão ao lado.")

        if b2.button("📊 Avaliar 20 episódios"):
            barra = st.progress(0.0)
            notas = []
            for i in range(20):
                _, total, _ = rodar_episodio(modelo, vento, i, gravar=False)
                notas.append(total)
                barra.progress((i + 1) / 20)
            serie = pd.Series(notas)
            st.metric("Recompensa média (20 episódios)", f"{serie.mean():.1f}", f"± {serie.std():.1f}", delta_color="off")
            st.bar_chart(serie.rename("Recompensa por episódio"))
