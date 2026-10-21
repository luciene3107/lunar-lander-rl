# Pouso lunar com aprendizado por reforço (Questão 3)

Trabalho Final de Lógica Nebulosa — Questão 3 (Aprendizado por reforço)

- **Aluna:** Luciene Borges
- **Professor:** Prof. Mardônio França
- **App online:** https://lunar-lander-rl-kmefdcuv9hvbjlfv8rkkya.streamlit.app/

Um agente aprende, por tentativa e erro, a pousar uma nave no ambiente **LunarLander-v3** (Gymnasium). Foram comparados dois algoritmos, **DQN** e **PPO** (Stable-Baselines3), e feita uma variação própria: **vento**.

## O que o enunciado pede e onde está

| Item da Questão 3 | Onde ver |
|---|---|
| 3a) Ambiente: estados, ações e recompensas | Seção 1 do notebook e aba "Projeto" do app |
| 3a) Agente | Seções 3 e 4 do notebook (DQN e PPO) |
| 3b) Evolução do aprendizado ao longo dos episódios | Seções 5 e 6 (curvas de recompensa) e Seção 10 (GIF do treino) |
| 3b) Caminho final encontrado pelo agente | Seção 8 do notebook e aba "Demos" do app (GIFs) |

## O ambiente

- **Estado (8 valores):** posição x e y, velocidade em x e y, ângulo, velocidade angular e contato de cada perna com o chão.
- **Ações (4):** não fazer nada, propulsor esquerdo, propulsor principal, propulsor direito.
- **Recompensa:** positiva ao se aproximar da plataforma, reduzir a velocidade e encostar as pernas; **+100** ao pousar; negativa ao gastar combustível e ao se afastar ou inclinar; **-100** ao bater. Recompensa média acima de **200** indica pouso consistente.

## Resultados (semente de treino 42, 100 episódios de avaliação)

| Agente | Recompensa média |
|---|---|
| DQN, 300 mil passos | 69,3 ± 65,2 |
| PPO, 300 mil passos | 79,8 ± 106,7 |
| **PPO, 1 milhão de passos (modelo final)** | **245,4 ± 52,0** |

O "melhor modelo" escolhido pelo callback de avaliação chegou a 256,1 ± 31,0, mas essa escolha tem viés de seleção, então o resultado principal é o modelo final.

**Variação própria: vento** (`wind_power=15`, `turbulence_power=1.5`)

| Treinou ↓ / Testou → | Sem vento | Com vento |
|---|---|---|
| Sem vento (PPO 1M) | 253,4 ± 39,2 | 103,4 ± 131,9 |
| Com vento (PPO 1M) | 118,5 ± 74,2 | 93,3 ± 100,7 |
| Com vento (PPO 2M, final) | não testado | 27,3 ± 156,9 |

O vento derruba o agente treinado sem vento, e treinar com vento não superou esse agente.

**Limitação:** todos os treinos usaram uma única semente (42).

## Estrutura do repositório

```
app.py                       # app Streamlit (vitrine + teste ao vivo)
requirements.txt             # dependências do app
packages.txt                 # pacote de sistema (swig), usado pelo Streamlit Cloud
lunar_lander_rl.ipynb        # notebook com todo o código (treino, avaliação, GIFs, vento)
assets/
  gifs/                      # GIFs gravados (antes/depois, falhas, vento, evolução do treino)
  img/                       # gráfico de recompensa
  models/                    # modelos treinados (.zip): ppo_1M, ppo_base, dqn_base
```

## Como executar

### 1. Notebook (Google Colab) — recomendado para reproduzir o treino

1. Abra `lunar_lander_rl.ipynb` no [Google Colab](https://colab.research.google.com) (menu *Arquivo > Abrir notebook > GitHub*, ou envie o arquivo).
2. Execute as células na ordem. A primeira instala `swig`, `gymnasium[box2d]`, `stable-baselines3` e `imageio`, e monta o Google Drive.
3. Os modelos, logs e GIFs são salvos em `MyDrive/rl_lunar`.

Tempos aproximados no Colab: DQN e PPO de 300 mil passos, alguns minutos cada; continuação do PPO até 1 milhão, cerca de 25 minutos; seções de vento, cerca de 1 hora cada.

Não há conjunto de dados: o agente aprende por interação com o ambiente simulado.

### 2. App online

Abra o link do app indicado no topo. A parte "Teste ao vivo" roda os modelos de `assets/models` e permite ligar o vento e trocar a semente do cenário. O app gratuito hiberna após alguns dias sem uso. Se aparecer o botão "wake up", clique e aguarde cerca de 30 segundos.

### 3. App no seu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

O pacote `box2d-py` precisa de `swig` e de ferramentas de compilação C++, e não tem versão pronta para Python muito recentes (por exemplo 3.14 no Windows). Use **Python 3.11 ou 3.12**. Sem o Box2D, as abas "Projeto", "Resultados" e "Demos" funcionam normalmente e a aba "Teste ao vivo" mostra um aviso.

## Tecnologias

Python, Gymnasium (LunarLander-v3), Stable-Baselines3 (DQN e PPO), Streamlit, imageio, pandas, matplotlib.
