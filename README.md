# laya-experiments

Notebooks para entender [Laya](https://huggingface.co/convaiinnovations/laya) midiéndola:
un modelo de decisión **no autoregresivo** (ModernBERT-large, 421M) que no genera texto —
devuelve probabilidades sobre preguntas que tú tipas, en ~25 ms en CPU.

Tres notebooks en español: cómo funciona, experimentos diagnósticos para saber si sus
números significan algo, y un caso real (enrutar mensajes a proyectos) con los resultados
medidos, incluidos **los que salieron mal**. El objetivo no es demostrar que el modelo
funciona, sino averiguar dónde funciona y dónde no.

## Arranque

```bash
uv venv --python 3.12 .venv-laya          # torch aún no trae wheels para 3.14
VIRTUAL_ENV=.venv-laya uv pip install -r requirements.txt
.venv-laya/bin/python -m ipykernel install --user --name laya --display-name "Python (laya)"
```

En VS Code el intérprete ya viene fijado en `.vscode/settings.json`. En los notebooks,
elige el kernel **Python (laya)**.

JupyterLab aparte:

```bash
.venv-laya/bin/jupyter lab
```

## Notebooks

| | qué hace |
|---|---|
| `01_laya_intro.ipynb` | cómo funciona: `choice` / `score` / `noul`, leer la salida, el Router, los tres límites |
| `02_experimentos.ipynb` | diagnósticos: redacción, control negativo, negación, trampa léxica, inyección de prompt, umbrales |
| `03_router_proyectos.ipynb` | prototipo: enrutar un WhatsApp a un proyecto, con prior bayesiano y fallback interactivo |

Corre el 01 y el 02 en orden. El 03 hay que editarlo con proyectos reales para que sirva.

## Hallazgos medidos

Todo esto sale de correr los notebooks en esta máquina, no del model card.

**Lo que funciona**
- Enrutar entre proyectos con vocabulario distinto: **7/7**.
- Clasificar bug / feature / pregunta / admin: **6/6**.
- Resistir inyección de prompt: no hay bucle de generación que secuestrar.

**Lo que no funciona**
- Decidir si un mensaje es de trabajo o personal: **5/10**, azar.
- `noul` abstractos (`"¿es de trabajo?"`, `"¿pide hacer algo?"`): invertidos o pegados a 0.
  Los `noul` concretos (`"threaten to cancel"` → 0.879) sí funcionan.
- Detectar "esto no es de ningún proyecto" por umbral: imposible. `choice` es un softmax
  de conjunto cerrado; fuera de dominio da distribuciones picudas y equivocadas
  (`"Acuérdate de comprar pan"` → `shop` con p=0.873, más confiado que un mensaje
  de proyecto real).
- `score` ordinales: colapsados a ~2.0–2.4 en todos los mensajes.

**El patrón:** discrimina bien por **vocabulario presente en el texto**, y mal cuando la
etiqueta exige un **juicio abstracto** sobre la naturaleza del mensaje.

**Lo que más sube la precisión, en orden:**
1. Descripciones de `choice` como **listas de keywords**, no prosa: 6/7 → 7/7, y los tokens
   caen de 150 a 81.
2. **Quitar la etiqueta `otros`**: 4/7 → 6/7. Es un atractor.
3. **Nombrar las etiquetas con palabras con significado**, no con códigos internos. El mismo
   banco con `acme_shop`/`fitleads` da 2/7, 4/7, 6/7 en las tres variantes; con
   `shop`/`leadgen`, 4/7, 6/7, 7/7. La clave del dict entra al modelo igual que su descripción.

El notebook 03 documenta también dos conclusiones que la medición **tumbó** (que añadir
keywords empeora, y que el checkpoint inglés va mejor): las dos eran artefactos de usar
etiquetas opacas.

## Lo que hay que saber antes de construir algo encima

- **≤10 opciones por `choice`.** El checkpoint trae la temperatura de calibración de 11+
  fuera de rango (salta un `RuntimeWarning` al cargar). Con más opciones la etiqueta sirve
  pero la probabilidad no, y cualquier umbral deja de significar nada.
- **`input_tokens` incluye tus preguntas y descripciones.** 512 en el checkpoint inglés;
  1024–8192 en el multilingüe. Las descripciones largas se comen el presupuesto.
- **Laya no tiene memoria.** Una pasada, sin estado. El historial va en tu código
  (prior sobre las probabilidades) o en un fine-tune.
- **Los `score` ordinales son el punto débil declarado.** Para ordenar, no para un `if`.
- **Sobreconfiado por defecto.** Calcula margen top-2 sobre `probabilities`; no te apoyes
  en el campo `confidence` (fórmula sin documentar).

## Ficheros

```
ejemplo_laya.py      script suelto de arranque rápido
notebooks/           los tres notebooks
decisiones.jsonl     historial del router (lo genera el 03; en .gitignore)
```

Los notebooks se publican **sin outputs** a propósito: se aprende ejecutándolos. Los tres
están verificados de principio a fin con `nbclient`, y las cifras de este README se
reproducen corriéndolos.

## Licencia y atribución

Este repositorio: MIT (ver [LICENSE](LICENSE)).

El modelo Laya es Apache 2.0, de [convaiinnovations](https://huggingface.co/convaiinnovations/laya)
([GitHub](https://github.com/NandhaKishorM/laya)). No está vinculado a este repositorio;
aquí solo se usa y se mide.
