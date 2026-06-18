# 🍽️ Food Recommendation System

Pick a few foods and get combinations that actually go well together. A deep
recommender, a **LightGCN graph neural network**, learns ingredient
complementarity from around 240k [RecipeNLG](https://recipenlg.cs.put.poznan.pl/)
recipes and serves real-time suggestions.

![Demo](assets/demo.gif)

**Live demo:** <https://food-recommendation.heigonsoldera77.workers.dev/>

---

## What it does

You select ingredients you have or like; the system suggests other ingredients
that pair well with them. Instead of returning *substitutes* (what cosine
similarity over co-occurrence usually gives, e.g. `tomato → tomatoes`), it learns
*complementarity* (given a recipe with A, what B goes well with it?) by framing
the task as recommendation on the recipe → ingredient graph.

## How it works

- **Data.** RecipeNLG recipes are reduced to clean ingredient baskets, with
  canonicalization (merging variants like `fresh basil → basil`) and junk removal.
- **Model.** A **LightGCN** graph neural network is trained with a **BPR ranking
  loss** and **popularity-corrected negative sampling**, so pantry staples stop
  dominating and genuine pairings surface. Trained on GPU (PyTorch/CUDA).
- **Serving.** Ingredient embeddings are exported and served by a lightweight
  NumPy-only FastAPI backend; suggestions drop the picks, near-duplicate variants,
  staples and junk.

## Components

This repository ties the project together. Frontend and training are included as
**git submodules**; the backend runs on a Hugging Face Space.

| Component | Stack | Link |
|-----------|-------|------|
| **Training** (submodule) | Python, PyTorch, LightGCN | [Food-Recommendation-Training](https://github.com/heigon77/Food-Recommendation-Training) |
| **Backend** (Hugging Face Space) | FastAPI, NumPy | [Food-Recommendation-Backend](https://huggingface.co/spaces/heigon77/Food-Recommendation-Backend) |
| **Frontend** (submodule) | Angular | [Food-Recommendation-Frontend](https://github.com/heigon77/Food-Recommendation-Frontend) |

## Clone with submodules

```bash
git clone --recurse-submodules https://github.com/heigon77/Food-Recommendation.git

# already cloned without --recurse-submodules?
git submodule update --init --recursive
```

Each part has its own README with setup and run instructions.

## Architecture

```
[ Angular frontend ] --POST /recommend {foods:[...]}--> [ FastAPI @ HF Space ]
        ^                                                        |
        |                                          LightGCN embeddings (NumPy)
        +------------- complementary ingredients + score <-------+
                                                                 ^
                                            ( Training repo trains & exports them )
```
