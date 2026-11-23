import os
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from datetime import datetime


TASK_DIR = "cert_work"
timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
run_tag = f"{timestamp}"
MAX_EVALUATION_JOBS = 1
MAX_PROPOSAL_JOBS = 1
MAX_DB_WORKERS = 1

search_task_sys_msg = (
    "You are an expert machine learning researcher specializing in optimization algorithms and learning rate scheduling."
    "\nYOUR TASK:"
    "You are given a Python function `compute_lr(step, total_steps, ...)` that implements a learning rate scheduler. Your goal is to IMPROVE this scheduler to achieve lower validation loss when training a neural network."
    "\nCRITICAL CONSTRAINTS:"
    "The function signature MUST keep parameters: `step`, `total_steps`."
    "If your scheduler doesn't need `step`, `total_steps`, you can ignore them in the function body, but they must remain in the signature"
    "\n2. Must return a POSITIVE, FINITE float for any valid step."
    "LR must be > 0 (zero or negative LR will cause validation to fail)"
    "LR must be finite (no inf or NaN values)"
    "\n3. Must be DETERMINISTIC (same inputs → same output)"
    "\n4. LR is applied ONCE PER EPOCH (not per batch)."
    "total_steps corresponds to the number of training epochs"
    "step is the current epoch index (0-based)"
    "\nTRAINING CONTEXT (fixed, do not change):"
    "Dataset: MNIST (5000 train / 1000 validation samples)."
    "Model: CNN-based classifier,"
    "Optimizer: Adam,"
    "Loss: CrossEntropyLoss,"
    "Batch size: 128,"
    "Training duration: total_steps epochs = 50"
    "\nAPPROACH:"
    "1. Analyze the provided code: What strategy does it use? What are its strengths/weaknesses?"
    "2. Identify opportunities for improvement: Can the warmup be better? Is the decay optimal? Are the parameter values well-chosen?"
    "3. Implement your improved version while respecting all constraints."
    "4. Ensure your code is clean and deterministic.")

evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    # use all three mutation patch types and set prob for all
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.45, 0.45, 0.1],
    # runs for 20 generations in sequence
    num_generations=350,
    max_patch_resamples=3,  # resample 3 times if patch fails
    max_patch_attempts=3,  # try 3 times to fix patch via reflection
    # runs locally using the local environment (no loading of conda/docker)
    job_type="local",
    language="python",
    # set LLMs for ensemble
    llm_models=["local/qwen3.8-27b@http://127.0.0.1:9931/v1"],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],  # uniform temperature sampling
        max_tokens=16384,
    ),
    # no embedding model
    embedding_model=None,  # "text-embedding-3-small"
    # no meta scratchpad
    meta_rec_interval=None,  # e.g. every 5 generations
    meta_llm_models=None,  # e.g. ["gpt-4.1"]
    meta_llm_kwargs={},  # same as above
    # Set path to initial program relative to repo root
    init_program_path=f"{TASK_DIR}/initial.py",
    results_dir=f"results/cert_work/{run_tag}",
    # each mutation has three chances of providing a novel solution
    max_novelty_attempts=3,
    # ensemble llm selection among candidates based on past performance
    llm_dynamic_selection=None,  # i think this parameter isn't used in program because i have only one llm-model
)

db_config = DatabaseConfig(
   # db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=20,
    elite_selection_ratio=0.3, #x% лучших программ из текущего поколения автоматически переходят в следующее без мутаций.
    num_archive_inspirations=1,#При генерации мутации LLM получает n случайные программы из архива как примеры для вдохновения.
    num_top_k_inspirations=1,#При генерации мутации LLM получает m лучшие программы из текущего поколения как примеры
    migration_interval=10, #каждые n поколений лучшие программы с одного острова переходят в другой фактически заменяя какие-то индивиды
    migration_rate=0.1, #процент от популяции сколько программ перейдет с острова на остров
    island_elitism=True,#Включает элитарный отбор внутри каждого острова (не только глобально)
    enforce_island_separation=True,#запрещает программам с разных островов быть слишком похожими, чтобы работал нужен эмбединг
    parent_selection_strategy="weighted", #параметры селекции
    parent_selection_lambda=10.0,
)

job_config = LocalJobConfig(
    eval_program_path=f"{TASK_DIR}/evaluate_temp.py",
#    activate_script=".venv/bin/activate"
)

runner = ShinkaEvolveRunner(
    evo_config=evo_config,
    job_config=job_config,
    db_config=db_config,
    max_evaluation_jobs=MAX_EVALUATION_JOBS,
    max_proposal_jobs=MAX_PROPOSAL_JOBS,
    max_db_workers=MAX_DB_WORKERS,
    verbose=True,
    #debug=True
)

runner.run()

