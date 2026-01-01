python eval_math500.py \
    --scorer_type logprob \
    --model_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --n_particles 32 \
    --max_steps 50 \
    --max_tokens_per_step 128 \
    --temperature 0.7 \
    --output_file math500_logprob_results.jsonl \
    --llm_device cuda:0 \
    --tp 1 \
    --gpu_mem 0.9 \
    --alpha 0.2 \
    --beta 10.0 \
    --rho 1.0 \
    --logprob_normalize

