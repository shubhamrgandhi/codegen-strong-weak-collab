python agentless_lite/retrieve_swe.py \
        --dataset princeton-nlp/SWE-bench_Lite \
        --num_threads 1 \
        --output_folder results/base \
        --output_file retrieval.jsonl \
        --embedding_folder voyage_lite \
        --embedding_model voyage-code-3 \
        --filter_model text-embedding-3-small \
        --filter_python \
        --entire_file