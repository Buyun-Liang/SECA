import argparse

def create_parser(args=None):
    # Create the parser
    parser = argparse.ArgumentParser(description="Semantically Equivalent & Coherent Attacks for Eliciting LLM Hallucinations")

    # Add the arguments
    parser.add_argument('--mmlu_subject', type=str, default='machine_learning', help='MMLU subject')
    parser.add_argument('--mmlu_question_idx', type=int, default=0, help='MMLU question index')
    parser.add_argument('--max_iteration', type=int, default=30, help='Max number of iterations')
    parser.add_argument('--candidate_size_M', type=int, default=3, help='Candidate size M. See line 6 in algorithm 1 in the paper')
    parser.add_argument('--top_N_most_adversarial', type=int, default=3, help='Select top N most adversarial candidates. See line 10 in algorithm 1 in the paper')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--target_llm', type=str, default='llama3_8b', help='LLM model type')
    parser.add_argument('--num_attack_trials_K', type=int, default=1, help='for the best-of-K attack')

    parser.add_argument('--rng_seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--mmlu_dataset_split_type', type=str, default='test', help='Dataset split type')
    parser.add_argument('--termination_confidence_threshold', type=float, default=1.0, help='Termination threshold')

    # Parse the arguments
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args