import numpy as np
from my_utils import get_prompt, obj_fun, semantic_equivalence_proposer, get_final_answer, get_second_largest_choice_index, wrap_preserve_newlines, feasibility_check
from tqdm import tqdm

def get_new_parents_list(args, candidate_parent_list, parents_list):
    '''Generate a new parents list based on the candidate_parent_list and parents_list.'''
    # Sort candidate_parent_list by obj_value in descending order
    sorted_candidates = sorted(candidate_parent_list, key=lambda x: x[1], reverse=True)

    # Take top from candidate list (as many as available, max args.top_N_most_adversarial)
    top_from_candidates = sorted_candidates[:min(args.top_N_most_adversarial, len(sorted_candidates))]

    # If fewer than args.top_N_most_adversarial, fill in from parents_list (excluding duplicates)
    if len(top_from_candidates) < args.top_N_most_adversarial:

        # Sort parents_list by obj_value descending
        sorted_parents = sorted(parents_list, key=lambda x: x[1], reverse=True)

        # Filter out those already in top_from_candidates
        fill_from_parents = sorted_parents[:3 - len(top_from_candidates)]

        # Combine both
        new_parents_list = top_from_candidates + fill_from_parents
    else:
        new_parents_list = top_from_candidates

    parents_list = new_parents_list
    return parents_list

import re

def seca(
    args, mmlu_question, model, tokenizer, device,
    semantic_equivalence_proposer_LLM,
    feasibility_evaluator_model
):
    # === Step 1: Extract initial question data ===
    query_x0 = mmlu_question['question']  # Original prompt
    choices = mmlu_question['choices']  # Answer choices
    ground_truth_idx = mmlu_question['answer']  # Ground-truth answer index
    subject = mmlu_question['subject']  # Subject name
    
    # === Step 2: Get model prediction for original query ===
    query_x = query_x0  # Working query (can be modified in iterations)
    full_query = get_prompt(args, query_x, mmlu_question)
    response = get_final_answer(args, full_query, model, tokenizer, device)
    if args.verbose:
        print(f"Initial Answer: \n{wrap_preserve_newlines(response, width=120)}\n")

    # === Step 3: Determine target choice index for attack objective ===
    target_choice_index = get_second_largest_choice_index(args, full_query, model, tokenizer, device, ground_truth_idx) # the target choice index in our dataset is only for llama3-8b

    if target_choice_index == -1:
        print(f"Ground truth is not asscociated with the highest confidence. Skip this question.")
        return None, None, None, None, None, None, None

    if args.verbose:
        print(f"Original Query        : {query_x0}")
        print(f"Ground Truth Index    : {ground_truth_idx}")
        print(f"Target Choice Index   : {target_choice_index}")

    # === Step 4: Evaluate objective on original query ===
    obj_value, _ = obj_fun(args, full_query, target_choice_index, model, tokenizer, device)
    best_obj = obj_value
    best_query_x = query_x
    best_idx_tuple = (0, -1)  # (index, parent_index)

    if args.verbose:
        print(f"Initial Objective Value: {obj_value:.6f}")
        print(f"Initial Target Confidence: {np.exp(obj_value):.6f}")

    # === Step 5: Initialize parent list ===
    self_index = 0
    parents_list = [(query_x, obj_value, (0, -1), 1.0)] * args.top_N_most_adversarial
    all_parents_list = [parents_list]
    all_children_list = [parents_list]

    # === Step 6: Main attack generation loop ===
    for i in tqdm(range(args.max_iteration)):
        if args.verbose:
            print(f"Iteration: {i}")
        children_list = [] # args.top_N_most_adversarial * args.candidate_size_M children

        for parent in parents_list:
            for _ in range(args.candidate_size_M):    
                # === Generate a semantically equivalent adversarial query ===
                query_x = parent[0]
                parent_index = parent[2][0]

                query_x_new = semantic_equivalence_proposer(
                    query_x, subject, ground_truth_idx, choices,
                    semantic_equivalence_proposer_LLM, verbose=args.verbose
                )

                # === Evaluate the new query ===
                full_query = get_prompt(args, query_x_new, mmlu_question)
                obj_value_new, _ = obj_fun(args, full_query, target_choice_index, model, tokenizer, device)
                self_index += 1

                if args.verbose:
                    # === Print debug info ===
                    print(f"\nQuery #{self_index} (Parent: {parent_index})")
                    print(f"- New Query         : {query_x_new}")
                    print(f"- Objective Value   : {obj_value_new:.6f}")
                    print(f"- Target Confidence : {np.exp(obj_value_new):.6f}")

                # Store child
                children_list.append((query_x_new, obj_value_new, (self_index, parent_index)))
        
        all_children_list.append(children_list)
        
        # === Step 7: Keep only children with improved objective ===
        filtered_children = [(q, o, idx) for (q, o, idx) in children_list if o > best_obj]
        candidate_parent_list = []

        # === Step 8: Check feasibility and update best if improved ===
        for query_x, obj_value, idx_tuple in filtered_children:
            is_feasible = feasibility_check(
                query_x, query_x0, choices, subject, ground_truth_idx,
                feasibility_evaluator_model
            )

            if is_feasible:
                if args.verbose:
                    print(f"Feasible query accepted! Index: {idx_tuple[0]} (Parent: {idx_tuple[1]})")
                candidate_parent_list.append((query_x, obj_value, idx_tuple))

                # Update best if this is the highest scoring feasible query so far
                if obj_value >= best_obj:
                    best_obj = obj_value
                    best_query_x = query_x
                    best_idx_tuple = idx_tuple
                    if args.verbose:
                        print(f"New Best Query Found!")
                        print(f"- Query Index: {best_idx_tuple[0]} (Parent: {best_idx_tuple[1]})")
                        print(f"- Query      : {best_query_x}")
                        print(f"- Objective  : {best_obj:.6f}")
            else:
                if args.verbose:
                    print(f"Infeasible query rejected!")
                    print(f"- Query      : {best_query_x}")
            
        # === Step 9: Form next generation of parents ===
        parents_list = get_new_parents_list(args, candidate_parent_list, parents_list)
        all_parents_list.append(parents_list)

        if args.verbose:
            print(f"Updated Parents List ({len(parents_list)} items)")

        # === Step 10: Check termination condition ===
        if best_obj > np.log(args.termination_confidence_threshold):
            print(f"Termination criterion met! Best Objective: {best_obj:.6f}")
            break

    # === Step 11: Final evaluation and output ===
    print(f"================================================")
    print(f"Initial Query        : {query_x0}")
    print(f"Initial Objective Value: {all_parents_list[0][0][1]:.6f}")
    print(f"Initial Target Confidence: {np.exp(all_parents_list[0][0][1]):.6f}")
    print(f"Options: A. {choices[0]}, B. {choices[1]}, C. {choices[2]}, D. {choices[3]}")

    print(f"Ground Truth Index   : {ground_truth_idx}")
    print(f"Target Choice Index  : {target_choice_index}")

    print(f"================================================")
    print(f"Best Query           : {best_query_x}\n")
    print(f"Best Objective Value : {best_obj:.6f}\n")
    print(f"Best Target Confidence     : {np.exp(best_obj):.6f}\n")
    print(f"Best Query Index     : {best_idx_tuple[0]} (Parent: {best_idx_tuple[1]})\n")

    seca_result = {
        "all_parents_list": all_parents_list,
        "all_children_list": all_children_list,
        "best_query_x": best_query_x,
        "best_obj": best_obj,
        "best_idx_tuple": best_idx_tuple,
        "target_choice_index": target_choice_index,
        "ground_truth_idx": ground_truth_idx,
    }

    return  seca_result